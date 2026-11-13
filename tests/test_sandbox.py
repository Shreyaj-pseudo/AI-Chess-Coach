"""Real loopback HTTP with mocked external services; no model or engine launches."""
import asyncio
import json
import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import chess.engine
import requests

from model_client import ModelConfigurationError
from sandbox_server import Sandbox, analysis_view, make_server
from test_T09_selection import evidence, position


PGN = "1. f3 e5 2. g4 Qh4# 0-1"


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = Sandbox()
        self.addCleanup(self.sandbox.close)

    def test_inspect_both_colors_and_selected_pre_move_frames(self):
        for color, plies in (("white", [1, 3]), ("black", [2, 4])):
            result = self.sandbox.execute(dict(action="inspect", pgn=PGN, color=color))
            self.assertEqual(result["total_plies"], 4)
            self.assertEqual([p["ply"] for p in result["positions"]], plies)
            self.assertTrue(all(p["fen"] == p["frame"]["fen"] for p in result["positions"]))

    def test_mate_line_replay_frames_and_concrete_facts(self):
        record = evidence(position(PGN, index=1), 1.5, line="g2g4 d8h4", played_score=chess.engine.Mate(-1))
        with patch.object(self.sandbox, "_analyze", new=AsyncMock(return_value=record)):
            result = self.sandbox.execute(dict(action="analyze", pgn=PGN, color="white", ply=3))
        line = next(l for l in result["lines"] if l["id"] == record.played.id)
        self.assertEqual([f["label"] for f in line["frames"]], ["Before move", "g4", "Qh4#"])
        self.assertIn(dict(kind="legal_check", subject="d8h4", value="Qh4#"), line["frames"][1]["facts"])
        self.assertIn(dict(kind="termination", subject="board", value="checkmate"), line["frames"][2]["facts"])
        self.assertEqual(self.sandbox.execute(dict(action="view", analysis_id=record.id)), result)

    def test_draft_uses_server_record_not_client_supplied_evidence(self):
        record = evidence(position(), 1.5)
        self.sandbox.records = {record.id: record}
        with patch("sandbox_server.ModelClient.from_env") as factory, patch("sandbox_server.generate_grounded_card") as generate:
            generate.return_value.model_dump.return_value = {"synthetic": True}
            result = self.sandbox.execute(dict(action="draft", analysis_id=record.id))
            generate.assert_called_once_with(factory.return_value, record)
        self.assertEqual(result, {"draft": {"synthetic": True}})
        with self.assertRaises(ValueError):
            self.sandbox.execute(dict(action="draft", analysis_id="unknown"))

    def test_external_sources_receive_selected_history_and_learner(self):
        for action, factory_name, method in (("opening", "OpeningExplorer", "lookup"), ("tablebase", "TablebaseClient", "probe")):
            with patch("sandbox_server." + factory_name) as factory:
                operation = getattr(factory.return_value.__enter__.return_value, method)
                operation.return_value.model_dump.return_value = {"status": "unavailable"}
                result = self.sandbox.execute(dict(action=action, pgn=PGN, color="white", ply=3))
                self.assertEqual(result[action]["status"], "unavailable")
                if action == "opening":
                    self.assertEqual(operation.call_args.args[0].history_uci, ("f2f3", "e7e5"))
                else:
                    self.assertEqual(len(operation.call_args.args[0].move_stack), 2)
                    self.assertEqual(operation.call_args.args[1], "white")

    def test_invalid_pgn_or_wrong_side_never_starts_engine(self):
        with patch.object(self.sandbox, "_analyze", new=AsyncMock()) as engine:
            for updates in ({"pgn": "1. e5 *"}, {"ply": 2}, {"color": "purple"}):
                with self.assertRaises(ValueError):
                    self.sandbox.execute({**dict(action="analyze", pgn=PGN, color="white", ply=3), **updates})
            engine.assert_not_awaited()

    def test_only_one_job_at_a_time_and_explicit_failure(self):
        entered, release = threading.Event(), threading.Event()
        def operation(payload):
            entered.set()
            release.wait(3)
            raise ValueError("Synthetic invalid input")
        with patch.object(self.sandbox, "execute", side_effect=operation):
            first = self.sandbox.submit({"action": "inspect"})
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaises(RuntimeError):
                    self.sandbox.submit({"action": "inspect"})
            finally:
                release.set()
            deadline = time.monotonic() + 3
            while self.sandbox.get_job(first)["status"] == "running" and time.monotonic() < deadline:
                time.sleep(.01)
        self.assertEqual(self.sandbox.get_job(first)["status"], "failed")
        self.assertEqual(self.sandbox.get_job(first)["error"], "Synthetic invalid input")


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.http = requests.Session()
        cls.http.trust_env = False

    @classmethod
    def tearDownClass(cls):
        cls.http.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.server.sandbox.close()
        cls.thread.join(3)

    def test_static_pages_security_headers_and_no_repository_exposure(self):
        page = self.http.get(self.url, timeout=3)
        self.assertEqual(page.status_code, 200)
        self.assertIn("Chess workshop", page.text)
        self.assertIn("script-src 'self'", page.headers["Content-Security-Policy"])
        for path in ("/app.js", "/app.css"):
            self.assertEqual(self.http.get(self.url + path, timeout=3).status_code, 200)
        for path in ("/.env", "/.git/config", "/model_client.py"):
            self.assertEqual(self.http.get(self.url + path, timeout=3).status_code, 404)

    def test_bad_host_origin_content_type_and_extra_fields_rejected(self):
        for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"}):
            response = self.http.post(self.url + "/api/run", json={"action": "engine"}, headers=headers, timeout=3)
            self.assertEqual(response.status_code, 403)
        response = self.http.post(self.url + "/api/run", data="not json", timeout=3)
        self.assertEqual(response.status_code, 400)
        response = self.http.post(self.url + "/api/run", json={"action": "draft", "api_key": "not-accepted"}, timeout=3)
        self.assertEqual(response.status_code, 400)

    def test_safe_status_never_contains_a_key(self):
        with patch("sandbox_server.ModelClient.from_env", side_effect=ModelConfigurationError("Not configured")):
            result = self.http.get(self.url + "/api/status", timeout=3).json()
        self.assertFalse(result["model"]["configured"])
        self.assertIn("mate", result["samples"])
        self.assertNotIn("api_key", json.dumps(result))

    def test_real_http_job_polling_and_svg_board(self):
        response = self.http.post(self.url + "/api/run", json=dict(action="inspect", pgn=PGN, color="white"), timeout=3)
        self.assertEqual(response.status_code, 202)
        deadline = time.monotonic() + 3
        while True:
            job = self.http.get(self.url + "/api/jobs/" + response.json()["id"], timeout=3).json()
            if job["status"] != "running" or time.monotonic() > deadline:
                break
            time.sleep(.01)
        self.assertEqual(job["status"], "done")
        board = self.http.get(self.url + "/api/board", params={"fen": job["result"]["positions"][1]["fen"]}, timeout=3)
        self.assertEqual(board.status_code, 200)
        self.assertIn("image/svg+xml", board.headers["Content-Type"])
        self.assertIn("<svg", board.text)
        self.assertEqual(self.http.get(self.url + "/api/board?fen=invalid", timeout=3).status_code, 400)


if __name__ == "__main__":
    unittest.main()
