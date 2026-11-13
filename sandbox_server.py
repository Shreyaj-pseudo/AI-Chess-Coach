"""Loopback-only test workbench for the existing T01–T10 services."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

import chess
import chess.svg

from analysis_records import PositionContext, source_game_id
from board_facts import extract_board_facts
from candidate_analysis import analyze_position
from engine_config import resolve_stockfish_path
from engine_session import StockfishSession
from game_review import review_game
from model_client import ModelClient, ModelConfigurationError
from grounded_cards import generate_grounded_card
from opening_explorer import OpeningExplorer
from pgn_input import MAX_PGN_BYTES, learner_positions, parse_pgn
from tablebase import TablebaseClient


WEB = Path(__file__).parent / "sandbox_web"
FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "pgn"
SAMPLES = {"mate": "played_blunder.pgn", "castling": "castling.pgn", "endgame": "tablebase_mate.pgn"}
MAX_BODY = 2 * MAX_PGN_BYTES


def board_frame(board, label="Before move", last=None):
    facts = extract_board_facts(board)
    return dict(fen=board.fen(), label=label, last=last,
                facts=[f.model_dump(mode="json") for f in facts if f.kind in {
                    "turn", "in_check", "termination", "material_points", "legal_check", "legal_capture",
                    "legal_check_count", "legal_capture_count", "absolute_pin", "passed_pawn", "isolated_pawn", "doubled_pawn",
                } and (f.kind not in {"absolute_pin", "passed_pawn", "isolated_pawn", "doubled_pawn"} or f.value is True)])


def analysis_view(record):
    lines = []
    seen = set()
    for candidate in (*record.candidates, record.played):
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        board = record.position.board()
        frames = [board_frame(board)]
        for uci in candidate.pv_uci:
            san = board.san(chess.Move.from_uci(uci))
            board.push_uci(uci)
            frames.append(board_frame(board, san, uci))
        lines.append(dict(id=candidate.id, frames=frames))
    return dict(analysis=record.model_dump(mode="json"), lines=lines)


class Sandbox:
    def __init__(self):
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chess-sandbox")
        self.job = None
        self.records = {}

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)

    def submit(self, payload):
        with self.lock:
            if self.job and self.job["status"] == "running":
                raise RuntimeError("An operation is already running. Wait for it to finish.")
            self.job = dict(id=uuid.uuid4().hex, status="running", action=payload.get("action"))
            job_id = self.job["id"]
        self.worker.submit(self._run, payload, job_id)
        return job_id

    def get_job(self, job_id):
        with self.lock:
            if self.job is None or self.job["id"] != job_id:
                return None
            return dict(self.job)

    def _run(self, payload, job_id):
        try:
            result = self.execute(payload)
            update = dict(status="done", result=result)
        except (ValueError, RuntimeError, OSError) as error:
            update = dict(status="failed", error=str(error) if not isinstance(error, OSError) else "A required local resource could not be accessed.")
        except Exception:
            update = dict(status="failed", error="The operation failed unexpectedly. No result was produced.")
        with self.lock:
            if self.job["id"] == job_id:
                self.job.update(update)

    def execute(self, payload):
        action = payload.get("action")
        if action == "engine":
            return asyncio.run(self._engine_check())
        if action in ("view", "draft"):
            record = self.records.get(payload.get("analysis_id"))
            if record is None:
                raise ValueError("Analyze a position or select a review moment first.")
            if action == "view":
                return analysis_view(record)
            result = generate_grounded_card(ModelClient.from_env(), record)
            return dict(draft=result.model_dump(mode="json"))
        if action not in ("inspect", "analyze", "review", "opening", "tablebase"):
            raise ValueError("Unknown sandbox operation.")
        pgn, color = payload.get("pgn"), payload.get("color")
        if not isinstance(pgn, str):
            raise ValueError("Paste a PGN game first.")
        game = parse_pgn(pgn)
        decisions = learner_positions(game, color)
        if action == "inspect":
            return dict(total_plies=game.end().ply() - game.ply(),
                        positions=[dict(ply=d.ply, san=d.san, fen=d.fen, frame=board_frame(d.board)) for d in decisions],
                        frame=board_frame(decisions[0].board if decisions else game.board()))
        if action == "review":
            result = asyncio.run(self._review(pgn, color))
            self.records = {c.analysis.id: c.analysis for c in result.deep_checks}
            return dict(review=result.model_dump(mode="json"))
        ply = payload.get("ply")
        if type(ply) is not int:
            raise ValueError("Choose a learner decision first.")
        decision = next((d for d in decisions if d.ply == ply), None)
        if decision is None:
            raise ValueError("This ply is not a decision for the chosen learner.")
        position = PositionContext.from_decision(decision, game_id=source_game_id(game))
        if action == "analyze":
            record = asyncio.run(self._analyze(position))
            self.records = {record.id: record}
            return analysis_view(record)
        if action == "opening":
            with OpeningExplorer() as explorer:
                return dict(opening=explorer.lookup(position).model_dump(mode="json"))
        with TablebaseClient() as client:
            return dict(tablebase=client.probe(decision.board, color).model_dump(mode="json"))

    @staticmethod
    async def _analyze(position):
        async with StockfishSession() as session:
            return await analyze_position(session, position)

    @staticmethod
    async def _review(pgn, color):
        async with StockfishSession() as session:
            return await review_game(session, pgn, color)

    @staticmethod
    async def _engine_check():
        async with StockfishSession() as session:
            await session.analyse(chess.Board(), time_limit=.1)
            pid = session.pid
            board = chess.Board()
            board.push_uci("e2e4")
            await session.analyse(board, time_limit=.1)
            reused = pid == session.pid
            name = session.engine_name
        return dict(engine=name, reused=reused, closed=session.pid is None and session.returncode == 0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # PGNs, model output and secrets are never written to request logs.

    def reply(self, status, value, content_type="application/json; charset=utf-8"):
        body = json.dumps(value, ensure_ascii=False).encode() if content_type.startswith("application/json") else value
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        return (self.headers.get("Host") in hosts
                and self.headers.get("Origin") in (None, *(f"http://{h}" for h in hosts)))

    def do_GET(self):
        if not self.local_request():
            return self.reply(403, {"error": "Use the local sandbox URL."})
        url = urlsplit(self.path)
        if url.path == "/favicon.ico":
            return self.reply(200, chess.svg.piece(chess.Piece(chess.KNIGHT, chess.BLACK)).encode(), "image/svg+xml")
        if url.path in ("/", "/app.css", "/app.js"):
            name, mime = {"/": ("index.html", "text/html"), "/app.css": ("app.css", "text/css"), "/app.js": ("app.js", "text/javascript")}[url.path]
            return self.reply(200, (WEB / name).read_bytes(), mime + "; charset=utf-8")
        if url.path == "/api/status":
            try:
                resolve_stockfish_path()
                engine = True
            except (OSError, ValueError, RuntimeError):
                engine = False
            try:
                client = ModelClient.from_env()
                model = dict(configured=True, backend=client.config.backend, model=client.config.model)
            except ModelConfigurationError:
                model = dict(configured=False)
            return self.reply(200, dict(engine_found=engine, model=model,
                                       samples={key: (FIXTURES / name).read_text(encoding="utf-8") for key, name in SAMPLES.items()}))
        if url.path.startswith("/api/jobs/"):
            job = self.server.sandbox.get_job(url.path.rsplit("/", 1)[-1])
            return self.reply(200 if job else 404, job or {"error": "That operation is no longer available."})
        if url.path == "/api/board":
            try:
                query = parse_qs(url.query)
                board = chess.Board(query["fen"][0])
                if not board.is_valid():
                    raise ValueError
                last = chess.Move.from_uci(query["last"][0]) if query.get("last") else None
                svg = chess.svg.board(board, size=480, orientation=query.get("color", ["white"])[0] == "white",
                                      lastmove=last, check=board.king(board.turn) if board.is_check() else None,
                                      colors={"square light": "#eee9da", "square dark": "#789283", "margin": "#eee9da", "coord": "#20372f"})
                return self.reply(200, svg.encode(), "image/svg+xml")
            except (ValueError, KeyError):
                return self.reply(400, {"error": "Invalid board position."})
        return self.reply(404, {"error": "Not found."})

    def do_POST(self):
        if not self.local_request():
            return self.reply(403, {"error": "Use the local sandbox URL."})
        if self.path != "/api/run":
            return self.reply(404, {"error": "Not found."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply(400, {"error": "Send a JSON operation under 512 KiB."})
            self.connection.settimeout(5)
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict) or set(payload) - {"action", "pgn", "color", "ply", "analysis_id"}:
                raise ValueError
            job_id = self.server.sandbox.submit(payload)
            return self.reply(202, {"id": job_id})
        except RuntimeError as error:
            return self.reply(409, {"error": str(error)})
        except (ValueError, OSError):
            return self.reply(400, {"error": "Invalid JSON operation. Check the input and try again."})


def make_server(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.sandbox = Sandbox()
    return server
