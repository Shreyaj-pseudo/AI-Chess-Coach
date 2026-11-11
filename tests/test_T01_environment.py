"""Offline environment/configuration checks. Real UCI smoke is check_setup --engine."""

import importlib
from importlib.metadata import version
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.engine
import chess.pgn
import dotenv

import engine_config
from chess_part import ChessAnalyzer
from scripts import check_setup


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_supported_python_and_pinned_runtime_are_installed(self):
        self.assertEqual(sys.version_info[:2], (3, 12), "Use conda activate .\\.venv (see README.md)")
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_-]+)==([^\s]+)", line)
            self.assertIsNotNone(match, f"Runtime requirement is not exactly pinned: {line}")
            package, expected = match.groups()
            self.assertEqual(version(package), expected, f"Reinstall requirements.txt: {package}")

    def test_existing_modules_import_without_running_analysis(self):
        for module in ("AI_process", "chess_part", "opening_agent", "middlegame_agent", "endgame_agent", "data.data_collection"):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))

    def test_installed_chess_supports_legal_pgn_replay(self):
        game = chess.pgn.read_game(io.StringIO("1. e4 e5 2. Nf3 Nc6 *"))
        self.assertFalse(game.errors)
        board = game.board()
        moves = []
        for move in game.mainline_moves():
            self.assertIn(move, board.legal_moves)
            moves.append(board.san(move))
            board.push(move)
        self.assertEqual(moves, ["e4", "e5", "Nf3", "Nc6"])
        self.assertEqual(board.piece_at(chess.C6), chess.Piece(chess.KNIGHT, chess.BLACK))

    def test_analyzer_uses_the_shared_resolver_and_preserves_override(self):
        with patch("chess_part.resolve_stockfish_path", return_value="resolved-engine") as resolve:
            self.assertEqual(ChessAnalyzer().stockfish_path, "resolved-engine")
            resolve.assert_called_once_with(None)
            resolve.reset_mock()
            self.assertEqual(ChessAnalyzer(stockfish_path="custom-engine").stockfish_path, "resolved-engine")
            resolve.assert_called_once_with("custom-engine")

    def test_setup_check_reports_missing_engine_without_traceback(self):
        from contextlib import redirect_stderr, redirect_stdout

        output = io.StringIO()
        with patch.object(check_setup, "resolve_stockfish_path", side_effect=FileNotFoundError("set STOCKFISH_PATH")):
            with redirect_stderr(output), redirect_stdout(output):
                self.assertEqual(check_setup.main([]), 1)
        self.assertIn("STOCKFISH_PATH", output.getvalue())
        self.assertIn("README.md", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_setup_check_does_not_start_engine_without_flag(self):
        from contextlib import redirect_stdout

        with patch.object(check_setup, "resolve_stockfish_path", return_value="configured-engine"):
            with patch("chess.engine.SimpleEngine.popen_uci") as launch, redirect_stdout(io.StringIO()):
                self.assertEqual(check_setup.main([]), 0)
                launch.assert_not_called()

    def test_setup_check_reports_uci_failure_without_traceback(self):
        from contextlib import redirect_stderr, redirect_stdout

        output = io.StringIO()
        with patch.object(check_setup, "resolve_stockfish_path", return_value="configured-engine"):
            with patch("chess.engine.SimpleEngine.popen_uci", side_effect=chess.engine.EngineTerminatedError("not a UCI engine")):
                with redirect_stderr(output), redirect_stdout(output):
                    self.assertEqual(check_setup.main(["--engine"]), 1)
        self.assertIn("Stockfish UCI check failed", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


class EnginePathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.binary = self.root / "local-stockfish.exe"
        self.binary.write_bytes(b"configuration test fixture, not a runnable engine")
        for target, kwargs in (
            ("environment", {}),
            ("config", {"PROJECT_ROOT": self.root, "LOCAL_BINARY": self.binary}),
        ):
            context = patch.dict(os.environ, {}, clear=True) if target == "environment" else patch.multiple(engine_config, **kwargs)
            context.start()
            self.addCleanup(context.stop)
        for target, result in (("load_dotenv", False), ("shutil.which", None), ("os.access", True)):
            context = patch(f"engine_config.{target}", return_value=result)
            context.start()
            self.addCleanup(context.stop)

    def test_explicit_path_beats_invalid_environment_and_path_lookup(self):
        os.environ["STOCKFISH_PATH"] = "missing-env-engine"
        self.assertEqual(engine_config.resolve_stockfish_path(self.binary), str(self.binary))
        engine_config.shutil.which.assert_not_called()
        engine_config.load_dotenv.assert_not_called()

    def test_environment_path_is_resolved_from_project_root(self):
        os.environ["STOCKFISH_PATH"] = self.binary.name
        self.assertEqual(engine_config.resolve_stockfish_path(), str(self.binary))
        engine_config.load_dotenv.assert_called_once_with(self.root / ".env", override=False)

    def test_actual_project_dotenv_resolves_relative_binary(self):
        (self.root / ".env").write_text(f"STOCKFISH_PATH={self.binary.name}\n", encoding="utf-8")
        engine_config.load_dotenv.side_effect = dotenv.load_dotenv
        self.assertEqual(engine_config.resolve_stockfish_path(), str(self.binary))

    def test_shell_value_overrides_actual_project_dotenv(self):
        (self.root / ".env").write_text("STOCKFISH_PATH=missing-from-dotenv.exe\n", encoding="utf-8")
        os.environ["STOCKFISH_PATH"] = str(self.binary)
        engine_config.load_dotenv.side_effect = dotenv.load_dotenv
        self.assertEqual(engine_config.resolve_stockfish_path(), str(self.binary))

    def test_path_engine_precedes_local_download(self):
        path_binary = self.root / "path-stockfish.exe"
        path_binary.write_bytes(b"PATH fixture")
        engine_config.shutil.which.return_value = str(path_binary)
        self.assertEqual(engine_config.resolve_stockfish_path(), str(path_binary))

    def test_blank_environment_uses_local_fallback(self):
        os.environ["STOCKFISH_PATH"] = ""
        self.assertEqual(engine_config.resolve_stockfish_path(), str(self.binary))

    def test_invalid_override_does_not_silently_use_local_engine(self):
        for invalid in (self.root / "missing.exe", self.root):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(FileNotFoundError, "STOCKFISH_PATH"):
                engine_config.resolve_stockfish_path(invalid)
        os.environ["STOCKFISH_PATH"] = "missing-env.exe"
        with self.assertRaisesRegex(FileNotFoundError, "STOCKFISH_PATH"):
            engine_config.resolve_stockfish_path()

    def test_missing_automatic_engine_has_actionable_error(self):
        with patch.object(engine_config, "LOCAL_BINARY", self.root / "missing.exe"):
            with self.assertRaisesRegex(FileNotFoundError, "Install Stockfish on PATH"):
                engine_config.resolve_stockfish_path()

    def test_nonexecutable_file_is_rejected(self):
        engine_config.os.access.return_value = False
        with self.assertRaisesRegex(PermissionError, "execute permissions"):
            engine_config.resolve_stockfish_path(self.binary)


if __name__ == "__main__":
    unittest.main()
