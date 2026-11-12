"""Synthetic provider metrics; real tablebase responses are checked separately."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

import chess
import chess.variant
from pydantic import ValidationError
import requests

import lichess_http
import opening_explorer
from scripts import inspect_tablebase as cli
from tablebase import ENDPOINT, TablebaseClient, TablebaseContext


FEN = "7k/8/8/8/8/8/R7/K7 w - - 0 1"
FIXTURES = Path(__file__).parent / "fixtures/pgn"


def raw_value(board, category, dtz, dtm=None, precise=True):
    if board.is_checkmate():
        category, dtz, dtm = "loss", -1, 0
    elif board.is_stalemate() or board.is_insufficient_material():
        category, dtz, dtm = "draw", 0, 0
    return dict(category=category, dtz=dtz, precise_dtz=dtz if precise else None, dtm=dtm,
                checkmate=board.is_checkmate(), stalemate=board.is_stalemate(),
                insufficient_material=board.is_insufficient_material(), variant_win=False, variant_loss=False)


def payload(board=None, category="win", child_category="loss", dtz=5, dtm=None, precise=True):
    board = chess.Board(FEN) if board is None else board
    raw = raw_value(board, category, dtz, dtm, precise)
    raw["moves"] = []
    for move in board.legal_moves:
        child = board.copy(stack=True)
        child.push(move)
        child_dtz = 0 if child_category == "draw" else -abs(dtz) if "loss" in child_category else abs(dtz)
        raw["moves"].append(dict(uci=move.uci(), san=board.san(move), zeroing=board.is_zeroing(move),
                                 **raw_value(child, child_category, child_dtz, precise=precise)))
    return raw


class Response:
    def __init__(self, raw=None, *, status=200, body=None, headers=None):
        self.status_code = status
        self.body = json.dumps(payload() if raw is None else raw).encode() if body is None else body
        self.headers = headers or {}
        self.close = Mock()

    def iter_content(self, chunk_size):
        for index in range(0, len(self.body), chunk_size):
            yield self.body[index:index + chunk_size]


class TablebaseTests(unittest.TestCase):
    def setUp(self):
        limiter = patch.object(lichess_http, "COOLDOWN_UNTIL", 0.0)
        limiter.start()
        self.addCleanup(limiter.stop)
        self.transport = Mock()
        self.transport.get.side_effect = lambda *a, **kw: Response(payload(chess.Board(kw["params"]["fen"])))
        factory = patch("tablebase.requests.Session", return_value=self.transport)
        factory.start()
        self.addCleanup(factory.stop)
        self.client = TablebaseClient()
        self.addCleanup(self.client.close)

    def probe(self, board=None, learner="white"):
        return self.client.probe(chess.Board(FEN) if board is None else board, learner)

    def test_supported_probe_uses_actual_clock_and_public_standard_endpoint(self):
        board = chess.Board(FEN)
        board.halfmove_clock = 17
        result = self.probe(board)
        call = self.transport.get.call_args
        self.assertEqual(call.args, (ENDPOINT,))
        self.assertEqual(call.kwargs["params"], {"fen": board.fen(), "dtc": "never"})
        self.assertEqual(call.kwargs["timeout"], (3.0, 5.0))
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertNotIn("Authorization", call.kwargs["headers"])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.outcome, "win")
        self.assertFalse(result.engine_fallback)
        self.assertEqual(result.rules.halfmove_clock, 17)
        self.assertEqual({m.uci for m in result.moves}, {m.uci() for m in board.legal_moves})

    def test_more_than_seven_pieces_and_castling_rights_skip_http(self):
        positions = (chess.Board(), chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1"))
        for board in positions:
            with self.subTest(fen=board.fen()):
                result = self.probe(board)
                self.assertEqual(result.status, "ineligible")
                self.assertEqual(result.outcome, "unknown")
                self.assertTrue(result.engine_fallback)
        self.transport.get.assert_not_called()

    def test_seven_piece_boundary_is_supported_but_eight_is_not(self):
        board = chess.Board("7k/8/8/8/8/8/PPPPP3/KR6 w - - 0 1")
        self.assertEqual(len(board.piece_map()), 8)
        self.assertEqual(self.probe(board).status, "ineligible")
        board.remove_piece_at(chess.E2)
        self.assertEqual(len(board.piece_map()), 7)
        self.assertEqual(self.probe(board).status, "ok")
        self.assertEqual(self.transport.get.call_count, 1)

    def test_terminal_checkmate_resolves_locally_from_either_learner_side(self):
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        board.push_uci("g6g7")
        for learner, expected in (("white", "win"), ("black", "loss")):
            with self.subTest(learner=learner):
                result = self.probe(board, learner)
                self.assertEqual((result.status, result.outcome), ("terminal", expected))
                self.assertFalse(result.engine_fallback)
                self.assertIsNone(result.value)
                self.assertIsNone(result.query_fen)
                self.assertEqual(result.rules.termination, "checkmate")
        self.transport.get.assert_not_called()

    def test_stalemate_insufficient_material_and_automatic_75_move_draw_skip_searches(self):
        for fen in ("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", "7k/8/8/8/8/8/8/K7 w - - 0 1",
                    FEN.replace("0 1", "150 76")):
            with self.subTest(fen=fen):
                result = self.probe(chess.Board(fen))
                self.assertEqual((result.status, result.outcome), ("terminal", "draw"))
                self.assertFalse(result.engine_fallback)
        self.transport.get.assert_not_called()

    def test_fifth_repetition_is_terminal_and_history_is_preserved(self):
        board = chess.Board(FEN)
        for move in ("a2a3", "h8g8", "a3a2", "g8h8") * 4:
            board.push_uci(move)
        result = self.probe(board)
        self.assertEqual(result.rules.termination, "fivefold_repetition")
        self.assertEqual(result.rules.repetition_count, 5)
        self.assertEqual(result.outcome, "draw")
        self.assertEqual(result.history_uci, tuple(m.uci() for m in board.move_stack))
        self.transport.get.assert_not_called()

    def test_black_learner_and_child_side_to_move_are_normalized_correctly(self):
        result = self.probe(learner="black")
        self.assertEqual((result.value.source_category, result.value.source_pov), ("win", "white"))
        self.assertEqual((result.value.category, result.value.pov, result.outcome), ("loss", "black", "loss"))
        self.assertEqual(result.value.dtz_plies, -5)
        child = next(m for m in result.moves if m.rules.termination is None)
        self.assertEqual((child.value.source_category, child.value.source_pov), ("loss", "black"))
        self.assertEqual((child.value.category, child.value.pov, child.value.outcome), ("loss", "black", "loss"))
        white = self.probe()
        white_child = next(m for m in white.moves if m.rules.termination is None)
        self.assertEqual(white_child.value.source_category, "loss")
        self.assertEqual(white_child.value.category, "win")
        self.assertEqual(white_child.value.dtz_plies, 5)

    def test_black_to_move_can_be_the_winning_learner(self):
        result = self.probe(chess.Board(FEN).mirror(), "black")
        self.assertEqual((result.value.source_pov, result.value.pov), ("black", "black"))
        self.assertEqual(result.outcome, "win")

    def test_cursed_win_and_blessed_loss_preserve_category_but_are_drawn_under_50_move_rule(self):
        for category, child, dtz in (("cursed-win", "blessed-loss", 101), ("blessed-loss", "cursed-win", -101)):
            self.transport.get.side_effect = lambda *a, **kw: Response(payload(category=category, child_category=child, dtz=dtz))
            with self.subTest(category=category):
                result = self.probe()
                self.assertEqual(result.status, "rule_dependent")
                self.assertEqual(result.value.category, category)
                self.assertEqual(result.outcome, "draw")
                self.assertEqual(result.value.dtz_plies, dtz)
                self.assertIsNone(result.value.dtm_plies)

    def test_rounded_dtz_and_unknown_categories_do_not_become_certain_wins(self):
        for category in ("syzygy-win", "syzygy-loss", "maybe-win", "maybe-loss", "unknown"):
            distance = -7 if "loss" in category else 7
            self.transport.get.side_effect = lambda *a, **kw: Response(payload(category=category, dtz=distance, precise=False))
            with self.subTest(category=category):
                result = self.probe()
                self.assertEqual(result.outcome, "unknown")
                self.assertTrue(result.engine_fallback)
                self.assertTrue(result.value.rule_dependent)
                self.assertIsNone(result.value.precise_dtz_plies)

    def test_known_root_with_unknown_move_requests_fallback_for_incomplete_comparison(self):
        raw = payload()
        raw["moves"][0]["category"] = "unknown"
        self.transport.get.side_effect = lambda *a, **kw: Response(raw)
        result = self.probe()
        self.assertEqual(result.outcome, "win")
        self.assertEqual(result.moves[0].value.outcome, "unknown")
        self.assertTrue(result.engine_fallback)
        self.assertIn("Use engine analysis", result.reason)

    def test_fifty_move_claim_is_an_option_not_an_automatic_draw_or_decisive_verdict(self):
        for clock in (99, 100):
            board = chess.Board(FEN)
            board.halfmove_clock = clock
            with self.subTest(clock=clock):
                result = self.probe(board)
                self.assertTrue(result.rules.can_claim_fifty_moves)
                self.assertIsNone(result.rules.termination)
                self.assertEqual(result.value.source_category, "win")
                self.assertEqual(result.outcome, "unknown")
                self.assertTrue(result.engine_fallback)
                self.assertEqual(result.query_fen, board.fen())

    def test_draw_category_remains_draw_even_with_claimable_context(self):
        board = chess.Board(FEN)
        board.halfmove_clock = 100
        self.transport.get.side_effect = lambda *a, **kw: Response(payload(board, "draw", "draw", 0))
        result = self.probe(board)
        self.assertEqual(result.outcome, "draw")
        self.assertTrue(result.value.rule_dependent)

    def test_current_threefold_and_prior_reversible_repetition_withhold_decisive_result(self):
        for cycles in (1, 2):
            board = chess.Board(FEN)
            for move in ("a2a3", "h8g8", "a3a2", "g8h8") * cycles:
                board.push_uci(move)
            with self.subTest(cycles=cycles):
                result = self.probe(board)
                self.assertTrue(result.rules.repetition_in_reversible_history)
                self.assertEqual(result.rules.repetition_count, cycles + 1)
                self.assertEqual(result.rules.can_claim_threefold_repetition, cycles == 2)
                self.assertEqual(result.outcome, "unknown")
                self.assertEqual(result.status, "rule_dependent")

    def test_zeroing_pawn_move_resets_child_clock_and_draw_context(self):
        board = chess.Board("7k/8/8/8/8/8/P7/1K6 w - - 99 1")
        result = self.probe(board)
        pawn_move = next(m for m in result.moves if m.uci == "a2a4")
        self.assertTrue(pawn_move.zeroing)
        self.assertEqual(pawn_move.rules.halfmove_clock, 0)
        self.assertFalse(pawn_move.rules.can_claim_fifty_moves)
        self.assertFalse(pawn_move.rules.repetition_in_reversible_history)
        self.assertEqual(pawn_move.value.outcome, "win")

    def test_en_passant_is_preserved_and_zeroing_flag_is_verified(self):
        board = chess.Board("7k/8/8/3pP3/8/8/8/7K w - d6 0 2")
        result = self.probe(board)
        self.assertEqual(self.transport.get.call_args.kwargs["params"]["fen"], board.fen())
        move = next(m for m in result.moves if m.uci == "e5d6")
        self.assertTrue(move.zeroing)
        self.assertIsNone(chess.Board(move.fen_after).piece_at(chess.D5))

    def test_dtz_is_not_mate_distance_and_actual_mating_child_is_checked_locally(self):
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        self.transport.get.side_effect = lambda *a, **kw: Response(payload(board, dtz=8, dtm=None))
        result = self.probe(board)
        self.assertEqual(result.value.dtz_plies, 8)
        self.assertIsNone(result.value.dtm_plies)
        self.assertNotIn('"kind": "mate"', result.model_dump_json())
        mate = next(m for m in result.moves if m.uci == "g6g7")
        self.assertEqual(mate.san, "Qg7#")
        self.assertEqual(mate.rules.termination, "checkmate")
        self.assertEqual(mate.value.outcome, "win")

    def test_fifty_move_context_does_not_override_checkmate_after_an_actual_move(self):
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 100 1")
        result = self.probe(board)
        self.assertEqual(result.outcome, "unknown")
        mate = next(m for m in result.moves if m.uci == "g6g7")
        self.assertEqual(mate.value.outcome, "win")
        self.assertEqual(mate.rules.termination, "checkmate")

    def test_timeout_bad_json_status_and_oversize_return_explicit_engine_fallback(self):
        for value in (requests.Timeout("synthetic timeout"), Response(body=b"not JSON"), Response(status=404),
                      Response(status=503), Response(status=302), Response(body=b"x" * (256 * 1024 + 1))):
            with self.subTest(value=value):
                self.transport.get.side_effect = value if isinstance(value, Exception) else None
                self.transport.get.return_value = value
                result = self.probe()
                self.assertEqual(result.status, "unavailable")
                self.assertEqual(result.outcome, "unknown")
                self.assertTrue(result.engine_fallback)
                self.assertIn("Use engine analysis", result.reason)
                if isinstance(value, Response):
                    self.assertTrue(value.close.called)

    def test_malformed_distances_flags_and_moves_are_not_accepted_as_endgame_truth(self):
        for mode in ("bool", "precision", "sign", "category", "illegal", "san", "duplicate", "missing", "zeroing", "mate", "variant", "outcome"):
            raw = payload()
            if mode == "bool": raw["dtz"] = True
            elif mode == "precision": raw["precise_dtz"] = 8
            elif mode == "sign": raw["dtz"] = raw["precise_dtz"] = -5
            elif mode == "category": raw["category"] = "made-up"
            elif mode == "illegal": raw["moves"][0]["uci"] = "a2a9"
            elif mode == "san": raw["moves"][0]["san"] = "Qh4#"
            elif mode == "duplicate": raw["moves"].append(raw["moves"][0])
            elif mode == "missing": raw["moves"].pop()
            elif mode == "zeroing": raw["moves"][0]["zeroing"] = True
            elif mode == "mate": raw["checkmate"] = True
            elif mode == "variant": raw["variant_win"] = True
            else:
                raw["category"] = "draw"
                raw["dtz"] = raw["precise_dtz"] = 0
            self.transport.get.side_effect = lambda *a, **kw: Response(raw)
            with self.subTest(mode=mode):
                self.assertEqual(self.probe().status, "unavailable")

    def test_rate_limit_is_shared_with_opening_adapter_without_blocking_sleep(self):
        self.transport.get.side_effect = None
        self.transport.get.return_value = Response(status=429, headers={"Retry-After": "120"})
        result = self.probe()
        self.assertEqual(result.status, "rate_limited")
        self.assertEqual(result.retry_after_seconds, 120)
        self.assertGreater(lichess_http.COOLDOWN_UNTIL, time.monotonic())
        self.assertIs(opening_explorer._REQUEST_LOCK, lichess_http.REQUEST_LOCK)
        self.assertEqual(self.probe().status, "rate_limited")
        self.assertEqual(self.transport.get.call_count, 1)

    def test_invalid_variant_boards_and_illegal_history_fail_before_http(self):
        null_history = chess.Board(FEN)
        null_history.push(chess.Move.null())
        for board in (chess.Board(None), chess.Board(chess960=True), chess.variant.CrazyhouseBoard(), null_history, "fen"):
            with self.subTest(board=board), self.assertRaises(ValueError):
                self.probe(board)
        with self.assertRaises(ValueError):
            self.probe(learner="both")
        self.transport.get.assert_not_called()

    def test_json_ids_and_board_history_are_preserved_and_false_derived_fields_rejected(self):
        board = chess.Board(FEN)
        board.push_uci("a2a3")
        before = (board.fen(), tuple(board.move_stack))
        result = self.probe(board)
        self.assertEqual(before, (board.fen(), tuple(board.move_stack)))
        self.assertEqual(TablebaseContext.model_validate_json(result.model_dump_json()), result)
        for mode in ("pov", "outcome", "clock", "child", "fallback"):
            raw = result.model_dump(mode="json", exclude={"id"})
            if mode == "pov": raw["value"]["pov"] = "black"
            elif mode == "outcome": raw["outcome"] = "draw"
            elif mode == "clock": raw["rules"]["halfmove_clock"] = 80
            elif mode == "child": raw["moves"][0]["fen_after"] = FEN
            else: raw["engine_fallback"] = True
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                TablebaseContext.model_validate(raw)

    def test_close_is_idempotent_permanent_and_cli_uses_no_engine(self):
        output = io.StringIO()
        with patch("chess.engine.UciProtocol.popen") as launch, redirect_stdout(output):
            code = cli.main([str(FIXTURES / "tablebase_mate.pgn"), "--color", "white", "--ply", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(TablebaseContext.model_validate_json(output.getvalue()).status, "ok")
        launch.assert_not_called()
        self.client.close()
        self.client.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.probe()

    def test_cli_rejects_bad_inputs_without_requesting_tablebases(self):
        for file, ply in ((FIXTURES / "absent.pgn", "1"), (FIXTURES / "tablebase_mate.pgn", "2")):
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = cli.main([str(file), "--color", "white", "--ply", ply])
            self.assertEqual(code, 1)
            self.assertIn("Tablebase inspection failed", errors.getvalue())
        self.transport.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
