"""Strict ingestion, history-safe learner positions, and original PGN fixtures."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import chess
import chess.pgn

from chess_part import ChessAnalyzer
from pgn_input import MAX_PGN_BYTES, PGNError, learner_positions, parse_pgn
from scripts import inspect_pgn


FIXTURES = Path(__file__).parent / "fixtures" / "pgn"


def fixture(name):
    return parse_pgn((FIXTURES / f"{name}.pgn").read_bytes())


class LegalReplayTests(unittest.TestCase):
    def test_all_fixtures_have_legal_independent_premove_snapshots(self):
        for path in sorted(FIXTURES.glob("*.pgn")):
            game = parse_pgn(path.read_bytes())
            for color in ("white", "black"):
                with self.subTest(fixture=path.name, color=color):
                    selected = {position.ply: position for position in learner_positions(game, color)}
                    board = game.board()
                    for ply, move in enumerate(game.mainline_moves(), start=1):
                        if board.turn == (color == "white"):
                            position = selected[ply]
                            self.assertEqual(position.fen, board.fen())
                            self.assertEqual(position.san, board.san(move))
                            self.assertEqual(position.uci, move.uci())
                            self.assertIn(position.played_move, position.board.legal_moves)
                            self.assertEqual(position.board.move_stack, board.move_stack)
                            self.assertIsNot(position.board, board)
                            prefix = parse_pgn(position.pgn_prefix)
                            self.assertEqual(prefix.end().board().fen(), board.fen())
                            self.assertEqual(list(prefix.mainline_moves()), list(board.move_stack))
                        board.push(move)
                    self.assertEqual(board.fen(), game.end().board().fen())

    def test_white_and_black_selection_is_explicit(self):
        game = parse_pgn("1.e4 e5 2.Nf3 Nc6*")
        self.assertEqual([p.ply for p in learner_positions(game, "White")], [1, 3])
        self.assertEqual([p.ply for p in learner_positions(game, "black")], [2, 4])
        for invalid in (None, True, "", "both", "red"):
            with self.subTest(color=invalid), self.assertRaisesRegex(PGNError, "learner color"):
                learner_positions(game, invalid)

    def test_castling_is_replayed_for_both_colors(self):
        game = fixture("castling")
        white = next(p for p in learner_positions(game, "white") if p.san == "O-O")
        self.assertEqual(white.ply, 9)
        board = white.board.copy()
        board.push(white.played_move)
        self.assertEqual(board.king(chess.WHITE), chess.G1)
        self.assertEqual(board.piece_at(chess.F1), chess.Piece(chess.ROOK, chess.WHITE))
        black = learner_positions(game, "black")[-1]
        self.assertEqual(black.san, "O-O")
        self.assertEqual(game.end().board().king(chess.BLACK), chess.G8)

    def test_en_passant_removes_the_captured_pawn(self):
        position = learner_positions(fixture("en_passant"), "white")[-1]
        self.assertEqual(position.ply, 5)
        self.assertEqual(position.uci, "e5d6")
        self.assertTrue(position.board.is_en_passant(position.played_move))
        position.board.push(position.played_move)
        self.assertIsNone(position.board.piece_at(chess.D5))
        self.assertEqual(position.board.piece_at(chess.D6), chess.Piece(chess.PAWN, chess.WHITE))

    def test_setup_promotion_preserves_initial_board_and_number(self):
        game = fixture("promotion")
        position = learner_positions(game, "white")[0]
        self.assertEqual(position.board.fullmove_number, 42)
        self.assertEqual(position.san, "a8=Q+")
        self.assertEqual(position.uci, "a7a8q")
        self.assertEqual(parse_pgn(position.pgn_prefix).board().fen(), game.board().fen())
        self.assertEqual(game.end().board().piece_at(chess.A8), chess.Piece(chess.QUEEN, chess.WHITE))

    def test_black_to_move_setup_uses_actual_turn_not_ply_parity(self):
        game = fixture("black_setup")
        self.assertEqual([p.ply for p in learner_positions(game, "black")], [1])
        white = learner_positions(game, "white")[0]
        self.assertEqual(white.ply, 2)
        self.assertEqual(white.board.fullmove_number, 43)
        self.assertIn("42... a1=Q+", white.pgn_prefix)

    def test_repetition_history_is_retained_not_reconstructed_from_fen(self):
        position = learner_positions(fixture("repetition"), "white")[-1]
        self.assertEqual(position.ply, 9)
        self.assertEqual(len(position.board.move_stack), 8)
        self.assertTrue(position.board.is_repetition(3))
        self.assertFalse(chess.Board(position.fen).is_repetition(3))

    def test_mate_ends_the_game_and_does_not_leak_into_prefix(self):
        game = fixture("checkmate")
        self.assertTrue(game.end().board().is_checkmate())
        self.assertEqual(game.headers["Result"], "0-1")
        last = learner_positions(game, "black")[-1]
        self.assertEqual(last.san, "Qh4#")
        self.assertNotIn("Qh4", last.pgn_prefix)
        self.assertNotIn("0-1", last.pgn_prefix)

    def test_positions_do_not_share_mutable_board_history(self):
        positions = learner_positions(fixture("castling"), "white")
        later_fen = positions[-1].fen
        positions[0].board.push(positions[0].played_move)
        self.assertEqual(positions[-1].fen, later_fen)

    def test_comments_headers_variations_and_nags_are_not_coaching_context(self):
        game = fixture("comments")
        self.assertEqual([p.san for p in learner_positions(game, "white")], ["e4", "Nf3"])
        for position in learner_positions(game, "black"):
            self.assertNotIn("Ignore", position.pgn_prefix)
            self.assertNotIn("reveal", position.pgn_prefix)
            self.assertNotIn("d4", position.pgn_prefix)
            self.assertNotIn("Event", position.pgn_prefix)
        for node in game.mainline():
            self.assertFalse(node.comment)
            self.assertFalse(node.nags)
            self.assertLessEqual(len(node.variations), 1)

    def test_ignored_variation_need_not_have_legal_moves(self):
        game = parse_pgn("1. e4 (1. impossible (more ignored text)) e5 *")
        self.assertEqual(len(list(game.mainline_moves())), 2)

    def test_bom_utf8_and_windows_newlines(self):
        raw = '\ufeff[Event "échecs"]\r\n\r\n1. e4 e5 *'.encode("utf-8")
        game = parse_pgn(raw)
        self.assertEqual(game.headers["Event"], "échecs")
        self.assertEqual(len(list(game.mainline_moves())), 2)

    def test_incomplete_legal_movetext_and_result_only_game(self):
        self.assertEqual(len(list(parse_pgn("1.e4 e5").mainline_moves())), 2)
        self.assertEqual(learner_positions(parse_pgn("*"), "white"), [])

    def test_terminal_setup_can_have_zero_review_decisions(self):
        terminal = fixture("checkmate").end().board().fen()
        game = parse_pgn(f'[SetUp "1"]\n[FEN "{terminal}"]\n\n0-1')
        self.assertTrue(game.board().is_checkmate())
        self.assertEqual(learner_positions(game, "white"), [])


class RejectedInputTests(unittest.TestCase):
    def test_unknown_tokens_and_illegal_moves_are_not_silently_skipped(self):
        for text in ("banana", "1. e4 banana e5 *", "1. e5 *", "1. e4 e4 *", "1. Ke2 *", "1. e9 *"):
            with self.subTest(pgn=text), self.assertRaises(PGNError):
                parse_pgn(text)

    def test_null_moves_are_not_legal_public_game_moves(self):
        for token in ("--", "Z0", "0000", "@@@@"):
            with self.subTest(token=token), self.assertRaises(PGNError):
                parse_pgn(f"1. {token} *")

    def test_false_capture_markers_are_rejected(self):
        for text in ("1.Nxf3 *", "1.e4 d5 2.ed5 *"):
            with self.subTest(pgn=text), self.assertRaisesRegex(PGNError, "mainline move"):
                parse_pgn(text)

    def test_empty_nontext_and_invalid_utf8_are_rejected(self):
        for value in ("", " \r\n", '{only a comment}', '[Event "headers only"]', None, 12, b"\xff", "\ud800"):
            with self.subTest(value=repr(value)), self.assertRaises(PGNError):
                parse_pgn(value)

    def test_byte_limit_counts_utf8_bytes_and_accepts_exact_boundary(self):
        base = "1. e4 *"
        valid = base + " " * (MAX_PGN_BYTES - len(base))
        self.assertEqual(len(list(parse_pgn(valid).mainline_moves())), 1)
        self.assertEqual(len(list(parse_pgn(valid.encode()).mainline_moves())), 1)
        for value in (valid + " ", (valid + " ").encode(), '{' + 'é' * (MAX_PGN_BYTES // 2) + '} 1. e4 *'):
            with self.subTest(length=len(value)), self.assertRaisesRegex(PGNError, "256 KiB"):
                parse_pgn(value)

    def test_multiple_games_and_trailing_movetext_are_rejected(self):
        for text in ("1.e4 * 1.d4 *", "1.e4 e5\n\n1.d4 d5 *", "1.e4 *\n\n[Event \"second\"]\n\n1.d4 *"):
            with self.subTest(pgn=text), self.assertRaisesRegex(PGNError, "exactly one game"):
                parse_pgn(text)

    def test_unbalanced_comments_and_variations_are_rejected(self):
        for text in ("1.e4 {oops", "1.e4 } e5 *", "1.e4 (1.d4 d5", "1.e4 ) e5 *", "(1.d4) 1.e4 *"):
            with self.subTest(pgn=text), self.assertRaises(PGNError):
                parse_pgn(text)

    def test_malformed_duplicate_or_late_headers_are_rejected(self):
        for text in ('[Broken]\n1.e4 *', '[Event "a"]\n[Event "b"]\n1.e4 *', '1.e4\n[Event "late"]\ne5 *'):
            with self.subTest(pgn=text), self.assertRaises(PGNError):
                parse_pgn(text)

    def test_unsupported_variants_are_rejected(self):
        for variant in ("Chess960", "Crazyhouse", "Antichess", "not-a-variant"):
            with self.subTest(variant=variant), self.assertRaisesRegex(PGNError, "Only standard chess"):
                parse_pgn(f'[Variant "{variant}"]\n\n*')

    def test_invalid_and_inconsistent_setup_headers_are_rejected(self):
        fen = '7k/P7/8/8/8/8/8/7K w - - 0 42'
        for headers in ('[SetUp "1"]', f'[FEN "{fen}"]', f'[SetUp "0"]\n[FEN "{fen}"]',
                        '[SetUp "2"]', '[SetUp "1"]\n[FEN "not a FEN"]',
                        '[SetUp "1"]\n[FEN "8/8/8/8/8/8/8/7k w - - 0 1"]'):
            with self.subTest(headers=headers), self.assertRaises(PGNError):
                parse_pgn(headers + "\n\n*")

    def test_wrong_or_dangling_move_numbers_are_rejected(self):
        for text in ("2.e4 *", "1...e4 *", "1.e4 1.e5 *", "1.", "1. *", "1.e4 1..."):
            with self.subTest(pgn=text), self.assertRaisesRegex(PGNError, "[Mm]ove number"):
                parse_pgn(text)

    def test_wrong_check_annotations_and_result_claims_are_rejected(self):
        for text in ("1.e4+ *", "1.e4# *", '[Result "0-1"]\n\n1.e4 1-0', '[Result "won"]\n\n*',
                     '1.f3 e5 2.g4 Qh4# 1-0'):
            with self.subTest(pgn=text), self.assertRaises(PGNError):
                parse_pgn(text)

    def test_moves_after_checkmate_and_automatic_draw_are_rejected(self):
        with self.assertRaisesRegex(PGNError, "terminal"):
            parse_pgn("1.f3 e5 2.g4 Qh4# 3.e4 *")
        cycles = " ".join(f"{2*n+1}.Nf3 Nf6 {2*n+2}.Ng1 Ng8" for n in range(4))
        self.assertTrue(parse_pgn(cycles + " *").end().board().is_fivefold_repetition())
        with self.assertRaisesRegex(PGNError, "terminal"):
            parse_pgn(cycles + " 9.Nf3 *")

    def test_200_ply_boundary_uses_played_count_not_fullmove_number(self):
        rng = random.Random(12)
        board = chess.Board()
        for _ in range(201):
            choices = list(board.legal_moves)
            rng.shuffle(choices)
            for move in choices:
                board.push(move)
                if not board.is_game_over():
                    break
                board.pop()
            else:
                self.fail("Could not build the deterministic long-game boundary fixture")
        exported = chess.pgn.Game.from_board(board)
        text = exported.accept(chess.pgn.StringExporter(headers=False, comments=False, variations=False))
        with self.assertRaisesRegex(PGNError, "200 plies"):
            parse_pgn(text)
        board.pop()
        text = chess.pgn.Game.from_board(board).accept(chess.pgn.StringExporter(headers=False))
        self.assertEqual(len(list(parse_pgn(text).mainline_moves())), 200)


class IntegrationTests(unittest.TestCase):
    def test_existing_opening_helper_uses_strict_parser(self):
        with patch("chess_part.resolve_stockfish_path", return_value="not-launched"):
            analyzer = ChessAnalyzer()
        self.assertEqual(analyzer.opening_moves("1.e4 e5 2.Nf3 Nc6 *"), ["e4", "e5", "Nf3", "Nc6"])
        with self.assertRaises(PGNError):
            analyzer.opening_moves("1.e4 garbage e5 *")

    def test_cli_selects_color_and_replays_without_engine_or_network(self):
        output = io.StringIO()
        with patch("chess.engine.SimpleEngine.popen_uci", side_effect=AssertionError("No engine in T02")):
            with patch("requests.get", side_effect=AssertionError("No API in T02")), redirect_stdout(output):
                result = inspect_pgn.main([str(FIXTURES / "castling.pgn"), "--color", "white"])
        self.assertEqual(result, 0)
        self.assertIn("14 plies; 7 white decisions", output.getvalue())
        self.assertIn("Ply 9: O-O (e1g1)", output.getvalue())

    def test_cli_missing_file_is_actionable(self):
        output = io.StringIO()
        with redirect_stderr(output):
            result = inspect_pgn.main([str(FIXTURES / "missing.pgn"), "--color", "black"])
        self.assertEqual(result, 1)
        self.assertIn("PGN rejected", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
