"""Selection policy fixtures use invented scores, not measured chess strength."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import random
import unittest
from unittest.mock import AsyncMock, patch

import chess
import chess.engine
import chess.pgn
from pydantic import ValidationError

from analysis_records import AnalysisRecord, Candidate, PositionContext, SearchProvenance, source_game_id
from candidate_analysis import BOUND_SCORE_WARNING
from engine_session import EngineSessionError
from game_review import GameReview, _reasons, _same_lesson, review_game
from pgn_input import learner_positions, parse_pgn
from scripts import review_game as cli


PGN = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 d6 *"


def position(pgn=PGN, color="white", index=0):
    game = parse_pgn(pgn)
    return PositionContext.from_decision(learner_positions(game, color)[index], game_id=source_game_id(game))


def evidence(pos, seconds, *, loss=0, strong=False, line=None, notes=(), best_score=None, played_score=None, depth=12):
    board = pos.board()
    actual = chess.Move.from_uci(pos.played_uci)
    others = [m for m in board.legal_moves if m != actual]
    best = actual if strong or not others else others[0]

    def candidate(move, score, pv=None):
        moves = [move] if pv is None else [chess.Move.from_uci(u) for u in pv.split()]
        after = board.copy(stack=True)
        for item in moves:
            after.push(item)
        if pv is None and not after.is_game_over():
            moves.append(next(iter(after.legal_moves)))
        search = SearchProvenance(engine="Synthetic selection fixture", config_sha256="a" * 64,
                                  root_fen=pos.fen, time_limit_seconds=float(seconds), depth=depth,
                                  nodes=100, elapsed_seconds=float(seconds))
        return Candidate.from_engine(board, {"pv": moves, "score": chess.engine.PovScore(score, board.turn)},
                                     learner_color=pos.learner_color, search=search)

    top = candidate(best, best_score if best_score is not None else chess.engine.Cp(100), line if best == actual else None)
    played = top if best == actual else candidate(actual, played_score if played_score is not None else chess.engine.Cp(100 - loss), line)
    candidates = [top]
    if strong and others:
        candidates.append(candidate(others[0], chess.engine.Cp(100 - loss)))
    return AnalysisRecord(position=pos, candidates=tuple(candidates), played=played, uncertainty=notes)


def long_game():
    rng = random.Random(19)
    board = chess.Board()
    for _ in range(84):
        if board.is_game_over():
            raise AssertionError("Long fixture unexpectedly ended")
        board.push(rng.choice(list(board.legal_moves)))
    return str(chess.pgn.Game.from_board(board))


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_candidate_pipeline_recovers_incomplete_scan_before_selection(self):
        from test_T05_candidates import row, session, three
        mating = row("g2g4 d8h4", score=chess.engine.PovScore(chess.engine.Mate(-1), chess.WHITE))
        alternatives = [row("e2e4 b8c6", 0)]
        engine = session(three(), [row("f2f3")], three(), [row("f2f3 e7e5", -50)],
                         alternatives, [mating], alternatives, [mating],
                         three(), [row("f2f3 e7e5", -50)])
        result = await review_game(engine, "1. f3 e5 2. g4 Qh4# 0-1", "white")
        self.assertFalse(result.partial)
        self.assertEqual(result.failures, ())
        self.assertEqual(len(result.scanned), 2)
        self.assertEqual(len(result.deep_checks), 2)
        self.assertEqual(result.scanned[0].analysis.played.pv_uci, ("f2f3", "e7e5"))
        self.assertTrue(any("recovered" in note for note in result.scanned[0].analysis.uncertainty))
        self.assertEqual(engine.analyse.await_count, 10)
        self.assertEqual(result.scanned[0].analysis.played.search.time_limit_seconds, .5)
        self.assertEqual(GameReview.model_validate_json(result.model_dump_json()), result)

    async def run_review(self, pgn=PGN, color="white", callback=None):
        async def analyze(session, pos, *, time_limit):
            return callback(pos, time_limit) if callback else evidence(pos, time_limit)
        with patch("game_review.analyze_position", side_effect=analyze) as mock:
            result = await review_game(object(), pgn, color)
        return result, mock

    async def test_quiet_game_has_no_padding_or_deep_searches_both_colors(self):
        for color, plies in (("white", [1, 3, 5, 7]), ("black", [2, 4, 6, 8])):
            result, mock = await self.run_review(color=color)
            self.assertEqual([s.analysis.position.ply for s in result.scanned], plies)
            self.assertEqual((result.moments, result.deep_checks, result.top_three), ((), (), ()))
            self.assertEqual(mock.call_count, 4)
            self.assertFalse(result.partial)

    async def test_near_equality_below_threshold_is_not_a_lesson(self):
        result, _ = await self.run_review(callback=lambda p, t: evidence(p, t, loss=99))
        self.assertFalse(result.moments)

    async def test_deep_confirmation_drops_shallow_noise_and_ranks_deep_losses(self):
        def trace(p, t):
            loss = 200 if t == 0.15 else {1: 20, 3: 150, 5: 450, 7: 250}[p.ply]
            return evidence(p, t, loss=loss, notes=("Finite search fixture uncertainty.",))
        result, mock = await self.run_review(callback=trace)
        self.assertEqual([a.position.ply for a in result.selected_analyses], [5, 7, 3])
        self.assertEqual(len(result.deep_checks), 4)
        self.assertEqual(mock.call_count, 8)
        self.assertEqual(result.top_three, result.selected_analyses)
        for moment, record in zip(result.moments, result.selected_analyses):
            self.assertEqual(moment.analysis_id, record.id)
            self.assertEqual(record.played.pv_uci[0], record.position.played_uci)
            self.assertGreaterEqual(len(record.played.pv_uci), 2)
            self.assertEqual(record.played.search.time_limit_seconds, 1.5)
            self.assertTrue(record.uncertainty)

    async def test_equal_rank_is_chronological_and_repeatable(self):
        callback = lambda p, t: evidence(p, t, loss=150)
        first, _ = await self.run_review(callback=callback)
        second, _ = await self.run_review(callback=callback)
        self.assertEqual(first.id, second.id)
        self.assertEqual([a.position.ply for a in first.selected_analyses], [1, 3, 5, 7])

    async def test_adjacent_shared_lesson_is_removed_from_final_selection(self):
        def trace(p, t):
            lines = {1: "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5", 3: "g1f3 b8c6 f1c4 f8c5"}
            return evidence(p, t, loss=200 if p.ply in lines else 0, line=lines.get(p.ply))
        result, _ = await self.run_review(callback=trace)
        self.assertEqual(len(result.deep_checks), 2)
        self.assertEqual([a.position.ply for a in result.selected_analyses], [1])

    async def test_persistent_bound_signal_is_deep_checked_but_not_selected(self):
        result, mock = await self.run_review(callback=lambda p, t: evidence(p, t, loss=200, notes=(BOUND_SCORE_WARNING,)))
        self.assertEqual(mock.call_count, 8)
        self.assertEqual(len(result.deep_checks), 4)
        self.assertTrue(all(s.reasons == ("evaluation_loss",) for s in result.scanned))
        self.assertFalse(result.moments)

    async def test_uncertain_f3_loss_gets_deep_confirmation(self):
        def trace(p, t):
            if p.ply == 1:
                return evidence(p, t, loss=118, notes=(BOUND_SCORE_WARNING,) if t == .15 else ())
            return evidence(p, t, line="g2g4 d8h4", played_score=chess.engine.Mate(-1))
        result, mock = await self.run_review(pgn="1. f3 e5 2. g4 Qh4# 0-1", callback=trace)
        self.assertEqual(mock.call_count, 4)
        self.assertEqual(len(result.deep_checks), 2)
        self.assertEqual([a.position.ply for a in result.selected_analyses], [3, 1])
        self.assertEqual(result.moments[1].reasons, ("evaluation_loss",))
        self.assertEqual(result.selection_version, "t09-v2")
        self.assertEqual(GameReview.model_validate_json(result.model_dump_json()), result)

    async def test_uncertain_loss_disappears_after_deep_search(self):
        result, _ = await self.run_review(callback=lambda p, t: evidence(
            p, t, loss=118 if t == .15 else 20, notes=(BOUND_SCORE_WARNING,) if t == .15 else ()))
        self.assertEqual(len(result.deep_checks), 4)
        self.assertFalse(result.moments)

    async def test_shallow_low_depth_signal_can_be_confirmed(self):
        result, _ = await self.run_review(callback=lambda p, t: evidence(p, t, loss=150, depth=5 if t == .15 else 12))
        self.assertEqual(len(result.deep_checks), 4)
        self.assertEqual(len(result.moments), 4)

    async def test_uncertain_near_equal_scores_do_not_trigger_deep_search(self):
        result, mock = await self.run_review(callback=lambda p, t: evidence(p, t, loss=99, notes=(BOUND_SCORE_WARNING,)))
        self.assertEqual(mock.call_count, 4)
        self.assertFalse(result.deep_checks)

    async def test_uncertain_shortlist_still_has_thirty_position_cap(self):
        result, mock = await self.run_review(pgn=long_game(), callback=lambda p, t: evidence(
            p, t, loss=200, notes=(BOUND_SCORE_WARNING,)))
        self.assertEqual(len(result.deep_checks), 30)
        self.assertEqual(mock.call_count, 72)
        self.assertFalse(result.moments)

    async def test_no_decisions_and_black_setup(self):
        empty, mock = await self.run_review(pgn="*")
        self.assertEqual(mock.call_count, 0)
        self.assertEqual(empty.total_plies, 0)
        setup = '[SetUp "1"]\n[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 12"]\n\n12... e5 13. e4 *'
        result, _ = await self.run_review(pgn=setup, color="black")
        self.assertEqual([s.analysis.position.ply for s in result.scanned], [1])
        self.assertEqual(result.total_plies, 2)

    async def test_deep_and_card_caps_and_top_three(self):
        result, mock = await self.run_review(pgn=long_game(), callback=lambda p, t: evidence(p, t, loss=200))
        self.assertEqual(len(result.scanned), 42)
        self.assertEqual(len(result.deep_checks), 30)
        self.assertEqual(len(result.moments), 15)
        self.assertEqual(len(result.top_three), 3)
        self.assertEqual(mock.call_count, 72)

    async def test_invalid_scan_is_explicit_partial_not_zero_score(self):
        def trace(p, t):
            if p.ply == 3:
                raise ValueError("PV continues after automatic draw")
            return evidence(p, t)
        result, _ = await self.run_review(callback=trace)
        self.assertTrue(result.partial)
        self.assertEqual([(f.position.ply, f.stage) for f in result.failures], [(3, "scan")])
        self.assertEqual(len(result.scanned), 3)
        self.assertFalse(result.moments)

    async def test_invalid_deep_evidence_does_not_fall_back_to_scan(self):
        def trace(p, t):
            if t == 1.5:
                raise ValueError("Deep metadata rejected")
            return evidence(p, t, loss=200)
        result, _ = await self.run_review(callback=trace)
        self.assertEqual(len(result.failures), 4)
        self.assertTrue(all(f.stage == "deep" for f in result.failures))
        self.assertFalse(result.moments)
        self.assertTrue(result.partial)

    async def test_engine_crash_and_cancellation_propagate(self):
        for error in (EngineSessionError("Engine died"), asyncio.CancelledError()):
            with patch("game_review.analyze_position", new=AsyncMock(side_effect=error)) as mock:
                with self.assertRaises(type(error)):
                    await review_game(object(), PGN, "white")
                self.assertEqual(mock.await_count, 1)

    async def test_invalid_budgets_pgn_and_color_never_search(self):
        with patch("game_review.analyze_position", new=AsyncMock()) as mock:
            for scan, deep in ((0, 1), (1, 1), (2, 1), (.1, 6), (True, 1), (.1, float("nan")), (.1, float("inf"))):
                with self.assertRaises(ValueError):
                    await review_game(object(), PGN, "white", scan_time=scan, deep_time=deep)
            for pgn, color in (("1. e5 *", "white"), (PGN, "purple")):
                with self.assertRaises(ValueError):
                    await review_game(object(), pgn, color)
            mock.assert_not_awaited()

    async def test_engine_wrong_budget_is_recorded_as_failure(self):
        result, _ = await self.run_review(callback=lambda p, t: evidence(p, 0.2))
        self.assertEqual(len(result.failures), 4)
        self.assertTrue(result.partial)

    async def test_round_trip_and_tamper_rejection(self):
        result, _ = await self.run_review(callback=lambda p, t: evidence(p, t, loss=200))
        self.assertEqual(GameReview.model_validate_json(result.model_dump_json()), result)
        for field, value in (("scanned", []), ("deep_checks", []), ("moments", []), ("partial", True), ("game_id", "f" * 64)):
            data = result.model_dump(mode="json", exclude={"id"})
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                GameReview.model_validate(data)
        with self.assertRaises(ValidationError):
            GameReview.model_validate_json(result.model_dump_json().replace(result.id, "f" * 64, 1))


class ReasonTests(unittest.TestCase):
    def test_good_quiet_decision_requires_meaningful_alternative(self):
        pos = position()
        self.assertEqual(_reasons(evidence(pos, 1.5, strong=True, loss=150)), ("strong_decision",))
        self.assertEqual(_reasons(evidence(pos, 1.5, strong=True, loss=99)), ())

    def test_forcing_choice_capture_context(self):
        pos = position("1. e4 d5 2. exd5 *", index=1)
        self.assertEqual(_reasons(evidence(pos, 1.5, strong=True, loss=150)), ("forcing_choice",))

    def test_castling_opening_heuristic_does_not_claim_theory(self):
        pos = position("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O *", index=3)
        self.assertEqual(_reasons(evidence(pos, 1.5)), ("opening_moment",))

    def test_mating_line_and_mate_swing_without_fake_cp(self):
        pos = position("1. f3 e5 2. g4 Qh4# 0-1", index=1)
        record = evidence(pos, 1.5, line="g2g4 d8h4", played_score=chess.engine.Mate(-1))
        self.assertEqual(_reasons(record), ("mate_swing", "mating_line"))
        self.assertIsNone(record.loss_cp)
        black = position("1. f3 e5 2. g4 Qh4# 0-1", color="black", index=1)
        record = evidence(black, 1.5, strong=True, line="d8h4", best_score=chess.engine.Mate(1))
        self.assertIn("mating_line", _reasons(record))

    def test_mate_distance_change_is_not_cp_loss_or_mate_swing(self):
        record = evidence(position(), 1.5, best_score=chess.engine.Mate(2), played_score=chess.engine.Mate(4))
        self.assertEqual(_reasons(record), ())
        self.assertIsNone(record.loss_cp)

    def test_bound_shallow_missing_or_contradictory_scores_not_promoted(self):
        pos = position()
        for kwargs in (dict(notes=("e4: " + BOUND_SCORE_WARNING,)), dict(depth=7),
                       dict(played_score=chess.engine.Cp(200), notes=("Unstable comparison",))):
            self.assertEqual(_reasons(evidence(pos, 1.5, loss=200, **kwargs)), ())
        raw = evidence(pos, 1.5, loss=200).model_dump(mode="json", exclude={"id"})
        raw["played"].pop("id")
        raw["played"]["score"] = {"kind": "unavailable", "pov": "white", "reason": "Missing"}
        self.assertEqual(_reasons(AnalysisRecord.model_validate(raw)), ())

    def test_shared_continuation_deduplicates_but_independent_error_remains(self):
        first = evidence(position(), 1.5, loss=200, line="e2e4 e7e5 g1f3 b8c6 f1c4 f8c5")
        same = evidence(position(index=1), 1.5, loss=200, line="g1f3 b8c6 f1c4 f8c5")
        different = evidence(position(index=1), 1.5, loss=200, line="g1f3 g8f6")
        self.assertTrue(_same_lesson(first, same))
        self.assertTrue(_same_lesson(same, first))
        self.assertFalse(_same_lesson(first, different))


class CliTests(unittest.TestCase):
    def test_json_success_and_actionable_error(self):
        result = asyncio.run(SelectionTests().run_review())[0]
        with patch.object(cli, "_review", new=AsyncMock(return_value=result)), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(["tests/fixtures/pgn/castling.pgn", "--color", "white"]), 0)
        self.assertEqual(GameReview.model_validate_json(output.getvalue()), result)
        with redirect_stderr(io.StringIO()) as error:
            self.assertEqual(cli.main(["missing-T09-file.pgn", "--color", "white"]), 1)
        self.assertIn("Game review failed:", error.getvalue())


if __name__ == "__main__":
    unittest.main()
