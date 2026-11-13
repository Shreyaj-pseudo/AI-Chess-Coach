"""Synthetic engine fixtures; live Stockfish checks are recorded separately."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import chess
import chess.engine
from pydantic import ValidationError

from analysis_records import AnalysisRecord, MateScore, PositionContext, UnavailableScore, source_game_id
from candidate_analysis import analyze_position
from engine_session import EngineSessionError, StockfishSession
from pgn_input import learner_positions, parse_pgn
from scripts import analyze_position as cli


FIXTURES = Path(__file__).parent / "fixtures" / "pgn"


def context(pgn="1. h4 *", color="white", index=0):
    game = parse_pgn(pgn)
    return PositionContext.from_decision(learner_positions(game, color)[index], game_id=source_game_id(game))


def row(line, cp=50, *, pov=chess.WHITE, **updates):
    info = dict(pv=[chess.Move.from_uci(move) for move in line.split()],
                score=chess.engine.PovScore(chess.engine.Cp(cp), pov), depth=14, nodes=2345, time=0.15)
    info.update(updates)
    return info


def session(*responses):
    return SimpleNamespace(engine_name="Synthetic fixture engine 1.0", engine_options={"Threads": 1, "Hash": 128},
                           analyse=AsyncMock(side_effect=responses))


def three():
    return [row("e2e4 e7e5 g1f3 b8c6", 70), row("d2d4 d7d5 c2c4", 25), row("g1f3 d7d5", -20)]


class CandidateTests(unittest.IsolatedAsyncioTestCase):
    async def test_played_move_outside_three_has_its_own_root_search_and_reply(self):
        engine = session(three(), [row("h2h4 e7e5 h4h5 d7d5", -180)])
        record = await analyze_position(engine, context())
        self.assertEqual(len(record.candidates), 3)
        self.assertNotIn(record.played.move_uci, [c.move_uci for c in record.candidates])
        self.assertEqual(record.played.pv_uci, ("h2h4", "e7e5", "h4h5", "d7d5"))
        self.assertEqual(record.played.pv_san, ("h4", "e5", "h5", "d5"))
        self.assertEqual(record.loss_cp, 250)
        self.assertEqual(engine.analyse.await_count, 2)
        first, forced = engine.analyse.call_args_list
        self.assertEqual(first.kwargs, {"time_limit": 1.5, "multipv": 3})
        self.assertEqual(forced.kwargs, {"time_limit": 1.5, "multipv": 1,
                                       "root_moves": (chess.Move.from_uci("h2h4"),)})
        self.assertEqual(first.args[0].fen(), forced.args[0].fen())
        self.assertEqual(record.played.search.root_moves, ("h2h4",))
        self.assertTrue(all(not c.search.root_moves for c in record.candidates))

    async def test_played_move_in_multipv_reuses_exact_evidence_without_search(self):
        engine = session(three())
        record = await analyze_position(engine, context("1. d4 *"))
        self.assertEqual(record.played, record.candidates[1])
        self.assertEqual(record.played.id, record.candidates[1].id)
        self.assertEqual(engine.analyse.await_count, 1)
        self.assertEqual(record.loss_cp, 45)

    async def test_best_score_pv_and_metadata_stay_with_the_same_result(self):
        infos = three()
        infos[1].update(depth=13, nodes=4321, time=0.149)
        record = await analyze_position(session(infos), context("1. e4 *"), time_limit=0.15)
        for candidate, info in zip(record.candidates, infos):
            self.assertEqual(candidate.pv_uci, tuple(move.uci() for move in info["pv"]))
            self.assertEqual(candidate.score.cp, info["score"].white().score())
            self.assertEqual((candidate.search.depth, candidate.search.nodes, candidate.search.elapsed_seconds),
                             (info["depth"], info["nodes"], info["time"]))
            self.assertEqual(candidate.search.time_limit_seconds, 0.15)
            self.assertEqual(candidate.search.engine, "Synthetic fixture engine 1.0")
        expected = hashlib.sha256(json.dumps(
            {"options": {"Threads": 1, "Hash": 128, "MultiPV": 3}, "limit": {"time": 0.15}},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(record.candidates[0].search.config_sha256, expected)

    async def test_multipv_width_is_in_provenance_and_qualified(self):
        record = await analyze_position(session(three(), [row("h2h4 e7e5", -100)]), context())
        self.assertNotEqual(record.candidates[0].search.config_sha256, record.played.search.config_sha256)
        self.assertTrue(any("MultiPV width" in note for note in record.uncertainty))
        self.assertEqual(record.candidates[0].search.time_limit_seconds, record.played.search.time_limit_seconds)

    async def test_all_black_scores_use_learner_viewpoint(self):
        infos = [row("e7e5 g1f3", -70), row("c7c5 g1f3", -25), row("e7e6 d2d4", 20)]
        forced = [row("h7h5 d2d4", 130)]
        record = await analyze_position(session(infos, forced), context("1. e4 h5 *", "black"))
        self.assertEqual([c.score.cp for c in record.candidates], [70, 25, -20])
        self.assertEqual(record.played.score.cp, -130)
        self.assertEqual(record.loss_cp, 200)
        self.assertTrue(all(c.score.pov == "black" for c in (*record.candidates, record.played)))

    async def test_clear_tactical_blunder_keeps_forced_mate_reply_not_centipawn_loss(self):
        pgn = (FIXTURES / "played_blunder.pgn").read_text()
        infos = [row("e2e4 b8c6", 0), row("b1c3 g8f6", -50), row("d2d3 d7d5", -100)]
        forced = row("g2g4 d8h4", score=chess.engine.PovScore(chess.engine.Mate(-1), chess.WHITE))
        record = await analyze_position(session(infos, [forced]), context(pgn, index=1))
        self.assertEqual(record.played.pv_san, ("g4", "Qh4#"))
        self.assertIsInstance(record.played.score, MateScore)
        self.assertEqual((record.played.score.moves, record.played.score.winning), (1, False))
        self.assertIsNone(record.loss_cp)

    async def test_checkmating_played_move_needs_no_opponent_reply(self):
        info = row("d8h4", score=chess.engine.PovScore(chess.engine.Mate(1), chess.BLACK))
        position = context((FIXTURES / "played_blunder.pgn").read_text(), "black", index=1)
        record = await analyze_position(session([info]), position)
        self.assertEqual(record.played.pv_san, ("Qh4#",))
        self.assertTrue(record.played.score.winning)
        self.assertIsNone(record.loss_cp)

    async def test_one_legal_move_and_terminal_draw_after_played_move(self):
        fen = "7k/8/8/8/8/8/6r1/7K w - - 0 1"
        position = context(f'[SetUp "1"]\n[FEN "{fen}"]\n\n1. Kxg2 1/2-1/2')
        self.assertEqual(position.board().legal_moves.count(), 1)
        record = await analyze_position(session([row("h1g2", 0)]), position)
        self.assertEqual(len(record.candidates), 1)
        self.assertEqual(record.played.pv_san, ("Kxg2",))

    async def test_illegal_and_missing_pv_roots_are_rejected(self):
        for info in (row("e2e5"), row("e2e4 e7e6 e4e6"), row(""), {"depth": 10, "nodes": 1, "time": 0.1}):
            with self.subTest(info=info), self.assertRaises(ValueError):
                await analyze_position(session([info]), context("1. e4 *"))

    async def test_restricted_result_must_begin_with_the_played_move(self):
        with self.assertRaisesRegex(ValueError, "restricted search"):
            await analyze_position(session(three(), [row("a2a3 e7e5")]), context())

    async def test_pv_cannot_continue_after_checkmate(self):
        position = context((FIXTURES / "played_blunder.pgn").read_text(), "black", index=1)
        bad = row("d8h4 e2e4", score=chess.engine.PovScore(chess.engine.Mate(1), chess.BLACK))
        with self.assertRaisesRegex(ValueError, "game over"):
            await analyze_position(session([bad]), position)

    async def test_pv_cannot_continue_after_automatic_insufficient_material_draw(self):
        fen = "7k/8/8/8/8/8/6r1/7K w - - 0 1"
        position = context(f'[SetUp "1"]\n[FEN "{fen}"]\n\n1. Kxg2 1/2-1/2')
        with self.assertRaisesRegex(ValueError, "game over"):
            await analyze_position(session([row("h1g2 h8h7", 0)]), position)

    async def test_terminal_roots_fail_revalidation_before_search(self):
        checkmate = parse_pgn((FIXTURES / "played_blunder.pgn").read_text()).end().board().fen()
        for fen, color, played in ((checkmate, "white", "e2e4"),
                                   ("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", "black", "h8h7"),
                                   ("7k/8/8/8/8/8/R7/K7 w - - 150 76", "white", "a2a3")):
            with self.subTest(fen=fen):
                position = PositionContext.model_construct(game_id="a" * 64, ply=1, initial_fen=fen,
                                                          fen=fen, history_uci=(), learner_color=color, played_uci=played)
                engine = session()
                with self.assertRaisesRegex(ValidationError, "game over"):
                    await analyze_position(engine, position)
                engine.analyse.assert_not_called()

    async def test_missing_opponent_reply_after_one_retry_is_an_actionable_failure(self):
        for infos, pgn in (([row("e2e4")], "1. e4 *"), (three(), "1. h4 *")):
            move = "e2e4" if "e4" in pgn else "h2h4"
            engine = session(infos, *([[row(move)]] if move == "e2e4" else [[row(move)], infos, [row(move)]]))
            with self.subTest(pgn=pgn), self.assertRaisesRegex(ValueError, "after one longer retry"):
                await analyze_position(engine, context(pgn))
            self.assertEqual(engine.analyse.await_count, 2 if move == "e2e4" else 4)

    async def test_incomplete_multipv_played_line_recovers_with_one_restricted_retry(self):
        engine = session([row("e2e4", 30)], [row("e2e4 e7e5 g1f3", 30, depth=17, nodes=700)])
        record = await analyze_position(engine, context("1. e4 *"), time_limit=.15)
        self.assertEqual(record.played.pv_uci, ("e2e4", "e7e5", "g1f3"))
        self.assertEqual(record.played.search.depth, 17)
        self.assertEqual(record.played.search.nodes, 700)
        self.assertEqual(record.played.search.time_limit_seconds, .5)
        self.assertEqual(record.played.search.root_moves, ())
        self.assertEqual(engine.analyse.await_count, 2)
        self.assertEqual(engine.analyse.call_args.kwargs, dict(time_limit=.5, multipv=3))
        self.assertTrue(any("recovered" in note for note in record.uncertainty))

    async def test_incomplete_forced_line_replaces_failed_score_and_warning(self):
        incomplete = row("h2h4", -500, upperbound=True)
        recovered = row("h2h4 e7e5", -100, depth=16)
        engine = session(three(), [incomplete], three(), [recovered])
        record = await analyze_position(engine, context())
        self.assertEqual(record.played.score.cp, -100)
        self.assertEqual(record.played.pv_uci, ("h2h4", "e7e5"))
        self.assertFalse(any("h4: engine score is a bound" in note for note in record.uncertainty))
        self.assertEqual(engine.analyse.await_count, 4)
        self.assertTrue(all(c.search.time_limit_seconds == 4.5 for c in (*record.candidates, record.played)))
        self.assertTrue(all(call.args[0].fen() == record.position.fen for call in engine.analyse.call_args_list))

    async def test_retry_budget_is_capped_and_larger_results_require_recovery_provenance(self):
        from candidate_analysis import matches_search_budget
        engine = session([row("e2e4")], [row("e2e4 e7e5")])
        record = await analyze_position(engine, context("1. e4 *"), time_limit=5)
        self.assertEqual([c.kwargs["time_limit"] for c in engine.analyse.call_args_list], [5, 5])
        self.assertTrue(matches_search_budget(record, 5))
        ordinary = await analyze_position(session([row("e2e4 e7e5")]), context("1. e4 *"), time_limit=.5)
        self.assertFalse(matches_search_budget(ordinary, .15))

    async def test_bad_retry_pv_and_engine_errors_are_not_retried_again(self):
        for retry, kind in (([row("d2d4 d7d5")], ValueError),
                            ([row("h2h4 e7e6 h4e6")], ValueError),
                            (EngineSessionError("crashed during retry"), EngineSessionError),
                            (asyncio.CancelledError(), asyncio.CancelledError)):
            engine = session(three(), [row("h2h4")], three(), retry)
            with self.subTest(retry=retry), self.assertRaises(kind):
                await analyze_position(engine, context())
            self.assertEqual(engine.analyse.await_count, 4)

    async def test_full_history_reaches_both_searches(self):
        position = context("1.Nf3 Nf6 2.Ng1 Ng8 3.h4 *", index=2)
        engine = session(three(), [row("h2h4 e7e5")])
        record = await analyze_position(engine, position)
        for call in engine.analyse.call_args_list:
            self.assertEqual(tuple(m.uci() for m in call.args[0].move_stack), position.history_uci)
        self.assertEqual(record.position.history_uci, position.history_uci)

    async def test_pv_cannot_continue_after_history_based_fifth_repetition(self):
        position = context("1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1 Ng8 5.Nf3 Nf6 6.Ng1 Ng8 7.Nf3 Nf6 8.Ng1 Ng8 *",
                           "black", index=7)
        with self.assertRaisesRegex(ValueError, "game over"):
            await analyze_position(session([row("f6g8 g1f3", pov=chess.BLACK)]), position)

    async def test_near_equal_played_move_and_alternatives_are_qualified(self):
        infos = [row("e2e4 e7e5", 30), row("d2d4 d7d5", 20), row("g1f3 d7d5", 5)]
        record = await analyze_position(session(infos), context("1. d4 *"))
        self.assertEqual(record.loss_cp, 10)
        self.assertTrue(any("uniquely correct" in note for note in record.uncertainty))
        self.assertTrue(any("does not establish a mistake" in note for note in record.uncertainty))

    async def test_negative_loss_and_inconsistent_candidate_order_are_preserved_with_warning(self):
        infos = [row("e2e4 e7e5", 20), row("d2d4 d7d5", 50)]
        record = await analyze_position(session(infos, [row("h2h4 e7e5", 90)]), context())
        self.assertEqual(record.loss_cp, -70)
        self.assertEqual([c.score.cp for c in record.candidates], [20, 50])
        self.assertTrue(any("ranking" in note for note in record.uncertainty))
        self.assertTrue(any("unstable" in note for note in record.uncertainty))

    async def test_conflicting_mate_distances_are_compared_without_fake_centipawns(self):
        infos = [row("e2e4 e7e5", score=chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)),
                 row("d2d4 d7d5", score=chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE))]
        record = await analyze_position(session(infos), context("1. d4 *"))
        self.assertIsNone(record.loss_cp)
        self.assertTrue(any("ranking" in note for note in record.uncertainty))
        self.assertTrue(any("unstable" in note for note in record.uncertainty))

    async def test_missing_score_bound_shallow_and_uneven_depth_are_visible(self):
        infos = [row("e2e4 e7e5", score=None, depth=3), row("d2d4 d7d5", lowerbound=True, depth=4)]
        record = await analyze_position(session(infos), context("1. d4 *"))
        self.assertIsInstance(record.candidates[0].score, UnavailableScore)
        self.assertIsNone(record.loss_cp)
        text = " ".join(record.uncertainty)
        for signal in ("no score", "bound", "shallow", "different depths"):
            self.assertIn(signal, text)

    async def test_missing_or_invalid_search_metadata_is_not_fabricated(self):
        for key, bad in (("depth", None), ("nodes", -1), ("time", float("nan"))):
            info = row("e2e4 e7e5")
            if bad is None:
                del info[key]
            else:
                info[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                await analyze_position(session([info]), context("1. e4 *"))

    async def test_invalid_count_duplicates_rank_and_malformed_results_fail(self):
        invalid = ([], {}, [None], three() + [row("c2c4 e7e5")],
                   [row("e2e4 e7e5"), row("e2e4 e7e5")], [row("e2e4 e7e5", multipv=2)])
        for infos in invalid:
            with self.subTest(infos=infos), self.assertRaises(ValueError):
                await analyze_position(session(infos), context("1. e4 *"))

    async def test_records_roundtrip_without_changing_content_ids(self):
        record = await analyze_position(session(three(), [row("h2h4 e7e5", -100)]), context())
        restored = AnalysisRecord.model_validate_json(record.model_dump_json())
        self.assertEqual(record, restored)
        self.assertEqual(record.id, restored.id)
        self.assertEqual(record.played.id, restored.played.id)

    async def test_search_failure_and_cancellation_propagate_without_partial_record(self):
        for error in (EngineSessionError("crashed"), asyncio.CancelledError()):
            engine = session(three(), error)
            with self.subTest(error=error), self.assertRaises(type(error)):
                await analyze_position(engine, context())


class SessionForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_multipv_and_root_restrictions_reach_protocol_and_keep_default_contract(self):
        protocol = SimpleNamespace(
            id={"name": "Synthetic protocol fixture"}, initialize=AsyncMock(), configure=AsyncMock(),
            analyse=AsyncMock(side_effect=[three(), [row("h2h4 e7e5")], row("e2e4 e7e5")]),
            returncode=asyncio.get_running_loop().create_future(),
        )

        async def quit():
            protocol.returncode.set_result(0)

        protocol.quit = quit
        transport = SimpleNamespace(close=lambda: None, get_pid=lambda: 123)
        with patch("engine_session.resolve_stockfish_path", return_value="fake"), \
                patch("chess.engine.UciProtocol.popen", new=AsyncMock(return_value=(transport, protocol))):
            async with StockfishSession() as engine:
                first = await engine.analyse(chess.Board(), multipv=3)
                roots = [chess.Move.from_uci("h2h4")]
                second = await engine.analyse(chess.Board(), multipv=1, root_moves=roots)
                default = await engine.analyse(chess.Board())
                self.assertIsInstance(first, list)
                self.assertIsInstance(second, list)
                self.assertIsInstance(default, dict)
        calls = protocol.analyse.call_args_list
        self.assertEqual(calls[0].kwargs, {"multipv": 3})
        self.assertEqual(calls[1].kwargs, {"multipv": 1, "root_moves": tuple(roots)})
        self.assertEqual(calls[2].kwargs, {})
        self.assertEqual(protocol.returncode.result(), 0)

    async def test_invalid_multipv_and_roots_fail_without_starting_engine(self):
        async with StockfishSession() as engine:
            with patch("chess.engine.UciProtocol.popen") as launch:
                for value in (0, 4, True, 1.5, "3"):
                    with self.subTest(multipv=value), self.assertRaises(ValueError):
                        await engine.analyse(chess.Board(), multipv=value)
                for roots in ([], [chess.Move.from_uci("e2e5")], ["e2e4"],
                              [chess.Move.from_uci("e2e4")] * 2):
                    with self.subTest(roots=roots), self.assertRaises(ValueError):
                        await engine.analyse(chess.Board(), root_moves=roots)
                launch.assert_not_called()


class CLITests(unittest.TestCase):
    def test_cli_selects_exact_learner_decision_and_emits_valid_record_json(self):
        position = context((FIXTURES / "played_blunder.pgn").read_text(), index=1)
        record = asyncio.run(analyze_position(session(
            [row("e2e4 b8c6")], [row("g2g4 d8h4", score=chess.engine.PovScore(chess.engine.Mate(-1), chess.WHITE))]
        ), position))
        output = io.StringIO()
        with patch.object(cli, "_analyze", new=AsyncMock(return_value=record)) as run, redirect_stdout(output):
            self.assertEqual(cli.main([str(FIXTURES / "played_blunder.pgn"), "--color", "white", "--ply", "3"]), 0)
        self.assertEqual(run.call_args.args, (position, 1.5))
        self.assertEqual(AnalysisRecord.model_validate_json(output.getvalue()), record)

    def test_cli_invalid_budget_fails_without_spawning_engine_or_emitting_a_record(self):
        for budget in ("0", "6", "nan"):
            with self.subTest(budget=budget):
                output, errors = io.StringIO(), io.StringIO()
                with patch("chess.engine.UciProtocol.popen") as launch, redirect_stdout(output), redirect_stderr(errors):
                    code = cli.main([str(FIXTURES / "played_blunder.pgn"), "--color", "white", "--ply", "3", "--time", budget])
                self.assertEqual(code, 1)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("time_limit", errors.getvalue())
                launch.assert_not_called()

    def test_cli_rejects_missing_file_wrong_side_ply_and_engine_failure_without_traceback(self):
        cases = (([str(FIXTURES / "missing.pgn"), "--color", "white", "--ply", "1"], None),
                 ([str(FIXTURES / "played_blunder.pgn"), "--color", "white", "--ply", "2"], None),
                 ([str(FIXTURES / "played_blunder.pgn"), "--color", "white", "--ply", "3"], EngineSessionError("STOCKFISH_PATH")))
        for args, error in cases:
            with self.subTest(args=args):
                output = io.StringIO()
                with patch.object(cli, "_analyze", new=AsyncMock(side_effect=error)) as run, redirect_stderr(output):
                    self.assertEqual(cli.main(args), 1)
                self.assertIn("Position analysis failed", output.getvalue())
                self.assertNotIn("Traceback", output.getvalue())
                if error is None:
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
