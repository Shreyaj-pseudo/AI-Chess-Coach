"""Deterministic lifecycle tests; these fakes are not the real-binary smoke."""

import asyncio
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import AsyncMock, patch

import chess
import chess.engine
import chess.variant

from engine_session import EngineSessionError, EngineTimeoutError, EngineUnavailableError, StockfishSession
from scripts.check_setup import engine_smoke


class FakeEngine:
    def __init__(self):
        self.id = {"name": "fake Stockfish (unit test only)"}
        self.returncode = asyncio.get_running_loop().create_future()
        self.events = {name: asyncio.Event() for name in ("initialize", "configure", "analyse", "quit")}
        self.gates = {}
        self.errors = {}
        self.boards = []
        self.limits = []
        self.options = None
        self.active = self.max_active = 0

    async def phase(self, name):
        self.events[name].set()
        if name in self.gates:
            await self.gates[name].wait()
        if self.returncode.done() and name != "quit":
            raise chess.engine.EngineTerminatedError("fake engine exited")
        if name in self.errors:
            raise self.errors[name]

    async def initialize(self):
        await self.phase("initialize")

    async def configure(self, options):
        self.options = options
        await self.phase("configure")

    async def analyse(self, board, limit):
        self.boards.append(board)
        self.limits.append(limit)
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            await self.phase("analyse")
            return {"pv": [next(iter(board.legal_moves))], "score": chess.engine.PovScore(chess.engine.Cp(10), board.turn)}
        finally:
            self.active -= 1

    async def quit(self):
        await self.phase("quit")
        if not self.returncode.done():
            self.returncode.set_result(0)


class FakeTransport:
    def __init__(self, engine, pid):
        self.engine = engine
        self.pid = pid
        self.close_calls = 0
        self.reap = True

    def get_pid(self):
        return self.pid

    def close(self):
        self.close_calls += 1
        if self.reap and not self.engine.returncode.done():
            self.engine.returncode.set_result(-9)
            for gate in self.engine.gates.values():
                gate.set()


class EngineSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engines = []
        self.transports = []
        self.prepared = []

        async def launch(path, **kwargs):
            engine = self.prepared.pop(0) if self.prepared else FakeEngine()
            transport = FakeTransport(engine, 100 + len(self.engines))
            self.engines.append(engine)
            self.transports.append(transport)
            return transport, engine

        resolver = patch("engine_session.resolve_stockfish_path", return_value="resolved-stockfish")
        self.resolve = resolver.start()
        self.addCleanup(resolver.stop)
        launcher = patch("chess.engine.UciProtocol.popen", new=AsyncMock(side_effect=launch))
        self.launch = launcher.start()
        self.addCleanup(launcher.stop)
        self.session = StockfishSession("chosen-path")
        self.session.STARTUP_TIMEOUT = 1.0
        self.session.RESPONSE_GRACE = 0.1
        self.session.SHUTDOWN_TIMEOUT = 1.0
        self.addAsyncCleanup(self.session.aclose)

    def prepare_blocked(self, phase):
        engine = FakeEngine()
        engine.gates[phase] = asyncio.Event()
        self.prepared.append(engine)
        return engine

    async def test_lazy_start_reuses_one_process_for_two_positions(self):
        self.launch.assert_not_called()
        board = chess.Board()
        first = await self.session.analyse(board, time_limit=0.15)
        pid = self.session.pid
        board.push_uci("e2e4")
        second = await self.session.analyse(board)
        self.assertIn(first["pv"][0], chess.Board().legal_moves)
        self.assertIn(second["pv"][0], board.legal_moves)
        self.assertEqual(self.session.pid, pid)
        self.assertEqual(self.launch.await_count, 1)
        self.resolve.assert_called_once_with("chosen-path")
        self.assertEqual(self.launch.call_args.args, ("resolved-stockfish",))
        self.assertEqual(self.engines[0].options, {"Threads": 1, "Hash": 128})
        self.assertEqual([limit.time for limit in self.engines[0].limits], [0.15, 1.5])
        self.assertEqual(self.session.engine_name, self.engines[0].id["name"])

    async def test_preserves_repetition_history_and_copies_board_before_queueing(self):
        engine = self.prepare_blocked("analyse")
        first = asyncio.create_task(self.session.analyse(chess.Board()))
        await engine.events["analyse"].wait()
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            board.push_uci(move)
        queued = asyncio.create_task(self.session.analyse(board))
        await asyncio.sleep(0)
        board.push_uci("e2e4")
        engine.gates["analyse"].set()
        await asyncio.gather(first, queued)
        snapshot = engine.boards[1]
        self.assertEqual(len(snapshot.move_stack), 8)
        self.assertTrue(snapshot.is_repetition(3))
        self.assertEqual(len(board.move_stack), 9)

    async def test_concurrent_searches_are_serialized(self):
        engine = self.prepare_blocked("analyse")
        tasks = [asyncio.create_task(self.session.analyse(chess.Board())) for _ in range(3)]
        await engine.events["analyse"].wait()
        await asyncio.sleep(0)
        self.assertEqual(len(engine.boards), 1)
        engine.gates["analyse"].set()
        await asyncio.gather(*tasks)
        self.assertEqual(len(engine.boards), 3)
        self.assertEqual(engine.max_active, 1)
        self.assertEqual(self.launch.await_count, 1)

    async def test_shutdown_is_reaped_idempotent_and_permanent(self):
        await self.session.analyse(chess.Board())
        await self.session.aclose()
        await self.session.aclose()
        self.assertIsNone(self.session.pid)
        self.assertEqual(self.session.returncode, 0)
        self.assertEqual(self.transports[0].close_calls, 1)
        with self.assertRaisesRegex(EngineSessionError, "closed"):
            await self.session.analyse(chess.Board())
        with self.assertRaisesRegex(EngineSessionError, "closed"):
            await self.session.__aenter__()
        self.assertEqual(self.launch.await_count, 1)

    async def test_context_body_exception_closes_engine(self):
        with self.assertRaisesRegex(ValueError, "body error"):
            async with self.session:
                await self.session.analyse(chess.Board())
                raise ValueError("body error")
        self.assertEqual(self.session.returncode, 0)

    async def test_unused_session_closes_without_launching(self):
        async with self.session:
            pass
        self.launch.assert_not_called()

    async def test_missing_binary_is_typed_and_actionable(self):
        self.resolve.side_effect = FileNotFoundError("Set STOCKFISH_PATH")
        with self.assertRaisesRegex(EngineUnavailableError, "STOCKFISH_PATH"):
            await self.session.analyse(chess.Board())
        self.launch.assert_not_called()

    async def test_spawn_failure_is_typed(self):
        self.launch.side_effect = OSError("incompatible executable")
        with self.assertRaisesRegex(EngineUnavailableError, "binary compatibility"):
            await self.session.analyse(chess.Board())
        self.assertIsNone(self.session.pid)

    async def test_initialization_and_configuration_failures_close_owned_child(self):
        for phase in ("initialize", "configure"):
            with self.subTest(phase=phase):
                engine = FakeEngine()
                engine.errors[phase] = chess.engine.EngineError("UCI failure")
                self.prepared.append(engine)
                with self.assertRaisesRegex(EngineUnavailableError, "startup failed"):
                    await self.session.analyse(chess.Board())
                self.assertEqual(engine.returncode.result(), -9)
                self.assertIsNone(self.session.pid)

    async def test_startup_timeout_reaps_child(self):
        engine = self.prepare_blocked("initialize")
        self.session.STARTUP_TIMEOUT = 0.01
        with self.assertRaisesRegex(EngineTimeoutError, "startup timed out"):
            await self.session.analyse(chess.Board())
        self.assertEqual(engine.returncode.result(), -9)

    async def test_search_timeout_reaps_child_and_next_search_recovers(self):
        engine = self.prepare_blocked("analyse")
        with self.assertRaisesRegex(EngineTimeoutError, "search timed out"):
            await self.session.analyse(chess.Board(), time_limit=0.001)
        self.assertEqual(engine.returncode.result(), -9)
        await self.session.analyse(chess.Board())
        self.assertEqual(self.launch.await_count, 2)

    async def test_engine_crash_reaps_child_and_next_search_recovers(self):
        engine = FakeEngine()
        engine.errors["analyse"] = chess.engine.EngineTerminatedError("exit code 1")
        self.prepared.append(engine)
        with self.assertRaisesRegex(EngineSessionError, "search failed"):
            await self.session.analyse(chess.Board())
        self.assertTrue(engine.returncode.done())
        self.assertEqual(self.transports[0].close_calls, 1)
        await self.session.analyse(chess.Board())
        self.assertEqual(self.launch.await_count, 2)
        self.assertEqual(self.session.pid, 101)

    async def test_unexpected_search_error_also_cleans_up(self):
        engine = FakeEngine()
        engine.errors["analyse"] = ValueError("unexpected result")
        self.prepared.append(engine)
        with self.assertRaisesRegex(ValueError, "unexpected result"):
            await self.session.analyse(chess.Board())
        self.assertEqual(engine.returncode.result(), -9)

    async def test_cancellation_during_startup_and_search_reaps_child(self):
        for phase in ("initialize", "configure", "analyse"):
            with self.subTest(phase=phase):
                engine = self.prepare_blocked(phase)
                task = asyncio.create_task(self.session.analyse(chess.Board()))
                await engine.events[phase].wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(engine.returncode.result(), -9)
                self.assertIsNone(self.session.pid)
        await self.session.analyse(chess.Board())
        self.assertEqual(self.launch.await_count, 4)

    async def test_cancelling_queued_search_does_not_kill_active_search(self):
        engine = self.prepare_blocked("analyse")
        active = asyncio.create_task(self.session.analyse(chess.Board()))
        await engine.events["analyse"].wait()
        queued = asyncio.create_task(self.session.analyse(chess.Board()))
        await asyncio.sleep(0)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertFalse(engine.returncode.done())
        engine.gates["analyse"].set()
        await active
        self.assertEqual(len(engine.boards), 1)

    async def test_cancellation_and_timeout_preserve_python_chess_analysis_future(self):
        for mode in ("cancel", "timeout"):
            with self.subTest(mode=mode):
                await self.session.analyse(chess.Board())
                engine, transport = self.engines[-1], self.transports[-1]
                result = chess.engine.AnalysisResult()
                entered = asyncio.Event()

                async def analyse(board, limit):
                    entered.set()
                    await result.wait()
                    return result.info

                original_close = transport.close

                def close():
                    # Mirrors UciProtocol.connection_lost; fails if wait() was cancelled.
                    result.set_exception(chess.engine.EngineTerminatedError("terminated"))
                    original_close()

                engine.analyse = analyse
                transport.close = close
                task = asyncio.create_task(self.session.analyse(chess.Board(), time_limit=0.001))
                await entered.wait()
                if mode == "cancel":
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError if mode == "cancel" else EngineTimeoutError):
                    await task
                self.assertFalse(result._finished.cancelled())
                self.assertEqual(engine.returncode.result(), -9)
                self.assertIsNone(self.session._command_task)

    async def test_shutdown_drains_active_search_and_rejects_queue(self):
        engine = self.prepare_blocked("analyse")
        active = asyncio.create_task(self.session.analyse(chess.Board()))
        await engine.events["analyse"].wait()
        queued = asyncio.create_task(self.session.analyse(chess.Board()))
        closing = asyncio.create_task(self.session.aclose())
        await asyncio.sleep(0)
        engine.gates["analyse"].set()
        await active
        with self.assertRaisesRegex(EngineSessionError, "closed"):
            await queued
        await closing
        self.assertEqual(engine.returncode.result(), 0)
        self.assertEqual(len(engine.boards), 1)

    async def test_unresponsive_quit_forces_termination(self):
        self.session.SHUTDOWN_TIMEOUT = 0.02
        engine = self.prepare_blocked("quit")
        await self.session.analyse(chess.Board())
        await self.session.aclose()
        self.assertEqual(engine.returncode.result(), -9)
        self.assertIsNone(self.session.pid)

    async def test_cancelled_shutdown_still_joins_cleanup(self):
        engine = self.prepare_blocked("quit")
        await self.session.analyse(chess.Board())
        closing = asyncio.create_task(self.session.aclose())
        await engine.events["quit"].wait()
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        engine.gates["quit"].set()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.assertEqual(engine.returncode.result(), 0)
        self.assertIsNone(self.session.pid)

    async def test_unconfirmed_exit_is_visible_and_blocks_new_process(self):
        self.session.SHUTDOWN_TIMEOUT = 0.1
        await self.session.analyse(chess.Board())
        engine, transport = self.engines[0], self.transports[0]
        engine.errors["analyse"] = chess.engine.EngineError("failed search")
        transport.reap = False
        with self.assertRaisesRegex(EngineSessionError, "did not exit"):
            await self.session.analyse(chess.Board())
        self.assertFalse(engine.returncode.cancelled())
        with self.assertRaisesRegex(EngineSessionError, "did not exit"):
            await self.session.analyse(chess.Board())
        self.assertEqual(self.launch.await_count, 1)
        transport.reap = True
        self.session.SHUTDOWN_TIMEOUT = 1.0
        await self.session.aclose()
        self.assertEqual(self.session.returncode, 0)

    async def test_invalid_budgets_fail_before_spawning(self):
        for budget in (0, -1, 5.01, 10**1000, float("inf"), float("nan"), True, "1.5", None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                await self.session.analyse(chess.Board(), time_limit=budget)
        self.launch.assert_not_called()

    async def test_invalid_and_variant_boards_fail_before_spawning(self):
        for board in (chess.Board(None), chess.Board(chess960=True), chess.variant.CrazyhouseBoard(), "fen"):
            with self.subTest(board=board), self.assertRaises(ValueError):
                await self.session.analyse(board)
        self.launch.assert_not_called()

    async def test_setup_smoke_exercises_two_searches_and_checks_shutdown(self):
        output = io.StringIO()
        with redirect_stdout(output):
            await engine_smoke("chosen-path")
        self.assertEqual(self.launch.await_count, 1)
        self.assertEqual(len(self.engines[0].boards), 2)
        self.assertEqual(self.engines[0].returncode.result(), 0)
        self.assertIn("reused for two positions; reaped with exit code 0", output.getvalue())
        self.assertEqual(output.getvalue().count("legal smoke line"), 2)


if __name__ == "__main__":
    unittest.main()
