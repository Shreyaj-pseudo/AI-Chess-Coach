"""One persistent Stockfish process, owned by one asyncio event loop.

Use ``async with StockfishSession() as session`` and await ``session.analyse``.
Concurrent callers are serialized. Cancel the search task to abort and reap its
engine; a later call starts fresh. Explicitly closing the session is permanent.
"""

import asyncio
import os
import subprocess

import chess
import chess.engine

from engine_config import resolve_stockfish_path


class EngineSessionError(RuntimeError):
    """Stockfish failed, or its session is closed."""


class EngineUnavailableError(EngineSessionError):
    """The configured executable cannot be started or initialized."""


class EngineTimeoutError(EngineSessionError, TimeoutError):
    """Stockfish exceeded its startup or search deadline."""


async def _finish_cleanup(coroutine):
    # Shield and join cleanup even if the owner is cancelled repeatedly.
    task = asyncio.create_task(coroutine)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


class StockfishSession:
    STARTUP_TIMEOUT = 5.0
    RESPONSE_GRACE = 1.0
    SHUTDOWN_TIMEOUT = 2.0
    MAX_SEARCH_SECONDS = 5.0

    def __init__(self, stockfish_path=None):
        self._path = stockfish_path
        self._lock = asyncio.Lock()
        self._engine = None
        self._transport = None
        self._command_task = None
        self._closed = False
        self._cleanup_failed = False
        self.engine_name = None
        self.returncode = None

    @property
    def pid(self):
        return self._transport.get_pid() if self._transport else None

    @property
    def engine_options(self):
        return {"Threads": 1, "Hash": 128}

    async def __aenter__(self):
        if self._closed:
            raise EngineSessionError("Stockfish session is closed; create a new session.")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.aclose()

    async def _start(self):
        if self._cleanup_failed:
            raise EngineSessionError("Stockfish cleanup failed; close this session before replacing it.")
        if self._engine is not None:
            return
        path = resolve_stockfish_path(self._path)
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        # Own the transport before UCI initialization, which can hang or fail.
        self._transport, self._engine = await chess.engine.UciProtocol.popen(path, **kwargs)
        self.returncode = None
        await self._command(self._engine.initialize())
        await self._command(self._engine.configure(self.engine_options))
        self.engine_name = self._engine.id.get("name", "unknown UCI engine")

    async def _command(self, coroutine):
        # python-chess 1.11.2 must receive process exit before its analysis future is cancelled.
        self._command_task = asyncio.create_task(coroutine)
        result = await asyncio.shield(self._command_task)
        self._command_task = None
        return result

    async def analyse(self, board: chess.Board, *, time_limit=1.5, multipv=None, root_moves=None):
        """Return InfoDict, or a list when multipv is supplied; preserve history.

        Budget is 0 < time_limit <= 5 seconds, plus one second for UCI response.
        Queue wait is separate. Startup has its own five-second deadline.
        Failures are surfaced, never silently retried; the next call can recover.
        MultiPV is limited to 1-3; root_moves restricts the original board's legal
        root choices without forcing any later reply.
        """
        if (isinstance(time_limit, bool) or not isinstance(time_limit, (int, float))
                or not 0 < time_limit <= self.MAX_SEARCH_SECONDS):
            raise ValueError("Stockfish time_limit must be finite, positive and at most 5 seconds.")
        if not isinstance(board, chess.Board) or not board.is_valid() or board.chess960 or board.uci_variant != "chess":
            raise ValueError("Stockfish requires a valid standard chess board.")
        position = board.copy(stack=True)
        search_options = {}
        if multipv is not None:
            if type(multipv) is not int or not 1 <= multipv <= 3:
                raise ValueError("Stockfish multipv must be an integer from 1 to 3.")
            search_options["multipv"] = multipv
        if root_moves is not None:
            roots = tuple(root_moves)
            if (not roots or any(not isinstance(move, chess.Move) or move not in position.legal_moves for move in roots)
                    or len(set(roots)) != len(roots)):
                raise ValueError("Stockfish root_moves must be nonempty, unique legal moves.")
            search_options["root_moves"] = roots
        async with self._lock:
            if self._closed:
                raise EngineSessionError("Stockfish session is closed; create a new session.")
            stage = "startup"
            try:
                async with asyncio.timeout(self.STARTUP_TIMEOUT):
                    await self._start()
                stage = "search"
                async with asyncio.timeout(time_limit + self.RESPONSE_GRACE):
                    return await self._command(self._engine.analyse(position, chess.engine.Limit(time=time_limit), **search_options))
            except TimeoutError as error:
                await _finish_cleanup(self._dispose())
                raise EngineTimeoutError(
                    f"Stockfish {stage} timed out; process closed. Check the binary/CPU or reduce the workload."
                ) from error
            except (OSError, chess.engine.EngineError) as error:
                await _finish_cleanup(self._dispose())
                kind = EngineUnavailableError if stage == "startup" else EngineSessionError
                raise kind(
                    f"Stockfish {stage} failed: {error}. Check STOCKFISH_PATH and binary compatibility; "
                    "the next search will start a fresh process."
                ) from error
            except BaseException:
                await _finish_cleanup(self._dispose())
                raise

    async def _dispose(self, graceful=False):
        if self._transport is None:
            return
        self._cleanup_failed = True
        try:
            if graceful and not self._engine.returncode.done():
                try:
                    async with asyncio.timeout(self.SHUTDOWN_TIMEOUT):
                        await self._engine.quit()
                except (TimeoutError, OSError, chess.engine.EngineError):
                    pass
        finally:
            self._transport.close()
        try:
            async with asyncio.timeout(self.SHUTDOWN_TIMEOUT):
                self.returncode = await asyncio.shield(self._engine.returncode)
                if self._command_task is not None:
                    await asyncio.shield(asyncio.gather(self._command_task, return_exceptions=True))
        except TimeoutError as error:
            raise EngineSessionError(
                "Stockfish did not exit after termination; retry aclose() and restart the worker if it persists."
            ) from error
        self._transport = self._engine = None
        self._command_task = None
        self._cleanup_failed = False

    async def aclose(self):
        """Permanently close, draining a bounded active search before quitting.

        Queued searches fail as closed. Cancelling aclose still waits for cleanup.
        """
        self._closed = True

        async def close_locked():
            async with self._lock:
                await self._dispose(graceful=True)

        await _finish_cleanup(close_locked())
