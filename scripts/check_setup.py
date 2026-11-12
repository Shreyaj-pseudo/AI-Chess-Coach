"""Run with python -m scripts.check_setup [--engine] from the project root."""

import argparse
import asyncio
import importlib
from importlib.metadata import version
import platform
import sys

from engine_config import resolve_stockfish_path
from engine_session import StockfishSession


async def engine_smoke(path):
    """Live, two-position check of process reuse and completed shutdown."""
    import chess

    async with StockfishSession(path) as session:
        first_pid = None
        for moves in ((), ("e2e4", "e7e5")):
            board = chess.Board()
            for move in moves:
                board.push_uci(move)
            info = await session.analyse(board, time_limit=0.1)
            if not info.get("pv") or info["pv"][0] not in board.legal_moves:
                raise ValueError("Stockfish did not return a legal first move")
            if first_pid is not None and session.pid != first_pid:
                raise ValueError("Stockfish process was not reused")
            first_pid = session.pid
            print(f"Stockfish: {session.engine_name}; PID: {session.pid}; legal smoke line: {board.variation_san(info['pv'])}")
    if session.returncode is None or session.pid is not None:
        raise ValueError("Stockfish shutdown was not confirmed")
    print(f"Stockfish process reused for two positions; reaped with exit code {session.returncode}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="store_true", help="Also analyze two positions in one Stockfish session")
    args = parser.parse_args(argv)
    try:
        print(f"Python: {platform.python_version()} ({sys.executable})")
        for package, module in (("chess", "chess"), ("requests", "requests"), ("python-dotenv", "dotenv"), ("pydantic", "pydantic")):
            importlib.import_module(module)
            print(f"{package}: {version(package)}")
        for module in ("AI_process", "chess_part", "opening_agent", "middlegame_agent", "endgame_agent", "data.data_collection", "analysis_records", "engine_session", "candidate_analysis", "board_facts"):
            importlib.import_module(module)
        print("Project modules: import successfully")
        path = resolve_stockfish_path()
        print(f"Stockfish path: {path}")
        if args.engine:
            asyncio.run(engine_smoke(path))
        print("Setup checks passed. This does not validate the complete coaching pipeline.")
        return 0
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"Setup check failed: {error}\nActivate the project Conda environment and follow README.md.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
