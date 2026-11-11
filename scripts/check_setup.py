"""Run with python -m scripts.check_setup [--engine] from the project root."""

import argparse
import importlib
from importlib.metadata import version
import platform
import sys

from engine_config import resolve_stockfish_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="store_true", help="Also start Stockfish and analyze one position")
    args = parser.parse_args(argv)
    try:
        print(f"Python: {platform.python_version()} ({sys.executable})")
        for package, module in (("chess", "chess"), ("requests", "requests"), ("python-dotenv", "dotenv"), ("pydantic", "pydantic")):
            importlib.import_module(module)
            print(f"{package}: {version(package)}")
        for module in ("AI_process", "chess_part", "opening_agent", "middlegame_agent", "endgame_agent", "data.data_collection", "analysis_records"):
            importlib.import_module(module)
        print("Project modules: import successfully")
        path = resolve_stockfish_path()
        print(f"Stockfish path: {path}")
        if args.engine:
            import chess
            import chess.engine

            board = chess.Board()
            try:
                with chess.engine.SimpleEngine.popen_uci(path, timeout=5.0) as engine:
                    info = engine.analyse(board, chess.engine.Limit(time=0.1))
                    if not info.get("pv") or info["pv"][0] not in board.legal_moves:
                        raise ValueError("Stockfish did not return a legal first move")
                    print(f"Stockfish: {engine.id.get('name', 'unknown')}; legal smoke move: {board.san(info['pv'][0])}")
            except chess.engine.EngineError as error:
                raise RuntimeError(f"Stockfish UCI check failed: {error}") from error
        print("Setup checks passed. This does not validate the complete coaching pipeline.")
        return 0
    except (ImportError, OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"Setup check failed: {error}\nActivate the project Conda environment and follow README.md.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
