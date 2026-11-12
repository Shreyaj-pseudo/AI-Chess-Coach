"""Engine-free opening context: python -m scripts.inspect_opening FILE --color white --ply 3."""

import argparse
from pathlib import Path
import sys

from analysis_records import PositionContext, source_game_id
from opening_explorer import DEFAULT_CACHE, OpeningExplorer
from pgn_input import MAX_PGN_BYTES, learner_positions, parse_pgn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--color", required=True, choices=("white", "black"))
    parser.add_argument("--ply", required=True, type=int, help="Relative half-move number from inspect_pgn")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Local SQLite cache path")
    args = parser.parse_args(argv)
    try:
        with args.file.open("rb") as stream:
            game = parse_pgn(stream.read(MAX_PGN_BYTES + 1))
        decision = next((d for d in learner_positions(game, args.color) if d.ply == args.ply), None)
        if decision is None:
            raise ValueError(f"No {args.color} decision at relative ply {args.ply}.")
        position = PositionContext.from_decision(decision, game_id=source_game_id(game))
        with OpeningExplorer(args.cache) as explorer:
            print(explorer.lookup(position).model_dump_json(indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Opening inspection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
