"""Inspect validated pre-move positions: python -m scripts.inspect_pgn FILE --color white."""

import argparse
from pathlib import Path
import sys

from pgn_input import MAX_PGN_BYTES, PGNError, learner_positions, parse_pgn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--color", required=True, choices=("white", "black"))
    args = parser.parse_args(argv)
    try:
        with args.file.open("rb") as stream:
            game = parse_pgn(stream.read(MAX_PGN_BYTES + 1))
        positions = learner_positions(game, args.color)
        print(f"Valid standard game: {game.end().ply() - game.ply()} plies; {len(positions)} {args.color} decisions")
        for position in positions:
            print(f"Ply {position.ply}: {position.san} ({position.uci}) | before: {position.fen}")
        print(f"Final FEN: {game.end().board().fen()}")
        return 0
    except (PGNError, OSError) as error:
        print(f"PGN rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
