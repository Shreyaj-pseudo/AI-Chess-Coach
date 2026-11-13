"""Select whole-game review evidence: python -m scripts.review_game FILE --color white."""

import argparse
import asyncio
from pathlib import Path
import sys

from engine_session import StockfishSession
from game_review import review_game
from pgn_input import MAX_PGN_BYTES


async def _review(source, color, scan_time, deep_time):
    async with StockfishSession() as session:
        return await review_game(session, source, color, scan_time=scan_time, deep_time=deep_time)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--color", required=True, choices=("white", "black"))
    parser.add_argument("--scan-time", type=float, default=0.15)
    parser.add_argument("--deep-time", type=float, default=1.5)
    args = parser.parse_args(argv)
    try:
        with args.file.open("rb") as stream:
            source = stream.read(MAX_PGN_BYTES + 1)
        result = asyncio.run(_review(source, args.color, args.scan_time, args.deep_time))
        print(result.model_dump_json(indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Game review failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
