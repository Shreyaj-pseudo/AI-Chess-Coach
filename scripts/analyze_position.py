"""Analyze one PGN decision: python -m scripts.analyze_position FILE --color white --ply 3."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

from analysis_records import PositionContext, source_game_id
from candidate_analysis import analyze_position
from board_facts import build_board_evidence
from engine_session import StockfishSession
from pgn_input import MAX_PGN_BYTES, learner_positions, parse_pgn


async def _analyze(position, time_limit):
    async with StockfishSession() as session:
        return await analyze_position(session, position, time_limit=time_limit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--color", required=True, choices=("white", "black"))
    parser.add_argument("--ply", type=int, required=True, help="Relative half-move number shown by inspect_pgn")
    parser.add_argument("--time", type=float, default=1.5, help="Seconds per search, positive and at most 5")
    parser.add_argument("--facts", action="store_true", help="Include board facts and changes at every line ply")
    args = parser.parse_args(argv)
    try:
        with args.file.open("rb") as stream:
            game = parse_pgn(stream.read(MAX_PGN_BYTES + 1))
        decision = next((d for d in learner_positions(game, args.color) if d.ply == args.ply), None)
        if decision is None:
            raise ValueError(f"No {args.color} decision at relative ply {args.ply}; inspect the PGN first.")
        position = PositionContext.from_decision(decision, game_id=source_game_id(game))
        record = asyncio.run(_analyze(position, args.time))
        if args.facts:
            evidence = build_board_evidence(record)
            print(json.dumps({"analysis": record.model_dump(mode="json"), "board_evidence": evidence.model_dump(mode="json")}, indent=2))
        else:
            print(record.model_dump_json(indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Position analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
