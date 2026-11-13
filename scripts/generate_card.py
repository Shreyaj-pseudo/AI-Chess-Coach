"""Generate one unapproved draft from an AnalysisRecord JSON, not a whole review."""

import argparse
from pathlib import Path
import sys

from analysis_records import AnalysisRecord
from model_client import MAX_INPUT_BYTES, ModelClient, ModelError
from grounded_cards import generate_grounded_card


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.file.open("rb") as stream:
            source = stream.read(MAX_INPUT_BYTES + 1)
        if len(source) > MAX_INPUT_BYTES:
            raise ValueError
        encoding = "utf-16" if source.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        analysis = AnalysisRecord.model_validate_json(source.decode(encoding))
    except (OSError, ValueError):
        print("Draft generation failed: supply a readable, valid AnalysisRecord JSON of at most 64 KiB.", file=sys.stderr)
        return 1
    try:
        result = generate_grounded_card(ModelClient.from_env(), analysis)
        print(result.model_dump_json(indent=2))
        return 0
    except ModelError as error:
        print(f"Draft generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
