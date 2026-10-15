"""Export one JSON Schema per published row model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .models import AGENTX_ROW, EVAL_ROW, FIXED_SEQUENCE_ROW, RUN_STATS_ROW

DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMAS: dict[str, tuple[str, TypeAdapter[Any]]] = {
    "fixed_sequence_row": ("FixedSequenceRow", FIXED_SEQUENCE_ROW),
    "agentx_row": ("AgentXRow", AGENTX_ROW),
    "eval_row": ("EvalRow", EVAL_ROW),
    "run_stats_row": ("RunStatsRow", RUN_STATS_ROW),
}


def export(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (title, adapter) in SCHEMAS.items():
        document = {"$schema": DIALECT, "title": title, **adapter.json_schema()}
        path = directory / f"{name}.schema.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m infx.results.schema", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_command = commands.add_parser("export", help="write <name>.schema.json files")
    export_command.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    for path in export(args.directory):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
