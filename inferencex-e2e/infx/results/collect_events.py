"""Join a sweep's per-job ``job_event.json`` records into one JSON Lines file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FILENAME = "job_event.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m infx.results.collect_events", description=__doc__
    )
    parser.add_argument(
        "artifacts", type=Path, help="directory of downloaded job_event_* artifacts"
    )
    parser.add_argument("output", type=Path, help="JSON Lines file to write")
    args = parser.parse_args(argv)
    lines = []
    for path in sorted(args.artifacts.rglob(FILENAME)):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"WARNING: skipping {path}: {error}", file=sys.stderr)
            continue
        if not isinstance(record, dict):
            print(f"WARNING: skipping {path}: not a JSON object", file=sys.stderr)
            continue
        lines.append(json.dumps(record, separators=(",", ":")) + "\n")
    args.output.write_text("".join(lines))
    print(f"Collected {len(lines)} job events into {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
