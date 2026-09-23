#!/usr/bin/env python3
"""Per-chip dedup pass.

For each results/<platform>/<chip>/ directory:
  1. Read every *.json run file.
  2. For each unique (op_type, args, backend) tuple, keep the row from the
     latest run (by run.started_at).
  3. Emit a single new run file with id/timestamp=NOW, run metadata copied
     from the most-recent input run for that chip.
  4. Delete the original files.

Run from the project root (the directory containing `results/`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
from pathlib import Path


def _row_key(row: dict) -> tuple:
    op = row["op"]
    args = op.get("args", {})
    # Deterministic ordering for dict; args are JSON-loaded so plain dicts.
    args_t = tuple(sorted(args.items(), key=lambda kv: kv[0]))
    return (op["type"], op.get("backend"), args_t)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(cluster: str | None) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)[:5]
    cl = cluster or "unknown"
    return f"{stamp}-{cl}-{suffix}"


def consolidate_chip(chip_dir: Path, dry_run: bool) -> None:
    files = sorted(chip_dir.glob("*.json"))
    if not files:
        return

    # Map key -> (started_at, row); keep latest.
    latest: dict[tuple, tuple[str, dict]] = {}
    latest_run_meta: dict | None = None
    latest_started = ""

    for f in files:
        data = json.loads(f.read_text())
        run = data.get("run", {})
        started = run.get("started_at", "")
        if started >= latest_started:
            latest_started = started
            latest_run_meta = run
        for row in data.get("rows", []):
            k = _row_key(row)
            prev = latest.get(k)
            if prev is None or started > prev[0]:
                latest[k] = (started, row)

    if not latest_run_meta:
        return

    # Build the consolidated run record.
    cluster = latest_run_meta.get("cluster")
    new_id = _run_id(cluster)
    now_iso = _utc_now_iso()
    new_run = {
        **latest_run_meta,
        "id": new_id,
        "started_at": now_iso,
        "finished_at": now_iso,
    }

    rows = [r for _, r in latest.values()]
    out = {"schema_version": "1", "run": new_run, "rows": rows}

    out_path = chip_dir / f"{new_id}.json"
    print(f"{chip_dir}: {len(files)} files -> 1 file ({len(rows)} unique rows) -> {out_path.name}")

    if dry_run:
        return

    out_path.write_text(json.dumps(out, indent=2))
    # Remove the originals (excluding the freshly-written one).
    for f in files:
        if f != out_path:
            f.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for chip_dir in sorted(args.results_dir.glob("*/*")):
        if chip_dir.is_dir():
            consolidate_chip(chip_dir, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
