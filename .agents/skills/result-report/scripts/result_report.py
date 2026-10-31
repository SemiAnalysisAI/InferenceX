#!/usr/bin/env python3
"""result-report: InferenceX benchmark run(s) -> report.md + data.json (+ report.html).

    python3 .agents/skills/result-report/scripts/result_report.py <run-url|run-id|dir>... [--html]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract  # noqa: E402
import fetch  # noqa: E402
import render  # noqa: E402


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "report"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="GitHub Actions run URL, run id, or local folder")
    ap.add_argument("--repo", default=fetch.DEFAULT_REPO, help="repo for bare run ids")
    ap.add_argument("--baseline", help="arm label to compare against (default: first arm)")
    ap.add_argument("--out", type=Path, help="output folder (default ~/bench-reports/<date>-<slug>)")
    ap.add_argument("--title")
    ap.add_argument("--html", action="store_true", help="also write report.html with charts")
    ap.add_argument("--findings", type=Path, help="markdown bullets that replace the Key findings placeholder")
    ap.add_argument("--max-full-zip-mb", type=float, default=100,
                    help="largest artifact zip to download whole when Range reads fail")
    ap.add_argument("--max-member-mb", type=float, default=extract.MAX_MEMBER_MB,
                    help="largest compressed server_metrics_export.json read for a search winner's cache tiers")
    args = ap.parse_args(argv)

    if args.findings and not args.findings.is_file():
        print(f"ERROR --findings file not found: {args.findings}", file=sys.stderr)
        return 2
    try:
        sources = [fetch.parse_source(a, args.repo) for a in args.inputs]
    except ValueError as err:
        print(f"ERROR {err}", file=sys.stderr)
        return 2
    today = datetime.date.today().isoformat()
    names = [s.run_id or s.path.name for s in sources]
    out = args.out or Path.home() / "bench-reports" / f"{today}-{slugify(args.title or '-'.join(names))}"
    cache = out / ".cache"
    fetch.FAILED_JOBS.clear()
    before = fetch.FETCHED["bytes"]

    try:
        configs, notes = [], []
        for src in sources:
            found, found_notes = fetch.discover(src, cache, args.max_full_zip_mb)
            configs += found
            notes += found_notes
        if not configs:
            print("ERROR nothing to report:\n" + "\n".join(f"  {n}" for n in notes), file=sys.stderr)
            return 1
        data = extract.build_data(configs, notes, baseline=args.baseline, title=args.title,
                                  inputs=[s.link for s in sources], generated=today, max_member_mb=args.max_member_mb)
    except ValueError as err:
        print(f"ERROR {err}", file=sys.stderr)
        return 2
    except fetch.TooLarge as err:
        print(f"ERROR {err}\nAsk the user before downloading it; then re-run with --max-full-zip-mb <MB>.",
              file=sys.stderr)
        return 3
    except FileNotFoundError as err:
        if err.filename != "gh":  # an input file vanished mid-run, not a missing gh
            print(f"ERROR {err}", file=sys.stderr)
            return 2
        print(f"ERROR {err}: the gh CLI is required for run links (install it and run `gh auth login`)",
              file=sys.stderr)
        return 4
    except subprocess.CalledProcessError as err:
        print(f"ERROR gh failed: {' '.join(err.cmd)}\n{(err.stderr or '').strip()}", file=sys.stderr)
        return 4

    this_run = fetch.FETCHED["bytes"] - before
    stamp = cache / "fetched.json"  # cumulative download size survives cached --findings re-runs
    try:
        prior = int(json.loads(stamp.read_text()).get("bytes", 0)) if stamp.is_file() else 0
    except (ValueError, TypeError, AttributeError, OSError):  # corrupt stamp: start the total over
        print(f"warning: ignoring unreadable {stamp}", file=sys.stderr)
        prior = 0
    data["bytes_fetched"] = prior + this_run
    if fetch.FAILED_JOBS:
        data["failed_jobs"] = list(fetch.FAILED_JOBS)
    if this_run:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = stamp.with_suffix(".tmp")
        tmp.write_text(json.dumps({"bytes": data["bytes_fetched"]}))
        os.replace(tmp, stamp)
    if args.findings:
        data["findings"] = args.findings.read_text().strip()
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
    (out / "report.md").write_text(render.to_markdown(render.build_blocks(data)))
    written = ["report.md", "data.json"]
    if args.html:
        (out / "report.html").write_text(render.to_html(render.build_blocks(data, charts=True), data["title"]))
        written.append("report.html")
    print(f"wrote {', '.join(str(out / w) for w in written)}")
    print(f"{len(data['points'])} config(s), {len(data['arms'])} arm(s), baseline {data['baseline']!r}, "
          f"downloaded {this_run / 1e6:.2f} MB this run (total {data['bytes_fetched'] / 1e6:.2f} MB), "
          f"{len(data['notes'])} note(s)")
    for note in data["notes"]:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
