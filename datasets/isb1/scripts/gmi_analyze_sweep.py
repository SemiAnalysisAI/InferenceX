#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

from isb1_results_db import render_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KV sweep runs from ISB1 SQLite results.")
    parser.add_argument("--db-path", required=True, help="Path to SQLite DB (isb1_results.db)")
    parser.add_argument("--output-dir", default=".", help="Directory to write summary outputs")
    parser.add_argument("--pareto", action="store_true", help="Also run plot_pareto.py")
    parser.add_argument(
        "--distributions",
        action="store_true",
        help="Also run analyze_benchmark_distributions.py",
    )
    parser.add_argument("--export-file", default=None, help="Export JSON for --distributions")
    parser.add_argument("--trace-dir", default=None, help="Trace directory for --distributions")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_concurrency(raw_result_json: str | None) -> int | None:
    if not raw_result_json:
        return None
    try:
        payload = json.loads(raw_result_json)
    except json.JSONDecodeError:
        return None
    return _to_int(payload.get("conc") or payload.get("max_concurrency"))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def load_rows(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
          id,
          offload_mode,
          ttft_p50_ms,
          ttft_p99_ms,
          throughput_tok_s,
          preemption_count,
          status,
          raw_result_json
        FROM benchmark_runs
        WHERE offload_mode IS NOT NULL
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()

    normalized: list[dict[str, Any]] = []
    for row in rows:
        concurrency = _extract_concurrency(row["raw_result_json"])
        normalized.append(
            {
                "offload_mode": row["offload_mode"],
                "concurrency": concurrency,
                "ttft_p50_ms": _to_float(row["ttft_p50_ms"]),
                "ttft_p99_ms": _to_float(row["ttft_p99_ms"]),
                "throughput_tok_s": _to_float(row["throughput_tok_s"]),
                "preemption_count": _to_int(row["preemption_count"]) or 0,
                "status": row["status"],
            }
        )
    return normalized


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["concurrency"] is None:
            continue
        key = (row["offload_mode"], row["concurrency"])
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (offload_mode, concurrency), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        ttft_p50_values = [x["ttft_p50_ms"] for x in items if x["ttft_p50_ms"] is not None]
        ttft_p99_values = [x["ttft_p99_ms"] for x in items if x["ttft_p99_ms"] is not None]
        throughput_values = [x["throughput_tok_s"] for x in items if x["throughput_tok_s"] is not None]
        preemptions = [x["preemption_count"] for x in items]
        success_count = sum(1 for x in items if x["status"] == "success")

        summary_rows.append(
            {
                "offload_mode": offload_mode,
                "concurrency": concurrency,
                "runs": len(items),
                "success_runs": success_count,
                "ttft_p50_ms": median(ttft_p50_values) if ttft_p50_values else None,
                "ttft_p99_ms": percentile(ttft_p99_values, 0.99),
                "throughput_tok_s": median(throughput_values) if throughput_values else None,
                "preemptions": int(median(preemptions)) if preemptions else 0,
            }
        )

    return {
        "total_rows": len(rows),
        "grouped_rows": len(summary_rows),
        "summary": summary_rows,
    }


def write_summary_json(output_dir: Path, summary: dict[str, Any]) -> Path:
    output_path = output_dir / "sweep_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path


def write_pareto_csv(output_dir: Path, summary: dict[str, Any]) -> Path:
    output_path = output_dir / "pareto_data.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["offload_mode", "concurrency", "throughput_tok_s", "ttft_p99_ms"])
        for row in summary["summary"]:
            writer.writerow(
                [
                    row["offload_mode"],
                    row["concurrency"],
                    row["throughput_tok_s"],
                    row["ttft_p99_ms"],
                ]
            )
    return output_path


def print_console_summary(summary: dict[str, Any]) -> None:
    headers = [
        "offload_mode",
        "concurrency",
        "runs",
        "success_runs",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "throughput_tok_s",
        "preemptions",
    ]
    rows = [
        [
            row["offload_mode"],
            row["concurrency"],
            row["runs"],
            row["success_runs"],
            row["ttft_p50_ms"],
            row["ttft_p99_ms"],
            row["throughput_tok_s"],
            row["preemptions"],
        ]
        for row in summary["summary"]
    ]

    print(f"Total rows: {summary['total_rows']}")
    print(f"Grouped rows: {summary['grouped_rows']}")
    if rows:
        print(render_table(headers, rows))
    else:
        print("No sweep rows with offload_mode + concurrency found.")


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(db_path)
    summary = summarize(rows)
    summary_path = write_summary_json(output_dir, summary)
    pareto_path = write_pareto_csv(output_dir, summary)

    print_console_summary(summary)
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {pareto_path}")

    script_dir = Path(__file__).resolve().parent

    if args.pareto:
        pareto_cmd = [
            sys.executable,
            str(script_dir / "plot_pareto.py"),
            "--db-path",
            str(db_path),
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(pareto_cmd, check=True)

    if args.distributions:
        dist_cmd = [
            sys.executable,
            str(script_dir / "analyze_benchmark_distributions.py"),
            "--output-dir",
            str(output_dir),
        ]
        if args.export_file:
            dist_cmd.extend(["--export-file", args.export_file])
        elif args.trace_dir:
            dist_cmd.extend(["--trace-dir", args.trace_dir])
        else:
            raise SystemExit("--distributions requires --export-file or --trace-dir")
        subprocess.run(dist_cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
