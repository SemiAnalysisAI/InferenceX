"""Analyze ISL/OSL distributions from AIPerf benchmark results.

Reads profile_export.jsonl and produces mean/median/p75/p90/p95 ISL and OSL
summary stats.

Usage:
    python analyze_benchmark_distributions.py path/to/aiperf_artifacts/ -o output_dir/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(artifacts_dir: Path) -> list[dict]:
    """Load per-request records from profile_export.jsonl."""
    jsonl_path = artifacts_dir / "profile_export.jsonl"
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _stats(values: list[int]) -> dict[str, float]:
    sv = sorted(values)
    n = len(sv)
    return {
        "n": n,
        "mean": sum(sv) / n,
        "median": sv[n // 2],
        "p75": sv[int(n * 0.75)],
        "p90": sv[int(n * 0.90)],
        "p95": sv[int(n * 0.95)],
    }


def _fmt(s: dict[str, float]) -> str:
    return (
        f"  n={s['n']:,}  mean={s['mean']:,.0f}  median={s['median']:,}  "
        f"p75={s['p75']:,}  p90={s['p90']:,}  p95={s['p95']:,}"
    )


def analyze(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_isl: list[int] = []
    all_osl: list[int] = []
    for r in records:
        metrics = r.get("metrics", {})
        if "input_sequence_length" not in metrics or "output_sequence_length" not in metrics:
            continue
        all_isl.append(metrics["input_sequence_length"]["value"])
        all_osl.append(metrics["output_sequence_length"]["value"])

    if not all_isl:
        print("No records with ISL/OSL metrics found.")
        return

    isl_stats = _stats(all_isl)
    osl_stats = _stats(all_osl)

    lines = [
        "=" * 70,
        "BENCHMARK WORKLOAD DISTRIBUTION ANALYSIS",
        "=" * 70,
        f"Total requests: {len(records):,}",
        "",
        "ALL REQUESTS ISL:",
        _fmt(isl_stats),
        "ALL REQUESTS OSL:",
        _fmt(osl_stats),
        "=" * 70,
    ]
    summary_text = "\n".join(lines)
    print(summary_text)
    (output_dir / "workload_distribution_summary.txt").write_text(summary_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark workload distributions")
    parser.add_argument("artifacts_dir", help="Path to aiperf_artifacts/ directory")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: same as artifacts_dir)",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output) if args.output else artifacts_dir

    aiperf_jsonl = artifacts_dir / "profile_export.jsonl"
    if not aiperf_jsonl.exists():
        print(f"No profile_export.jsonl found in {artifacts_dir}")
        return

    records = load_records(artifacts_dir)
    print(f"Loaded {len(records):,} records from {artifacts_dir}")

    analyze(records, output_dir)


if __name__ == "__main__":
    main()
