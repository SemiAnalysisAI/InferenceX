#!/usr/bin/env python3
"""Analyze ISL/OSL/turn distributions from AIPerf benchmark results.

Reads profile_export.jsonl and produces summary stats + distribution plots
to verify the benchmark workload matches the intended Qwen trace profile.

Usage:
    python analyze_benchmark_distributions.py path/to/aiperf_artifacts/ -o output_dir/
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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


def analyze(records: list[dict], output_dir: Path) -> None:
    """Run distribution analysis and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by conversation
    convos: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cid = r["metadata"]["conversation_id"]
        ti = r["metadata"]["turn_index"]
        isl = r["metrics"]["input_sequence_length"]["value"]
        osl = r["metrics"]["output_sequence_length"]["value"]
        convos[cid].append({"turn": ti, "isl": isl, "osl": osl})

    # Sort turns within each conversation
    for v in convos.values():
        v.sort(key=lambda x: x["turn"])

    # Turn count distribution
    turn_counts = Counter(len(v) for v in convos.values())
    total_convos = len(convos)
    total_requests = len(records)

    lines = []
    lines.append("=" * 70)
    lines.append("BENCHMARK WORKLOAD DISTRIBUTION ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Total conversations: {total_convos:,}")
    lines.append(f"Total requests: {total_requests:,}")
    lines.append(f"Avg turns/conv: {total_requests / total_convos:.2f}")
    lines.append("")

    lines.append("TURN COUNT DISTRIBUTION:")
    lines.append(f"  {'Turns':>5s}  {'Count':>6s}  {'Pct':>6s}   Target")
    target = {1: 59, 2: 20, 3: 10, 4: 5, 5: 3, 6: 2, 7: 1}
    for k in sorted(turn_counts.keys()):
        pct = 100 * turn_counts[k] / total_convos
        tgt = f"{target.get(k, 0):.0f}%" if k in target else ""
        lines.append(f"  {k:5d}  {turn_counts[k]:6,}  {pct:5.1f}%   {tgt}")

    # ISL/OSL by turn index
    lines.append("")
    lines.append("ISL BY TURN INDEX:")
    lines.append(
        f"  {'Turn':>4s}  {'N':>6s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'P5':>8s}  {'P95':>8s}"
    )
    max_turn = max(t["turn"] for v in convos.values() for t in v)
    for ti in range(max_turn + 1):
        vals = sorted(t["isl"] for v in convos.values() for t in v if t["turn"] == ti)
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        median = vals[n // 2]
        p5 = vals[int(n * 0.05)]
        p95 = vals[int(n * 0.95)]
        lines.append(
            f"  {ti:4d}  {n:6,}  {mean:8.0f}  {median:8.0f}  {std:8.0f}  {p5:8.0f}  {p95:8.0f}"
        )

    lines.append("")
    lines.append("OSL BY TURN INDEX:")
    lines.append(
        f"  {'Turn':>4s}  {'N':>6s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'P5':>8s}  {'P95':>8s}"
    )
    for ti in range(max_turn + 1):
        vals = sorted(t["osl"] for v in convos.values() for t in v if t["turn"] == ti)
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        median = vals[n // 2]
        p5 = vals[int(n * 0.05)]
        p95 = vals[int(n * 0.95)]
        lines.append(
            f"  {ti:4d}  {n:6,}  {mean:8.0f}  {median:8.0f}  {std:8.0f}  {p5:8.0f}  {p95:8.0f}"
        )

    # Overall ISL/OSL stats
    all_isl = sorted(t["isl"] for v in convos.values() for t in v)
    all_osl = sorted(t["osl"] for v in convos.values() for t in v)
    n = len(all_isl)
    isl_mean = sum(all_isl) / n
    osl_mean = sum(all_osl) / n
    lines.append("")
    lines.append("ALL REQUESTS ISL:")
    lines.append(
        f"  n={n:,}  mean={isl_mean:.0f}  median={all_isl[n//2]}  "
        f"p5={all_isl[int(n*0.05)]}  p95={all_isl[int(n*0.95)]}"
    )
    lines.append("ALL REQUESTS OSL:")
    lines.append(
        f"  n={n:,}  mean={osl_mean:.0f}  median={all_osl[n//2]}  "
        f"p5={all_osl[int(n*0.05)]}  p95={all_osl[int(n*0.95)]}"
    )

    # ISL context growth (shows accumulation across turns)
    lines.append("")
    lines.append("ISL CONTEXT GROWTH (sample multi-turn conversations):")
    multi = [(cid, v) for cid, v in convos.items() if len(v) >= 3][:10]
    for cid, turns in multi:
        isls = " -> ".join(str(t["isl"]) for t in turns)
        lines.append(f"  {cid}: {isls}")

    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)

    # Save summary
    (output_dir / "workload_distribution_summary.txt").write_text(summary_text)

    # Try to generate plots (matplotlib may not be available)
    try:
        _generate_plots(convos, records, output_dir)
    except ImportError:
        print("matplotlib not available, skipping plots")


def _generate_plots(
    convos: dict[str, list[dict]], records: list[dict], output_dir: Path
) -> None:
    """Generate distribution plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Benchmark Workload Distribution Analysis", fontsize=14)

    # (0,0) Turn count distribution
    ax = axes[0, 0]
    turn_counts = Counter(len(v) for v in convos.values())
    turns = sorted(turn_counts.keys())
    counts = [turn_counts[t] for t in turns]
    total = sum(counts)
    bars = ax.bar(turns, [100 * c / total for c in counts], edgecolor="black", alpha=0.7)
    for bar, t in zip(bars, turns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("% of Conversations")
    ax.set_title(f"Turn Count Distribution (n={total:,})")
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) Turn 0 ISL histogram
    ax = axes[0, 1]
    t0_isl = [t["isl"] for v in convos.values() for t in v if t["turn"] == 0]
    clip = min(3000, int(sorted(t0_isl)[int(len(t0_isl) * 0.99)] * 1.2))
    ax.hist([v for v in t0_isl if v <= clip], bins=60, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(sorted(t0_isl)[len(t0_isl) // 2], color="red", linestyle="--", label=f"Median: {sorted(t0_isl)[len(t0_isl)//2]:,}")
    ax.set_xlabel("Input Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title("Turn 0 ISL Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,2) Turn 1+ ISL histogram
    ax = axes[0, 2]
    later_isl = [t["isl"] for v in convos.values() for t in v if t["turn"] > 0]
    if later_isl:
        clip = min(5000, int(sorted(later_isl)[int(len(later_isl) * 0.99)] * 1.2))
        ax.hist([v for v in later_isl if v <= clip], bins=60, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(sorted(later_isl)[len(later_isl) // 2], color="red", linestyle="--", label=f"Median: {sorted(later_isl)[len(later_isl)//2]:,}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Input Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title("Turn 1+ ISL Distribution (cumulative context)")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,0) Turn 0 OSL histogram
    ax = axes[1, 0]
    t0_osl = [t["osl"] for v in convos.values() for t in v if t["turn"] == 0]
    clip = min(3000, int(sorted(t0_osl)[int(len(t0_osl) * 0.99)] * 1.2))
    ax.hist([v for v in t0_osl if v <= clip], bins=60, edgecolor="black", alpha=0.7, color="coral")
    ax.axvline(sorted(t0_osl)[len(t0_osl) // 2], color="red", linestyle="--", label=f"Median: {sorted(t0_osl)[len(t0_osl)//2]:,}")
    ax.set_xlabel("Output Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title("Turn 0 OSL Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) Turn 1+ OSL histogram
    ax = axes[1, 1]
    later_osl = [t["osl"] for v in convos.values() for t in v if t["turn"] > 0]
    if later_osl:
        clip = min(3000, int(sorted(later_osl)[int(len(later_osl) * 0.99)] * 1.2))
        ax.hist([v for v in later_osl if v <= clip], bins=60, edgecolor="black", alpha=0.7, color="coral")
        ax.axvline(sorted(later_osl)[len(later_osl) // 2], color="red", linestyle="--", label=f"Median: {sorted(later_osl)[len(later_osl)//2]:,}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Output Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title("Turn 1+ OSL Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,2) ISL growth by turn number (mean + std)
    ax = axes[1, 2]
    max_turn = max(t["turn"] for v in convos.values() for t in v)
    turn_range = range(min(max_turn + 1, 8))
    means, stds, ns = [], [], []
    for ti in turn_range:
        vals = [t["isl"] for v in convos.values() for t in v if t["turn"] == ti]
        if vals:
            m = sum(vals) / len(vals)
            means.append(m)
            stds.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
            ns.append(len(vals))
        else:
            means.append(0)
            stds.append(0)
            ns.append(0)
    ax.bar(list(turn_range), means, yerr=stds, capsize=3, edgecolor="black", alpha=0.7, color="mediumseagreen")
    for i, (m, n) in enumerate(zip(means, ns)):
        if n > 0:
            ax.text(i, m, f"n={n:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Mean ISL (tokens)")
    ax.set_title("ISL Growth by Turn (context accumulation)")
    ax.grid(True, alpha=0.3, axis="y")

    # (2,0) All requests ISL histogram
    ax = axes[2, 0]
    all_isl = [t["isl"] for v in convos.values() for t in v]
    clip = min(6000, int(sorted(all_isl)[int(len(all_isl) * 0.99)] * 1.2))
    ax.hist([v for v in all_isl if v <= clip], bins=80, edgecolor="black", alpha=0.7, color="steelblue")
    all_isl_sorted = sorted(all_isl)
    median_isl = all_isl_sorted[len(all_isl) // 2]
    mean_isl = sum(all_isl) / len(all_isl)
    ax.axvline(median_isl, color="red", linestyle="--", label=f"Median: {median_isl:,}")
    ax.axvline(mean_isl, color="orange", linestyle="--", label=f"Mean: {mean_isl:,.0f}")
    ax.set_xlabel("Input Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title(f"All Requests ISL (n={len(all_isl):,})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (2,1) All requests OSL histogram
    ax = axes[2, 1]
    all_osl = [t["osl"] for v in convos.values() for t in v]
    clip = min(3000, int(sorted(all_osl)[int(len(all_osl) * 0.99)] * 1.2))
    ax.hist([v for v in all_osl if v <= clip], bins=80, edgecolor="black", alpha=0.7, color="coral")
    all_osl_sorted = sorted(all_osl)
    median_osl = all_osl_sorted[len(all_osl) // 2]
    mean_osl = sum(all_osl) / len(all_osl)
    ax.axvline(median_osl, color="red", linestyle="--", label=f"Median: {median_osl:,}")
    ax.axvline(mean_osl, color="orange", linestyle="--", label=f"Mean: {mean_osl:,.0f}")
    ax.set_xlabel("Output Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title(f"All Requests OSL (n={len(all_osl):,})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (2,2) ISL vs OSL scatter
    ax = axes[2, 2]
    ax.scatter(all_isl, all_osl, alpha=0.15, s=3, c="purple")
    ax.set_xlabel("ISL (tokens)")
    ax.set_ylabel("OSL (tokens)")
    ax.set_title("ISL vs OSL (all requests)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "workload_distribution_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark workload distributions"
    )
    parser.add_argument("artifacts_dir", help="Path to aiperf_artifacts/ directory")
    parser.add_argument(
        "-o", "--output", default=None, help="Output directory (default: same as artifacts_dir)"
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output) if args.output else artifacts_dir

    records = load_records(artifacts_dir)
    print(f"Loaded {len(records):,} records from {artifacts_dir}")
    analyze(records, output_dir)


if __name__ == "__main__":
    main()
