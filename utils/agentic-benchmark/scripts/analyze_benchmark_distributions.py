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


def load_trace_replay_records(trace_replay_dir: Path) -> list[dict]:
    """Load per-request records from trace_replay detailed_results.csv.

    Converts to the same format as AIPerf JSONL records so the analyze()
    function can process both formats identically.
    """
    import csv
    import sys
    csv.field_size_limit(sys.maxsize)

    csv_path = trace_replay_dir / "detailed_results.csv"
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("success") != "True":
                continue
            records.append({
                "metadata": {
                    "x_correlation_id": row["trace_id"],
                    "conversation_id": row["trace_id"],
                    "turn_index": int(row["request_idx"]),
                    "benchmark_phase": "profiling",
                },
                "metrics": {
                    "input_sequence_length": {"value": int(row["input_tokens"])},
                    "output_sequence_length": {"value": int(row["output_tokens_actual"])},
                },
            })
    return records


def analyze(records: list[dict], output_dir: Path) -> None:
    """Run distribution analysis and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by conversation
    convos: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        metrics = r.get("metrics", {})
        if "input_sequence_length" not in metrics or "output_sequence_length" not in metrics:
            continue
        # Use x_correlation_id (unique per session) not conversation_id (template, reused)
        cid = r["metadata"].get("x_correlation_id") or r["metadata"]["conversation_id"]
        ti = r["metadata"]["turn_index"]
        isl = metrics["input_sequence_length"]["value"]
        osl = metrics["output_sequence_length"]["value"]
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

    # Per-conversation stats
    conv_max_isl = sorted(max(t["isl"] for t in v) for v in convos.values())
    conv_total_osl = sorted(sum(t["osl"] for t in v) for v in convos.values())
    nc = len(conv_max_isl)
    lines.append("")
    lines.append("PER-CONVERSATION MAX ISL (final context size):")
    lines.append(
        f"  n={nc:,}  mean={sum(conv_max_isl)/nc:.0f}  median={conv_max_isl[nc//2]}  "
        f"p5={conv_max_isl[int(nc*0.05)]}  p95={conv_max_isl[int(nc*0.95)]}"
    )
    lines.append("PER-CONVERSATION TOTAL OSL:")
    lines.append(
        f"  n={nc:,}  mean={sum(conv_total_osl)/nc:.0f}  median={conv_total_osl[nc//2]}  "
        f"p5={conv_total_osl[int(nc*0.05)]}  p95={conv_total_osl[int(nc*0.95)]}"
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

    # (0,1) All requests ISL histogram
    ax = axes[0, 1]
    all_isl = [t["isl"] for v in convos.values() for t in v]
    clip = int(sorted(all_isl)[int(len(all_isl) * 0.99)] * 1.2)
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

    # (0,2) All requests OSL histogram
    ax = axes[0, 2]
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

    # (1,0) Average new prefill tokens by turn index (ISL delta per turn)
    ax = axes[1, 0]
    # Collect deltas grouped by turn index
    deltas_by_turn: dict[int, list[int]] = defaultdict(list)
    for v in convos.values():
        for i, t in enumerate(v):
            if i == 0:
                deltas_by_turn[t["turn"]].append(t["isl"])
            else:
                deltas_by_turn[t["turn"]].append(max(0, t["isl"] - v[i - 1]["isl"]))
    if deltas_by_turn:
        turn_indices = sorted(deltas_by_turn.keys())
        means = [sum(deltas_by_turn[ti]) / len(deltas_by_turn[ti]) for ti in turn_indices]
        ns = [len(deltas_by_turn[ti]) for ti in turn_indices]
        ax.plot(turn_indices, means, marker="o", markersize=3, linewidth=1, color="mediumseagreen")
        ax.fill_between(turn_indices, 0, means, alpha=0.2, color="mediumseagreen")
        # Label first and last points
        if len(turn_indices) > 0:
            ax.annotate(f"{means[0]:,.0f}", (turn_indices[0], means[0]), fontsize=7, ha="left", va="bottom")
        if len(turn_indices) > 1:
            ax.annotate(f"{means[-1]:,.0f}\n(n={ns[-1]})", (turn_indices[-1], means[-1]), fontsize=7, ha="right", va="bottom")
    # Overall mean/median across all deltas
    all_deltas = [d for dlist in deltas_by_turn.values() for d in dlist]
    if all_deltas:
        overall_mean = sum(all_deltas) / len(all_deltas)
        all_deltas_sorted = sorted(all_deltas)
        overall_median = all_deltas_sorted[len(all_deltas) // 2]
        ax.axhline(overall_mean, color="orange", linestyle="--", linewidth=1, label=f"Mean: {overall_mean:,.0f}")
        ax.axhline(overall_median, color="red", linestyle="--", linewidth=1, label=f"Median: {overall_median:,}")
        ax.legend(fontsize=7)
    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Mean New Prefill Tokens")
    ax.set_title("Avg New Prefill Tokens by Turn")
    ax.grid(True, alpha=0.3)

    # (1,1) ISL vs OSL scatter
    ax = axes[1, 1]
    ax.scatter(all_isl, all_osl, alpha=0.15, s=3, c="purple")
    ax.set_xlabel("ISL (tokens)")
    ax.set_ylabel("OSL (tokens)")
    ax.set_title("ISL vs OSL (all requests)")
    ax.grid(True, alpha=0.3)

    # (1,2) Per-conversation max ISL vs num turns scatter
    ax = axes[1, 2]
    conv_turns = [len(v) for v in convos.values()]
    conv_max_isl_list = [max(t["isl"] for t in v) for v in convos.values()]
    ax.scatter(conv_turns, conv_max_isl_list, alpha=0.3, s=8, c="steelblue")
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Max ISL (tokens)")
    ax.set_title("Final Context Size vs Turn Count")
    ax.grid(True, alpha=0.3)

    # (2,0) Per-conversation max ISL (final context size per conversation)
    ax = axes[2, 0]
    conv_max_isl = [max(t["isl"] for t in v) for v in convos.values()]
    clip = int(sorted(conv_max_isl)[int(len(conv_max_isl) * 0.99)] * 1.2)
    ax.hist([v for v in conv_max_isl if v <= clip], bins=60, edgecolor="black", alpha=0.7, color="steelblue")
    conv_max_isl_sorted = sorted(conv_max_isl)
    median_max = conv_max_isl_sorted[len(conv_max_isl) // 2]
    mean_max = sum(conv_max_isl) / len(conv_max_isl)
    ax.axvline(median_max, color="red", linestyle="--", label=f"Median: {median_max:,}")
    ax.axvline(mean_max, color="orange", linestyle="--", label=f"Mean: {mean_max:,.0f}")
    ax.set_xlabel("Max ISL per Conversation (tokens)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Conversation Final Context Size (n={len(conv_max_isl):,})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (3,1) Per-conversation total OSL (sum of all output tokens across turns)
    ax = axes[2, 1]
    conv_total_osl = [sum(t["osl"] for t in v) for v in convos.values()]
    clip = int(sorted(conv_total_osl)[int(len(conv_total_osl) * 0.99)] * 1.2)
    ax.hist([v for v in conv_total_osl if v <= clip], bins=60, edgecolor="black", alpha=0.7, color="coral")
    conv_total_osl_sorted = sorted(conv_total_osl)
    median_tosl = conv_total_osl_sorted[len(conv_total_osl) // 2]
    mean_tosl = sum(conv_total_osl) / len(conv_total_osl)
    ax.axvline(median_tosl, color="red", linestyle="--", label=f"Median: {median_tosl:,}")
    ax.axvline(mean_tosl, color="orange", linestyle="--", label=f"Mean: {mean_tosl:,.0f}")
    ax.set_xlabel("Total OSL per Conversation (tokens)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Conversation Total Output Tokens (n={len(conv_total_osl):,})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (2,2) is empty — already placed scatter at (1,2)
    axes[2, 2].axis("off")

    plt.tight_layout()
    out = output_dir / "workload_distribution_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark workload distributions"
    )
    parser.add_argument("artifacts_dir", help="Path to aiperf_artifacts/ or trace_replay/ directory")
    parser.add_argument(
        "-o", "--output", default=None, help="Output directory (default: same as artifacts_dir)"
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output) if args.output else artifacts_dir

    # Auto-detect format
    trace_replay_csv = artifacts_dir / "detailed_results.csv"
    aiperf_jsonl = artifacts_dir / "profile_export.jsonl"

    if trace_replay_csv.exists():
        records = load_trace_replay_records(artifacts_dir)
        print(f"Loaded {len(records):,} records from {artifacts_dir} (trace replay)")
    elif aiperf_jsonl.exists():
        records = load_records(artifacts_dir)
        print(f"Loaded {len(records):,} records from {artifacts_dir} (AIPerf)")
    else:
        print(f"No recognized data files in {artifacts_dir}")
        return

    analyze(records, output_dir)


if __name__ == "__main__":
    main()
