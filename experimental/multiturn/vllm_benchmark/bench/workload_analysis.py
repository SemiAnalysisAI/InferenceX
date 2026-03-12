#!/usr/bin/env python3
"""
Workload analysis for multi-turn benchmark runs.

Generates histograms of turns, input/output token lengths from
per-request client metrics. Can be called programmatically after a
benchmark run or standalone on any metrics_client_metrics.csv.

Usage:
    python -m bench.workload_analysis results/metrics_client_metrics.csv
    python -m bench.workload_analysis results/metrics_client_metrics.csv -o results/workload.png
"""

import argparse
import csv
import os
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class WorkloadStats(NamedTuple):
    input_num_turns: int
    input_num_tokens: int
    output_num_tokens: int


def load_from_csv(csv_path: str) -> list[WorkloadStats]:
    """Load workload stats from a metrics_client_metrics.csv file."""
    stats = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats.append(WorkloadStats(
                input_num_turns=int(row["input_num_turns"]),
                input_num_tokens=int(row["input_num_tokens"]),
                output_num_tokens=int(row["output_num_tokens"]),
            ))
    return stats


def generate_workload_plots(
    stats: list[WorkloadStats],
    output_path: str,
) -> None:
    """Generate workload analysis plots.

    Args:
        stats: List of per-request workload stats (from RequestStats or CSV).
        output_path: Path to save the output PNG.
    """
    if not stats:
        print("No workload data to plot")
        return

    # input_num_turns is len(messages) = number of messages in context (1, 3, 5, ...)
    # Normalize to conversation turns: 1 message = turn 1, 3 messages = turn 2, etc.
    raw_msg_counts = np.array([s.input_num_turns for s in stats])
    turns = (raw_msg_counts + 1) // 2  # 1->1, 3->2, 5->3, ...
    input_tokens = np.array([s.input_num_tokens for s in stats])
    output_tokens = np.array([s.output_num_tokens for s in stats])
    tpt = input_tokens / np.maximum(turns, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Workload Analysis ({len(stats):,} requests)", fontsize=14)

    # 1. Histogram of turns per request
    ax = axes[0, 0]
    max_t = min(int(turns.max()), 20)
    bins = np.arange(0.5, max_t + 1.5, 1)
    counts, _, bars = ax.hist(
        turns[turns <= max_t], bins=bins, edgecolor="black", alpha=0.7
    )
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(count)}",
                ha="center", va="bottom", fontsize=7,
            )
    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("Number of Requests")
    ax.set_title("Which Turn of the Conversation")
    ax.set_xticks(range(1, max_t + 1))
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Histogram of input tokens per turn
    ax = axes[0, 1]
    clip = min(np.percentile(tpt, 99), 2000)
    ax.hist(tpt[tpt <= clip], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(np.median(tpt), color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {np.median(tpt):.0f}")
    ax.axvline(np.mean(tpt), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(tpt):.0f}")
    ax.set_xlabel("Input Tokens per Turn")
    ax.set_ylabel("Number of Requests")
    ax.set_title("Avg Input Tokens per Turn")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Histogram of total input tokens per request
    ax = axes[1, 0]
    clip_in = min(np.percentile(input_tokens, 99), 30000)
    ax.hist(input_tokens[input_tokens <= clip_in], bins=50,
            edgecolor="black", alpha=0.7, color="coral")
    ax.axvline(np.median(input_tokens), color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {np.median(input_tokens):.0f}")
    ax.axvline(np.mean(input_tokens), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(input_tokens):.0f}")
    ax.set_xlabel("Total Input Tokens")
    ax.set_ylabel("Number of Requests")
    ax.set_title("Input Tokens per Request")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Histogram of output tokens per request
    ax = axes[1, 1]
    clip_out = min(np.percentile(output_tokens, 99), 5000)
    ax.hist(output_tokens[output_tokens <= clip_out], bins=50,
            edgecolor="black", alpha=0.7, color="mediumseagreen")
    ax.axvline(np.median(output_tokens), color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {np.median(output_tokens):.0f}")
    ax.axvline(np.mean(output_tokens), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(output_tokens):.0f}")
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Number of Requests")
    ax.set_title("Output Tokens per Request")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved workload analysis to {output_path}")
    plt.close()


def generate_from_request_stats(
    client_metrics: list,
    output_path: str,
) -> None:
    """Generate workload plots from a list of RequestStats (in-process call)."""
    stats = [
        WorkloadStats(
            input_num_turns=m.input_num_turns,
            input_num_tokens=m.input_num_tokens,
            output_num_tokens=m.output_num_tokens,
        )
        for m in client_metrics
    ]
    generate_workload_plots(stats, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate workload analysis plots from benchmark client metrics."
    )
    parser.add_argument(
        "csv_path",
        help="Path to metrics_client_metrics.csv",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG path (default: <csv_dir>/metrics_workload.png)",
    )
    args = parser.parse_args()

    if args.output is None:
        csv_dir = os.path.dirname(args.csv_path) or "."
        args.output = os.path.join(csv_dir, "metrics_workload.png")

    stats = load_from_csv(args.csv_path)
    generate_workload_plots(stats, args.output)
