#!/usr/bin/env python3
"""
SLA-constrained throughput visualization for benchmark results.
Shows scatter plots of ISL vs throughput with SLA frontier curves.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# SLA thresholds (in seconds)
TTFT_THRESHOLDS = [0.5, 1.0, 2.0, 5.0, 10.0]  # 500ms to 10s
TPOT_THRESHOLDS = [0.05, 0.075, 0.1, 0.2]  # 50ms to 200ms


def load_results(results_dir: Path) -> List[Dict]:
    """Load all benchmark JSON files."""
    results = []
    pattern = r"tp(\d+)_isl(\d+)_osl(\d+)_conc(\d+)"
    for filepath in results_dir.glob("*.json"):
        match = re.search(pattern, filepath.name)
        if match:
            tp, isl, osl, conc = map(int, match.groups())
            try:
                with open(filepath) as f:
                    data = json.load(f)
                data["tp"] = tp
                data["isl"] = isl
                data["osl"] = osl
                data["conc"] = conc
                results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping {filepath.name}: {e}")
    return results


def compute_frontier(results: List[Dict], sla_thresholds: List[float],
                     latency_key: str, throughput_key: str) -> Dict[float, Dict[int, float]]:
    """For each ISL, find max normalized throughput meeting each SLA threshold."""
    frontiers = {threshold: {} for threshold in sla_thresholds}
    isl_levels = sorted(set(r["isl"] for r in results))

    for threshold in sla_thresholds:
        for isl in isl_levels:
            valid = [r for r in results
                     if r["isl"] == isl and r[latency_key] < threshold]
            if valid:
                # Normalize throughput by TP (number of GPUs)
                frontiers[threshold][isl] = max(r[throughput_key] / r["tp"] for r in valid)

    return frontiers


def plot_sla_frontier(results: List[Dict], output_path: Path):
    """Create scatter plots with SLA frontier curves."""
    if not results:
        print("No results to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    isl_levels = sorted(set(r["isl"] for r in results))

    # Colors for SLA frontier curves (use a sequential colormap)
    frontier_cmap = plt.cm.viridis

    # --- Prefill Graph (TTFT) ---
    # Scatter all data points
    all_isls = [r["isl"] for r in results]
    all_input_tput = [r["input_throughput"] / r["tp"] for r in results]
    ax1.scatter(all_isls, all_input_tput, marker='o', c='gray', s=60, alpha=0.6, zorder=3)

    # Compute and plot TTFT frontiers as stacked bands
    ttft_frontiers = compute_frontier(results, TTFT_THRESHOLDS, "p99_ttft", "input_throughput")
    frontier_colors = frontier_cmap([i / (len(TTFT_THRESHOLDS) - 1) for i in range(len(TTFT_THRESHOLDS))])

    # Plot from strictest to most permissive (stacked bands)
    prev_throughputs = None
    for i, threshold in enumerate(TTFT_THRESHOLDS):
        frontier = ttft_frontiers[threshold]
        if frontier:
            isls = sorted(frontier.keys())
            throughputs = [frontier.get(isl, 0) for isl in isl_levels]
            lower = prev_throughputs if prev_throughputs else [0] * len(isl_levels)
            label = f'TTFT < {int(threshold * 1000)}ms' if threshold < 1 else f'TTFT < {int(threshold)}s'
            ax1.fill_between(isl_levels, lower, throughputs, color=frontier_colors[i], alpha=0.5,
                             edgecolor=frontier_colors[i], linewidth=1.5, label=label, zorder=1)
            prev_throughputs = throughputs

    ax1.set_xlabel("Token Position", fontsize=11)
    ax1.set_ylabel("Input Throughput (tok/s/GPU)", fontsize=11)
    ax1.set_title("Prefill", fontsize=12)
    ax1.set_xticks(isl_levels)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Decode Graph (TPOT) ---
    # Scatter all data points
    all_output_tput = [r["output_throughput"] / r["tp"] for r in results]
    ax2.scatter(all_isls, all_output_tput, marker='o', c='gray', s=60, alpha=0.6, zorder=3)

    # Compute and plot TPOT frontiers as stacked bands
    tpot_frontiers = compute_frontier(results, TPOT_THRESHOLDS, "p99_tpot", "output_throughput")
    frontier_colors = frontier_cmap([i / (len(TPOT_THRESHOLDS) - 1) for i in range(len(TPOT_THRESHOLDS))])

    # Plot from strictest to most permissive (stacked bands)
    prev_throughputs = None
    for i, threshold in enumerate(TPOT_THRESHOLDS):
        frontier = tpot_frontiers[threshold]
        if frontier:
            isls = sorted(frontier.keys())
            throughputs = [frontier.get(isl, 0) for isl in isl_levels]
            lower = prev_throughputs if prev_throughputs else [0] * len(isl_levels)
            label = f'TPOT < {int(threshold * 1000)}ms'
            ax2.fill_between(isl_levels, lower, throughputs, color=frontier_colors[i], alpha=0.5,
                             edgecolor=frontier_colors[i], linewidth=1.5, label=label, zorder=1)
            prev_throughputs = throughputs

    ax2.set_xlabel("Token Position", fontsize=11)
    ax2.set_ylabel("Output Throughput (tok/s/GPU)", fontsize=11)
    ax2.set_title("Decode", fontsize=12)
    ax2.set_xticks(isl_levels)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("MiniMax M2.1 vLLM Benchmark (P99 SLA)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot SLA frontier visualization")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "sla_frontier_plot.png",
        help="Output plot filename"
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")

    if results:
        plot_sla_frontier(results, args.output)
    else:
        print("No results found!")


if __name__ == "__main__":
    main()