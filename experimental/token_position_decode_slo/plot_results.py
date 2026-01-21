#!/usr/bin/env python3
"""
Visualization script for MiniMax M2.1 benchmark results.
Plots prefill and decode throughput vs token position (ISL).
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


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


def plot_results(results: List[Dict], output_path: Path):
    """Create prefill and decode throughput plots."""
    if not results:
        print("No results to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    conc_levels = sorted(set(r["conc"] for r in results))
    isl_levels = sorted(set(r["isl"] for r in results))

    # Plot lines for each concurrency
    for conc in conc_levels:
        subset = [r for r in results if r["conc"] == conc]
        subset.sort(key=lambda x: x["isl"])

        isls = [r["isl"] for r in subset]
        input_tput = [r["input_throughput"] for r in subset]
        output_tput = [r["output_throughput"] for r in subset]

        ax1.plot(isls, input_tput, 'o-', label=f'conc {conc}')
        ax2.plot(isls, output_tput, 'o-', label=f'conc {conc}')

    # Configure axes
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_title("Prefill")
    ax1.set_xticks(isl_levels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_title("Decode")
    ax2.set_xticks(isl_levels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("MiniMax M2.1 vLLM Benchmark")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "result_plot.png",
        help="Output plot filename"
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")

    if results:
        plot_results(results, args.output)
    else:
        print("No results found!")


if __name__ == "__main__":
    main()