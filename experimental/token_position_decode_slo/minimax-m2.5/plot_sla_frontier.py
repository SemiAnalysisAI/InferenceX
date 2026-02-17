#!/usr/bin/env python3
"""
SLA frontier merged visualization.
1x2 layout: Prefill / Decode. Three SLA tiers layered in each subplot.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from itertools import combinations


def load_results(results_dir: Path) -> List[Dict]:
    results = []
    pattern = r"tep(\d+)_isl(\d+)_osl(\d+)_conc(\d+)"
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


def compute_frontier(results, isl_levels, threshold, latency_key, throughput_key):
    xs, ys = [], []
    for isl in isl_levels:
        valid = [r[throughput_key] / r["tp"] for r in results
                 if r["isl"] == isl and r[latency_key] < threshold]
        if valid:
            xs.append(isl)
            ys.append(max(valid))
    return xs, ys


def pick_3_distinct(values, candidates, min_points=5):
    """Pick 3 thresholds from candidates that produce most distinct point sets."""
    cand_sets = {}
    for c in candidates:
        pts = set(i for i in range(len(values)) if values[i] < c)
        if len(pts) >= min_points:
            cand_sets[c] = pts
    best_combo, best_score = None, float('inf')
    for combo in combinations(cand_sets.keys(), 3):
        sets = [cand_sets[c] for c in combo]
        sizes = [len(s) for s in sets]
        if len(set(sizes)) < 3:
            continue
        max_jaccard = 0
        for a, b in combinations(range(3), 2):
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            max_jaccard = max(max_jaccard, inter / union if union else 0)
        if max_jaccard < best_score:
            best_score = max_jaccard
            best_combo = combo
    return sorted(best_combo) if best_combo else candidates[:3]


def plot_sla_merged(results: List[Dict], output_path: Path):
    if not results:
        print("No results to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    isl_levels = sorted(set(r["isl"] for r in results))
    isls = np.array([r["isl"] for r in results])
    in_tput = np.array([r["input_throughput"] / r["tp"] for r in results])
    out_tput = np.array([r["output_throughput"] / r["tp"] for r in results])

    ttft_candidates = [5, 10, 15, 20, 25, 30]
    tpot_candidates_ms = [20, 40, 80, 120, 160, 200]
    ttfts = np.array([r["p99_ttft"] for r in results])
    tpots_ms = np.array([r["p99_tpot"] * 1000 for r in results])

    ttft_picks = pick_3_distinct(ttfts, ttft_candidates, min_points=5)
    tpot_picks_ms = pick_3_distinct(tpots_ms, tpot_candidates_ms, min_points=5)

    tier_names = ["Interactive", "Balanced", "Throughput"]
    tier_colors = ["#4a90d9", "#41ab5d", "#e05252"]


    # Gray scatter background
    ax1.scatter(isls, in_tput, marker='o', color='gray', s=40, alpha=0.75, zorder=1)
    ax2.scatter(isls, out_tput, marker='o', color='gray', s=40, alpha=0.75, zorder=1)

    # --- Prefill: draw loosest first (back), tightest last (front) ---
    prefill_handles = []
    for i in range(2, -1, -1):
        tt = ttft_picks[i]
        color = tier_colors[i]
        name = tier_names[i]
        fx, fy = compute_frontier(results, isl_levels, tt, "p99_ttft", "input_throughput")
        if fx:
            front = 2 - i  # 0=Throughput(back), 2=Interactive(front)
            ax1.fill_between(fx, 0, fy, color=color, alpha=0.5, linewidth=0, zorder=2 + front)
            ax1.plot(fx, fy, color=color, linewidth=2, alpha=0.9, zorder=5 + front)
            ax1.scatter(fx, fy, color=color, s=50, zorder=8 + front,
                        edgecolors='white', linewidths=0.5)

    # Build legend in order (Interactive first)
    for i, (name, tt) in enumerate(zip(tier_names, ttft_picks)):
        prefill_handles.append(mpatches.Patch(
            color=tier_colors[i], alpha=0.75,
            label=f'{name} (TTFT < {tt}s)'))

    ax1.legend(handles=prefill_handles, loc='upper right', fontsize=10, framealpha=0.9)
    ax1.set_xlabel("Token Position", fontsize=11)
    ax1.set_ylabel("Input Throughput (tok/s/GPU)", fontsize=11)
    ax1.set_title("Prefill", fontsize=12)
    ax1.set_xticks(isl_levels)
    ax1.set_xticklabels([f'{x//1024}K' for x in isl_levels], fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # --- Decode: draw loosest first (back), tightest last (front) ---
    decode_handles = []
    for i in range(2, -1, -1):
        tp_ms = tpot_picks_ms[i]
        t = tp_ms / 1000
        color = tier_colors[i]
        fx, fy = compute_frontier(results, isl_levels, t, "p99_tpot", "output_throughput")
        if fx:
            front = 2 - i
            ax2.fill_between(fx, 0, fy, color=color, alpha=0.5, linewidth=0, zorder=2 + front)
            ax2.plot(fx, fy, color=color, linewidth=2, alpha=0.9, zorder=5 + front)
            ax2.scatter(fx, fy, color=color, s=50, zorder=8 + front,
                        edgecolors='white', linewidths=0.5)

    for i, (name, tp_ms) in enumerate(zip(tier_names, tpot_picks_ms)):
        interactivity = 1000 / tp_ms
        decode_handles.append(mpatches.Patch(
            color=tier_colors[i], alpha=0.75,
            label=f'{name} ({interactivity:.0f} tok/s/user, TPOT < {tp_ms}ms)'))

    ax2.legend(handles=decode_handles, loc='upper right', fontsize=10, framealpha=0.9)
    ax2.set_xlabel("Token Position", fontsize=11)
    ax2.set_ylabel("Output Throughput (tok/s/GPU)", fontsize=11)
    ax2.set_title("Decode", fontsize=12)
    ax2.set_xticks(isl_levels)
    ax2.set_xticklabels([f'{x//1024}K' for x in isl_levels], fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("MiniMax M2.5 SLA Frontier\n8xH200 TEP8 vLLM", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    for name, tt, tp_ms in zip(tier_names, ttft_picks, tpot_picks_ms):
        print(f"  {name}: TTFT < {tt}s, TPOT < {tp_ms}ms")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot merged SLA frontier")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "minimax-m25_sla_frontier.png")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")

    if results:
        plot_sla_merged(results, args.output)
    else:
        print("No results found!")


if __name__ == "__main__":
    main()
