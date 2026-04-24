#!/usr/bin/env python3
"""Generate overview plots for sweep results.

Produces:
- throughput_vs_concurrency.png: Throughput & cache hit rate vs concurrent sessions per TP
- workload_consistency.png: ISL distribution box plots per experiment to verify consistent workload

Usage:
    python plot_sweep_overview.py <pareto_input_dir> [<output_dir>]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_throughput_vs_concurrency(df: pd.DataFrame, output_dir: Path) -> None:
    """Throughput and cache hit rate vs concurrent sessions, per TP."""
    tps = sorted(df["tp"].unique())
    n = len(tps)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Throughput & Cache Hit Rate vs Concurrent Sessions", fontsize=15)

    for idx, tp in enumerate(tps):
        tp_df = df[df["tp"] == tp].sort_values("bs")
        off = tp_df[tp_df["offload"] == "off"].sort_values("bs")
        on = tp_df[tp_df["offload"] == "on"].sort_values("bs")

        # --- Top row: Throughput ---
        ax = axes[0, idx]
        if len(off) > 0:
            ax.plot(off["bs"], off["total_tps_per_gpu"], "o-", color="#d62728",
                    linewidth=2.5, markersize=7, label="Offload OFF")
        if len(on) > 0:
            ax.plot(on["bs"], on["total_tps_per_gpu"], "s-", color="#2ca02c",
                    linewidth=2.5, markersize=7, label="Offload ON")

        # Annotate max gain
        if len(off) > 0 and len(on) > 0:
            merged = pd.merge(off[["bs", "total_tps_per_gpu"]], on[["bs", "total_tps_per_gpu"]],
                              on="bs", suffixes=("_off", "_on"))
            if len(merged) > 0:
                merged["gain_pct"] = ((merged["total_tps_per_gpu_on"] - merged["total_tps_per_gpu_off"])
                                      / merged["total_tps_per_gpu_off"] * 100)
                max_row = merged.loc[merged["gain_pct"].idxmax()]
                if max_row["gain_pct"] > 20:
                    ax.annotate(f"+{max_row['gain_pct']:.0f}%",
                                xy=(max_row["bs"], max_row["total_tps_per_gpu_on"]),
                                xytext=(0, 15), textcoords="offset points",
                                fontsize=11, fontweight="bold", color="green", ha="center")

        ax.set_xlabel("Concurrent Sessions", fontsize=10)
        ax.set_ylabel("Throughput/GPU (tok/s)", fontsize=10)
        ax.set_title(f"TP{tp} — Throughput", fontsize=13, fontweight="bold")
        max_tput = df["total_tps_per_gpu"].max()
        ax.set_ylim(0, max_tput * 1.15 if max_tput > 0 else 15000)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # --- Bottom row: Cache hit rate ---
        ax = axes[1, idx]
        if len(off) > 0:
            ax.plot(off["bs"], off["gpu_hit_rate"], "o-", color="#d62728",
                    linewidth=2, markersize=6, label="GPU Hit — OFF")
        if len(on) > 0:
            ax.plot(on["bs"], on["gpu_hit_rate"], "s-", color="#2ca02c",
                    linewidth=2, markersize=6, label="GPU Hit — ON")
            cpu_hit = on["cpu_hit_rate"].fillna(0)
            if cpu_hit.max() > 1:
                ax.plot(on["bs"], cpu_hit, "v--", color="#9467bd",
                        linewidth=2, markersize=6, label="CPU Hit — ON")

        ax.set_xlabel("Concurrent Sessions", fontsize=10)
        ax.set_ylabel("Cache Hit Rate (%)", fontsize=10)
        ax.set_title(f"TP{tp} — Cache Hit Rate", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = output_dir / "throughput_vs_concurrency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_workload_consistency(pareto_input_dir: Path, output_dir: Path) -> None:
    """ISL distribution box plots per experiment to verify consistent workload."""
    csv.field_size_limit(sys.maxsize)

    tps = set()
    data_by_tp: dict[int, list[tuple[int, str, list[float]]]] = defaultdict(list)

    for exp_dir in sorted(pareto_input_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("tp"):
            continue
        if "offloadon" in exp_dir.name:
            continue  # Only use offload-off for consistency check

        parts = exp_dir.name.split("_")
        try:
            tp = int(parts[0].replace("tp", ""))
            bs = int(parts[1].replace("bs", ""))
        except (IndexError, ValueError):
            continue

        tps.add(tp)

        # Try trace replay CSV
        csv_path = exp_dir / "trace_replay" / "detailed_results.csv"
        if not csv_path.exists():
            # Try aiperf JSONL
            continue

        isls = []
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("success") == "True":
                        isls.append(int(row["input_tokens"]) / 1000)  # k tokens
        except Exception:
            continue

        if isls:
            data_by_tp[tp].append((bs, exp_dir.name, isls))

    if not data_by_tp:
        print("No workload data found for consistency plot")
        return

    sorted_tps = sorted(data_by_tp.keys())
    n = len(sorted_tps)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle("Workload Consistency — ISL Distribution Per Experiment (Offload OFF)", fontsize=14)

    for idx, tp in enumerate(sorted_tps):
        ax = axes[idx]
        entries = sorted(data_by_tp[tp], key=lambda x: x[0])

        box_data = [e[2] for e in entries]
        labels = [str(e[0]) for e in entries]
        means = [np.mean(e[2]) for e in entries]

        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)

        ax.plot(range(1, len(means) + 1), means, "o--", color="orange", linewidth=2,
                markersize=6, label=f"Mean ({np.mean(means):.0f}k ± {np.std(means):.0f}k)", zorder=5)

        overall_mean = np.mean(means)
        overall_std = np.std(means)
        ax.axhspan(overall_mean - overall_std, overall_mean + overall_std,
                   alpha=0.1, color="orange", label="±1σ band")
        ax.axhline(overall_mean, color="orange", linestyle=":", alpha=0.5)

        ax.set_xlabel("Concurrent Sessions", fontsize=11)
        ax.set_ylabel("ISL (k tokens)", fontsize=11)
        ax.set_title(f"TP{tp}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylim(0, 140)

    plt.tight_layout()
    out = output_dir / "workload_consistency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pareto_input_dir> [<output_dir>]")
        sys.exit(1)

    pareto_input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else pareto_input_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment summary
    summary_csv = pareto_input_dir / "experiment_summary.csv"
    if not summary_csv.exists():
        # Try parent
        summary_csv = output_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"No summary CSV found in {pareto_input_dir} or {output_dir}")
        return

    df = pd.read_csv(summary_csv)

    # Ensure required columns exist
    required = ["tp", "bs", "offload", "total_tps_per_gpu", "gpu_hit_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns in summary: {missing}")
        return

    plot_throughput_vs_concurrency(df, output_dir)
    plot_workload_consistency(pareto_input_dir, output_dir)


if __name__ == "__main__":
    main()
