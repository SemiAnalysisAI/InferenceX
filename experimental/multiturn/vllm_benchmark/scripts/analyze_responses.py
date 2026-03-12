#!/usr/bin/env python3
"""
Analyze actual benchmark request/response data across a sweep.

Reads metrics_client_metrics.csv from each experiment directory and produces
holistic statistics and plots on the real workload that was executed:
  - Output token length distributions
  - Input token length distributions
  - Conversation turn depth
  - Context growth over turns
  - Latency breakdown by turn number
  - Cache hit progression over turns
  - Per-config workload summary

Usage:
    python analyze_responses.py <pareto_input_dir> [output_dir]
    python analyze_responses.py ~/Downloads/multiturn_aggregated/pareto_input
"""

import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_all_experiments(results_dir: Path) -> pd.DataFrame:
    """Load client metrics CSVs from all experiment directories."""
    frames = []
    pattern = re.compile(r"tp(\d+)_bs(\d+)_offload(on|off|noprefix)")

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        m = pattern.match(exp_dir.name)
        if not m:
            continue

        status_file = exp_dir / "status.txt"
        if status_file.exists() and status_file.read_text().strip() != "SUCCESS":
            continue

        csv_file = exp_dir / "metrics_client_metrics.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)
        df["tp"] = int(m.group(1))
        df["users"] = int(m.group(2))
        df["offload"] = m.group(3)
        df["exp_name"] = exp_dir.name
        frames.append(df)

    if not frames:
        print("No experiment data found!")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} requests across {len(frames)} experiments")
    return combined


def plot_workload_summary(df: pd.DataFrame, output_dir: Path):
    """High-level workload summary table printed + saved."""
    summary = df.groupby(["tp", "users", "offload"]).agg(
        total_requests=("output_num_tokens", "count"),
        unique_convs=("conversation_id", "nunique"),
        max_turn=("input_num_turns", "max"),
        median_turn=("input_num_turns", "median"),
        mean_input_tokens=("input_num_tokens", "mean"),
        median_input_tokens=("input_num_tokens", "median"),
        mean_output_tokens=("output_num_tokens", "mean"),
        median_output_tokens=("output_num_tokens", "median"),
        mean_ttft_ms=("ttft_ms", "mean"),
        mean_tpot_ms=("tpot_ms", "mean"),
    ).reset_index()
    summary = summary.sort_values(["tp", "users", "offload"])

    out_file = output_dir / "workload_summary.csv"
    summary.to_csv(out_file, index=False, float_format="%.1f")
    print(f"\nSaved workload summary to {out_file}")
    print(summary.to_string(index=False))
    return summary


def plot_token_distributions(df: pd.DataFrame, output_dir: Path):
    """Histograms of input and output token lengths (global aggregate)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token Length Distributions (All Experiments)", fontsize=14)

    # Output tokens
    ax = axes[0, 0]
    ax.hist(df["output_num_tokens"], bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Count")
    ax.set_title("Output Token Length Distribution")
    ax.axvline(df["output_num_tokens"].median(), color="red", ls="--", lw=1.5,
               label=f"Median: {df['output_num_tokens'].median():.0f}")
    ax.axvline(df["output_num_tokens"].mean(), color="orange", ls="--", lw=1.5,
               label=f"Mean: {df['output_num_tokens'].mean():.0f}")
    ax.legend(fontsize=9)

    # Input tokens
    ax = axes[0, 1]
    ax.hist(df["input_num_tokens"], bins=100, color="coral", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Count")
    ax.set_title("Input Token Length Distribution")
    ax.axvline(df["input_num_tokens"].median(), color="red", ls="--", lw=1.5,
               label=f"Median: {df['input_num_tokens'].median():.0f}")
    ax.axvline(df["input_num_tokens"].mean(), color="orange", ls="--", lw=1.5,
               label=f"Mean: {df['input_num_tokens'].mean():.0f}")
    ax.legend(fontsize=9)

    # Output tokens by turn number
    ax = axes[1, 0]
    turn_groups = df.groupby("input_num_turns")["output_num_tokens"]
    turns = sorted(df["input_num_turns"].unique())
    bp_data = [turn_groups.get_group(t).values for t in turns if len(turn_groups.get_group(t)) > 10]
    bp_turns = [t for t in turns if len(turn_groups.get_group(t)) > 10]
    if bp_data:
        bp = ax.boxplot(bp_data, positions=bp_turns, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="red", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Output Tokens")
    ax.set_title("Output Tokens by Turn Number")
    ax.set_xlim(0, max(bp_turns) + 1 if bp_turns else 10)

    # Input tokens by turn number
    ax = axes[1, 1]
    turn_groups_in = df.groupby("input_num_turns")["input_num_tokens"]
    bp_data_in = [turn_groups_in.get_group(t).values for t in turns if len(turn_groups_in.get_group(t)) > 10]
    if bp_data_in:
        bp = ax.boxplot(bp_data_in, positions=bp_turns, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="red", lw=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("coral")
            patch.set_alpha(0.6)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Input Tokens")
    ax.set_title("Input Context Size by Turn Number (Context Growth)")
    ax.set_xlim(0, max(bp_turns) + 1 if bp_turns else 10)

    plt.tight_layout()
    out_file = output_dir / "token_distributions.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved token distributions to {out_file}")
    plt.close()


def plot_turn_depth(df: pd.DataFrame, output_dir: Path):
    """How many turns conversations actually reached."""
    # For each conversation in each experiment, find max turn reached
    conv_max_turn = df.groupby(["exp_name", "conversation_id"])["input_num_turns"].max().reset_index()
    conv_max_turn.columns = ["exp_name", "conversation_id", "max_turn"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Conversation Turn Depth", fontsize=14)

    # Global histogram
    ax = axes[0]
    ax.hist(conv_max_turn["max_turn"], bins=range(1, conv_max_turn["max_turn"].max() + 2),
            color="mediumpurple", edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Max Turn Reached")
    ax.set_ylabel("Number of Conversations")
    ax.set_title("Distribution of Max Turn Reached (All Experiments)")
    ax.axvline(conv_max_turn["max_turn"].median(), color="red", ls="--", lw=1.5,
               label=f"Median: {conv_max_turn['max_turn'].median():.0f}")
    ax.legend()

    # By TP — aggregate across users/offload
    ax = axes[1]
    # Parse tp from exp_name
    conv_max_turn["tp"] = conv_max_turn["exp_name"].str.extract(r"tp(\d+)").astype(int)
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    for tp in sorted(conv_max_turn["tp"].unique()):
        subset = conv_max_turn[conv_max_turn["tp"] == tp]
        counts, bins = np.histogram(subset["max_turn"],
                                    bins=range(1, subset["max_turn"].max() + 2))
        ax.step(bins[:-1], counts, where="mid", color=tp_colors.get(tp, "gray"),
                label=f"TP={tp}", lw=1.5, alpha=0.8)
    ax.set_xlabel("Max Turn Reached")
    ax.set_ylabel("Number of Conversations")
    ax.set_title("Turn Depth by TP")
    ax.legend()

    plt.tight_layout()
    out_file = output_dir / "turn_depth.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved turn depth plot to {out_file}")
    plt.close()


def plot_latency_by_turn(df: pd.DataFrame, output_dir: Path):
    """How TTFT, TPOT, and E2E latency change with turn number."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Latency vs Turn Number (Median across all experiments)", fontsize=14)

    # Only include turns with sufficient data
    turn_counts = df["input_num_turns"].value_counts()
    valid_turns = turn_counts[turn_counts >= 20].index
    df_valid = df[df["input_num_turns"].isin(valid_turns)]

    offload_colors = {"on": "green", "off": "steelblue", "noprefix": "red"}
    offload_labels = {"on": "Prefix+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}

    metrics = [
        (axes[0, 0], "ttft_ms", "TTFT (ms)", "Median TTFT vs Turn"),
        (axes[0, 1], "tpot_ms", "TPOT (ms)", "Median TPOT vs Turn"),
        (axes[1, 0], "latency_ms", "E2E Latency (ms)", "Median E2E Latency vs Turn"),
        (axes[1, 1], "interactivity_tok_per_sec", "Interactivity (tok/s)", "Median Interactivity vs Turn"),
    ]

    for ax, col, ylabel, title in metrics:
        for offload in ["on", "off", "noprefix"]:
            subset = df_valid[df_valid["offload"] == offload]
            if len(subset) == 0:
                continue
            medians = subset.groupby("input_num_turns")[col].median()
            medians = medians.sort_index()
            ax.plot(medians.index, medians.values,
                    color=offload_colors[offload], lw=2, alpha=0.8,
                    label=offload_labels[offload], marker=".", markersize=4)

        ax.set_xlabel("Turn Number")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = output_dir / "latency_by_turn.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved latency by turn plot to {out_file}")
    plt.close()


def plot_cache_hit_by_turn(df: pd.DataFrame, output_dir: Path):
    """How prefix cache hit rate evolves over conversation turns."""
    # Only for experiments with caching (off and on modes)
    df_cached = df[df["offload"].isin(["on", "off"])].copy()
    if len(df_cached) == 0:
        print("No cached experiments found, skipping cache hit plot")
        return

    turn_counts = df_cached["input_num_turns"].value_counts()
    valid_turns = turn_counts[turn_counts >= 20].index
    df_valid = df_cached[df_cached["input_num_turns"].isin(valid_turns)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prefix Cache Hit Rate vs Turn Number", fontsize=14)

    offload_colors = {"on": "green", "off": "steelblue"}
    offload_labels = {"on": "Prefix+Offload", "off": "Prefix Only"}

    # By offload mode (aggregate across TP/users)
    ax = axes[0]
    for offload in ["on", "off"]:
        subset = df_valid[df_valid["offload"] == offload]
        if len(subset) == 0:
            continue
        medians = subset.groupby("input_num_turns")["approx_cached_percent"].median()
        medians = medians.sort_index()
        ax.plot(medians.index, medians.values,
                color=offload_colors[offload], lw=2, alpha=0.8,
                label=offload_labels[offload], marker=".", markersize=4)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Approx Cached %")
    ax.set_title("Cache Hit Rate by Turn (Median)")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # By TP (only prefix-only mode for clarity)
    ax = axes[1]
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    df_prefix = df_valid[df_valid["offload"] == "off"]
    for tp in sorted(df_prefix["tp"].unique()):
        subset = df_prefix[df_prefix["tp"] == tp]
        medians = subset.groupby("input_num_turns")["approx_cached_percent"].median()
        medians = medians.sort_index()
        ax.plot(medians.index, medians.values,
                color=tp_colors.get(tp, "gray"), lw=2, alpha=0.8,
                label=f"TP={tp}", marker=".", markersize=4)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Approx Cached %")
    ax.set_title("Cache Hit Rate by Turn × TP (Prefix Only)")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = output_dir / "cache_hit_by_turn.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved cache hit by turn plot to {out_file}")
    plt.close()


def plot_requests_over_time(df: pd.DataFrame, output_dir: Path):
    """Request completion rate over benchmark time, by config."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Request Completion Over Time", fontsize=14)

    tp_values = sorted(df["tp"].unique())
    tp_colors_map = {1: "blue", 2: "green", 4: "orange", 8: "red"}

    for idx, tp in enumerate(tp_values):
        ax = axes[idx // 2, idx % 2]
        df_tp = df[df["tp"] == tp]

        for offload in ["on", "off", "noprefix"]:
            offload_colors = {"on": "green", "off": "steelblue", "noprefix": "red"}
            offload_labels = {"on": "P+O", "off": "Prefix", "noprefix": "NoPfx"}

            for users in sorted(df_tp["users"].unique()):
                subset = df_tp[(df_tp["offload"] == offload) & (df_tp["users"] == users)]
                if len(subset) == 0:
                    continue
                # Cumulative requests over time
                times = subset["relative_time_sec"].sort_values().values
                cumulative = np.arange(1, len(times) + 1)
                alpha = 0.4 if users <= 64 else 0.7
                lw = 1 if users <= 64 else 1.5
                ax.plot(times, cumulative, color=offload_colors[offload],
                        alpha=alpha, lw=lw)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Completed Requests")
        ax.set_title(f"TP={tp}")
        ax.grid(True, alpha=0.3)
        # Simple legend for offload modes
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="green", lw=2, label="Prefix+Offload"),
            Line2D([0], [0], color="steelblue", lw=2, label="Prefix Only"),
            Line2D([0], [0], color="red", lw=2, label="No Prefix"),
        ]
        ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    out_file = output_dir / "requests_over_time.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved requests over time plot to {out_file}")
    plt.close()


def plot_global_stats(df: pd.DataFrame, output_dir: Path):
    """Print and save aggregate statistics across the entire sweep."""
    stats = {
        "Total requests": len(df),
        "Unique conversations": df["conversation_id"].nunique(),
        "": "",
        "--- Output Tokens ---": "",
        "Mean": f"{df['output_num_tokens'].mean():.1f}",
        "Median": f"{df['output_num_tokens'].median():.0f}",
        "P5": f"{df['output_num_tokens'].quantile(0.05):.0f}",
        "P25": f"{df['output_num_tokens'].quantile(0.25):.0f}",
        "P75": f"{df['output_num_tokens'].quantile(0.75):.0f}",
        "P95": f"{df['output_num_tokens'].quantile(0.95):.0f}",
        "P99": f"{df['output_num_tokens'].quantile(0.99):.0f}",
        "Max": f"{df['output_num_tokens'].max():.0f}",
        " ": "",
        "--- Input Tokens ---": "",
        "Mean ": f"{df['input_num_tokens'].mean():.1f}",
        "Median ": f"{df['input_num_tokens'].median():.0f}",
        "P5 ": f"{df['input_num_tokens'].quantile(0.05):.0f}",
        "P95 ": f"{df['input_num_tokens'].quantile(0.95):.0f}",
        "Max ": f"{df['input_num_tokens'].max():.0f}",
        "  ": "",
        "--- Turns ---": "",
        "Mean  ": f"{df['input_num_turns'].mean():.1f}",
        "Median  ": f"{df['input_num_turns'].median():.0f}",
        "Max  ": f"{df['input_num_turns'].max():.0f}",
    }

    print("\n" + "=" * 50)
    print("GLOBAL WORKLOAD STATISTICS")
    print("=" * 50)
    for k, v in stats.items():
        if v == "":
            print(k)
        else:
            print(f"  {k.strip():.<30} {v}")
    print("=" * 50)

    # Save as text
    with open(output_dir / "global_stats.txt", "w") as f:
        for k, v in stats.items():
            if v == "":
                f.write(f"{k}\n")
            else:
                f.write(f"  {k.strip():.<30} {v}\n")
    print(f"Saved global stats to {output_dir / 'global_stats.txt'}")


def plot_output_by_config(df: pd.DataFrame, output_dir: Path):
    """Output token distribution broken down by TP and user count."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Output Token Distribution by Config", fontsize=14)

    tp_values = sorted(df["tp"].unique())
    for idx, tp in enumerate(tp_values):
        ax = axes[idx // 2, idx % 2]
        df_tp = df[df["tp"] == tp]

        user_vals = sorted(df_tp["users"].unique())
        bp_data = []
        bp_labels = []
        for u in user_vals:
            subset = df_tp[df_tp["users"] == u]["output_num_tokens"]
            if len(subset) > 5:
                bp_data.append(subset.values)
                bp_labels.append(str(u))

        if bp_data:
            bp = ax.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True,
                            showfliers=False, medianprops=dict(color="red", lw=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor("steelblue")
                patch.set_alpha(0.6)

        ax.set_xlabel("Concurrent Users")
        ax.set_ylabel("Output Tokens")
        ax.set_title(f"TP={tp}")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = output_dir / "output_tokens_by_config.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved output tokens by config to {out_file}")
    plt.close()


def plot_inflight_concurrency(df: pd.DataFrame, output_dir: Path):
    """Analyze actual in-flight concurrency vs configured max (users).

    Two plots:
      1. Time-series of in-flight requests for selected configs
      2. Summary bar chart of mean ratio-to-max across all configs
    """
    from matplotlib.lines import Line2D

    experiments = df.groupby(["tp", "users", "offload"])
    exp_keys = sorted(experiments.groups.keys())

    # --- Compute in-flight stats for every experiment ---
    records = []
    timeseries = {}  # store a few for plotting
    for tp, users, offload in exp_keys:
        sub = experiments.get_group((tp, users, offload))
        starts = sub["start_time_ms"].values
        ends = (sub["start_time_ms"] + sub["latency_ms"]).values
        t_min, t_max = starts.min(), ends.max()

        # 1-second sample grid
        ticks = np.arange(t_min, t_max, 1000)
        inflight = np.array(
            [np.sum((starts <= t) & (ends > t)) for t in ticks]
        )

        # Steady-state window: skip first 30s warmup, last 10s drain
        mask = (ticks >= t_min + 30_000) & (ticks <= t_max - 10_000)
        steady = inflight[mask]
        if len(steady) == 0:
            continue

        records.append({
            "tp": tp, "users": users, "offload": offload,
            "mean_inflight": steady.mean(),
            "median_inflight": np.median(steady),
            "ratio": steady.mean() / users,
        })

        # Keep time-series for a representative set (offload=off only, keep it readable)
        if offload == "off":
            rel_sec = (ticks - t_min) / 1000
            timeseries[(tp, users)] = (rel_sec, inflight, users)

    stats_df = pd.DataFrame(records)

    # ---- Figure: 3 panels ----
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # --- Panel 1: time-series for TP=1 (prefix-only) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax, tp_show in [(ax1, 1), (ax2, 8)]:
        cmap = plt.cm.viridis
        relevant = {k: v for k, v in timeseries.items() if k[0] == tp_show}
        user_vals = sorted(set(k[1] for k in relevant))
        norm = plt.Normalize(0, len(user_vals) - 1)
        for i, u in enumerate(user_vals):
            key = (tp_show, u)
            if key not in relevant:
                continue
            rel_sec, inflight, max_u = relevant[key]
            color = cmap(norm(i))
            ax.plot(rel_sec, inflight, color=color, alpha=0.7, lw=1.2, label=f"users={u}")
            ax.axhline(max_u, color=color, ls=":", alpha=0.3, lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("In-Flight Requests")
        ax.set_title(f"In-Flight Over Time — TP={tp_show} (Prefix Only)")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)

    # --- Panel 2: ratio-to-max bar chart by user count ---
    ax3 = fig.add_subplot(gs[1, 0])
    offload_colors = {"on": "green", "off": "steelblue", "noprefix": "salmon"}
    offload_labels = {"on": "P+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}
    user_vals = sorted(stats_df["users"].unique())
    bar_width = 0.25
    x = np.arange(len(user_vals))

    for i, offload in enumerate(["off", "on", "noprefix"]):
        ratios = []
        for u in user_vals:
            row = stats_df[(stats_df["offload"] == offload) & (stats_df["users"] == u)]
            # Average ratio across TP values
            ratios.append(row["ratio"].mean() if len(row) > 0 else 0)
        ax3.bar(x + i * bar_width, ratios, bar_width,
                color=offload_colors[offload], alpha=0.8,
                label=offload_labels[offload], edgecolor="white", lw=0.5)

    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels([str(u) for u in user_vals], fontsize=8)
    ax3.set_xlabel("Configured Users (max-num-seqs)")
    ax3.set_ylabel("Mean In-Flight / Users")
    ax3.set_title("Avg Utilization Ratio (across all TP values)")
    ax3.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5, label="100% (always full)")
    ax3.set_ylim(0, 1.15)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: ratio by TP ---
    ax4 = fig.add_subplot(gs[1, 1])
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_vals = sorted(stats_df["tp"].unique())
    bar_width_tp = 0.18

    for i, tp in enumerate(tp_vals):
        ratios = []
        for u in user_vals:
            row = stats_df[(stats_df["tp"] == tp) & (stats_df["users"] == u)]
            ratios.append(row["ratio"].mean() if len(row) > 0 else 0)
        ax4.bar(x + i * bar_width_tp, ratios, bar_width_tp,
                color=tp_colors[tp], alpha=0.8,
                label=f"TP={tp}", edgecolor="white", lw=0.5)

    ax4.set_xticks(x + 1.5 * bar_width_tp)
    ax4.set_xticklabels([str(u) for u in user_vals], fontsize=8)
    ax4.set_xlabel("Configured Users (max-num-seqs)")
    ax4.set_ylabel("Mean In-Flight / Users")
    ax4.set_title("Avg Utilization Ratio by TP (avg across offload modes)")
    ax4.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5, label="100%")
    ax4.set_ylim(0, 1.15)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle("In-Flight Concurrency Analysis", fontsize=15, y=1.01)
    plt.savefig(output_dir / "inflight_concurrency.png", dpi=150, bbox_inches="tight")
    print(f"Saved in-flight concurrency plot to {output_dir / 'inflight_concurrency.png'}")
    plt.close()

    # Also save the stats table
    stats_df.to_csv(output_dir / "inflight_stats.csv", index=False, float_format="%.2f")
    print(f"Saved in-flight stats to {output_dir / 'inflight_stats.csv'}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_responses.py <pareto_input_dir> [output_dir]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir.parent / "response_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {results_dir}")
    print(f"Output directory:  {output_dir}")

    df = load_all_experiments(results_dir)

    # Global stats
    plot_global_stats(df, output_dir)

    # Workload summary table
    plot_workload_summary(df, output_dir)

    # Token distributions
    plot_token_distributions(df, output_dir)

    # Turn depth analysis
    plot_turn_depth(df, output_dir)

    # Latency vs turn number
    plot_latency_by_turn(df, output_dir)

    # Cache hit rate progression
    plot_cache_hit_by_turn(df, output_dir)

    # Requests over time
    plot_requests_over_time(df, output_dir)

    # Output tokens by config
    plot_output_by_config(df, output_dir)

    # In-flight concurrency analysis
    plot_inflight_concurrency(df, output_dir)

    print(f"\nAll analysis outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
