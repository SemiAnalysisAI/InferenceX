#!/usr/bin/env python3
"""
Plot Pareto frontiers for prefix caching modes.
Modes: on (prefix + offload), off (prefix only), noprefix (no prefix caching)
Pareto frontier: throughput vs latency trade-off.

Usage:
    python plot_pareto.py <results_dir>
    python plot_pareto.py ~/sweep_results_20260204_062339
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_experiment_data(exp_dir: Path) -> dict | None:
    """Load and aggregate metrics from an experiment directory."""
    client_metrics_file = exp_dir / "metrics_client_metrics.csv"
    server_metrics_file = exp_dir / "metrics_server_metrics.csv"
    status_file = exp_dir / "status.txt"

    # Check if experiment completed successfully
    if not status_file.exists():
        return None
    status = status_file.read_text().strip()
    if status != "SUCCESS":
        return None

    if not client_metrics_file.exists():
        return None

    try:
        df = pd.read_csv(client_metrics_file)

        # Load server metrics for cache hit rates
        gpu_hit_rate = None
        cpu_hit_rate = None
        if server_metrics_file.exists():
            server_df = pd.read_csv(server_metrics_file)
            # Get final cumulative values
            final_row = server_df.iloc[-1]
            if final_row["prefix_cache_queries"] > 0:
                gpu_hit_rate = 100 * final_row["prefix_cache_hits"] / final_row["prefix_cache_queries"]
            if final_row["cpu_prefix_cache_queries"] > 0:
                cpu_hit_rate = 100 * final_row["cpu_prefix_cache_hits"] / final_row["cpu_prefix_cache_queries"]
        if len(df) == 0:
            return None

        # Parse experiment name: tp{N}_bs{M}_offload{on|off}
        exp_name = exp_dir.name
        parts = exp_name.split("_")
        tp = int(parts[0].replace("tp", ""))
        bs = int(parts[1].replace("bs", ""))
        offload = parts[2].replace("offload", "")

        # Calculate metrics
        total_time_sec = df["relative_time_sec"].max() - df["relative_time_sec"].min()
        if total_time_sec <= 0:
            total_time_sec = df["latency_ms"].sum() / 1000  # fallback

        num_requests = len(df)
        throughput_rps = num_requests / total_time_sec if total_time_sec > 0 else 0

        # Input token throughput (prefill)
        total_input_tokens = df["input_num_tokens"].sum()
        input_throughput_tps = total_input_tokens / total_time_sec if total_time_sec > 0 else 0

        # Total token throughput (input + output)
        total_output_tokens = df["output_num_tokens"].sum()
        total_tokens = total_input_tokens + total_output_tokens
        total_throughput_tps = total_tokens / total_time_sec if total_time_sec > 0 else 0

        # Normalized throughput (per GPU)
        input_tps_per_gpu = input_throughput_tps / tp
        total_tps_per_gpu = total_throughput_tps / tp

        return {
            "exp_name": exp_name,
            "tp": tp,
            "bs": bs,
            "offload": offload,
            "num_requests": num_requests,
            "throughput_rps": throughput_rps,
            "input_throughput_tps": input_throughput_tps,
            "total_throughput_tps": total_throughput_tps,
            "input_tps_per_gpu": input_tps_per_gpu,
            "total_tps_per_gpu": total_tps_per_gpu,
            "mean_ttft_ms": df["ttft_ms"].mean(),
            "p50_ttft_ms": df["ttft_ms"].median(),
            "p90_ttft_ms": df["ttft_ms"].quantile(0.9),
            "p99_ttft_ms": df["ttft_ms"].quantile(0.99),
            "mean_tpot_ms": df["tpot_ms"].mean(),
            "p50_tpot_ms": df["tpot_ms"].median(),
            "p90_tpot_ms": df["tpot_ms"].quantile(0.9),
            "p99_tpot_ms": df["tpot_ms"].quantile(0.99),
            "p999_tpot_ms": df["tpot_ms"].quantile(0.999),
            "mean_latency_ms": df["latency_ms"].mean(),
            "p50_latency_ms": df["latency_ms"].median(),
            "p90_latency_ms": df["latency_ms"].quantile(0.9),
            "p99_latency_ms": df["latency_ms"].quantile(0.99),
            "p999_latency_ms": df["latency_ms"].quantile(0.999),
            "p999_ttft_ms": df["ttft_ms"].quantile(0.999),
            # Prefill speed: ISL / TTFT (tokens/sec per request)
            "p50_prefill_tps": (df["input_num_tokens"] / (df["ttft_ms"] / 1000)).median(),
            "p90_prefill_tps": (df["input_num_tokens"] / (df["ttft_ms"] / 1000)).quantile(0.90),
            "p99_prefill_tps": (df["input_num_tokens"] / (df["ttft_ms"] / 1000)).quantile(0.99),
            "p999_prefill_tps": (df["input_num_tokens"] / (df["ttft_ms"] / 1000)).quantile(0.999),
            # Cache hit rates
            "gpu_hit_rate": gpu_hit_rate,
            "cpu_hit_rate": cpu_hit_rate,
        }
    except Exception as e:
        print(f"Error loading {exp_dir}: {e}")
        return None


def compute_pareto_frontier(points: list[tuple[float, float]], maximize_x: bool = False) -> list[tuple[float, float]]:
    """
    Compute Pareto frontier for (x, y) points.
    Y is always maximized. X is minimized by default, or maximized if maximize_x=True.

    For minimize X, maximize Y (e.g., latency vs throughput):
        - Frontier goes bottom-left to top-right
        - Low latency = low throughput, high latency = high throughput

    For maximize X, maximize Y (e.g., interactivity vs throughput):
        - Frontier goes top-left to bottom-right
        - Trade-off between the two "goods"

    Returns points sorted by X ascending for plotting.
    """
    if not points:
        return []

    # Remove invalid points
    points = [(x, y) for x, y in points if x > 0 and y > 0]
    if not points:
        return []

    frontier = []
    sorted_points = sorted(points, key=lambda p: p[0])

    if maximize_x:
        # Maximize both X and Y: frontier goes top-left to bottom-right
        # Traverse from high X to low X, keep points with increasing Y
        max_y = float('-inf')
        for x, y in reversed(sorted_points):
            if y > max_y:
                frontier.append((x, y))
                max_y = y
        return sorted(frontier, key=lambda p: p[0])
    else:
        # Minimize X, maximize Y: frontier goes bottom-left to top-right
        # Traverse from low X to high X, keep points with increasing Y
        max_y = float('-inf')
        for x, y in sorted_points:
            if y > max_y:
                frontier.append((x, y))
                max_y = y
        return frontier


def compute_pareto_frontier_with_metadata(df_subset: pd.DataFrame, x_col: str, y_col: str, maximize_x: bool = False) -> pd.DataFrame:
    """
    Compute Pareto frontier and return the rows from the dataframe that are on the frontier.
    """
    if len(df_subset) == 0:
        return pd.DataFrame()

    # Get valid points
    valid_mask = (df_subset[x_col] > 0) & (df_subset[y_col] > 0)
    df_valid = df_subset[valid_mask].copy()

    if len(df_valid) == 0:
        return pd.DataFrame()

    # Sort by x
    df_sorted = df_valid.sort_values(x_col).reset_index(drop=True)

    frontier_indices = []
    max_y = float('-inf')

    if maximize_x:
        # Traverse from high X to low X
        for i in range(len(df_sorted) - 1, -1, -1):
            y = df_sorted.iloc[i][y_col]
            if y > max_y:
                frontier_indices.append(i)
                max_y = y
        frontier_indices = frontier_indices[::-1]  # Reverse to get ascending X order
    else:
        # Traverse from low X to high X
        for i in range(len(df_sorted)):
            y = df_sorted.iloc[i][y_col]
            if y > max_y:
                frontier_indices.append(i)
                max_y = y

    return df_sorted.iloc[frontier_indices]


def generate_pareto_only_figure(df: pd.DataFrame, results_dir: Path):
    """Generate a clean figure showing only Pareto frontier points with concurrency labels."""

    # Compute interactivity
    df = df.copy()
    df["interactivity"] = 1000.0 / df["p50_tpot_ms"]

    # Get available modes and create subsets
    available_modes = sorted(df["offload"].unique())
    mode_titles = {"on": "Prefix+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}
    df_subsets = {mode: df[df["offload"] == mode] for mode in available_modes}

    # Create figure with columns for each mode
    num_cols = len(available_modes)
    fig, axes = plt.subplots(4, num_cols, figsize=(6 * num_cols, 18))
    fig.suptitle("Pareto Frontiers Only (with Concurrency Labels)", fontsize=14)

    # Handle single column case
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Color by TP
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    # Metrics configs: (row, x_col, y_col, metric_name, x_label, y_label, maximize_x)
    metrics_configs = [
        (0, "p50_ttft_ms", "input_tps_per_gpu", "TTFT", "Median TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity", "total_tps_per_gpu", "Interactivity", "Interactivity (1000/TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p50_latency_ms", "total_tps_per_gpu", "E2E Latency", "Median E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p50_prefill_tps", "total_tps_per_gpu", "Prefill Speed", "Median Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, metric_name, x_label, y_label, maximize_x in metrics_configs:
        for col, mode in enumerate(available_modes):
            ax = axes[row, col]
            df_subset = df_subsets[mode]
            title = f"{metric_name} ({mode_titles.get(mode, mode)})"

            # Get Pareto frontier points with metadata
            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                # Plot frontier line
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle='-', linewidth=2, alpha=0.5, color="black")

                # Plot points colored by TP
                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors="black", linewidths=1,
                              label=f"TP={tp}", zorder=5)

                # Add concurrency labels
                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=(5, 5),
                               fontsize=8,
                               alpha=0.8)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if len(frontier_df) > 0:
                ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_clean.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved clean Pareto plot to {output_file}")
    plt.close()


def generate_pareto_only_figure_p90(df: pd.DataFrame, results_dir: Path):
    """Generate a clean figure showing only Pareto frontier points with p90 latencies."""

    df = df.copy()
    df["interactivity_p90"] = 1000.0 / df["p90_tpot_ms"]

    available_modes = sorted(df["offload"].unique())
    mode_titles = {"on": "Prefix+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}
    df_subsets = {mode: df[df["offload"] == mode] for mode in available_modes}

    num_cols = len(available_modes)
    fig, axes = plt.subplots(4, num_cols, figsize=(6 * num_cols, 18))
    fig.suptitle("Pareto Frontiers (P90 Latencies) with Concurrency Labels", fontsize=14)

    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    metrics_configs = [
        (0, "p90_ttft_ms", "input_tps_per_gpu", "TTFT", "P90 TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity_p90", "total_tps_per_gpu", "Interactivity", "Interactivity (1000/P90 TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p90_latency_ms", "total_tps_per_gpu", "E2E Latency", "P90 E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p90_prefill_tps", "total_tps_per_gpu", "Prefill Speed", "P90 Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, metric_name, x_label, y_label, maximize_x in metrics_configs:
        for col, mode in enumerate(available_modes):
            ax = axes[row, col]
            df_subset = df_subsets[mode]
            title = f"{metric_name} ({mode_titles.get(mode, mode)})"

            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle='-', linewidth=2, alpha=0.5, color="black")

                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors="black", linewidths=1,
                              label=f"TP={tp}", zorder=5)

                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=(5, 5),
                               fontsize=8,
                               alpha=0.8)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if len(frontier_df) > 0:
                ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_clean_p90.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved clean P90 Pareto plot to {output_file}")
    plt.close()


def generate_pareto_overlay_figure_p90(df: pd.DataFrame, results_dir: Path):
    """Generate a figure with all prefix cache modes overlaid using p90 latencies."""

    df = df.copy()
    df["interactivity_p90"] = 1000.0 / df["p90_tpot_ms"]

    available_modes = df["offload"].unique()

    mode_styles = {
        "on": ("-", "black", "black", (5, 8), "normal"),
        "off": ("--", "none", "gray", (5, -12), "italic"),
        "noprefix": (":", "red", "red", (5, -25), "oblique"),
    }
    mode_labels = {
        "on": "Prefix+Offload",
        "off": "Prefix Only",
        "noprefix": "No Prefix",
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 18))
    fig.suptitle("Pareto Frontiers (P90 Latencies): Mode Comparison", fontsize=14)

    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    plot_configs = [
        (0, "p90_ttft_ms", "input_tps_per_gpu", "TTFT vs Input Throughput/GPU", "P90 TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity_p90", "total_tps_per_gpu", "Interactivity vs Total Throughput/GPU", "Interactivity (1000/P90 TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p90_latency_ms", "total_tps_per_gpu", "E2E Latency vs Total Throughput/GPU", "P90 E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p90_prefill_tps", "total_tps_per_gpu", "Prefill Speed vs Total Throughput/GPU", "P90 Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, title, x_label, y_label, maximize_x in plot_configs:
        ax = axes[row]

        for mode in ["on", "off", "noprefix"]:
            if mode not in available_modes:
                continue

            df_subset = df[df["offload"] == mode]
            linestyle, marker_edge, line_color, label_offset, font_style = mode_styles[mode]

            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle=linestyle, linewidth=2, alpha=0.6, color=line_color,
                       label=f"Pareto ({mode_labels[mode]})")

                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    label = f"TP={tp}" if mode == "on" else None
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors=marker_edge, linewidths=1.5,
                              label=label, zorder=5)

                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=label_offset,
                               fontsize=7,
                               alpha=0.7,
                               style=font_style)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_overlay_p90.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved overlay P90 Pareto plot to {output_file}")
    plt.close()


def generate_pareto_only_figure_p99(df: pd.DataFrame, results_dir: Path):
    """Generate a clean figure showing only Pareto frontier points with p99 latencies."""

    # Compute interactivity using p99
    df = df.copy()
    df["interactivity_p99"] = 1000.0 / df["p99_tpot_ms"]

    # Get available modes and create subsets
    available_modes = sorted(df["offload"].unique())
    mode_titles = {"on": "Prefix+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}
    df_subsets = {mode: df[df["offload"] == mode] for mode in available_modes}

    # Create figure with columns for each mode
    num_cols = len(available_modes)
    fig, axes = plt.subplots(4, num_cols, figsize=(6 * num_cols, 18))
    fig.suptitle("Pareto Frontiers (P99 Latencies) with Concurrency Labels", fontsize=14)

    # Handle single column case
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Color by TP
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    # Metrics configs: (row, x_col, y_col, metric_name, x_label, y_label, maximize_x)
    metrics_configs = [
        (0, "p99_ttft_ms", "input_tps_per_gpu", "TTFT", "P99 TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity_p99", "total_tps_per_gpu", "Interactivity", "Interactivity (1000/P99 TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p99_latency_ms", "total_tps_per_gpu", "E2E Latency", "P99 E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p99_prefill_tps", "total_tps_per_gpu", "Prefill Speed", "P99 Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, metric_name, x_label, y_label, maximize_x in metrics_configs:
        for col, mode in enumerate(available_modes):
            ax = axes[row, col]
            df_subset = df_subsets[mode]
            title = f"{metric_name} ({mode_titles.get(mode, mode)})"

            # Get Pareto frontier points with metadata
            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                # Plot frontier line
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle='-', linewidth=2, alpha=0.5, color="black")

                # Plot points colored by TP
                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors="black", linewidths=1,
                              label=f"TP={tp}", zorder=5)

                # Add concurrency labels
                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=(5, 5),
                               fontsize=8,
                               alpha=0.8)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if len(frontier_df) > 0:
                ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_clean_p99.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved clean P99 Pareto plot to {output_file}")
    plt.close()


def generate_pareto_overlay_figure_p99(df: pd.DataFrame, results_dir: Path):
    """Generate a figure with all prefix cache modes overlaid using p99 latencies."""

    # Compute interactivity using p99
    df = df.copy()
    df["interactivity_p99"] = 1000.0 / df["p99_tpot_ms"]

    # Get available modes
    available_modes = df["offload"].unique()

    # Mode styles
    mode_styles = {
        "on": ("-", "black", "black", (5, 8), "normal"),
        "off": ("--", "none", "gray", (5, -12), "italic"),
        "noprefix": (":", "red", "red", (5, -25), "oblique"),
    }
    mode_labels = {
        "on": "Prefix+Offload",
        "off": "Prefix Only",
        "noprefix": "No Prefix",
    }

    # Create 4x1 figure
    fig, axes = plt.subplots(4, 1, figsize=(10, 18))
    fig.suptitle("Pareto Frontiers (P99 Latencies): Mode Comparison", fontsize=14)

    # Color by TP
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    # Plot configs
    plot_configs = [
        (0, "p99_ttft_ms", "input_tps_per_gpu", "TTFT vs Input Throughput/GPU", "P99 TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity_p99", "total_tps_per_gpu", "Interactivity vs Total Throughput/GPU", "Interactivity (1000/P99 TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p99_latency_ms", "total_tps_per_gpu", "E2E Latency vs Total Throughput/GPU", "P99 E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p99_prefill_tps", "total_tps_per_gpu", "Prefill Speed vs Total Throughput/GPU", "P99 Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, title, x_label, y_label, maximize_x in plot_configs:
        ax = axes[row]

        for mode in ["on", "off", "noprefix"]:
            if mode not in available_modes:
                continue

            df_subset = df[df["offload"] == mode]
            linestyle, marker_edge, line_color, label_offset, font_style = mode_styles[mode]

            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle=linestyle, linewidth=2, alpha=0.6, color=line_color,
                       label=f"Pareto ({mode_labels[mode]})")

                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    label = f"TP={tp}" if mode == "on" else None
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors=marker_edge, linewidths=1.5,
                              label=label, zorder=5)

                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=label_offset,
                               fontsize=7,
                               alpha=0.7,
                               style=font_style)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_overlay_p99.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved overlay P99 Pareto plot to {output_file}")
    plt.close()


def generate_pareto_only_figure_p999(df: pd.DataFrame, results_dir: Path):
    """Generate a clean figure showing only Pareto frontier points with p99.9 latencies."""

    df = df.copy()
    df["interactivity_p999"] = 1000.0 / df["p999_tpot_ms"]

    available_modes = sorted(df["offload"].unique())
    mode_titles = {"on": "Prefix+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}
    df_subsets = {mode: df[df["offload"] == mode] for mode in available_modes}

    num_cols = len(available_modes)
    fig, axes = plt.subplots(4, num_cols, figsize=(6 * num_cols, 18))
    fig.suptitle("Pareto Frontiers (P99.9 Latencies) with Concurrency Labels", fontsize=14)

    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    metrics_configs = [
        (0, "p999_ttft_ms", "input_tps_per_gpu", "TTFT", "P99.9 TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity_p999", "total_tps_per_gpu", "Interactivity", "Interactivity (1000/P99.9 TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p999_latency_ms", "total_tps_per_gpu", "E2E Latency", "P99.9 E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p999_prefill_tps", "total_tps_per_gpu", "Prefill Speed", "P99.9 Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, metric_name, x_label, y_label, maximize_x in metrics_configs:
        for col, mode in enumerate(available_modes):
            ax = axes[row, col]
            df_subset = df_subsets[mode]
            title = f"{metric_name} ({mode_titles.get(mode, mode)})"

            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle='-', linewidth=2, alpha=0.5, color="black")

                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors="black", linewidths=1,
                              label=f"TP={tp}", zorder=5)

                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=(5, 5),
                               fontsize=8,
                               alpha=0.8)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if len(frontier_df) > 0:
                ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_clean_p999.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved clean P99.9 Pareto plot to {output_file}")
    plt.close()


def generate_pareto_overlay_figure_p999(df: pd.DataFrame, results_dir: Path):
    """Generate a figure with all prefix cache modes overlaid using p99.9 latencies."""

    df = df.copy()
    df["interactivity_p999"] = 1000.0 / df["p999_tpot_ms"]

    available_modes = df["offload"].unique()

    mode_styles = {
        "on": ("-", "black", "black", (5, 8), "normal"),
        "off": ("--", "none", "gray", (5, -12), "italic"),
        "noprefix": (":", "red", "red", (5, -25), "oblique"),
    }
    mode_labels = {
        "on": "Prefix+Offload",
        "off": "Prefix Only",
        "noprefix": "No Prefix",
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 18))
    fig.suptitle("Pareto Frontiers (P99.9 Latencies): Mode Comparison", fontsize=14)

    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    plot_configs = [
        (0, "p999_ttft_ms", "input_tps_per_gpu", "TTFT vs Input Throughput/GPU", "P99.9 TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity_p999", "total_tps_per_gpu", "Interactivity vs Total Throughput/GPU", "Interactivity (1000/P99.9 TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p999_latency_ms", "total_tps_per_gpu", "E2E Latency vs Total Throughput/GPU", "P99.9 E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p999_prefill_tps", "total_tps_per_gpu", "Prefill Speed vs Total Throughput/GPU", "P99.9 Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, title, x_label, y_label, maximize_x in plot_configs:
        ax = axes[row]

        for mode in ["on", "off", "noprefix"]:
            if mode not in available_modes:
                continue

            df_subset = df[df["offload"] == mode]
            linestyle, marker_edge, line_color, label_offset, font_style = mode_styles[mode]

            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle=linestyle, linewidth=2, alpha=0.6, color=line_color,
                       label=f"Pareto ({mode_labels[mode]})")

                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    label = f"TP={tp}" if mode == "on" else None
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors=marker_edge, linewidths=1.5,
                              label=label, zorder=5)

                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=label_offset,
                               fontsize=7,
                               alpha=0.7,
                               style=font_style)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_overlay_p999.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved overlay P99.9 Pareto plot to {output_file}")
    plt.close()


def generate_combined_pareto_figure(df: pd.DataFrame, results_dir: Path,
                                    percentile: str = "p50"):
    """Generate a combined Pareto frontier across ALL offload modes.

    Points are colored by TP and edge-styled by offload mode so the viewer
    can see both the overall optimal frontier and which config each point
    comes from.

    percentile: one of "p50", "p90", "p99", "p999"
    """
    from matplotlib.lines import Line2D

    pct = percentile  # e.g. "p50"
    pct_label = {"p50": "Median", "p90": "P90", "p99": "P99", "p999": "P99.9"}[pct]
    suffix = "" if pct == "p50" else f"_{pct}"

    df = df.copy()
    interactivity_col = f"interactivity{suffix}"
    df[interactivity_col] = 1000.0 / df[f"{pct}_tpot_ms"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    fig.suptitle(f"Combined Pareto Frontier — {pct_label} SLA (All Configs)", fontsize=14)

    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    mode_edge = {
        "on":       {"edgecolors": "black",  "linewidths": 1.8},
        "off":      {"edgecolors": "gray",   "linewidths": 1.2},
        "noprefix": {"edgecolors": "#cc0000", "linewidths": 1.2},
    }
    mode_short = {"on": "P+O", "off": "P", "noprefix": "NP"}

    metrics_configs = [
        (0, f"{pct}_ttft_ms",     "input_tps_per_gpu", "TTFT",          f"{pct_label} TTFT (ms)",                       "Input Throughput/GPU (tok/s)", False),
        (1, interactivity_col,    "total_tps_per_gpu", "Interactivity", f"Interactivity (1000/{pct_label} TPOT)",       "Total Throughput/GPU (tok/s)", True),
        (2, f"{pct}_latency_ms",  "total_tps_per_gpu", "E2E Latency",   f"{pct_label} E2E Latency (ms)",               "Total Throughput/GPU (tok/s)", False),
        (3, f"{pct}_prefill_tps", "total_tps_per_gpu", "Prefill Speed",  f"{pct_label} Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, metric_name, x_label, y_label, maximize_x in metrics_configs:
        ax = axes[row]

        # All-data scatter (faded background)
        for tp in sorted(df["tp"].unique()):
            tp_data = df[df["tp"] == tp]
            ax.scatter(tp_data[x_col], tp_data[y_col],
                       c=tp_colors.get(tp, "purple"),
                       marker=tp_markers.get(tp, "x"),
                       s=40, alpha=0.15, linewidths=0.3,
                       edgecolors="gray")

        # Combined Pareto frontier
        frontier_df = compute_pareto_frontier_with_metadata(df, x_col, y_col, maximize_x)

        if len(frontier_df) > 0:
            ax.plot(frontier_df[x_col], frontier_df[y_col],
                    linestyle='-', linewidth=2, alpha=0.5, color="black",
                    label="Pareto Frontier", zorder=4)

            for _, pt in frontier_df.iterrows():
                tp = pt["tp"]
                mode = pt["offload"]
                edge_kw = mode_edge.get(mode, {"edgecolors": "black", "linewidths": 1})
                ax.scatter(pt[x_col], pt[y_col],
                           c=tp_colors.get(tp, "purple"),
                           marker=tp_markers.get(tp, "x"),
                           s=160, alpha=0.9, zorder=5,
                           **edge_kw)

            for _, pt in frontier_df.iterrows():
                ax.annotate(
                    f"conc={int(pt['bs'])} {mode_short.get(pt['offload'], '')}",
                    (pt[x_col], pt[y_col]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.85)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{metric_name} — All Configs Combined")
        ax.grid(True, alpha=0.3)

        handles = [Line2D([0], [0], color="black", lw=2, label="Pareto Frontier")]
        for tp in sorted(df["tp"].unique()):
            handles.append(Line2D([0], [0], marker=tp_markers[tp], color="w",
                                  markerfacecolor=tp_colors[tp], markersize=8,
                                  markeredgecolor="black", label=f"TP={tp}"))
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
                              markersize=8, markeredgecolor="black", markeredgewidth=1.8,
                              label="Edge: P+Offload"))
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
                              markersize=8, markeredgecolor="gray", markeredgewidth=1.2,
                              label="Edge: Prefix Only"))
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
                              markersize=8, markeredgecolor="#cc0000", markeredgewidth=1.2,
                              label="Edge: No Prefix"))
        ax.legend(handles=handles, fontsize=7,
                  loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()
    fname = f"pareto_frontiers_combined{suffix}.png"
    output_file = results_dir / fname
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved combined {pct_label} Pareto plot to {output_file}")
    plt.close()


def generate_pareto_overlay_figure(df: pd.DataFrame, results_dir: Path):
    """Generate a figure with all prefix cache modes overlaid for direct comparison."""

    # Compute interactivity
    df = df.copy()
    df["interactivity"] = 1000.0 / df["p50_tpot_ms"]

    # Get available modes
    available_modes = df["offload"].unique()

    # Mode styles: (linestyle, marker_edge, line_color, label_offset, font_style)
    mode_styles = {
        "on": ("-", "black", "black", (5, 8), "normal"),       # Prefix + Offload
        "off": ("--", "none", "gray", (5, -12), "italic"),     # Prefix only
        "noprefix": (":", "red", "red", (5, -25), "oblique"),  # No prefix caching
    }
    mode_labels = {
        "on": "Prefix+Offload",
        "off": "Prefix Only",
        "noprefix": "No Prefix",
    }

    # Create 4x1 figure
    fig, axes = plt.subplots(4, 1, figsize=(10, 18))
    fig.suptitle("Pareto Frontiers: Prefix Caching Mode Comparison", fontsize=14)

    # Color by TP
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    # Plot configs: (row, x_col, y_col, title, x_label, y_label, maximize_x)
    plot_configs = [
        (0, "p50_ttft_ms", "input_tps_per_gpu", "TTFT vs Input Throughput/GPU", "Median TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity", "total_tps_per_gpu", "Interactivity vs Total Throughput/GPU", "Interactivity (1000/TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p50_latency_ms", "total_tps_per_gpu", "E2E Latency vs Total Throughput/GPU", "Median E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p50_prefill_tps", "total_tps_per_gpu", "Prefill Speed vs Total Throughput/GPU", "Median Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, title, x_label, y_label, maximize_x in plot_configs:
        ax = axes[row]

        # Plot all available modes
        for mode in ["on", "off", "noprefix"]:
            if mode not in available_modes:
                continue

            df_subset = df[df["offload"] == mode]
            linestyle, marker_edge, line_color, label_offset, font_style = mode_styles[mode]

            frontier_df = compute_pareto_frontier_with_metadata(df_subset, x_col, y_col, maximize_x)

            if len(frontier_df) > 0:
                # Plot frontier line
                ax.plot(frontier_df[x_col], frontier_df[y_col],
                       linestyle=linestyle, linewidth=2, alpha=0.6, color=line_color,
                       label=f"Pareto ({mode_labels[mode]})")

                # Plot points colored by TP
                for tp in sorted(frontier_df["tp"].unique()):
                    tp_data = frontier_df[frontier_df["tp"] == tp]
                    # Only add TP to legend once (for first mode)
                    label = f"TP={tp}" if mode == "on" else None
                    ax.scatter(tp_data[x_col], tp_data[y_col],
                              c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                              s=150, alpha=0.9, edgecolors=marker_edge, linewidths=1.5,
                              label=label, zorder=5)

                # Add concurrency labels
                for _, point in frontier_df.iterrows():
                    ax.annotate(f"conc={point['bs']}",
                               (point[x_col], point[y_col]),
                               textcoords="offset points",
                               xytext=label_offset,
                               fontsize=7,
                               alpha=0.7,
                               style=font_style)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers_overlay.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved overlay Pareto plot to {output_file}")
    plt.close()


def main(results_dir: Path):
    # Load all experiments
    experiments = []
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith("tp"):
            data = load_experiment_data(exp_dir)
            if data:
                experiments.append(data)

    if not experiments:
        print("No experiment data found!")
        return

    df = pd.DataFrame(experiments)
    print(f"Loaded {len(df)} experiments")
    print(df[["exp_name", "tp", "bs", "offload", "input_tps_per_gpu", "total_tps_per_gpu", "p50_ttft_ms"]].to_string())

    # Compute interactivity = 1000 / TPOT (tokens per second for decode)
    df["interactivity"] = 1000.0 / df["p50_tpot_ms"]

    # Get available modes and create subsets
    available_modes = sorted(df["offload"].unique())
    mode_titles = {"on": "Prefix+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}
    df_subsets = {mode: df[df["offload"] == mode] for mode in available_modes}

    # Create figure with columns for each mode
    num_cols = len(available_modes)
    fig, axes = plt.subplots(4, num_cols, figsize=(6 * num_cols, 18))
    fig.suptitle("Pareto Frontiers: Throughput/GPU vs Latency (All Points)", fontsize=14)

    # Handle single column case
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Color by TP
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    # Metrics configs: (row, x_col, y_col, metric_name, x_label, y_label, maximize_x)
    metrics_configs = [
        (0, "p50_ttft_ms", "input_tps_per_gpu", "TTFT", "Median TTFT (ms)", "Input Throughput/GPU (tok/s)", False),
        (1, "interactivity", "total_tps_per_gpu", "Interactivity", "Interactivity (1000/TPOT)", "Total Throughput/GPU (tok/s)", True),
        (2, "p50_latency_ms", "total_tps_per_gpu", "E2E Latency", "Median E2E Latency (ms)", "Total Throughput/GPU (tok/s)", False),
        (3, "p50_prefill_tps", "total_tps_per_gpu", "Prefill Speed", "Median Prefill Speed (ISL/TTFT tok/s)", "Total Throughput/GPU (tok/s)", True),
    ]

    for row, x_col, y_col, metric_name, x_label, y_label, maximize_x in metrics_configs:
        for col, mode in enumerate(available_modes):
            ax = axes[row, col]
            df_subset = df_subsets[mode]
            title = f"{metric_name} ({mode_titles.get(mode, mode)})"

            # Compute and plot Pareto frontier
            points = list(zip(df_subset[x_col], df_subset[y_col]))
            frontier = compute_pareto_frontier(points, maximize_x=maximize_x)

            if frontier:
                fx, fy = zip(*frontier)
                ax.plot(fx, fy, linestyle='-', linewidth=2, alpha=0.8, color="black", label="Pareto frontier")

            # Plot points colored by TP
            for tp in sorted(df_subset["tp"].unique()):
                tp_data = df_subset[df_subset["tp"] == tp]
                ax.scatter(tp_data[x_col], tp_data[y_col],
                          c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                          s=100, alpha=0.8, edgecolors="black", linewidths=0.5,
                          label=f"TP={tp}")

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower right" if not maximize_x else "upper right")

    plt.tight_layout()

    output_file = results_dir / "pareto_frontiers.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close()

    # Also save summary CSV
    summary_file = results_dir / "experiment_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    # Generate clean Pareto-only figure
    generate_pareto_only_figure(df, results_dir)

    # Generate combined Pareto frontier (all configs pooled) for each SLA percentile
    for pct in ("p50", "p90", "p99", "p999"):
        generate_combined_pareto_figure(df, results_dir, percentile=pct)

    # Generate overlay figure (on vs off comparison)
    generate_pareto_overlay_figure(df, results_dir)

    # Generate P90 versions
    generate_pareto_only_figure_p90(df, results_dir)
    generate_pareto_overlay_figure_p90(df, results_dir)

    # Generate P99 versions
    generate_pareto_only_figure_p99(df, results_dir)
    generate_pareto_overlay_figure_p99(df, results_dir)

    # Generate P99.9 versions
    generate_pareto_only_figure_p999(df, results_dir)
    generate_pareto_overlay_figure_p999(df, results_dir)

    # Generate cache hit rate plot
    generate_cache_hit_rate_figure(df, results_dir)


def generate_cache_hit_rate_figure(df: pd.DataFrame, results_dir: Path):
    """Generate plot showing throughput vs cache hit rates (GPU and CPU)."""

    # Get available modes
    available_modes = sorted(df["offload"].unique())
    mode_titles = {"on": "Prefix+Offload", "off": "Prefix Only", "noprefix": "No Prefix"}

    # Create 2x3 figure (GPU hit rate row, CPU hit rate row, columns for each mode)
    num_cols = len(available_modes)
    fig, axes = plt.subplots(2, num_cols, figsize=(6 * num_cols, 10))
    fig.suptitle("Cache Hit Rate vs Throughput", fontsize=14)

    # Handle single column case
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Color by TP
    tp_colors = {1: "blue", 2: "green", 4: "orange", 8: "red"}
    tp_markers = {1: "o", 2: "s", 4: "^", 8: "D"}

    # Plot configs: (row, hit_rate_col, title_prefix)
    hit_rate_configs = [
        (0, "gpu_hit_rate", "GPU"),
        (1, "cpu_hit_rate", "CPU"),
    ]

    for row, hit_rate_col, hit_type in hit_rate_configs:
        for col, mode in enumerate(available_modes):
            ax = axes[row, col]
            df_subset = df[df["offload"] == mode].dropna(subset=[hit_rate_col])

            if len(df_subset) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{hit_type} Hit Rate ({mode_titles.get(mode, mode)})")
                continue

            # Plot points colored by TP
            for tp in sorted(df_subset["tp"].unique()):
                tp_data = df_subset[df_subset["tp"] == tp]
                ax.scatter(tp_data[hit_rate_col], tp_data["total_tps_per_gpu"],
                          c=tp_colors.get(tp, "purple"), marker=tp_markers.get(tp, "x"),
                          s=100, alpha=0.8, edgecolors="black", linewidths=0.5,
                          label=f"TP={tp}")

            # Add concurrency labels
            for _, point in df_subset.iterrows():
                ax.annotate(f"bs={int(point['bs'])}",
                           (point[hit_rate_col], point["total_tps_per_gpu"]),
                           textcoords="offset points",
                           xytext=(5, 5),
                           fontsize=7,
                           alpha=0.7)

            ax.set_xlabel(f"{hit_type} Cache Hit Rate (%)")
            ax.set_ylabel("Total Throughput/GPU (tok/s)")
            ax.set_title(f"{hit_type} Hit Rate ({mode_titles.get(mode, mode)})")
            ax.set_xlim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()

    output_file = results_dir / "cache_hit_rates.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved cache hit rate plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_pareto.py <results_dir>")
        print("Example: python plot_pareto.py ~/sweep_results_20260204_062339")
        sys.exit(1)

    results_dir = Path(sys.argv[1]).expanduser()
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    main(results_dir)
