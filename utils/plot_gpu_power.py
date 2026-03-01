#!/usr/bin/env python3
"""
GPU Power Over Time — Visualization Tool

Reads nvidia-smi CSV output and generates power-over-time graphs.
Saves plots as PNG artifacts.

Usage:
    python3 utils/plot_gpu_power.py gpu_power.csv [--output-dir .] [--title "Run Label"]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def load_power_csv(path: str) -> pd.DataFrame:
    """Load nvidia-smi CSV, clean column names, parse timestamps."""
    df = pd.read_csv(path, skipinitialspace=True)
    # nvidia-smi columns have trailing units like " [W]", " [%]", " [MiB]"
    df.columns = [c.strip() for c in df.columns]

    # Parse numeric columns (strip units)
    for col in df.columns:
        if col in ("timestamp", "name"):
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce",
            )

    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip(), format="mixed")
    df["index"] = df["index"].astype(int)
    return df


def compute_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed_s column (seconds since first sample)."""
    t0 = df["timestamp"].min()
    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()
    return df


def plot_power_per_gpu(df: pd.DataFrame, title: str, out: Path):
    """Individual GPU power draw over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    gpus = sorted(df["index"].unique())

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gpus), 1)))
    for i, gpu_id in enumerate(gpus):
        gdf = df[df["index"] == gpu_id].sort_values("elapsed_s")
        ax.plot(
            gdf["elapsed_s"], gdf["power.draw [W]"],
            label=f"GPU {gpu_id}", color=colors[i], alpha=0.85, linewidth=0.8,
        )

    # Power limit reference line (use first non-NaN value)
    plimit = df["power.limit [W]"].dropna().iloc[0] if "power.limit [W]" in df.columns else None
    if plimit and plimit > 0:
        ax.axhline(plimit, color="red", linestyle="--", alpha=0.5, label=f"Power limit ({plimit:.0f} W)")

    ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax.set_ylabel("Power Draw (W)", fontsize=11)
    ax.set_title(f"GPU Power Draw Over Time — {title}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / "gpu_power_per_gpu.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'gpu_power_per_gpu.png'}")


def plot_total_power(df: pd.DataFrame, title: str, out: Path):
    """Total (sum across GPUs) power draw over time."""
    total = df.groupby("elapsed_s").agg(
        total_power=("power.draw [W]", "sum"),
        num_gpus=("index", "nunique"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(total["elapsed_s"], total["total_power"], alpha=0.3, color="steelblue")
    ax.plot(total["elapsed_s"], total["total_power"], color="steelblue", linewidth=1)

    avg_power = total["total_power"].mean()
    peak_power = total["total_power"].max()
    ax.axhline(avg_power, color="orange", linestyle="--", alpha=0.7,
               label=f"Avg: {avg_power:.0f} W")
    ax.axhline(peak_power, color="red", linestyle=":", alpha=0.5,
               label=f"Peak: {peak_power:.0f} W")

    ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax.set_ylabel("Total Power Draw (W)", fontsize=11)
    ax.set_title(f"Total GPU Power Draw — {title}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / "gpu_power_total.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'gpu_power_total.png'}")


def plot_temperature(df: pd.DataFrame, title: str, out: Path):
    """GPU temperature over time."""
    if "temperature.gpu" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    gpus = sorted(df["index"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gpus), 1)))

    for i, gpu_id in enumerate(gpus):
        gdf = df[df["index"] == gpu_id].sort_values("elapsed_s")
        ax.plot(
            gdf["elapsed_s"], gdf["temperature.gpu"],
            label=f"GPU {gpu_id}", color=colors[i], alpha=0.85, linewidth=0.8,
        )

    ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax.set_ylabel("Temperature (\u00b0C)", fontsize=11)
    ax.set_title(f"GPU Temperature Over Time — {title}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out / "gpu_temperature.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'gpu_temperature.png'}")


def plot_utilization(df: pd.DataFrame, title: str, out: Path):
    """GPU utilization over time."""
    if "utilization.gpu [%]" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    gpus = sorted(df["index"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gpus), 1)))

    for i, gpu_id in enumerate(gpus):
        gdf = df[df["index"] == gpu_id].sort_values("elapsed_s")
        ax.plot(
            gdf["elapsed_s"], gdf["utilization.gpu [%]"],
            label=f"GPU {gpu_id}", color=colors[i], alpha=0.85, linewidth=0.8,
        )

    ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax.set_ylabel("GPU Utilization (%)", fontsize=11)
    ax.set_title(f"GPU Utilization Over Time — {title}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out / "gpu_utilization.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'gpu_utilization.png'}")


def plot_power_utilization_correlation(df: pd.DataFrame, title: str, out: Path):
    """Scatter plot: power vs GPU utilization."""
    if "utilization.gpu [%]" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    gpus = sorted(df["index"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gpus), 1)))

    for i, gpu_id in enumerate(gpus):
        gdf = df[df["index"] == gpu_id]
        ax.scatter(
            gdf["utilization.gpu [%]"], gdf["power.draw [W]"],
            label=f"GPU {gpu_id}", color=colors[i], alpha=0.3, s=8,
        )

    ax.set_xlabel("GPU Utilization (%)", fontsize=11)
    ax.set_ylabel("Power Draw (W)", fontsize=11)
    ax.set_title(f"Power vs Utilization — {title}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "gpu_power_vs_utilization.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'gpu_power_vs_utilization.png'}")


def print_summary(df: pd.DataFrame, title: str):
    """Print summary statistics to stdout."""
    gpus = sorted(df["index"].unique())
    duration = df["elapsed_s"].max()

    print(f"\n{'='*60}")
    print(f"GPU Power Summary — {title}")
    print(f"{'='*60}")
    print(f"Duration: {duration:.0f}s | GPUs: {len(gpus)} | Samples/GPU: {len(df)//len(gpus)}")
    print(f"{'─'*60}")
    print(f"{'GPU':>5} | {'Avg W':>8} | {'Peak W':>8} | {'Avg °C':>7} | {'Peak °C':>8} | {'Avg Util%':>9}")
    print(f"{'─'*60}")

    total_avg = 0
    total_peak = 0
    for gpu_id in gpus:
        gdf = df[df["index"] == gpu_id]
        avg_p = gdf["power.draw [W]"].mean()
        peak_p = gdf["power.draw [W]"].max()
        avg_t = gdf["temperature.gpu"].mean() if "temperature.gpu" in gdf.columns else 0
        peak_t = gdf["temperature.gpu"].max() if "temperature.gpu" in gdf.columns else 0
        avg_u = gdf["utilization.gpu [%]"].mean() if "utilization.gpu [%]" in gdf.columns else 0
        total_avg += avg_p
        total_peak += peak_p
        print(f"  {gpu_id:>3} | {avg_p:>7.1f} | {peak_p:>7.1f} | {avg_t:>6.1f} | {peak_t:>7.0f} | {avg_u:>8.1f}")

    print(f"{'─'*60}")
    print(f"Total | {total_avg:>7.0f} | {total_peak:>7.0f} |")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot GPU power data from nvidia-smi CSV")
    parser.add_argument("csv_file", help="Path to gpu_power.csv")
    parser.add_argument("--output-dir", default=".", help="Directory for output PNGs")
    parser.add_argument("--title", default="", help="Label for the run (e.g. 'gptoss 1k8k fp4 h100')")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = load_power_csv(str(csv_path))
    df = compute_elapsed(df)

    title = args.title or csv_path.stem
    print(f"Generating plots for: {title}")
    print(f"  Rows: {len(df)}, GPUs: {df['index'].nunique()}, Duration: {df['elapsed_s'].max():.0f}s")

    plot_power_per_gpu(df, title, out)
    plot_total_power(df, title, out)
    plot_temperature(df, title, out)
    plot_utilization(df, title, out)
    plot_power_utilization_correlation(df, title, out)
    print_summary(df, title)

    print("Done — all plots saved.")


if __name__ == "__main__":
    main()
