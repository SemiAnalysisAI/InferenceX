#!/usr/bin/env python3
"""Bar chart of mean pointer-chase hop latency per cache level across AMD GPUs.

Reads the per-CU summaries written by l2_pointer_chase
(``results/<sku>_<level>_cu_info_<date>.csv``), averages ``mean_latency`` over
all CUs of each SKU and level, and plots grouped bars in shader-clock cycles
and in nanoseconds (cycles divided by the shader clock the run reported).

Usage:
    python plot_latency_bars.py                       # all SKUs found in results/
    python plot_latency_bars.py --skus mi300x,mi355x --clock mi355x=2400
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

LEVELS = [("l2_4mb", "L2\n(4 MB chain)"),
          ("infinity_cache_64mb", "Infinity Cache\n(64 MB chain)"),
          ("hbm_1gb", "HBM\n(1 GB chain)")]
# Shader clock reported by hipDeviceAttributeClockRate on the nodes that produced
# the committed results; override with --clock sku=MHz.
DEFAULT_CLOCK_MHZ = {"mi300x": 2100, "mi325x": 2100, "mi355x": 2400}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--skus", help="Comma-separated SKU prefixes in display order (default: all found)")
    ap.add_argument("--clock", action="append", default=[], help="sku=MHz shader clock override")
    ap.add_argument("--out", help="Output PNG (default: results/amd_hop_latency_by_level.png)")
    args = ap.parse_args()

    clocks = dict(DEFAULT_CLOCK_MHZ)
    for spec in args.clock:
        sku, mhz = spec.split("=")
        clocks[sku.lower()] = float(mhz)

    files = glob.glob(os.path.join(args.results, "*_cu_info_*.csv"))
    data = {}
    for f in files:
        m = re.match(r"([a-z0-9]+)_(l2_4mb|infinity_cache_64mb|hbm_1gb)_cu_info_\d{4}-\d{2}-\d{2}\.csv$", os.path.basename(f))
        if not m:
            continue
        sku, level = m.group(1), m.group(2)
        data.setdefault(sku, {})[level] = pd.read_csv(f)["mean_latency"].mean()
    skus = args.skus.split(",") if args.skus else sorted(data)
    missing = [s for s in skus if s not in data]
    if missing:
        raise SystemExit(f"no cu_info files for {missing} in {args.results}")

    print(f"{'SKU':8s} {'clock':>7s}  " + "  ".join(f"{lvl[0]:>22s}" for lvl in LEVELS))
    for sku in skus:
        row = [data[sku].get(lvl, float('nan')) for lvl, _ in LEVELS]
        mhz = clocks.get(sku)
        cells = [f"{c:6.1f} cyc / {c / mhz * 1e3:5.0f} ns" if mhz else f"{c:6.1f} cyc" for c in row]
        print(f"{sku.upper():8s} {str(mhz) + ' MHz' if mhz else '':>7s}  " + "  ".join(f"{c:>22s}" for c in cells))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    width = 0.8 / len(skus)
    for ax, unit in zip(axes, ("cycles", "ns")):
        ax.set_facecolor(SURFACE)
        for i, sku in enumerate(skus):
            vals = [data[sku].get(lvl, float("nan")) for lvl, _ in LEVELS]
            if unit == "ns":
                vals = [v / clocks[sku] * 1e3 for v in vals]
            xs = [j + (i - (len(skus) - 1) / 2) * width for j in range(len(LEVELS))]
            bars = ax.bar(xs, vals, width=width * 0.92, color=SERIES_COLORS[i], label=sku.upper(),
                          edgecolor=SURFACE, linewidth=1.0, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8, color=INK2, zorder=4)
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([label for _, label in LEVELS], color=INK, fontsize=10)
        ax.set_ylabel("mean hop latency (shader-clock cycles)" if unit == "cycles" else "mean hop latency (ns)",
                      color=INK2, fontsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle("Pointer-chase hop latency by cache level, one wavefront per CU", color=INK, fontsize=12)
    fig.text(0.5, 0.01, "mean over all CUs and addresses; ns = cycles / shader clock reported by HIP "
             f"({', '.join(f'{s.upper()} {clocks[s]:.0f} MHz' for s in skus)})",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = args.out or os.path.join(args.results, "amd_hop_latency_by_level.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
