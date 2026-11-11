"""Plot fixed-stationary matmul II and latency vs M, one line per N (needs matplotlib).

  python ubenchX/trn3_matmul/plot_m.py fixed.csv [more.csv ...] fixed_m.png

TensorE cycles (2.4 GHz), median over the profiled executions of mm_bf16_fixed (bf16, K=128, one
stationary reused), for every N that was swept over M. II = start-to-start interval between consecutive
MATMULs; latency = start-to-end of one MATMUL. M sets how many of the 128 PE columns hold weights, so
it changes FLOPs, not timing. II and latency get their own panel because N=512's II (518) and N=128's
latency (523) would overlap on one axis.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

*csvs, out = sys.argv[1:]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, assigned by N ascending
PANELS = [("II_cycles", "II"), ("dur_cycles", "Latency")]

df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
df = df[df.kernel == "mm_bf16_fixed"]
g = df.groupby(["N", "M"])[["II_cycles", "dur_cycles"]].median()
ns = [n for n in g.index.get_level_values("N").unique() if len(g.loc[n]) > 1]  # N values swept over M
ms = sorted(g.loc[ns].index.get_level_values("M").unique())
top = float(g.loc[ns].to_numpy().max())
yt = [2 ** k for k in range(6, int(np.ceil(np.log2(top * 1.3))) + 1)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE, sharey=True)
for ax, (col, name) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    for n, color in zip(ns, COLORS):
        s = g.loc[n][col]
        ax.plot(s.index, s, color=color, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3, label=f"N={n}")
        ax.annotate(f"{s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", color=INK2, fontsize=10)
    ax.set_title(name, loc="left", color=INK, fontsize=12)
    ax.xaxis.set_major_locator(FixedLocator(ms))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlim(0, ms[-1] + 32)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(y) for y in yt])
    ax.set_ylim(yt[0], yt[-1])
    ax.set_xlabel("M (stationary columns per matmul)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
axes[0].set_ylabel("TensorE cycles", color=INK2)
h, l = axes[0].get_legend_handles_labels()
axes[1].legend(h[::-1], l[::-1], loc="lower right", frameon=False, labelcolor=INK)  # empty band below 512
fig.suptitle("MM Performance Doesn't Change with M", x=0.02, y=0.97, ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
