"""Plot fixed-stationary matmul II and latency vs M at N=512 (needs matplotlib).

  python ubenchX/trn3_matmul/plot_m.py runs/sweep_r0.csv fixed_m.png

TensorE cycles (2.4 GHz), median over the profiled executions of mm_bf16_fixed (bf16, K=128, one
stationary reused). II = start-to-start interval between consecutive MATMULs; latency = start-to-end
of one MATMUL. M sets how many of the 128 PE columns hold weights, so it changes FLOPs, not timing.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

csv, out = sys.argv[1:]
N = 512

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("dur_cycles", "Latency", "#eb6834", "s"), ("II_cycles", "II", "#2a78d6", "o")]

df = pd.read_csv(csv)
df = df[(df.kernel == "mm_bf16_fixed") & (df.N == N)]
g = df.groupby("M")[["II_cycles", "dur_cycles"]].median()
ms = list(g.index)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for col, name, color, marker in SERIES:
    ax.plot(ms, g[col], color=color, lw=2, marker=marker, ms=8, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3, label=name)
    ax.annotate(f"{name} {g[col].iloc[-1]:.0f} cycles", (ms[-1], g[col].iloc[-1]), xytext=(10, 0),
                textcoords="offset points", ha="left", va="center", color=INK2, fontsize=10)

ax.xaxis.set_major_locator(FixedLocator(ms))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(m) for m in ms])
ax.set_xlim(0, ms[-1] + 40)
ax.set_ylim(0, 1000)
ax.set_xlabel("M (stationary columns per matmul)", color=INK2)
ax.set_ylabel("TensorE cycles", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
ax.legend(loc="lower left", frameon=False, labelcolor=INK)
fig.suptitle(f"MM Performance Doesn't Change with M (N={N})", x=0.03, y=0.97, ha="left", color=INK,
             fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
