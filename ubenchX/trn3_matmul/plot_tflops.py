"""Plot ls_bf16_alt TFLOP/s and utilization of the bf16 peak vs N at M=128 (needs matplotlib).

  python ubenchX/trn3_matmul/plot_tflops.py ls_alt.csv alt_tflops_n.png

ls_bf16_alt issues a new stationary for every matmul (bf16, K=128). TFLOP/s = 2 K M N / II, median
over the profiled executions. Utilization = TFLOP/s / peak, where peak is 128 x 128 PEs x 2 FLOP x
2.4 GHz = 78.6 TFLOP/s (bf16); the right axis is the same line in % of peak.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

csv, out = sys.argv[1:]
PEAK = 128 * 128 * 2 * 2.4e9 / 1e12

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"

df = pd.read_csv(csv)
df = df[(df.kernel == "ls_bf16_alt") & (df.M == 128)]
g = df.groupby("N").TFLOPS.median()
ns = list(g.index)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(PEAK, color=INK2, lw=1.5, ls=(0, (4, 3)), zorder=1)
ax.text(ns[0] / 1.25, PEAK + 1.5, f"bf16 peak {PEAK:.1f} TFLOP/s", color=INK2, fontsize=10, va="bottom")
ax.plot(ns, g, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
for n, v in g.items():
    low = v < 15  # near the x axis: label to the lower right instead of below
    ax.annotate(f"{v:.1f}\n{v / PEAK:.0%}", (n, v), xytext=(12, -6) if low else (0, -14), textcoords="offset points",
                ha="left" if low else "center", va="top", color=INK2, fontsize=9.5, linespacing=1.1)

ax.set_xscale("log", base=2)
ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.3)
ax.xaxis.set_major_locator(FixedLocator(ns))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(n) for n in ns])
ax.set_ylim(0, 90)
ax.set_xlabel("N (moving columns per matmul)", color=INK2)
ax.set_ylabel("TFLOP/s (bf16)", color=INK2)
# right axis: the same line as % of peak (a fixed rescale, not a second measure)
ax2 = ax.secondary_yaxis("right", functions=(lambda t: t / PEAK * 100, lambda u: u * PEAK / 100))
ax2.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
ax2.set_yticklabels([f"{u}%" for u in (0, 25, 50, 75, 100)])
ax2.set_ylabel("Utilization (% of bf16 peak)", color=INK2)
ax2.spines["right"].set_color(AXIS)
ax2.tick_params(colors=MUTED, labelcolor=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
fig.suptitle("Alternating Stationary Throughput across N (M=128)", x=0.03, y=0.97, ha="left", color=INK,
             fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
