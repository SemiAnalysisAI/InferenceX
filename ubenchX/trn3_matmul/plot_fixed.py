"""Plot fixed-stationary matmul II or latency vs N at M=128 with its prediction (needs matplotlib).

  python ubenchX/trn3_matmul/plot_fixed.py runs/sweep_r0.csv fixed_ii.png             # II
  python ubenchX/trn3_matmul/plot_fixed.py --latency runs/sweep_r0.csv fixed_lat.png  # latency

Values are TensorE cycles (2.4 GHz), median over the profiled executions of mm_bf16_fixed (bf16, K=128,
one stationary reused); the dashed line is the prediction.
  II       start-to-start interval between consecutive MATMULs. Prediction N + 6: N moving columns
           streamed one per cycle plus a 6-cycle per-instruction overhead.
  latency  start-to-end of one MATMUL. Prediction N + 393 = N + (R-1) + (C-1) + 139: stream, fill down
           the 128 rows, drain across the 128 columns, and ~139 cycles of fixed overhead.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

lat = "--latency" in sys.argv
csv, out = [a for a in sys.argv[1:] if a != "--latency"]
if lat:
    COL, OVERHEAD, YLABEL, TITLE = "dur_cycles", 393, "Latency (TensorE cycles)", "Fixed Stationary Latency across N (M=128)"
else:
    COL, OVERHEAD, YLABEL, TITLE = "II_cycles", 6, "II (TensorE cycles)", "Fixed Stationary Initiation Interval across N (M=128)"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"

df = pd.read_csv(csv)
df = df[(df.kernel == "mm_bf16_fixed") & (df.M == 128)]
ii = df.groupby("N")[COL].median()
ns = list(ii.index)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
x = np.geomspace(ns[0] / 1.3, ns[-1] * 1.3, 200)
ax.plot(x, x + OVERHEAD, color=INK2, lw=1.5, ls=(0, (4, 3)), zorder=2, label=f"Prediction: N + {OVERHEAD}")
ax.plot(ns, ii, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
        label="Measured")
for n, v in ii.items():
    first = n == ns[0]  # label the first point on its right, away from the y axis
    ax.annotate(f"{v:.0f}", (n, v), xytext=(8, 8) if first else (-8, 8), textcoords="offset points",
                ha="left" if first else "right", va="bottom", color=INK2, fontsize=10)

ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(ns))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.3)
top = float(ii.max())
if lat:
    ax.set_ylim(0, 1000 * np.ceil(top * 1.08 / 1000))  # linear from 0 so the ~393-cycle fixed part is visible
else:
    ax.set_yscale("log", base=2)
    yt = [2 ** k for k in range(3, int(np.ceil(np.log2(top * 1.3))) + 1)]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(y) for y in yt])
    ax.set_ylim(yt[0], yt[-1])
ax.set_xlabel("N (moving columns per matmul)", color=INK2)
ax.set_ylabel(YLABEL, color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", frameon=False, labelcolor=INK)
fig.suptitle(TITLE, x=0.03, y=0.97, ha="left", color=INK,
             fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
