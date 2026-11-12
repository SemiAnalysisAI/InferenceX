"""Plot local-copy op latency vs bytes per partition at 128 partitions (HBM write granularity; needs matplotlib).

  python ubenchX/trn3_lnc_comm/plot_fine128.py a.csv [b.csv ...] out.png

Local copy (SBUF -> private HBM -> SBUF, no peer) of a [128, F] bf16 tile, F x 2 bytes per partition from 4 to
256 B, combining the fine 128-partition sweep with the earlier 128-partition points (CSVs without a P column
are 128-partition runs). Op latency in 1.2 GHz NeuronCore cycles, median over both physical cores, the profiled
executions and repeated runs. Filled markers: rows that are a multiple of 32 bytes; hollow: other sizes. The
faint vertical lines mark multiples of 32 B; only the aligned points are labeled.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

*csvs, out = sys.argv[1:]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_T = "#2a78d6", "#1f5fae"
ORANGE, ORANGE_T = "#eb6834", "#c4501f"
MAX_PPB = 256

frames = []
for c in csvs:
    f = pd.read_csv(c)
    if "P" not in f:
        f["P"] = 128
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
df = df[(df.kernel == "local_copy") & (df.dtype == "bf16") & (df.P.fillna(128) == 128)]
df["ppb"] = df.bytes // 128
g = df[df.ppb <= MAX_PPB].groupby("ppb").op_cycles.median()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for x in range(32, MAX_PPB + 1, 32):
    ax.axvline(x, color=GRID, lw=1, ls=(0, (2, 3)), zorder=1)
ax.plot(g.index, g, color=MUTED, lw=1.5, zorder=2)
aligned = g[g.index % 32 == 0]
other = g[g.index % 32 != 0]
ax.plot(aligned.index, aligned, ls="none", marker="o", ms=9, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5,
        zorder=3, label="multiple of 32 B")
ax.plot(other.index, other, ls="none", marker="o", ms=9, markerfacecolor=SURFACE, markeredgecolor=ORANGE,
        markeredgewidth=2, zorder=3, label="other sizes")
for n, v in aligned.items():  # label only the aligned (multiple of 32 B) points
    ax.annotate(f"{v:,.0f}", (n, v), xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                color=BLUE_T, fontsize=9, fontweight="bold")

ax.set_yscale("log")
yt = [1000, 2000, 5000, 10000]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels([f"{y:,}" for y in yt])
ax.set_ylim(1000, 14000)
xt = [0] + list(range(32, MAX_PPB + 1, 32))
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlim(-6, MAX_PPB + 8)
ax.set_xlabel("Bytes per partition (128 partitions; message = 128 x this)", color=INK2)
ax.set_ylabel("Op latency (cycles)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
ax.legend(loc="upper right", frameon=False, labelcolor=INK)
fig.suptitle("Local Copy through HBM at 128 Partitions: Fast Only at Multiples of 32 B per Partition (bf16)",
             x=0.02, y=0.97, ha="left", color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
