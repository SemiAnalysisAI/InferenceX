"""Plot op latency vs bytes per partition at a fixed message size (HBM small-write test; needs matplotlib).

  python ubenchX/trn3_lnc_comm/plot_shape.py shape.csv out.png

The same message (512 B or 2 KiB per rank) is sent as tiles of different shapes ([P, F] bf16, P = 128,
32, 8, 2), so only the bytes each partition holds changes (F x 2 B). Left: shared HBM + core_barrier;
right: the same copies into private HBM (local copy). Op latency in 1.2 GHz NeuronCore cycles, median over
both physical cores and the profiled executions; the point label is the partition count. The shaded band
is where the slowdown ends (between 16 and 64 B per partition).
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

csv, out = sys.argv[1:]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SIZES = {512: ("512 B message", "#2a78d6", "#1f5fae"), 2048: ("2 KiB message", "#eb6834", "#c4501f")}
PANELS = [("shm_barrier", "Shared HBM + core_barrier"), ("local_copy", "Local copy (private HBM)")]

df = pd.read_csv(csv)
df["ppb"] = df.F * 2
g = df.groupby(["kernel", "bytes", "ppb", "P"]).op_cycles.median().reset_index()
xs = sorted(g.ppb.unique())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE, sharey=True)
for ax, (kernel, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.axvspan(16, 64, color=GRID, alpha=0.5, lw=0, zorder=0)
    for b, (name, color, tcolor) in SIZES.items():
        s = g[(g.kernel == kernel) & (g.bytes == b)].sort_values("ppb")
        ax.plot(s.ppb, s.op_cycles, color=color, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3, label=name)
        other = g[(g.kernel == kernel) & (g.bytes != b)].set_index("ppb").op_cycles
        for r in s.itertuples():  # value and partition count; at each x the higher line above, the lower below
            up = r.ppb not in other.index or r.op_cycles >= other[r.ppb]
            ax.annotate(f"{r.op_cycles:,.0f}\n(P={r.P})", (r.ppb, r.op_cycles), xytext=(0, 9 if up else -9),
                        textcoords="offset points", ha="center", va="bottom" if up else "top", color=tcolor,
                        fontsize=8.5, fontweight="bold", linespacing=1.0)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{x} B" for x in xs])
    ax.set_xlim(xs[0] / 1.8, xs[-1] * 1.8)
    ax.set_yscale("log")
    yt = [500, 1000, 2000, 5000, 10000, 20000]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{y:,}" for y in yt])
    ax.set_ylim(600, 25000)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5)
    ax.set_xlabel("Bytes per partition (message / partitions)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
axes[0].set_ylabel("Op latency (cycles)", color=INK2)
axes[1].legend(loc="upper right", frameon=False, labelcolor=INK)
fig.suptitle("Small Writes to HBM: Same Message, Different Tile Shape (LNC=2, bf16)", x=0.02, y=0.98, ha="left",
             color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
