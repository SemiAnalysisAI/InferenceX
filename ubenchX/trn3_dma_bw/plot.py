"""Plot achieved bandwidth vs message size per DGE mode from a parse.py CSV (needs matplotlib).

  python ubenchX/trn3_dma_bw/plot.py single.csv dge_modes.png            # one NeuronCore
  python ubenchX/trn3_dma_bw/plot.py --chip allcore.csv chip_modes.png   # all cores of one chip

--chip plots the sum over cores of each core's median while all cores run (solo runs are dropped).
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

chip = "--chip" in sys.argv
csv, out = [a for a in sys.argv[1:] if a != "--chip"]
CHIP_PEAK = 4900  # GB/s, Trainium3 chip HBM
PEAK = CHIP_PEAK if chip else CHIP_PEAK / 8  # one NeuronCore's share of chip HBM

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODES = [("static", "#2a78d6", "o"), ("hwdge", "#eb6834", "s"), ("swdge", "#1baf7a", "^")]

df = pd.read_csv(csv)
df = df[df.pdim == 128]
if chip:
    df = df[df["run"] != "solo"]
    g = df.groupby(["kernel", "transfer_bytes", "core"]).GBps.median().groupby(["kernel", "transfer_bytes"]).sum()
    g = g.reset_index()
else:
    g = df.groupby(["kernel", "transfer_bytes"]).GBps.median().reset_index()
g["direction"] = g.kernel.str.split("_").str[0]
g["mode"] = g.kernel.str.split("_").str[1]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, facecolor=SURFACE)
sizes = sorted(g.transfer_bytes.unique())
labels = [f"{s // 1024} KiB" if s < 2**20 else f"{s // 2**20} MiB" for s in sizes]

for ax, (direction, title) in zip(axes, [("load", "Load (HBM → SBUF)"), ("store", "Store (SBUF → HBM)")],
                                  strict=True):
    ax.set_facecolor(SURFACE)
    ax.axhline(PEAK, color=INK2, lw=1.5, ls=(0, (2, 3)), zorder=1)
    for mode, color, marker in MODES:
        r = g[(g.direction == direction) & (g["mode"] == mode)].sort_values("transfer_bytes")
        ax.plot(r.transfer_bytes, r.GBps, color=color, lw=2, marker=marker, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=mode, zorder=3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(labels)
    ax.set_xlim(sizes[0] / 2.1, sizes[-1] * 1.25)
    ax.set_ylim(0, PEAK * 680 / 612.5)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)
    ax.set_xlabel("DMA Copy Message Size", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.text(sizes[0] / 2.0, PEAK * 1.02, f"Peak {PEAK / 1000:g} TB/s" if chip else f"Peak {PEAK:g} GB/s",
            color=INK2, fontsize=10, va="bottom")

    # Direct labels just left of the smallest-size point, where the lines are furthest apart.
    first = g[(g.direction == direction) & (g.transfer_bytes == sizes[0])].set_index("mode").GBps
    offsets = {"static": 0, "hwdge": 7, "swdge": -7}
    for mode, _, _ in MODES:
        ax.annotate(mode, (sizes[0], first[mode]), xytext=(-9, offsets[mode]), textcoords="offset points",
                    color=INK, fontsize=10, va="center", ha="right")

axes[0].set_ylabel("Achieved bandwidth (GB/s)", color=INK2)
axes[1].legend(loc="lower right", frameon=False, labelcolor=INK)
scope = "full chip" if chip else "one NeuronCore"
fig.suptitle(f"Trainium3 DMA bandwidth by descriptor mode, {scope}", x=0.05, y=0.985, ha="left",
             color=INK, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
