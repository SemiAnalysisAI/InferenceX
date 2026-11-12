"""Plot intra-chip collectives at LNC=2: time (cycles) and bus bandwidth vs size, HBM vs SBUF (needs matplotlib).

  python ubenchX/trn3_intrachip_cc/plot.py cc.csv out.png

nccom-test on one Trainium3 chip at LNC=2, 4 ranks = all 4 logical cores. Columns are ops; top row is the average collective time in µs (nccom-test's time:avg), bottom
row busbw (GB/s). Color: HBM buffers vs SBUF buffers (--sb2sb, up to 4 MiB; all_to_all is HBM-only). Size convention (nccom-test): all_gather = output bytes, reduce_scatter = input bytes,
all_reduce / all_to_all = buffer bytes per rank.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

csv, out = sys.argv[1:]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"HBM": ("#2a78d6", "#1f5fae"), "SBUF": ("#eb6834", "#c4501f")}
STYLES = {"lnc2_r4": ("-", "")}  # 4 ranks: all 4 logical cores of the chip
OPS = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
LABEL_SIZES = [4 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20, 256 << 20]


def fmt_bytes(b):
    return f"{b // 2**20}M" if b >= 2**20 else (f"{b // 2**10}K" if b >= 2**10 else f"{b}")


df = pd.read_csv(csv)
df = df[df.placement.isin(STYLES)]
sizes = sorted(df.bytes.unique())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(2, 4, figsize=(17, 8), facecolor=SURFACE, sharex=True)
for c, op in enumerate(OPS):
    for r, (col, ylabel) in enumerate([("time_us", "Time per collective (µs)"), ("busbw_GBps", "Bus bandwidth (GB/s)")]):
        ax = axes[r, c]
        ax.set_facecolor(SURFACE)
        for pl, (ls, plname) in STYLES.items():
            for buf, (color, tcolor) in COLORS.items():
                s = df[(df.placement == pl) & (df.op == op) & (df.buffer == buf)].set_index("bytes")[col].sort_index()
                if s.empty:
                    continue
                ax.plot(s.index, s, color=color, lw=2, ls=ls, marker="o", ms=5, markeredgecolor=SURFACE,
                        markeredgewidth=1, zorder=3, label=buf)
                if pl == "lnc2_r4":
                    # label 6 sizes per line (4K .. 256M, every other tick); bandwidth only from 20 GB/s (smaller values sit
                    # on the x axis; the time row labels those sizes);
                    # where both buffers have a point, the higher is labeled above and the lower below-right
                    other = df[(df.placement == pl) & (df.op == op) & (df.buffer != buf)].set_index("bytes")[col]
                    for n in [n for n in LABEL_SIZES if n in s.index and (col == "time_us" or s[n] >= 20)]:
                        up = n not in other.index or s[n] >= other[n]
                        if up and n == s.index[-1]:
                            off, ha, va = (-4, 7), "right", "bottom"  # last point: keep inside the axes
                        elif up:
                            off, ha, va = (0, 7), "center", "bottom"
                        else:
                            off, ha, va = (6, -6), "left", "top"
                        txt = f"{s[n]:,.0f}" if s[n] >= 100 or (col == "busbw_GBps" and s[n] >= 10) else f"{s[n]:.1f}"
                        ax.annotate(txt, (n, s[n]), xytext=off, textcoords="offset points", ha=ha, va=va,
                                    color=tcolor, fontsize=8, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(sizes[::2]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([fmt_bytes(b) for b in sizes[::2]])
        ax.set_xlim(sizes[0] / 1.6, sizes[-1] * 1.6)
        if col == "time_us":
            ax.set_yscale("log")
            yt = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
            ax.yaxis.set_major_locator(FixedLocator(yt))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticklabels([f"{y:,}" for y in yt])
            ax.set_ylim(2, 3000)
            ax.set_title(op, loc="left", color=INK, fontsize=12)
        else:
            ax.set_ylim(0, 600)
            ax.set_xlabel("Size (bytes)", color=INK2)
        if c == 0:
            ax.set_ylabel(ylabel, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, frameon=False, labelcolor=INK, fontsize=10, bbox_to_anchor=(0.99, 0.985),
           handlelength=3.2)
fig.suptitle("Collectives across the 4 Logical Cores of One Chip, LNC=2 (nccom-test, bf16)", x=0.02, y=0.98, ha="left", color=INK,
             fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
