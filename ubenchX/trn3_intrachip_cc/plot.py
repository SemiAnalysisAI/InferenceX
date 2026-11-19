"""Plot intra-chip collectives at LNC=2: bus bandwidth vs size, HBM vs SBUF (needs matplotlib).

  python ubenchX/trn3_intrachip_cc/plot.py cc.csv [more.csv ...] out.png [--label "run name"]

nccom-test on one Trainium3 chip at LNC=2, 4 ranks = all 4 logical cores. One panel per op, busbw (GB/s) vs size.
Color: HBM buffers vs SBUF buffers (--sb2sb, up to 8 MiB, above which SBUF collectives do not run; all_to_all is
HBM-only). Labels: each line's peak and the first point after the drop. Marker: the algorithm the library picked (nccom-test --show-algorithm).
HBM runs to 4 GiB; where the library switches to its Ring algorithm (512 MiB to 2 GiB) busbw drops ~3x, drawn as a dashed
segment into down-triangles. Later CSVs replace earlier ones per (placement, op, buffer, size). Size convention
(nccom-test): all_gather = output bytes, reduce_scatter = input bytes, all_reduce / all_to_all = buffer bytes per rank.
"""
import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("files", nargs="+", help="CSVs, then the output PNG")
ap.add_argument("--label", help="appended to the title, to tell runs apart")
a = ap.parse_args()
*csvs, out = a.files
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"HBM": ("#2a78d6", "#1f5fae"), "SBUF": ("#eb6834", "#c4501f")}
MARKERS = {"Mesh": "o", "RDH": "s", "Ring": "v"}  # anything else (a future algorithm name) falls back to a diamond
PL = "lnc2_r4"  # 4 ranks: all 4 logical cores of the chip
OPS = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
TICKS = [8, 1 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20, 4 << 30]


def fmt_bytes(b):
    for unit, sh in (("G", 30), ("M", 20), ("K", 10)):
        if b >= 1 << sh:
            return f"{b / (1 << sh):g}{unit}"
    return f"{b}"


df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
df = df[df.placement == PL].drop_duplicates(["placement", "op", "buffer", "bytes"], keep="last")
shown = df

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 4, figsize=(17, 5.0), facecolor=SURFACE)
for ax, op in zip(axes, OPS):
    ax.set_facecolor(SURFACE)
    for buf, (color, tcolor) in COLORS.items():
        d = shown[(shown.op == op) & (shown.buffer == buf)].sort_values("bytes")
        if d.empty:
            continue
        s = d.set_index("bytes").busbw_GBps
        ring = d.algorithm.eq("Ring").to_numpy()
        pre = s[~ring]  # Mesh / RDH: the plateau
        ax.plot(pre.index, pre, color=color, lw=2, zorder=3)
        if ring.any():  # the drop: dashed from the plateau end into the Ring points
            drop = s[s.index >= pre.index[-1]] if len(pre) else s
            ax.plot(drop.index, drop, color=color, lw=1.6, ls=(0, (4, 2.5)), zorder=3)
        for alg, g in d.groupby("algorithm"):
            ax.plot(g.bytes, g.busbw_GBps, ls="none", marker=MARKERS.get(alg, "D"), ms=7 if alg == "Ring" else 5.5,
                    color=color, markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
        # label the peak, and the first Ring point (the drop) below its marker
        top = s.idxmax()
        ax.annotate(f"{s[top]:,.0f}", (top, s[top]), xytext=(0, 7), textcoords="offset points",
                    ha="right" if top >= 256 << 20 else "center", va="bottom", color=tcolor, fontsize=8.5,
                    fontweight="bold")
        if ring.any():
            n = s[ring].index[0]
            ax.annotate(f"{s[n]:,.0f}", (n, s[n]), xytext=(0, -9), textcoords="offset points", ha="center",
                        va="top", color=tcolor, fontsize=8.5, fontweight="bold")
    ax.set_title(op, loc="left", color=INK, fontsize=12)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([fmt_bytes(b) for b in TICKS])
    ax.set_xlim(shown.bytes.min() / 1.6, shown.bytes.max() * 1.6)
    ax.set_ylim(0, 600)
    ax.set_xlabel("Size (bytes)", color=INK2)
    if op == OPS[0]:
        ax.set_ylabel("Bus bandwidth (GB/s)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
handles = [Line2D([], [], color=c, lw=2, label=b) for b, (c, _) in COLORS.items()]
handles += [Line2D([], [], ls="none", marker=m, ms=6, color=INK2, label=f"{a} algorithm")
            for a, m in MARKERS.items() if (shown.algorithm == a).any()]
fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, labelcolor=INK, fontsize=10,
           bbox_to_anchor=(0.012, 0.93), handlelength=2.4)
fig.suptitle("Collective Bus Bandwidth across the 4 Logical Cores of One Chip, LNC=2 (nccom-test, bf16)"
             + (f": {a.label}" if a.label else ""), x=0.02,
             y=0.985, ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
