"""Plot intra-chip collectives at LNC=2: bus bandwidth vs size, HBM vs SBUF (needs matplotlib).

  python ubenchX/trn3_intrachip_cc/plot.py cc.csv [more.csv ...] out.png

nccom-test on one Trainium3 chip at LNC=2, 4 ranks = all 4 logical cores. One panel per op, busbw (GB/s) vs size.
Color: HBM buffers vs SBUF buffers (--sb2sb; up to the mesh-algorithm limit, 4 MiB all_gather, 8 MiB all_reduce /
reduce_scatter; all_to_all is HBM-only). Marker: the algorithm the library picked (nccom-test --show-algorithm).
HBM is shown up to 1 GiB and only while the library stays on Mesh / RDH: the bandwidth plateau. Sizes where it
switched to Ring (busbw drops ~3x) are left out and summarized in a note in the panel. Later CSVs replace earlier
ones per (placement, op, buffer, size). Size convention (nccom-test): all_gather = output bytes, reduce_scatter =
input bytes, all_reduce / all_to_all = buffer bytes per rank.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

*csvs, out = sys.argv[1:]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"HBM": ("#2a78d6", "#1f5fae"), "SBUF": ("#eb6834", "#c4501f")}
MARKERS = {"Mesh": "o", "RDH": "s"}  # anything else (a future algorithm name) falls back to a diamond
PL = "lnc2_r4"  # 4 ranks: all 4 logical cores of the chip
OPS = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
MAX_BYTES = 1 << 30  # plateau view: HBM to 1 GiB
LABEL_SIZES = [1 << 20, 4 << 20, 16 << 20]  # plus the last point of each line (the plateau)
TICKS = [4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20, 1 << 30]


def fmt_bytes(b):
    for unit, sh in (("G", 30), ("M", 20), ("K", 10)):
        if b >= 1 << sh:
            return f"{b / (1 << sh):g}{unit}"
    return f"{b}"


df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
df = df[df.placement == PL].drop_duplicates(["placement", "op", "buffer", "bytes"], keep="last")
ring_all = df[df.algorithm == "Ring"]
shown = df[(df.algorithm != "Ring") & (df.bytes <= MAX_BYTES)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 4, figsize=(17, 5.0), facecolor=SURFACE)
for ax, op in zip(axes, OPS):
    ax.set_facecolor(SURFACE)
    for buf, (color, tcolor) in COLORS.items():
        d = shown[(shown.op == op) & (shown.buffer == buf)].sort_values("bytes")
        if d.empty:
            continue
        s = d.set_index("bytes").busbw_GBps
        ax.plot(s.index, s, color=color, lw=2, zorder=3)
        for alg, g in d.groupby("algorithm"):
            ax.plot(g.bytes, g.busbw_GBps, ls="none", marker=MARKERS.get(alg, "D"), ms=5.5, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
        # labels from 20 GB/s (smaller values sit on the x axis); where both buffers have a point, the higher is
        # labeled above and the lower below-right
        other = shown[(shown.op == op) & (shown.buffer != buf)].set_index("bytes").busbw_GBps
        for n in [n for n in s.index if (n in LABEL_SIZES or n == s.index[-1]) and s[n] >= 20]:
            up = n not in other.index or s[n] >= other[n]
            if up and n == s.index[-1] and buf == "HBM":
                off, ha, va = (-2, 7), "right", "bottom"  # last point: keep inside the axes
            elif up:
                off, ha, va = (0, 7), "center", "bottom"
            else:
                off, ha, va = (6, -6), "left", "top"
            ax.annotate(f"{s[n]:,.0f}", (n, s[n]), xytext=off, textcoords="offset points", ha=ha, va=va,
                        color=tcolor, fontsize=8, fontweight="bold")
    # what happens past the plateau (not drawn): the Ring switch, or no switch through the largest size run
    r = ring_all[(ring_all.op == op) & (ring_all.buffer == "HBM")].sort_values("bytes")
    hbm = df[(df.op == op) & (df.buffer == "HBM")].sort_values("bytes")
    if not r.empty:
        last = hbm[hbm.bytes < r.bytes.iloc[0]].iloc[-1]
        note = (f"HBM on Ring from {fmt_bytes(r.bytes.iloc[0])} (not shown):\n"
                f"{last.busbw_GBps:,.0f} → {r.busbw_GBps.iloc[0]:,.0f} GB/s")
    elif not hbm.empty and hbm.bytes.max() > MAX_BYTES:
        big = hbm[hbm.bytes > MAX_BYTES]
        note = f"HBM stays on {'/'.join(big.algorithm.unique())} through {fmt_bytes(big.bytes.max())}:\n" \
               f"{big.busbw_GBps.min():,.0f}-{big.busbw_GBps.max():,.0f} GB/s"
    else:
        note = ""
    ax.text(0.02, 0.98, note, transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=8.5)
    ax.set_title(op, loc="left", color=INK, fontsize=12)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([fmt_bytes(b) for b in TICKS])
    ax.set_xlim(shown.bytes.min() / 1.6, MAX_BYTES * 1.6)
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
fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, labelcolor=INK, fontsize=10,
           bbox_to_anchor=(0.99, 0.975), handlelength=2.4)
fig.suptitle("Collective Bus Bandwidth across the 4 Logical Cores of One Chip, LNC=2 (nccom-test, bf16)", x=0.02,
             y=0.965, ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
