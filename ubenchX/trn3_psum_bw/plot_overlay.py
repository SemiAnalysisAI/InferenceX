"""Overlay PSUM -> SBUF bytes per cycle per engine: each engine alone vs both concurrently (needs matplotlib).

  python ubenchX/trn3_psum_bw/plot_overlay.py psum.csv dual.csv overlay_bpc.png

Lines are the single-engine runs (copy_vector, copy_scalar); hollow rings are the same engine's rate
while the other engine copies at the same time (dual_alternate). Bytes per cycle = PSUM bytes read
per ns / 1.2 GHz, median over traces at pdim=128; the x axis is the size of one copy on one engine.
The dashed line is one engine's maximum: 128 partitions x 4 bytes (one fp32 per lane per cycle).
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

solo_csv, dual_csv, out = sys.argv[1:]
CLOCK_GHZ = 1.2  # VectorE and ScalarE clock (profiler Metadata)
MAX_BPC = 128 * 4

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ENGINES = [("Scalar", "ScalarE", "#eb6834", "s"), ("Vector", "VectorE", "#2a78d6", "o")]

s = pd.read_csv(solo_csv)
s = s[s.pdim == 128].groupby(["engine_seen", "copy_src_bytes"]).GBps.median().unstack(0) / CLOCK_GHZ
d = pd.read_csv(dual_csv)
d = d[d.pdim == 128].groupby("copy_src_bytes")[["Vector_GBps", "Scalar_GBps"]].median() / CLOCK_GHZ
d.columns = ["Vector", "Scalar"]
sizes = sorted(set(s.index) | set(d.index))
labels = [f"{x // 1024} KiB" if x < 2**20 else f"{x // 2**20} MiB" for x in sizes]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(MAX_BPC, color=INK2, lw=1.5, ls=(0, (2, 3)), zorder=1)
ax.text(sizes[0] / 1.35, MAX_BPC * 1.015, f"Max {MAX_BPC} B/cycle per engine", color=INK2, fontsize=10, va="bottom")
for col, name, color, marker in ENGINES:
    ax.plot(s.index, s[col], color=color, lw=2, marker=marker, ms=5, markeredgewidth=0, zorder=3,
            label=f"{name} alone")
for col, name, color, marker in ENGINES:
    ax.plot(d.index, d[col], ls="none", marker=marker, ms=14, markerfacecolor="none", markeredgecolor=color,
            markeredgewidth=2, zorder=4, label=f"{name}, both engines running")

ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(sizes))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(labels)
ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.4)
ax.set_ylim(0, MAX_BPC * 1.12)
ax.set_xlabel("Copy Message Size per engine", color=INK2)
ax.set_ylabel("PSUM bytes read per cycle", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = ax.get_legend_handles_labels()
order = [1, 3, 0, 2]  # VectorE pair, then ScalarE pair
ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", frameon=False, labelcolor=INK)
fig.suptitle("Trainium3 PSUM → SBUF tensor_copy, fp32: each engine alone vs both concurrently", x=0.03, y=0.97,
             ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
