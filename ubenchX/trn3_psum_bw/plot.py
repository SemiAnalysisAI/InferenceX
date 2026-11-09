"""Plot time per PSUM -> SBUF tensor_copy vs message size, one line per engine (needs matplotlib).

  python ubenchX/trn3_psum_bw/plot.py psum.csv copy_time.png

Uses the pdim=128 points: median ns per copy over all traces. The legend gives each engine's
issue overhead: the intercept of the fit ns per copy = issue overhead + bytes / streaming rate.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

csv, out = sys.argv[1], sys.argv[2]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# ScalarE is drawn first with small filled squares; VectorE on top with larger see-through circles,
# so both stay visible where the lines coincide.
ENGINES = [("copy_scalar", "ScalarE", "#eb6834", "s", "#eb6834", 7, 2),
           ("copy_vector", "VectorE", "#2a78d6", "o", "none", 12, 4)]

df = pd.read_csv(csv)
g = df[df.pdim == 128].groupby(["kernel", "copy_src_bytes"]).ns_per_copy.median().reset_index()
sizes = sorted(g.copy_src_bytes.unique())
labels = [f"{s // 1024} KiB" if s < 2**20 else f"{s // 2**20} MiB" for s in sizes]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for kernel, name, color, marker, face, ms, z in ENGINES:
    r = g[g.kernel == kernel].sort_values("copy_src_bytes")
    slope, fixed = np.polyfit(r.copy_src_bytes.to_numpy(float), r.ns_per_copy.to_numpy(float), 1)
    edge = color if face == "none" else SURFACE
    ax.plot(r.copy_src_bytes, r.ns_per_copy, color=color, lw=2, marker=marker, ms=ms, markerfacecolor=face,
            markeredgecolor=edge, markeredgewidth=2, zorder=z, label=f"{name}: {fixed:.0f} ns issue overhead")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.xaxis.set_major_locator(FixedLocator(sizes))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(labels)
yt = [50, 100, 200, 500, 1000, 2000, 5000]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels([f"{y:,}" for y in yt])
ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.4)
ax.set_ylim(40, 6000)
ax.set_xlabel("tensor_copy Message Size", color=INK2)
ax.set_ylabel("Time per copy (ns)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", frameon=False, labelcolor=INK)
fig.suptitle("Trainium3 PSUM → SBUF tensor_copy time per copy, fp32, one NeuronCore", x=0.03, y=0.97,
             ha="left", color=INK, fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
