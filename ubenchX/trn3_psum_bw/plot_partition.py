"""Plot time per PSUM -> SBUF tensor_copy against copy message size (pdim=128), per engine
(needs matplotlib).

  python ubenchX/trn3_psum_bw/plot_partition.py psum.csv copy_time_partition.png

Both axes are linear; message size = 128 partitions x bytes per partition, and dotted lines mark
the 2 KiB-per-partition PSUM bank boundaries (every 256 KiB of message).
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# ScalarE first with small filled squares; VectorE on top with see-through circles.
ENGINES = [("copy_scalar", "ScalarE", "#eb6834", "s", "#eb6834", 6, 2),
           ("copy_vector", "VectorE", "#2a78d6", "o", "none", 10, 4)]

df = pd.read_csv(csv)
g = df[df.pdim == 128].groupby(["kernel", "copy_src_bytes"]).ns_per_copy.median().reset_index()
g["kib"] = g.copy_src_bytes / 1024  # message size, KiB

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
for kernel, name, color, marker, face, ms, z in ENGINES:
    r = g[g.kernel == kernel].sort_values("kib")
    edge = color if face == "none" else SURFACE
    kw = dict(color=color, lw=2, marker=marker, ms=ms, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=2,
              zorder=z)
    ax.plot(r.kib, r.ns_per_copy, label=name, **kw)

kmax = g.kib.max()
bank = 2 * 128  # KiB of message per 2 KiB-per-partition PSUM bank
ticks = np.arange(bank, kmax + 1, bank)
for a in (ax,):
    a.set_facecolor(SURFACE)
    for b in ticks[:-1]:
        a.axvline(b, color=AXIS, lw=0.8, ls=(0, (1, 3)), zorder=1)
    a.set_xlim(0, kmax * 1.04)
    a.set_xticks(ticks)
    a.set_xticklabels([f"{t:g} KiB" if t < 1024 else f"{t / 1024:g} MiB" for t in ticks])
    a.set_xlabel("Copy Message Size", color=INK2)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(AXIS)
    a.tick_params(colors=MUTED, labelcolor=INK2)
ax.set_ylim(0, None)
ax.set_ylabel("Time per copy (ns)", color=INK2)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", frameon=False, labelcolor=INK)
fig.suptitle("Trainium3 PSUM → SBUF tensor_copy time per copy, fp32, one NeuronCore", x=0.02, y=0.98, ha="left", color=INK,
             fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
