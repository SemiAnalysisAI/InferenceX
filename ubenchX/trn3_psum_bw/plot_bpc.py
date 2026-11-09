"""Plot PSUM -> SBUF tensor_copy bytes per cycle vs message size, one line per engine (needs matplotlib).

  python ubenchX/trn3_psum_bw/plot_bpc.py dual.csv dual_bpc.png          # dual_alternate: both engines at once
  python ubenchX/trn3_psum_bw/plot_bpc.py --solo psum.csv solo_bpc.png   # copy_vector / copy_scalar, each alone

Bytes per cycle = PSUM bytes read per ns / engine clock (1.2 GHz for VectorE and ScalarE), median over
traces at pdim=128. In the dual run each engine's rate is its bytes over its own busy window. The
dashed line is one engine's maximum: 128 partitions x 4 bytes (one fp32 per lane per cycle).
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

solo = "--solo" in sys.argv
csv, out = [a for a in sys.argv[1:] if a != "--solo"]
CLOCK_GHZ = 1.2  # VectorE and ScalarE clock (profiler Metadata)
MAX_BPC = 128 * 4

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# ScalarE is drawn first with small filled squares; VectorE on top with larger see-through circles,
# so both stay visible where the lines coincide.
ENGINES = [("Scalar", "ScalarE", "#eb6834", "s", "#eb6834", 7, 2),
           ("Vector", "VectorE", "#2a78d6", "o", "none", 12, 4)]

df = pd.read_csv(csv)
df = df[df.pdim == 128]
if solo:
    g = df.groupby(["engine_seen", "copy_src_bytes"]).GBps.median().unstack(0)
else:
    g = df.groupby("copy_src_bytes")[["Vector_GBps", "Scalar_GBps"]].median()
    g.columns = ["Vector", "Scalar"]
g = g / CLOCK_GHZ
sizes = list(g.index)
labels = [f"{s // 1024} KiB" if s < 2**20 else f"{s // 2**20} MiB" for s in sizes]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(MAX_BPC, color=INK2, lw=1.5, ls=(0, (2, 3)), zorder=1)
ax.text(sizes[0] / 1.35, MAX_BPC * 1.015, f"Max {MAX_BPC} B/cycle per engine", color=INK2, fontsize=10, va="bottom")
for col, name, color, marker, face, ms, z in ENGINES:
    edge = color if face == "none" else SURFACE
    ax.plot(sizes, g[col], color=color, lw=2, marker=marker, ms=ms, markerfacecolor=face,
            markeredgecolor=edge, markeredgewidth=2, zorder=z, label=name)

ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(sizes))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(labels)
ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 1.4)
ax.set_ylim(0, MAX_BPC * 1.12)
ax.set_xlabel("Copy Message Size" if solo else "Copy Message Size per engine", color=INK2)
ax.set_ylabel("PSUM bytes read per cycle", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="lower right", frameon=False, labelcolor=INK)
scope = "each engine alone" if solo else "VectorE and ScalarE concurrently"
fig.suptitle(f"Trainium3 PSUM → SBUF tensor_copy, fp32, {scope}", x=0.03, y=0.97, ha="left", color=INK, fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
