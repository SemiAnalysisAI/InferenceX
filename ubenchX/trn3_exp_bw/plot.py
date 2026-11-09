"""Plot softmax exp (benchmark A) vs fdim, VectorE vs ScalarE (needs matplotlib).

  python ubenchX/trn3_exp_bw/plot.py act.csv exp_cycles.png          # cycles per instruction
  python ubenchX/trn3_exp_bw/plot.py --epc act.csv exp_epc.png       # elements per cycle, whole engine

Cycles per instruction = II, the start-to-start interval between consecutive instructions in cycles at
the 1.2 GHz engine clock, median over traces. Elements per cycle = 128 * fdim fp32 input elements / II,
for all 128 lanes of the engine; the dashed lines are each engine's fp32 data-path width from the
Trainium3 architecture guide (VectorE 256, ScalarE 128 elements per cycle). Both kernels compute
exp(x - row_max) on fp32 [128, fdim] -> bf16 with a running row sum: exp_vector_softmax is nisa.exponential on VectorE, exp_scalar_softmax is
nisa.activation(exp) on ScalarE.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

epc = "--epc" in sys.argv
csv, out = [a for a in sys.argv[1:] if a != "--epc"]
FP32_WIDTH = {"exp_vector_softmax": ("VectorE", 256), "exp_scalar_softmax": ("ScalarE", 128)}  # elements per cycle

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# ScalarE is drawn first with small filled squares; VectorE on top with larger see-through circles,
# so both stay visible where the lines coincide.
KERNELS = [("exp_scalar_softmax", "ScalarE  nisa.activation(nl.exp)", "#eb6834", "s", "#eb6834", 7, 2),
           ("exp_vector_softmax", "VectorE  nisa.exponential", "#2a78d6", "o", "none", 12, 4)]

df = pd.read_csv(csv)
df = df[df.pdim == 128]
if epc:
    g = (128 * df.groupby(["kernel", "fdim"]).elem_per_cycle.median()).unstack(0)
else:
    g = df.groupby(["kernel", "fdim"]).II_cycles.median().unstack(0)
fdims = list(g.index)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
if epc:
    colors = {k[0]: k[2] for k in KERNELS}
    for kernel, (name, width) in FP32_WIDTH.items():
        ax.axhline(width, color=colors[kernel], lw=1.5, ls=(0, (2, 3)), zorder=1)
        ax.text(fdims[0] / 1.35, width + 3, f"{name} max {width} elements/cycle", color=INK2, fontsize=10,
                va="bottom")
for kernel, name, color, marker, face, ms, z in KERNELS:
    edge = color if face == "none" else SURFACE
    ax.plot(fdims, g[kernel], color=color, lw=2, marker=marker, ms=ms, markerfacecolor=face,
            markeredgecolor=edge, markeredgewidth=2, zorder=z, label=name)

ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(fdims))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(f) for f in fdims])
ax.set_xlim(fdims[0] / 1.4, fdims[-1] * 1.4)
if epc:
    ax.set_ylim(0, 256 * 1.12)
    ax.set_ylabel("Elements per cycle", color=INK2)
    legend_loc, metric = "lower right", "elements per cycle"
else:
    ax.set_yscale("log")
    yt = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{y:,}" for y in yt])
    ax.set_ylim(18, 6000)
    ax.set_ylabel("Cycles per instruction", color=INK2)
    legend_loc, metric = "upper left", "cycles per instruction"
ax.set_xlabel("fdim", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc=legend_loc, frameon=False, labelcolor=INK)
fig.suptitle(f"Trainium3 softmax exp {metric}, fp32 → bf16, one NeuronCore",
             x=0.03, y=0.97, ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
