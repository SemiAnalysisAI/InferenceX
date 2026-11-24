"""Plot softmax exp (benchmark A) vs fdim, VectorE vs ScalarE (needs matplotlib).

  python ubenchX/trn3_exp_bw/plot.py act.csv exp_cycles.png          # cycles per instruction
  python ubenchX/trn3_exp_bw/plot.py --epc act.csv exp_epc.png       # elements per cycle, whole engine

Cycles per instruction = II, the start-to-start interval between consecutive instructions in cycles at
the 1.2 GHz engine clock, median over traces. Elements per cycle = 128 * fdim input elements / II,
for all 128 lanes of the engine; the dashed lines are upper bounds: each engine's fp32 data-path width
from the Trainium3 architecture guide (VectorE 256, ScalarE 128 elements per cycle) and, for bf16 on
VectorE, 512 elements per cycle, the same 1,024 bytes per cycle as 256 fp32. Every kernel computes
exp(x - row_max) on a [128, fdim] tile with a running row sum: exp_vector_softmax is nisa.exponential on
VectorE and exp_scalar_softmax is nisa.activation(exp) on ScalarE, both fp32 -> bf16;
exp_vector_softmax_bf16 is nisa.exponential bf16 -> bf16. Each legend entry carries its precision, and
kernels missing from the CSV are skipped.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

epc = "--epc" in sys.argv
csv, out = [a for a in sys.argv[1:] if a != "--epc"]
# upper bound in elements per cycle
WIDTH = {"exp_vector_softmax_bf16": ("VectorE bf16", 512), "exp_vector_softmax": ("VectorE fp32", 256),
         "exp_scalar_softmax": ("ScalarE fp32", 128)}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# ScalarE is drawn first with small filled squares; VectorE on top with larger see-through circles,
# and VectorE bf16 last with small filled diamonds, so all stay visible where the lines coincide.
KERNELS = [("exp_scalar_softmax", "ScalarE  nisa.activation(nl.exp)", "#eb6834", "s", "#eb6834", 7, 2),
           ("exp_vector_softmax", "VectorE  nisa.exponential", "#2a78d6", "o", "none", 12, 4),
           ("exp_vector_softmax_bf16", "VectorE  nisa.exponential", "#1baf7a", "D", "#1baf7a", 8, 5)]

df = pd.read_csv(csv)
df = df[df.pdim == 128]
if "src_dtype" not in df:
    df["src_dtype"] = "fp32"
prec = df.groupby("kernel")[["src_dtype", "dst_dtype"]].first()
KERNELS = [k for k in KERNELS if k[0] in prec.index]
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
    for kernel, (name, width) in WIDTH.items():
        if kernel in colors:
            ax.axhline(width, color=colors[kernel], lw=1.5, ls=(0, (2, 3)), zorder=1)
            ax.text(fdims[0] / 1.35, width + 3, f"{name} max {width}/cycle", color=INK2, fontsize=10, va="bottom")
for kernel, name, color, marker, face, ms, z in KERNELS:
    edge = color if face == "none" else SURFACE
    ax.plot(fdims, g[kernel], color=color, lw=2, marker=marker, ms=ms, markerfacecolor=face,
            markeredgecolor=edge, markeredgewidth=2, zorder=z,
            label=f"{name},  {prec.src_dtype[kernel]} → {prec.dst_dtype[kernel]}")

ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(fdims))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(f) for f in fdims])
ax.set_xlim(fdims[0] / 1.4, fdims[-1] * 1.4)
if epc:
    ax.set_ylim(0, max(WIDTH[k[0]][1] for k in KERNELS) * 1.12)
    ax.set_ylabel("Elements per cycle", color=INK2)
    # Above the axes: the bound lines and their labels leave no clear corner inside for three entries.
    legend_at, metric = {"loc": "lower left", "bbox_to_anchor": (0, 1.01), "borderaxespad": 0}, "elements per cycle"
else:
    ax.set_yscale("log")
    yt = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{y:,}" for y in yt])
    ax.set_ylim(18, 6000)
    ax.set_ylabel("Cycles per instruction", color=INK2)
    legend_at, metric = {"loc": "upper left"}, "cycles per instruction"
ax.set_xlabel("fdim", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], frameon=False, labelcolor=INK, **legend_at)
fig.suptitle(f"Trainium3 softmax exp {metric}, one NeuronCore",
             x=0.03, y=0.97, ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
