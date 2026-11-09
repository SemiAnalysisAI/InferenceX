"""Plot ScalarE activation elements per cycle vs fdim (benchmark B; needs matplotlib).

  python ubenchX/trn3_exp_bw/plot_act.py act.csv act_epc.png

Elements per cycle = 128 * fdim fp32 input elements / II (start-to-start interval in cycles at the
1.2 GHz engine clock), median over traces, for all 128 lanes of ScalarE. The six act_* kernels
(nisa.activation with silu -> bf16; sigmoid, rsqrt, softplus, sqrt, tanh -> fp32) measure the same to
within 0.002 cycles per instruction, so they are drawn as one line; the script checks that. The dashed
line is ScalarE's fp32 data-path width (128 elements per cycle, Trainium3 architecture guide).
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

csv, out = sys.argv[1:]
SCALAR_FP32_WIDTH = 128  # elements per cycle
FUNCS = ["silu", "sigmoid", "rsqrt", "softplus", "sqrt", "tanh"]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ORANGE = "#eb6834"

df = pd.read_csv(csv)
df = df[(df.pdim == 128) & df.kernel.isin([f"act_{f}" for f in FUNCS])]
g = (128 * df.groupby(["kernel", "fdim"]).elem_per_cycle.median()).unstack(0)
spread = (g.max(axis=1) / g.min(axis=1) - 1).max()
if spread > 1e-3:
    raise SystemExit(f"activations differ by up to {spread:.2%}; plot them as separate lines")
epc = g.mean(axis=1)
fdims = list(epc.index)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(SCALAR_FP32_WIDTH, color=ORANGE, lw=1.5, ls=(0, (2, 3)), zorder=1)
ax.text(fdims[0] / 1.35, SCALAR_FP32_WIDTH + 1.5, f"ScalarE max {SCALAR_FP32_WIDTH} elements/cycle", color=INK2,
        fontsize=10, va="bottom")
ax.plot(fdims, epc, color=ORANGE, lw=2, marker="s", ms=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
        label="ScalarE  nisa.activation: " + ", ".join(FUNCS))

ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(fdims))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(f) for f in fdims])
ax.set_xlim(fdims[0] / 1.4, fdims[-1] * 1.4)
ax.set_ylim(0, SCALAR_FP32_WIDTH * 1.12)
ax.set_xlabel("fdim", color=INK2)
ax.set_ylabel("Elements per cycle", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
ax.legend(loc="lower right", frameon=False, labelcolor=INK)
fig.suptitle("Trainium3 ScalarE activation elements per cycle, fp32 input, one NeuronCore",
             x=0.03, y=0.97, ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
