"""Plot fixed-stationary matmul II or latency vs N at M=128, bf16 vs quad MX (MXFP8) (needs matplotlib).

  python ubenchX/trn3_quad_mx_matmul/plot_fixed.py [--latency] --bf16 a.csv [b.csv ...] \
      --mx c.csv [d.csv ...] out.png

Values are TensorE cycles (2.4 GHz), median over the profiled executions in which TensorE was released
from its startup throttle (parse.py's throttle_release_ns is set), of mm_bf16_fixed (bf16, K=128) and
mm_mx_fixed (MXFP8 e4m3fn_x4 via nc_matmul_mx, K=512), each reusing one stationary. Later CSVs
replace earlier ones per (kernel, N), so a re-run can override sweep points. The dashed line is the bf16
prediction from ../trn3_matmul/plot_fixed.py:
  II       N + 6: N moving columns streamed one per cycle plus a 6-cycle per-instruction overhead.
  latency  N + 393: stream, fill down 128 rows, drain across 128 columns, ~139 cycles fixed overhead.
quad MX contracts four packed values per PE in the same cycle, so the same cycles carry 4x the FLOPs.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

args = sys.argv[1:]
lat = "--latency" in args
args = [a for a in args if a != "--latency"]
out = args.pop()
files, key = {"--bf16": [], "--mx": []}, None
for a in args:
    if a in files:
        key = a
    else:
        files[key].append(a)

if lat:
    COL, OVERHEAD, YLABEL, TITLE = "dur_cycles", 393, "Latency (TensorE cycles)", "Fixed Stationary Latency, bf16 vs MXFP8 (M=128)"
else:
    COL, OVERHEAD, YLABEL, TITLE = "II_cycles", 6, "II (TensorE cycles)", "Fixed Stationary Initiation Interval, bf16 vs MXFP8 (M=128)"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# bf16 first with small filled markers; MX on top with larger see-through circles, so both stay
# visible where the lines coincide (they nearly do up to N=512).
SERIES = [("--bf16", "mm_bf16_fixed", "bf16 (K=128)", "#2a78d6", "o", None, 7, 3),
          ("--mx", "mm_mx_fixed", "MXFP8 quad (K=512)", "#1baf7a", "o", "none", 12, 4)]


def load(paths, kernel):
    # later files win per (kernel, N): lets a re-run replace a sweep's points
    frames = [pd.read_csv(p).assign(_src=i) for i, p in enumerate(paths)]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.kernel == kernel]
    last = df.groupby("N")._src.transform("max")
    df = df[df._src == last]
    df = df[(df.M == 128) & df.throttle_release_ns.notna()]
    return df.groupby("N")[COL].median()


data = {flag: load(files[flag], kernel) for flag, kernel, *_ in SERIES}
ns = sorted(set().union(*(s.index for s in data.values())))
top = max(float(s.max()) for s in data.values())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
x = np.geomspace(ns[0] / 1.3, ns[-1] * 1.3, 200)
ax.plot(x, x + OVERHEAD, color=INK2, lw=1.5, ls=(0, (4, 3)), zorder=2, label=f"Prediction: N + {OVERHEAD}")
for flag, kernel, name, color, marker, face, ms, z in SERIES:
    s = data[flag]
    ax.plot(s.index, s, color=color, lw=2, marker=marker, ms=ms, markerfacecolor=face or color,
            markeredgecolor=color if face == "none" else SURFACE, markeredgewidth=2, zorder=z, label=name)

# labels: 5 points per line, spread over the axis (each line's first N, then 64, 256, 1024, 4096), in the
# line's color (a darker step of it so the text keeps contrast): MX above-left, bf16 below-right
LABEL_NS = [64, 256, 1024, 4096]
TEXT = {"--bf16": "#1f5fae", "--mx": "#0f7a55"}
for flag, off, ha, va in (("--mx", (-8, 8), "right", "bottom"), ("--bf16", (8, -8), "left", "top")):
    s = data[flag]
    for n in [s.index[0]] + [n for n in LABEL_NS if n in s.index]:
        o, h = off, ha
        if n == ns[0] and flag == "--mx":  # keep the first label off the y axis
            o, h = (8, off[1]), "left"
        ax.annotate(f"{s[n]:.0f}", (n, s[n]), xytext=o, textcoords="offset points", ha=h, va=va,
                    color=TEXT[flag], fontsize=10, fontweight="bold")

ax.set_xscale("log", base=2)
xt = [2 ** k for k in range(int(np.log2(ns[0])), int(np.log2(ns[-1])) + 1)]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(n) for n in xt])
ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.3)
if lat:
    ax.set_ylim(0, 1000 * np.ceil(top * 1.08 / 1000))  # linear from 0 so the ~393-cycle fixed part is visible
else:
    ax.set_yscale("log", base=2)
    yt = [2 ** k for k in range(3, int(np.ceil(np.log2(top * 1.3))) + 1)]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(y) for y in yt])
    ax.set_ylim(yt[0], yt[-1])
ax.set_xlabel("N", color=INK2)
ax.set_ylabel(YLABEL, color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", frameon=False, labelcolor=INK)
fig.suptitle(TITLE, x=0.03, y=0.97, ha="left", color=INK, fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
