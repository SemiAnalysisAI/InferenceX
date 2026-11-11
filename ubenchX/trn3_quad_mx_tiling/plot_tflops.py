"""Plot alternating-stationary quad MX (MXFP8) TFLOP/s vs N for K = 128, 256, 512, untiled vs row-tiled.

  python ubenchX/trn3_quad_mx_tiling/plot_tflops.py --k512 a.csv [b.csv ...] --small c.csv [d.csv ...] out.png

--k512: ../trn3_quad_mx_matmul parse.py CSVs (ls_mx_alt, kp = 128 partitions, the whole PE array).
--small: this directory's CSVs (ls_mx_alt_untiled / ls_mx_alt_tiled, kp = 32 or 64, K = 4 kp).
Every matmul loads a new stationary. TFLOP/s = 2 K M N / II, median over profiled executions in which
TensorE's throttle was released; II is the full-rate window (until a re-throttle, if any; points
whose full-rate window is under 4 matmuls have no value). PSUM-capacity-limited tiled points (EXCLUDE)
are left out. Later CSVs
in a list replace earlier ones per (kernel, K, N). Color = K; solid = row-tiled (128 / kp tiles running
side by side), dashed = untiled (one kp-partition matmul at a time). Dashed grey: MX peak, 128 x 128 PEs
x 4 MACs x 2 FLOP x 2.4 GHz = 314.6 TFLOP/s.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

args = sys.argv[1:]
out = args.pop()
files, key = {"--k512": [], "--small": []}, None
for a in args:
    if a in files:
        key = a
    else:
        files[key].append(a)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PEAK = 4 * 128 * 128 * 2 * 2.4e9 / 1e12
# (source, kernel, K, legend, line color, text color, linestyle)
SERIES = [("--k512", "ls_mx_alt", 512, "K=512 (full array)", "#1baf7a", "#0f7a55", "-"),
          ("--small", "ls_mx_alt_tiled", 256, "K=256, 2 row tiles", "#2a78d6", "#1f5fae", "-"),
          ("--small", "ls_mx_alt_untiled", 256, "K=256, untiled", "#2a78d6", "#1f5fae", (0, (4, 2))),
          ("--small", "ls_mx_alt_tiled", 128, "K=128, 4 row tiles", "#eb6834", "#c4501f", "-"),
          ("--small", "ls_mx_alt_untiled", 128, "K=128, untiled", "#eb6834", "#c4501f", (0, (4, 2)))]
LABEL_NS = [64]  # direct labels per line, kept few (5 lines share the axis): N=64 and each line's last N
# PSUM-capacity-limited points left out: T row tiles in flight need T PSUM destinations, but one [128, N]
# fp32 result takes N/512 of the 8 banks, so at N = 2048 (K=128, 4 tiles) and N = 4096 (both tiled K)
# the tiles share destinations and serialize; that measures PSUM capacity, not tiling.
EXCLUDE = {("ls_mx_alt_tiled", 128): [2048, 4096], ("ls_mx_alt_tiled", 256): [4096]}
PLACE = {("K=256, 2 row tiles", 2048): ((0, -12), "center", "top"),  # default (right of the point) collides
         ("K=128, 4 row tiles", 1024): ((0, -12), "center", "top")}


def load(paths, kernel, K):
    frames = [pd.read_csv(p).assign(_src=i) for i, p in enumerate(paths)]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.kernel == kernel) & (df.K == K) & (df.M == 128)]
    df = df[df._src == df.groupby("N")._src.transform("max")]
    df = df[df.throttle_release_ns.notna() & ~df.N.isin(EXCLUDE.get((kernel, K), []))]
    return df.groupby("N").TFLOPS.median().dropna()


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(PEAK, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
data = {}
for src, kernel, K, name, color, tcolor, ls in SERIES:
    s = load(files[src], kernel, K)
    data[name] = s
    if K == 512:  # large see-through rings under the others: it coincides with K=256 tiled
        ax.plot(s.index, s, color=color, lw=2, ls=ls, marker="o", ms=12, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=2, zorder=2, label=name)
    else:
        ax.plot(s.index, s, color=color, lw=2, ls=ls, marker="o", ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3, label=name)
ns = sorted(set().union(*(s.index for s in data.values())))
ax.text(ns[0] / 1.25, PEAK + 4, f"MX peak {PEAK:.1f} TFLOP/s", color=INK2, fontsize=10, va="bottom")
for src, kernel, K, name, color, tcolor, ls in SERIES:
    s = data[name]
    for n in [n for n in LABEL_NS if n in s.index] + [s.index[-1]]:
        off, ha, va = PLACE.get((name, n), ((10, 0), "left", "center"))
        ax.annotate(f"{s[n]:.0f}", (n, s[n]), xytext=off, textcoords="offset points", ha=ha, va=va, color=tcolor,
                    fontsize=9.5, fontweight="bold")

ax.set_xscale("log", base=2)
xt = [2 ** k for k in range(3, 13) if ns[0] <= 2 ** k <= ns[-1]]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(n) for n in xt])
ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.5)
ax.set_ylim(0, PEAK * 1.12)
ax.set_xlabel("N", color=INK2)
ax.set_ylabel("TFLOP/s", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, labelcolor=INK, fontsize=10,
          handlelength=3.6)
fig.suptitle("Alternating Stationary MXFP8 Throughput across K (M=128)", x=0.03, y=0.97, ha="left", color=INK,
             fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
