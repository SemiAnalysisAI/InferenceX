"""Plot alternating-stationary matmul TFLOP/s vs N at M=128, bf16 vs quad MX (MXFP8) (needs matplotlib).

  python ubenchX/trn3_quad_mx_matmul/plot_tflops.py --bf16 a.csv [b.csv ...] --mx c.csv [d.csv ...] out.png

ls_bf16_alt (bf16, K=128) and ls_mx_alt (MXFP8 e4m3fn_x4 via nc_matmul_mx, K=512) load a new stationary for every
matmul. TFLOP/s = 2 K M N / II, median over profiled executions. Later CSVs replace earlier ones per (kernel, N) (a re-run overrides sweep points); then runs with a warm-up
(warmup_mm > 0) replace runs without one; without a warm-up only executions in which TensorE was
released from its startup throttle count (with or without a warm-up). The solid line is the full-rate II (the matmuls between the
throttle release and, if it comes back, the re-throttle); the throttled rate after a re-throttle
(II_rethrottled_cycles) is not drawn. Each line is labeled at 5 points (bf16 N=8, MX N=16, then 64, 256,
1024, 4096) with TFLOP/s
and % of its own peak. Dashed lines
are the peaks: 128 x 128 PEs x 2.4 GHz x 2 FLOP x (1 bf16 | 4 MX) values per PE.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

args = sys.argv[1:]
out = args.pop()
files, key = {"--bf16": [], "--mx": []}, None
for a in args:
    if a in files:
        key = a
    else:
        files[key].append(a)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PEAK1 = 128 * 128 * 2 * 2.4e9 / 1e12  # bf16, one row per PE
# flag, kernel, legend, line color, text color (darker step), peak
SERIES = [("--bf16", "ls_bf16_alt", "bf16 (K=128)", "#2a78d6", "#1f5fae", PEAK1),
          ("--mx", "ls_mx_alt", "MXFP8 quad (K=512)", "#1baf7a", "#0f7a55", 4 * PEAK1)]
LABEL_NS = [64, 256, 1024, 4096]
# small-N labels where the two lines cross: placed by hand
PLACE = {("--bf16", 8): ((8, -4), "left", "top"), ("--bf16", 64): ((8, -10), "left", "top"),
         ("--mx", 16): ((10, -8), "left", "top"), ("--mx", 64): ((12, 0), "left", "center")}
FIRST_LABEL = {"--mx": 16}  # MX N=8 sits on the bf16 N=8 point near the x axis


def load(paths, kernel):
    # later files win per (kernel, N): lets a re-run replace a sweep's points
    frames = [pd.read_csv(p).assign(_src=i) for i, p in enumerate(paths)]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.kernel == kernel]
    last = df.groupby("N")._src.transform("max")
    df = df[df._src == last]
    df = df[df.M == 128]
    if "warmup_mm" not in df:
        df["warmup_mm"] = 0
    df["warmup_mm"] = df.warmup_mm.fillna(0)
    df = df[df.throttle_release_ns.notna()]  # executions in which TensorE was released at all
    warm = df[df.warmup_mm > 0]
    cold = df[(df.warmup_mm == 0) & ~df.N.isin(warm.N)]
    df = pd.concat([warm, cold])
    g = df.groupby("N")
    return g.TFLOPS.median()


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ns = set()
for flag, kernel, name, color, tcolor, peak in SERIES:
    full = load(files[flag], kernel)
    ns |= set(full.index)
    ax.axhline(peak, color=color, lw=1.2, ls=(0, (4, 3)), zorder=1, alpha=0.8)
    ax.plot(full.index, full, color=color, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3, label=name)
    for n in [FIRST_LABEL.get(flag, full.index[0])] + [n for n in LABEL_NS if n in full.index]:
        v = full[n]
        off, ha, va = PLACE.get((flag, n), ((0, -12), "center", "top"))  # default: centered below
        ax.annotate(f"{v:.1f} ({v / peak:.0%})", (n, v), xytext=off, textcoords="offset points", ha=ha, va=va,
                    color=tcolor, fontsize=9.5, fontweight="bold")
ns = sorted(ns)
for flag, kernel, name, color, tcolor, peak in SERIES:
    ax.text(ns[0] / 1.25, peak + 2, f"{name.split(' (')[0]} peak {peak:.1f} TFLOP/s", color=tcolor, fontsize=10,
            va="bottom")

ax.set_xscale("log", base=2)
xt = [2 ** k for k in range(3, 13) if ns[0] <= 2 ** k <= ns[-1]]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([str(n) for n in xt])
ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.3)
ax.set_ylim(0, 4 * PEAK1 * 1.12)
ax.set_xlabel("N", color=INK2)
ax.set_ylabel("TFLOP/s", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2)
ax.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=10)
fig.suptitle("Alternating Stationary Throughput, bf16 vs MXFP8 (M=128)", x=0.03, y=0.97, ha="left", color=INK,
             fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("wrote", out)
