"""Plot intra-LNC communication: op latency (cycles) and bandwidth (GB/s) vs message size (needs matplotlib).

  python ubenchX/trn3_lnc_comm/plot.py lnc.csv [more.csv ...] OUT   # writes OUT_sendrecv.png, OUT_allreduce.png, OUT_shm.png

One figure per comparison, latency (left) and bandwidth (right): sendrecv over the standard DMA vs GpSimd's internal DMA; sendrecv +
VectorE add vs the fused sendrecv_compute (2-rank all-reduce); shared HBM + core_barrier vs the same
copies into private HBM without a peer. Left: op latency = time per operation, including cross-core
synchronization (operations do not overlap), in 1.2 GHz NeuronCore cycles. Right: bytes per rank /
op time, i.e. what each core receives per second (one direction). Medians over both physical cores and
the profiled executions; bf16 (fp32 measures the same per byte for the add pair).
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

*csvs, out = sys.argv[1:]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = ("#2a78d6", "#1f5fae"), ("#eb6834", "#c4501f")
PAIRS = [("sendrecv, DMA vs GpSimd", [("sr_dma", "sendrecv (DMA)", BLUE), ("sr_gpsimd", "sendrecv (gpsimd_dma)", ORANGE)]),
         ("2-Rank All-Reduce", [("sr_add", "sendrecv + tensor_tensor add", BLUE),
                                ("src_add", "sendrecv_compute (fused add)", ORANGE)]),
         ("Shared HBM vs Local Copy", [("shm_barrier", "shared HBM + core_barrier", BLUE),
                                       ("local_copy", "local copy (private HBM)", ORANGE)])]
SLUGS = ["sendrecv", "allreduce", "shm"]
METRICS = [("op_cycles", "Op latency (cycles)"), ("bw_GBps", "Bandwidth per core (GB/s)")]


def fmt_bytes(b):
    return f"{b // 2**20}M" if b >= 2**20 else (f"{b // 2**10}K" if b >= 2**10 else f"{b}")


df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
df = df[df.dtype == "bf16"]
g = df.groupby(["kernel", "bytes"])[["op_cycles", "bw_GBps"]].median()
# ticks: every other common size up to 2M, then the largest (8M/12M sit too close to it to tick)
sizes = [int(b) for b in sorted(df[df.kernel == "sr_dma"].bytes.unique())]
ticks = [b for b in sizes if b <= 2 << 20 and (b.bit_length() % 2 == 0)] + [sizes[-1]]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
# labeled sizes per panel, plus each line's last point (2M and 4M sit too close for both latency labels)
LABEL_SIZES = {"op_cycles": [512, 8192, 131072, 2097152], "bw_GBps": [131072, 2097152]}
for slug, (title, series) in zip(SLUGS, PAIRS):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=SURFACE)
    for ax, (col, ylabel) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for kernel, name, (color, tcolor) in series:
            s = g.loc[kernel][col]
            ax.plot(s.index, s, color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    zorder=3, label=name)
            # labels at LABEL_SIZES and the line's last point (bandwidth only from 20 GB/s, smaller values sit on
            # the x axis); where both lines have a point, the higher is labeled above, the lower below
            other = g.loc[[k for k, _, _ in series if k != kernel][0]][col]
            pts = sorted({n for n in LABEL_SIZES[col] if n in s.index} | {s.index[-1]})
            for n in [n for n in pts if col == "op_cycles" or s[n] >= 20]:
                above = n not in other.index or s[n] >= other[n]
                ax.annotate(f"{s[n]:,.0f}", (n, s[n]), xytext=(0, 8 if above else -9), textcoords="offset points",
                            ha="center", va="bottom" if above else "top", color=tcolor, fontsize=9,
                            fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([fmt_bytes(b) for b in ticks])
        ax.set_xlim(sizes[0] / 1.6, sizes[-1] * 1.6)
        if col == "op_cycles":
            ax.set_yscale("log")
            yt = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
            ax.yaxis.set_major_locator(FixedLocator(yt))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticklabels([f"{y:,}" for y in yt])
            ax.set_ylim(400, 200000)
            ax.legend(loc="upper left", frameon=False, labelcolor=INK, fontsize=9.5)
        else:
            ax.set_ylim(0, 650)
        ax.set_title(ylabel, loc="left", color=INK, fontsize=11.5)
        ax.set_xlabel("Message per rank (bytes)", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK2)
    fig.suptitle(f"{title}: the Two Cores of an LNC=2 Logical Core (bf16)", x=0.02, y=0.98, ha="left", color=INK,
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = f"{out}_{slug}.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)
