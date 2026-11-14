"""Plot collective bus bandwidth vs size (needs matplotlib).

  python ubenchX/trn3_internode_cc/plot.py internode.csv out_dir --intrachip intrachip.csv

Writes two images with one panel per op, all at LNC=2, bf16, nccl-tests sizes:
  rank_scaling.png  1 chip (trn3_intrachip_cc lnc2_r4, HBM), 1 node, 1 rack (2n_rack), 1 UltraServer (4n)
  two_node.png      2 nodes over intra-rack NeuronLink, cross-rack NeuronLink and EFA
Failed sizes are left out; a config with no passing size is listed in the panel.
"""
import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"chip": ("#2a78d6", "o"), "1n": ("#eb6834", "s"), "2n_rack": ("#1baf7a", "^"),
         "2n_cable": ("#eda100", "v"), "2n_efa": ("#e87ba4", "D"), "4n": ("#008300", "P")}
FIGURES = [  # file, title, [(key, label)]
    ("rank_scaling", "Rank Scaling from One Chip to One UltraServer", [
        ("chip", "1 chip (4 ranks)"), ("1n", "1 node (64 ranks)"), ("2n_rack", "1 rack, 2 nodes (128 ranks)"),
        ("4n", "1 UltraServer, 4 nodes (256 ranks)")]),
    ("two_node", "Two Nodes over Each Link Type (128 ranks)", [
        ("2n_rack", "Intra-rack NeuronLink"), ("2n_cable", "Cross-rack NeuronLink (same slot)"),
        ("2n_efa", "EFA (cross-rack, other slot)")]),
]
OPS = ["all_reduce", "all_gather", "reduce_scatter", "alltoall"]
TICKS = [8, 1 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20, 4 << 30]


def fmt_bytes(b: int) -> str:
    for unit, sh in (("G", 30), ("M", 20), ("K", 10)):
        if b >= 1 << sh:
            return f"{b / (1 << sh):g}{unit}"
    return f"{b}"


def style(ax, ymax: float) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([fmt_bytes(b) for b in TICKS])
    ax.set_xlim(6, 6 << 30)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Size (bytes)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--intrachip", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    df = df[(df.status == "ok") & (df.buffer == "HBM")][["topology", "op", "bytes", "busbw_GBps"]]
    c = pd.read_csv(a.intrachip)
    c = c[(c.placement == "lnc2_r4") & (c.buffer == "HBM") & c.busbw_GBps.notna()].assign(topology="chip")
    c["op"] = c.op.replace({"all_to_all": "alltoall"})
    df = pd.concat([c[df.columns], df], ignore_index=True).drop_duplicates(["topology", "op", "bytes"], keep="last")
    os.makedirs(a.out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    ymax = df.busbw_GBps.max() * 1.08

    for fname, title, series in FIGURES:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), facecolor=SURFACE, sharey=True)
        for ax, op in zip(axes, OPS):
            style(ax, ymax)
            peaks = []
            for key, label in series:
                color, marker = COLOR[key]
                d = df[(df.topology == key) & (df.op == op)].sort_values("bytes")
                h = Line2D([], [], color=color, lw=2, marker=marker, ms=6, markeredgecolor=SURFACE)
                if d.empty:
                    peaks.append((h, "all sizes failed"))
                    continue
                ax.plot(d.bytes, d.busbw_GBps, color=color, lw=2, marker=marker, ms=5, markevery=2,
                        markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
                top = d.loc[d.busbw_GBps.idxmax()]
                peaks.append((h, f"{top.busbw_GBps:,.0f} GB/s at {fmt_bytes(int(top.bytes))}B"))
            leg = ax.legend([h for h, _ in peaks], [t for _, t in peaks], title="Peak busbw", loc="upper left",
                            frameon=False, fontsize=8.5, title_fontsize=8.5, labelcolor=INK2,
                            handlelength=1.6, borderaxespad=0.2)
            leg.get_title().set_color(MUTED)
            leg._legend_box.align = "left"
            ax.set_title(op, loc="left", color=INK, fontsize=12)
        axes[0].set_ylabel("Bus bandwidth (GB/s)", color=INK2)
        handles = [Line2D([], [], color=COLOR[k][0], lw=2, marker=COLOR[k][1], ms=6, markeredgecolor=SURFACE,
                          label=lab) for k, lab in series]
        fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, labelcolor=INK,
                   fontsize=10, bbox_to_anchor=(0.008, 0.9))
        fig.suptitle(f"Trainium3 {title}", x=0.012, y=0.99, ha="left", color=INK, fontsize=14)
        fig.text(0.012, 0.905, "nccom-test, bf16, LNC=2, HBM buffers, nccl-tests sizes 8 B to 4 GiB. busbw uses the "
                 "nccl-tests factors.", color=MUTED, fontsize=9, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0, 1, 0.87))
        out = os.path.join(a.out_dir, f"{fname}.png")
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
