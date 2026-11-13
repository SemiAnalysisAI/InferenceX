"""nisa.sendrecv on every core pair of a Trainium3 chip, reached with the core-map shim (sendrecv/XSENG.md).

  python3 c2c/figure_xseng.py            # writes figures/core-sendrecv-xseng.png

Same 8 x 8 layout as figures/c2c-stock/core-sendrecv.png, which shows the stock reach (the 4 LNC pairs only). Here
sendrecv/ncmap_shim.c relabels which two physical cores form the logical core, so the unmodified compiled
kernel exchanges SBUF tiles between any two cores. Above the diagonal: 512 B per exchange. Below: 2 MiB, with
the bandwidth per direction. Cycles at 1.2 GHz, default DMA engine; the 512 B same-SEngine cells show the stock
GPSIMD engine instead (it does not cross an SEngine and stops at 128 KiB tiles, so it has no 2 MiB number).
One measurement per route class: logical NC0 with NCb (results/core-sendrecv-xseng.json), applied to every
pair with that XOR mask. Red frame: the pairs every collective tested refuses stock (no_hier no_mesh).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
from figure_c2c import CLK_GHZ, ci  # noqa: E402
from figure_split import matrix  # noqa: E402

HOPS = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2}   # SEngine grid hops for XOR mask b
SMALL, LARGE = 512, 2097152


def topology(ax, cyc, gps, INK, MUTED):
    """The four SEngines as a 2 x 2 grid with the measured 512 B cost on each route."""
    from matplotlib.patches import FancyBboxPatch
    RED, LINK = "#C0392B", "#2F7F8F"
    W, H = 3.7, 2.9
    pos = {0: (.5, 6.3), 2: (5.8, 6.3), 4: (.5, 1.2), 6: (5.8, 1.2)}     # first core of the SEngine -> lower-left corner
    ctr = {k: (x + W / 2, y + H / 2) for k, (x, y) in pos.items()}
    r = lambda *bs: "-".join(dict.fromkeys(ci(v) for v in sorted(cyc[(b, SMALL)] for b in bs)))  # noqa: E731

    def link(a, b, label, vertical):
        (xa, ya), (xb, yb) = ctr[a], ctr[b]
        if vertical:
            ax.plot([xa, xa], [pos[a][1], pos[b][1] + H], color=LINK, lw=7, solid_capstyle="butt", zorder=1)
            left = xa < 5   # keep the label on the outer side, clear of the diagonal note
            ax.text(xa + (-.25 if left else .25), (pos[a][1] + pos[b][1] + H) / 2, label, ha="right" if left else "left",
                    va="center", fontsize=11.5, color=INK, fontweight="bold")
        else:
            ax.plot([pos[a][0] + W, pos[b][0]], [ya, ya], color=LINK, lw=7, solid_capstyle="butt", zorder=1)
            ax.text((pos[a][0] + W + pos[b][0]) / 2, ya + .3, label, ha="center", va="bottom", fontsize=11.5, color=INK,
                    fontweight="bold")

    link(0, 2, r(2, 3), False)
    link(4, 6, r(2, 3), False)
    link(0, 4, r(4, 5), True)
    link(2, 6, r(4, 5), True)
    for a, b in ((0, 6), (2, 4)):   # no diagonal link: XY routing takes two hops
        (xa, ya), (xb, yb) = ctr[a], ctr[b]
        sx = 1 if xb > xa else -1
        ax.plot([xa + sx * W / 2, xb - sx * W / 2], [pos[a][1], pos[b][1] + H], color=RED, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(5, 5.25, f"no diagonal link\n2 hops: {r(6, 7)}", ha="center", va="center", fontsize=11.5, color=RED,
            fontweight="bold", bbox=dict(fc="white", ec="none", pad=3), zorder=4)
    for k, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=.15", fc="#F4F6F8", ec=INK, lw=1.5,
                                    zorder=2))
        ax.text(x + .15, y + H - .12, "SEngine", ha="left", va="top", fontsize=10, color=MUTED, zorder=3)
        for i in (0, 1):
            cx = x + .25 + i * 2.2
            ax.add_patch(FancyBboxPatch((cx, y + 1.05), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=.08", fc="white",
                                        ec=INK, lw=1.4, zorder=3))
            ax.text(cx + .5, y + 1.55, f"NC{k + i}", ha="center", va="center", fontsize=12, color=INK, fontweight="bold",
                    zorder=4)
        ax.annotate("", (x + 2.45, y + 1.55), (x + 1.25, y + 1.55), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3),
                    zorder=4)
        ax.text(x + 1.85, y + 1.68, ci(cyc[(1, SMALL)]), ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="bold",
                zorder=4)   # default engine, like the links; GPSIMD below
        ax.text(x + 1.85, y + 1.42, f"GPSIMD {ci(gps)}", ha="center", va="top", fontsize=7.5, color=MUTED, zorder=4)
        ax.add_patch(FancyBboxPatch((x + .25, y + .22), W - .5, .55, boxstyle="round,pad=0,rounding_size=.06", fc="#DCE6EA",
                                    ec="none", zorder=3))
        ax.text(x + W / 2, y + .5, "shared HBM", ha="center", va="center", fontsize=10, color=INK, zorder=4)
    for y, name in ((6.3 + H / 2, "die 1"), (1.2 + H / 2, "die 0")):
        ax.text(10.15, y, name, ha="center", va="center", rotation=270, fontsize=12.5, color=INK, fontweight="bold")
    ax.set_title("The same numbers on the chip: four SEngines in a 2 x 2 grid, XY routing.\n"
                 "Cycles per 512 B exchange between a core of each SEngine.", fontsize=12, color=INK, loc="left", pad=10)
    ax.text(.5, .55, f"Default DMA engine on every route; GPSIMD ({ci(gps)}) only works inside an SEngine.\n"
            "Grid and XY routing as described by AWS; die assignment from the driver source.", fontsize=9.5, color=MUTED,
            va="top")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-.9, 9.6)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    sys.path.insert(0, str(HERE / "common"))
    try:
        from ubench_style import configure, INK, MUTED
        configure()
    except Exception:  # noqa: BLE001
        INK, MUTED = "#172F3B", "#5E707A"
    rows = [r for r in json.loads((HERE / "results" / "core-sendrecv-xseng.json").read_text())["rows"]
            if r["kernel"] == "xchg_dma" and r["deviceOk"]]
    cyc = {(r["partner"], r["tileBytes"]): r["cycles"] for r in rows}
    # Inside an SEngine the GPSIMD engine is the fast path at 512 B (it tops out at 128 KiB, and hangs across SEngines).
    grow = {r["tileBytes"]: r for r in json.loads((HERE / "results" / "core-sendrecv-xseng.json").read_text())["rows"]
            if r["kernel"] == "xchg_gpsimd" and r["partner"] == 1 and r["deviceOk"] and not r["swap"]}
    gps = grow[SMALL]["cycles"]
    bw = {r["partner"]: r["GBpsPerDirection"] for r in rows if r["tileBytes"] == LARGE}
    show = np.full((8, 8), np.nan)
    for a in range(8):
        for b in range(8):
            if a != b:
                show[a, b] = gps if (a < b and (a ^ b) == 1) else cyc[(a ^ b, SMALL if a < b else LARGE)]
    vir = plt.get_cmap("viridis")
    cmap = LinearSegmentedColormap.from_list("c2c", [vir(x) for x in np.linspace(.85, .02, 64)])
    fig = plt.figure(figsize=(18.6, 10.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1], wspace=.1)
    ax, tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    def sub(a, b):
        if a < b:
            h = HOPS[a ^ b]
            return "SEngine, GPSIMD" if h == 0 else f"{h} hop{'s' if h > 1 else ''}"
        return f"{bw[a ^ b]:.0f} GB/s" + (", default" if (a ^ b) == 1 else "")

    matrix(ax, plt, show, ci, cmap, Normalize(500, 9500), INK, sub=sub)
    fig.axes[-1].set_ylabel(f"cycles per exchange ({CLK_GHZ} GHz NeuronCore clock)", fontsize=12)
    for a in range(8):
        for b in range(8):
            if (a ^ b) in (6, 7):
                ax.add_patch(Rectangle((b - .46, a - .46), .92, .92, fill=False, ec="#C0392B", lw=1.8))
    ax.set_title("Above the diagonal: 512 B per exchange. Below: 2 MiB per exchange, with GB/s per direction.\n"
                 "Red frame: pairs every collective tested refuses on the stock runtime (no_hier no_mesh).",
                 fontsize=12, color=INK, loc="left", pad=10)
    c = lambda b: ci(cyc[(b, SMALL)])  # noqa: E731
    # what the text says about the plotted rows: chain lengths, and any cell that is not from chip 0
    small = [r for r in rows if r["tileBytes"] == SMALL]
    rest = [r for r in rows if r["tileBytes"] == LARGE] + [grow[SMALL]]
    span = lambda rs: f"{min(min(r['chainLengths']) for r in rs)}-{max(max(r['chainLengths']) for r in rs)}"  # noqa: E731
    off = [r for r in small + rest if r["chip"] != 0]
    away = "; " + ", ".join(f"one {'512 B' if r['tileBytes'] == SMALL else '2 MiB'} cell from chip {r['chip']}" for r in off) if off else ""
    fig.suptitle(f"Trainium3 nisa.sendrecv on every core pair via a core-map shim: {c(1)} in the SEngine ({ci(gps)} with GPSIMD), "
                 f"{c(2)}-{c(4)} one hop, {c(6)} two", x=.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(.02, .935, f"Stock, sendrecv only reaches the other core of an LNC pair. An LD_PRELOAD shim swaps entries of the "
             "driver's logical-to-physical core map, so the same compiled kernel\nruns on any two cores; compiler, runtime "
             f"and driver are unchanged. Default DMA engine, except the 512 B same-SEngine cells: GPSIMD engine, stock. It is faster there\n({ci(gps)} against {c(1)} cycles on the default engine), but stops at 128 KiB tiles and hangs across SEngines. Cost = slope of whole-NEFF time against chain length "
             f"({span(small)} for the default engine at 512 B, up to {max(max(r['chainLengths']) for r in rest)} otherwise).\n"
             "Device output checked at every length; physical cores confirmed by the driver's per-core counters. "
             f"Logical core ids. One measurement per route class (NC0 with NCb on chip 0 of one node{away}), not one per pair.",
             fontsize=10, color=MUTED, va="top")
    fig.subplots_adjust(left=.065, right=.985, top=.8, bottom=.05)
    topology(tx, cyc, gps, INK, MUTED)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "core-sendrecv-xseng.png", dpi=150)
    print("wrote figures/core-sendrecv-xseng.png")


if __name__ == "__main__":
    main()
