"""One Trainium3 topology diagram that every result in results/ is consistent with.

  python3 c2c/figure_topology.py            # writes figures/topology.png

Left: one chip. Four SEngines (two NeuronCores + one HBM stack each) in a ring: R links inside a die, D2D links
between the dies, no diagonal. One external port per SEngine. Link numbers are nisa.sendrecv through the core-map
shim (results/core-sendrecv-xseng.json), the only method that runs one protocol on all 28 pairs.
Right: one sled (two mirrored chips, two partner links) and the two switch planes (one per chip parity).
Below: each measurement and the part of the drawing it fixes, what is protocol rather than wiring, and what is
not known. Sources for the constants: c2c/C2C.md, p2p/P2P.md, sendrecv/XSENG.md, results/core-xchip.json,
results/core-nccom.json, results/core-xcoll.json, data/inventory.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
from figure_c2c import ci  # noqa: E402

TEAL, ORANGE, GREEN, BLUE, RED = "#2F7F8F", "#C8741A", "#2BAE66", "#2F5D8C", "#C0392B"
SMALL, LARGE = 512, 2097152
# Logical NC pair (even chip) -> (physical NCs, SENG / HBM stack / port number). Odd chips: see ODD.
EVEN = {0: ("nc4 nc5", 2), 2: ("nc6 nc7", 3), 4: ("nc2 nc3", 1), 6: ("nc0 nc1", 0)}
ODD = {0: 3, 2: 2, 4: 0, 6: 1}


def chip_panel(ax, cyc, bw, INK, MUTED):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")
    W, H = 4.6, 3.3
    pos = {0: (2.6, 7.2), 2: (9.0, 7.2), 4: (2.6, 1.6), 6: (9.0, 1.6)}
    mean = lambda vs: sum(vs) / len(vs)  # noqa: E731
    c512 = lambda *bs: ci(mean([cyc[(b, SMALL)] for b in bs]))  # noqa: E731   either core of the far SEngine costs the same
    gb = lambda *bs: f"{mean([bw[b] for b in bs]):.0f}"  # noqa: E731

    for y, name in ((6.8, "die 1"), (1.2, "die 0")):
        ax.add_patch(FancyBboxPatch((2.2, y), 11.8, 4.1, boxstyle="round,pad=0,rounding_size=.2", fc="#FAFBFC", ec=MUTED,
                                    lw=1.2, ls=(0, (5, 3)), zorder=0))
        top = name == "die 1"
        ax.text(2.3, y + 4.2 if top else y - .1, name, ha="left", va="bottom" if top else "top", fontsize=12, color=MUTED,
                fontweight="bold", zorder=1)

    # R links (inside a die) and D2D links (between the dies): one per pair of neighbouring SEngines
    for a, b in ((0, 2), (4, 6)):
        y = pos[a][1] + H / 2
        ax.plot([pos[a][0] + W, pos[b][0]], [y, y], color=TEAL, lw=9, solid_capstyle="butt", zorder=1)
        ax.text(8.1, y + .32, f"R  {c512(2, 3)}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(8.1, y - .32, f"{gb(2, 3)} GB/s", ha="center", va="top", fontsize=10.5, color=TEAL, fontweight="bold")
    for a, b, left in ((0, 4, True), (2, 6, False)):
        x = pos[a][0] + W / 2
        ax.plot([x, x], [pos[b][1] + H, pos[a][1]], color=ORANGE, lw=9, solid_capstyle="butt", zorder=1)
        ax.text(x + (-.3 if left else .3), 6.05, f"D2D  {c512(4, 5)}\n{gb(4, 5)} GB/s", ha="right" if left else "left",
                va="center", fontsize=10.5, color=INK, fontweight="bold", linespacing=1.35)
    for a, b in ((0, 6), (2, 4)):
        sx = 1 if pos[b][0] > pos[a][0] else -1
        xa, xb = pos[a][0] + W / 2 + sx * W / 2, pos[b][0] + W / 2 - sx * W / 2
        ax.plot([xa, xb], [pos[a][1], pos[b][1] + H], color=RED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(8.1, 6.05, f"no diagonal link\n{c512(6, 7)}\n{gb(6, 7)} GB/s", ha="center", va="center", fontsize=10,
            color=RED, fontweight="bold", linespacing=1.3, bbox=dict(fc="white", ec="none", pad=2), zorder=4)

    for k, (x, y) in pos.items():
        phys, s = EVEN[k]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=.15", fc="#F1F4F6", ec=INK, lw=1.6,
                                    zorder=2))
        ax.text(x + .18, y + H - .14, f"SEngine {s}", ha="left", va="top", fontsize=10.5, color=INK, fontweight="bold",
                zorder=3)
        for i in (0, 1):
            cx = x + .3 + i * 2.75
            ax.add_patch(FancyBboxPatch((cx, y + 1.0), 1.25, 1.45, boxstyle="round,pad=0,rounding_size=.08", fc="white",
                                        ec=INK, lw=1.4, zorder=3))
            ax.text(cx + .625, y + 1.95, f"NC{k + i}", ha="center", va="center", fontsize=13, color=INK,
                    fontweight="bold", zorder=4)
            ax.text(cx + .625, y + 1.33, phys.split()[i], ha="center", va="center", fontsize=8.5, color=MUTED, zorder=4)
        ax.annotate("", (x + 3.03, y + 1.72), (x + 1.57, y + 1.72), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3),
                    zorder=4)
        ax.text(x + 2.3, y + 1.84, ci(cyc[(1, SMALL)]), ha="center", va="bottom", fontsize=10.5, color=INK,
                fontweight="bold", zorder=4)
        ax.text(x + 2.3, y + 1.6, f"{bw[1]:.0f} GB/s", ha="center", va="top", fontsize=9, color=MUTED, zorder=4)
        ax.add_patch(FancyBboxPatch((x + .3, y + .2), W - .6, .55, boxstyle="round,pad=0,rounding_size=.06", fc="#D5E0E6",
                                    ec="none", zorder=3))
        ax.text(x + W / 2, y + .475, f"HBM stack {s}", ha="center", va="center", fontsize=9.5, color=INK, zorder=4)
        # external port
        left = x < 5
        col, name = (BLUE, "switch plane") if left else (GREEN, "partner chip")
        yy = y + H / 2
        x0, x1 = (x, .25) if left else (x + W, 15.75)
        ax.plot([x0, x1], [yy, yy], color=col, lw=5, solid_capstyle="butt", zorder=1)
        ax.text(x1 + (.05 if left else -.05), yy + .28, f"port S{s}", ha="left" if left else "right", va="bottom",
                fontsize=10.5, color=col, fontweight="bold")
        ax.text(x1 + (.05 if left else -.05), yy - .28, name, ha="left" if left else "right", va="top",
                fontsize=9.5, color=col)

    ax.set_title("One chip: four SEngines in a ring",
                 fontsize=13, color=INK, loc="left", fontweight="bold", pad=4)
    ax.text(0, .45, "Big: the core id you launch on (even chip). Small: the physical core. On links: cycles per 512 B sendrecv "
            "exchange, and GB/s per core pair at 2 MiB (latency-bound,\nnot link capacity). Measured from NC0 on one chip through the core-map shim; twin links show the same value.", fontsize=9.5, color=MUTED, va="top")


def sled_panel(ax, INK, MUTED):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")
    CW, CH, SW, SH = 4.6, 4.6, 1.85, 1.75
    Y0 = 3.85

    def chip(x0, title, cols, ports):
        ax.add_patch(FancyBboxPatch((x0, Y0), CW, CH, boxstyle="round,pad=0,rounding_size=.15", fc="#FAFBFC", ec=INK, lw=1.6,
                                    zorder=2))
        ax.text(x0 + CW / 2, Y0 + CH + .15, title, ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="bold")
        ax.plot([x0 + .15, x0 + CW - .15], [Y0 + CH / 2] * 2, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=3)
        out = {}
        for ci_, col in enumerate(cols):
            for ri, k in enumerate(col):      # ri 0 = die 1 (top), 1 = die 0 (bottom)
                x, y = x0 + .3 + ci_ * (SW + .3), Y0 + CH - .35 - SH - ri * (SH + .4)
                ax.add_patch(FancyBboxPatch((x, y), SW, SH, boxstyle="round,pad=0,rounding_size=.1", fc="#F1F4F6", ec=INK,
                                            lw=1.3, zorder=3))
                ax.text(x + SW / 2, y + SH / 2 + .22, f"NC{k} NC{k + 1}", ha="center", va="center", fontsize=10, color=INK,
                        fontweight="bold", zorder=4)
                ax.text(x + SW / 2, y + SH / 2 - .32, f"port S{ports[k]}", ha="center", va="center", fontsize=8.5,
                        color=MUTED, zorder=4)
                out[k] = (x, y + SH / 2)
        ax.text(x0 + .12, Y0 + CH - .08, "die 1", fontsize=7.5, color=MUTED, va="top", zorder=4)
        ax.text(x0 + .12, Y0 + .06, "die 0", fontsize=7.5, color=MUTED, va="bottom", zorder=4)
        return out

    xe, xo = 2.6, 8.8
    e = chip(xe, "even chip c", ((0, 4), (2, 6)), {k: v[1] for k, v in EVEN.items()})
    o = chip(xo, "odd chip c+1 (mirrored)", ((2, 6), (0, 4)), ODD)
    ax.add_patch(FancyBboxPatch((1.2, Y0 - .45), 13.6, CH + 1.5, boxstyle="round,pad=0,rounding_size=.2", fc="none",
                                ec="#D08A1F", lw=1.3, ls=(0, (5, 3)), zorder=1))
    ax.text(8.0, Y0 - .32, "one tray = two chips", ha="center", va="bottom", fontsize=9.5, color="#B46F0C")

    # partner links: the facing SEngines, die to die
    for k in (2, 6):
        y = e[k][1]
        ax.plot([e[k][0] + SW, o[k][0]], [y, y], color=GREEN, lw=6, solid_capstyle="butt", zorder=3)
    ax.text(8.0, Y0 + CH / 2, "tray\nlinks\n6.9k\n385 GB/s", ha="center", va="center", fontsize=9.5, color=GREEN, fontweight="bold",
            linespacing=1.15, bbox=dict(fc="#FAFBFC", ec="none", pad=1), zorder=5)

    # switch planes: two rails each (the die 1 ports and the die 0 ports never reach each other)
    def plane(yb, label, pts, side):
        ax.add_patch(FancyBboxPatch((.3, yb), 15.4, 1.45, boxstyle="round,pad=0,rounding_size=.15", fc="#EAF1F8", ec=BLUE,
                                    lw=1.3, zorder=1))
        ax.text(8.0, yb + (1.2 if side > 0 else .25), label, ha="center", va="center", fontsize=10.5, color=BLUE,
                fontweight="bold", zorder=4)
        rails = {0: yb + (.3 if side > 0 else 1.15), 4: yb + (.75 if side > 0 else .7)}
        for k, y in rails.items():
            ax.plot([.7, 15.3], [y, y], color=BLUE, lw=3.5, zorder=2)
            ax.text(15.25, y + side * .13, f"NC{k}-{k + 3} rail", ha="right", va="bottom" if side > 0 else "top", fontsize=7.5,
                    color=BLUE, zorder=4)
        for k, (x, y) in pts.items():
            xv = (x - (.45 if k == 0 else .95)) if side > 0 else (x + SW + (.45 if k == 0 else .95))
            xs = x if side > 0 else x + SW
            ax.plot([xs, xv, xv], [y, y, rails[k]], color=BLUE, lw=3.5, solid_joinstyle="miter", zorder=2)
            ax.plot([xv], [rails[k]], marker="o", ms=6, color=BLUE, zorder=3)

    plane(10.2, "switch plane A: the even chips of one rack (2 nodes)", {k: e[k] for k in (0, 4)}, +1)
    plane(.35, "switch plane B: the odd chips of one rack (2 nodes)", {k: o[k] for k in (0, 4)}, -1)

    note = dict(fontsize=9.5, color=INK, va="center", linespacing=1.35)
    ax.text(15.7, 9.86, "same plane: 5.8k, 245 GB/s inside a node; 149 GB/s to the rack's other node", ha="right", **note)
    ax.text(.3, 2.6, "other plane of the same rack, not the partner: two hops (7.9k, 112 GB/s)", ha="left", color=RED,
            **{k: v for k, v in note.items() if k != "color"})
    ax.set_title("One tray and the two switch planes (chip to chip: all_gather cycles)",
                 fontsize=13, color=INK, loc="left", fontweight="bold", pad=4)


SOFTWARE = [
    "all_gather's fast path (4.8k-5.3k, LNC pair and i+4) is a handshake, not a wire.",
    "Stock collectives refuse the diagonals; with the core-map shim all but SBUF all_gather run (16.4k-16.9k).",
    "GPSIMD sendrecv (858) only pairs a core with its SEngine sibling.",
    "Two rails per plane = the chip's two switch ports (runtime log); permute refuses NC0-3 to NC4-7.",
]
UNKNOWN = [
    "Where the SEngines and ports sit on a die: the layout is schematic.",
    "Whether a D2D link is one link or one lane per core.",
    "Which port carries the rack-to-rack cable.",
    "The rack split is read from the driver, not measured.",
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE / "common"))
    try:
        from ubench_style import configure, INK, MUTED
        configure()
    except Exception:  # noqa: BLE001
        INK, MUTED = "#172F3B", "#5E707A"
    rows = json.loads((HERE / "results" / "core-sendrecv-xseng.json").read_text())["rows"]
    rows = [r for r in rows if r["kernel"] == "xchg_dma" and r["deviceOk"]]
    cyc = {(r["partner"], r["tileBytes"]): r["cycles"] for r in rows}   # analyze_xseng.py keeps one row per (partner, tile)
    bw = {r["partner"]: r["GBpsPerDirection"] for r in rows if r["tileBytes"] == LARGE}
    assert all((b, SMALL) in cyc and b in bw for b in range(1, 8)), "missing a route in core-sendrecv-xseng.json"

    fig = plt.figure(figsize=(22, 10.6))
    fig.suptitle("Trainium3 topology inside one rack: a wiring that fits our measurements", x=.02, y=.975, ha="left", fontsize=17,
                 fontweight="bold", color=INK)
    fig.text(.02, .93, "Chip: 2 dies x 2 SEngines x 2 NeuronCores. Tray: 2 mirrored chips. UltraServer: 2 racks x 2 nodes x 8 trays x "
             "2 chips = 64 chips. Cycles at 1.2 GHz per 512 B exchange. Chip to chip GB/s: per chip, one direction, 64 MiB per core.", fontsize=11, color=MUTED, va="top")
    chip_panel(fig.add_axes([.02, .19, .47, .68]), cyc, bw, INK, MUTED)
    sled_panel(fig.add_axes([.51, .19, .47, .68]), INK, MUTED)
    for x, head, col, lines in ((.02, "Software, not wiring", INK, SOFTWARE), (.51, "Not known", RED, UNKNOWN)):
        fig.text(x, .168, head, fontsize=13, fontweight="bold", color=col, va="top")
        for i, t in enumerate(lines):
            fig.text(x, .134 - i * .027, t, fontsize=11, color=INK, va="top")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "topology.png", dpi=150)
    print("wrote figures/topology.png")


if __name__ == "__main__":
    main()
