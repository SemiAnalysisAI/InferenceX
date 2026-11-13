"""Simplified Trainium3 core-to-core diagram: cores, dies (from the Neuron driver source) and the links.

  python3 c2c/figure_c2c_simple.py      -> figures/c2c-stock/core-c2c-simple.png

The SEngine ring of figures/topology.png: links join SEngines (LNC pair + HBM stack), R inside a die, D2D across,
no diagonal. Link colours are the link kind (as c2c/figure_c2c.py), not a cost; no numbers or matrix.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
import figure_c2c as C  # noqa: E402


def main():
    hbm, sbuf, via, cls, two, stalls, chips = C.load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import FancyBboxPatch, Rectangle
    sys.path.insert(0, str(HERE / "common"))
    try:
        from ubench_style import configure, INK, MUTED
        configure()
    except Exception:  # noqa: BLE001
        INK, MUTED = "#172F3B", "#5E707A"
    vir = plt.get_cmap("viridis")
    cmap = LinearSegmentedColormap.from_list("c2c", [vir(x) for x in np.linspace(.85, .02, 64)])
    vals = np.concatenate([hbm[np.isfinite(hbm)], sbuf[np.isfinite(sbuf)], list(C.SR_PAIR.values())])
    norm = Normalize(vals.min(), vals.max())
    col = lambda c: cmap(norm(c))  # noqa: E731

    fig, ax = plt.subplots(figsize=(10, 9.95))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.1, 8.85)
    ax.set_aspect("equal")
    ax.axis("off")
    X = [1.3, 3.5, 6.5, 8.7]
    YA, YB = 6.6, 3.6
    HW, HH = .78, .58
    PAD = .2  # gap between the NC boxes and the LNC-pair outline
    HT = 1.33  # outer edge of the HBM stack bar, from the NC row
    YM = (YA + YB) / 2
    pos = {i: (X[i], YA) for i in range(4)} | {i + 4: (X[i], YB) for i in range(4)}
    for y0, lab, ly, va in ((YA - 1.0, "die 1 (driver)", YA + 1.77, "bottom"), (YB - 1.67, "die 0 (driver)", YB - 1.77, "top")):
        ax.add_patch(FancyBboxPatch((.25, y0), 9.5, 2.67, boxstyle="round,pad=0.05", fc="#F6F8F9", ec=INK, lw=1.4,
                                    zorder=0))
        ax.text(.3, ly, lab, fontsize=15, color=INK, va=va, fontweight="bold")

    for p, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):  # one HBM stack per LNC pair (driver ids)
        (xa, y), (xb, _) = pos[a], pos[b]
        up = p < 2
        hy = y + HT - .4 if up else y - HT
        ax.add_patch(Rectangle((xa - HW, hy), xb - xa + 2 * HW, .4, fc="#DCE6EA", ec=MUTED, lw=.8, zorder=1))
        ax.text((xa + xb) / 2, hy + .2, f"HBM stack {(2, 3, 1, 0)[p]} (odd chips: {(3, 2, 0, 1)[p]})", fontsize=13,
                color=INK, ha="center", va="center", zorder=2)
        for c in (a, b):
            ax.plot([pos[c][0]] * 2, [hy, y + HH] if up else [hy + .4, y - HH], color=MUTED, lw=1, zorder=1)

    def edge(p, q, c, lw=7):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, solid_capstyle="butt", zorder=2)

    for a in (0, 2, 4, 6):  # inside an SEngine: the LNC pair
        edge((pos[a][0] + HW, pos[a][1]), (pos[a + 1][0] - HW, pos[a][1]), C.C_LNC, lw=6)
    for a in (0, 2):  # D2D link: SEngine to SEngine across the dies ({0,1}-{4,5}, {2,3}-{6,7})
        xm = (X[a] + X[a + 1]) / 2
        edge((xm, YA - HH - PAD), (xm, YB + HH + PAD), C.C_D2D)
    for a in (1, 5):  # R link: the two SEngines of a die
        edge((pos[a][0] + HW + PAD, pos[a][1]), (pos[a + 1][0] - HW - PAD, pos[a][1]), C.C_R)
    # links join SEngines, not single cores (either core of the far SEngine costs the same): outline each
    # SEngine (the driver's SENG: 2 cores + 1 HBM stack + 32 DMA engines)
    for a in (0, 2, 4, 6):
        lo, hi = ((-HH - PAD, HT + .12) if a < 4 else (-HT - .12, HH + PAD))  # the pair's HBM stack is inside
        ax.add_patch(FancyBboxPatch((pos[a][0] - HW - PAD, pos[a][1] + lo), pos[a + 1][0] - pos[a][0] + 2 * (HW + PAD),
                                    hi - lo, boxstyle="round,pad=0.02", fc="#E9EFF8", ec="#9DB0CF", lw=1.2, zorder=.5))
    for c, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - HW, y - HH), 2 * HW, 2 * HH, boxstyle="round,pad=0.02", fc="white", ec=INK,
                                    lw=1.3, zorder=4))
        ax.text(x, y, f"NC{c}", fontsize=20, color=INK, fontweight="bold", ha="center", va="center", zorder=5)
    for k, (c, txt) in enumerate((
            (C.C_LNC, "inside an SEngine: the LNC pair, two cores on one HBM stack"),
            (C.C_R, "R link: the two SEngines of a die"),
            (C.C_D2D, "D2D link: SEngine to SEngine across the dies"))):
        y = 1.0 - .55 * k
        ax.plot([.4, 1.5], [y, y], color=c, lw=6, solid_capstyle="butt")
        ax.text(1.75, y, txt, fontsize=14, color=INK, va="center")
    ax.text(.4, -.62, "The four SEngines form a ring: no diagonal link ({0,1}-{6,7}, {2,3}-{4,5}). Links join SEngines, not\n"
            "single cores. Positions schematic; one D2D line per SEngine pair (one link or one lane per core: not known).",
            fontsize=11.5, color=MUTED, va="center")
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    fig.savefig(HERE / "figures" / "c2c-stock" / "core-c2c-simple.png", dpi=150)
    print("wrote figures/c2c-stock/core-c2c-simple.png")


if __name__ == "__main__":
    main()
