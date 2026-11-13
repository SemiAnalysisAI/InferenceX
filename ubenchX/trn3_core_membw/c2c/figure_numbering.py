"""Explainer: the core ids you launch on (logical) against the driver's hardware ids (physical), even and odd chips.

  python3 c2c/figure_numbering.py      -> figures/core-numbering.png

Source: nc_mapping_v0_seng_swap_pds in /usr/src/aws-neuronx-2.31.7.0/v4/neuron_dhal_v4.c and v4/address_map.h.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
# hardware, the same on every chip: (die, column) -> (physical cores, HBM stack)
HW = {(1, 0): ((4, 5), 2), (1, 1): ((6, 7), 3), (0, 0): ((2, 3), 1), (0, 1): ((0, 1), 0)}
# logical core -> physical core
MAP = {"even": (4, 5, 6, 7, 2, 3, 0, 1), "odd": (6, 7, 4, 5, 0, 1, 2, 3)}
PAIR_COL = ["#1F77B4", "#D95F02", "#2E7D32", "#8E44AD"]  # logical pair 0,1 / 2,3 / 4,5 / 6,7


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    sys.path.insert(0, str(HERE / "common"))
    try:
        from ubench_style import configure, INK, MUTED
        configure()
    except Exception:  # noqa: BLE001
        INK, MUTED = "#172F3B", "#5E707A"
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
    for ax, par, chips in zip(axes, ("even", "odd"), ("chips 0, 2, 4, ... 14", "chips 1, 3, 5, ... 15")):
        ax.set_xlim(0, 10)
        ax.set_ylim(-.3, 10.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(.2, 10.1, f"{par.capitalize()}-numbered chip", fontsize=17, fontweight="bold", color=INK, va="center")
        ax.text(4.15, 10.1, f"({chips})", fontsize=13, color=MUTED, va="center")
        logical = {p: l for l, p in enumerate(MAP[par])}  # physical -> logical
        for die, y0 in ((1, 5.5), (0, .4)):
            ax.add_patch(FancyBboxPatch((.2, y0), 9.6, 3.9, boxstyle="round,pad=0.04", fc="#F6F8F9", ec=INK, lw=1.3))
            ax.text(.35, y0 + 3.9 + .22 if die else y0 - .28, f"die {die}", fontsize=13.5, fontweight="bold", color=INK,
                    va="center")
            for colx, x0 in ((0, .7), (1, 5.4)):
                (pa, pb), hbm = HW[(die, colx)]
                hy = y0 + 3.05 if die else y0 + .35  # HBM bar: outer edge of the die
                cy = y0 + 1.35 if die else y0 + 2.55  # core boxes: inner edge
                ax.add_patch(Rectangle((x0, hy), 3.9, .5, fc="#DCE6EA", ec=MUTED, lw=.8))
                ax.text(x0 + 1.95, hy + .25, f"HBM stack {hbm}", fontsize=13, color=INK, ha="center", va="center")
                for k, p in enumerate((pa, pb)):
                    x = x0 + .05 + 2.0 * k
                    c = PAIR_COL[logical[p] // 2]
                    ax.plot([x + .9, x + .9], [hy if die else hy + .5, cy + .75 if die else cy - .75], color=MUTED, lw=1)
                    ax.add_patch(FancyBboxPatch((x, cy - .75), 1.8, 1.5, boxstyle="round,pad=0.02", fc="white", ec=c,
                                                lw=2.6))
                    ax.text(x + .9, cy + .2, f"NC{logical[p]}", fontsize=21, fontweight="bold", color=c, ha="center",
                            va="center")
                    ax.text(x + .9, cy - .42, f"hardware core {p}", fontsize=10.5, color=MUTED, ha="center", va="center")
    fig.suptitle("Trainium3 core numbering: the hardware is identical on every chip, but the core names you launch on "
                 "swap sides on odd chips", x=.015, ha="left", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(.015, .14, "Large coloured label = the core id you launch on (NEURON_RT_VISIBLE_CORES, and every figure "
             "here). Grey text = the driver's fixed id for that piece of silicon.\nEach hardware pair is wired to one "
             "HBM stack. Follow the blue pair: NC0 and NC1 sit on HBM stack 2 on an even chip and on HBM stack 3 on an "
             "odd chip.\nNC0-3 are always die 1 and NC4-7 always die 0; only the two pairs inside each die trade "
             "places.\nSource: the driver source (nc_mapping_v0_seng_swap_pds). Positions inside a die are schematic.", fontsize=12.5, color=INK, va="top", linespacing=1.35)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.15, wspace=.03)
    fig.savefig(HERE / "figures" / "core-numbering.png", dpi=150)
    print("wrote figures/core-numbering.png")


if __name__ == "__main__":
    main()
