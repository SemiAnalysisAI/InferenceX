"""How the 16 chips of a Trainium3 node reach each other, as the measurements show it.

  python3 p2p/analyze_xchip.py && python3 c2c/figure_chip_topology.py      -> figures/chip-topology.png

Drawn from: results/core-xchip.json (p2p/xchip.sh: cycles per 512 B exchange, which pairs run) and the per-rank
packet traces of p2p/xchip_routes.sh (which CCdma_S port carries each chip distance). The layout is schematic:
neuron-ls lists the same connectivity per chip (its partner, then the 7 chips of its parity; ultraserver-type Switch),
so the "shared link" is a switch plane; see c2c/figure_chip_topology_sketch.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
from figure_c2c import kc  # noqa: E402

GREEN, BLUE, RED, GREY = "#2BAE66", "#2F5D8C", "#C0392B", "#5E707A"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    sys.path.insert(0, str(HERE / "common"))
    try:
        from ubench_style import configure, INK, MUTED
        configure()
    except Exception:  # noqa: BLE001
        INK, MUTED = "#172F3B", "#5E707A"
    runs = json.loads((HERE / "results" / "core-xchip.json").read_text())["runs"]
    ag = {r["chipMask"]: r for r in runs if r["kernel"] == "ag_hbm" and r["coreMask"] == 0 and r["ok"]}
    cp = {r["chipMask"]: r for r in runs if r["kernel"] == "cp_hbm" and r["coreMask"] == 0}
    G = json.loads((HERE / "results" / "chain-steady.json").read_text())   # 64 MiB per core, GB/s per chip, one direction
    g = G["gbpsByChipMask"]
    med = lambda ms: float(np.median([ag[m]["cyclesMedian"] for m in ms]))  # noqa: E731
    even = [m for m in ag if m % 2 == 0]
    odd = [m for m in ag if m % 2 == 1 and m != 1]
    def rng(ms):
        if not ms:
            return "n/a"
        lo, hi = kc(min(ag[m]["cyclesMedian"] for m in ms)), kc(max(ag[m]["cyclesMedian"] for m in ms))
        return lo if lo == hi else f"{lo}-{hi}"

    def parity(par):   # same-parity offsets, split by the parity of the chips: range over offsets of the median over pairs
        vs = [float(np.median(c)) for c in ([p["cycles"] for p in ag[m]["pairs"] if p["chipA"] % 2 == par and p["chipB"] % 2 == par]
                                            for m in even) if c]
        return rng(even) if not vs else kc(min(vs)) if kc(min(vs)) == kc(max(vs)) else f"{kc(min(vs))}-{kc(max(vs))}"

    def both():   # the title's same-parity range: spans the even and the odd plane labels
        ends = sorted({float(x.rstrip("k")) for par in (0, 1) for x in parity(par).split("-")})
        return f"{ends[0]:g}k" if len(ends) == 1 else f"{ends[0]:g}k-{ends[-1]:g}k"

    fig = plt.figure(figsize=(19, 10.2))
    ax = fig.add_axes([.02, .1, .62, .74])
    ax.set_xlim(-1.2, 16.4)
    ax.set_ylim(-3.4, 5.4)
    ax.axis("off")
    YE, YO, W, H = 2.2, -.2, 1.45, 1.0   # even-chip row, odd-chip row
    X = lambda p: 2 * p + .3  # noqa: E731

    def chip(x, y, c):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=INK, lw=1.6, zorder=3))
        ax.text(x + W / 2, y + H / 2, f"chip {c}", ha="center", va="center", fontsize=13, fontweight="bold", color=INK, zorder=4)

    # shared links: one bar per parity
    YB_E, YB_O = 4.3, -2.3
    ax.plot([X(0) + .2, X(7) + W - .2], [YB_E, YB_E], color=BLUE, lw=7, solid_capstyle="round", zorder=1)
    ax.plot([X(0) + .2, X(7) + W - .2], [YB_O, YB_O], color=BLUE, lw=7, solid_capstyle="round", zorder=1)
    for p in range(8):
        x = X(p)
        chip(x, YE, 2 * p)
        chip(x, YO, 2 * p + 1)
        for dx in (-.28, .28):  # two ports per link, one per die
            ax.plot([x + W / 2 + dx] * 2, [YO + H, YE], color=GREEN, lw=3.2, zorder=2)
            ax.plot([x + W / 2 + dx] * 2, [YE + H, YB_E], color=BLUE, lw=2.2, zorder=2)
            ax.plot([x + W / 2 + dx] * 2, [YO, YB_O], color=BLUE, lw=2.2, zorder=2)
    box = dict(boxstyle="round,pad=0.3", fc="white", lw=1.4)
    ax.text(X(3) + W / 2, (YE + YO + H) / 2, f"tray link\n{kc(ag[1]['cyclesMedian'])}\n{g['1']} GB/s", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=INK, bbox=dict(box, ec=GREEN), zorder=5)
    ax.text(X(3) + W + .28, YB_E + .55, f"switch plane, even chips: {parity(0)} cycles, {g['2']} GB/s",
            ha="center", va="center", fontsize=12, fontweight="bold", color=INK, bbox=dict(box, ec=BLUE), zorder=5)
    ax.text(X(3) + W + .28, YB_O - .55, f"switch plane, odd chips: {parity(1)} cycles, {g['2']} GB/s",
            ha="center", va="center", fontsize=12, fontweight="bold", color=INK, bbox=dict(box, ec=BLUE), zorder=5)
    # the pair with no single link: chip 0 to chip 3
    ax.annotate("", xy=(X(1) + .15, YO + H - .05), xytext=(X(0) + W - .15, YE + .05),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2, ls=(0, (4, 3))), zorder=6)
    ax.text(X(1) + W + .28, YO + H + .7, f"two hops\n{rng(odd)}\n{g['3']} GB/s", ha="center", va="center", fontsize=10.5,
            color=RED, fontweight="bold", bbox=dict(box, ec=RED, ls=":"), zorder=7)
    ax.text(-1.1, YE + H / 2, "even\nchips", ha="center", va="center", fontsize=12, color=MUTED, fontweight="bold")
    ax.text(-1.1, YO + H / 2, "odd\nchips", ha="center", va="center", fontsize=12, color=MUTED, fontweight="bold")

    # ---------------- one chip's four ports
    bx = fig.add_axes([.66, .1, .33, .74])
    bx.set_xlim(0, 10)
    bx.set_ylim(-1.2, 10.6)
    bx.axis("off")
    bx.text(0, 10.3, "One chip: four ports, one per LNC pair", fontsize=14, fontweight="bold", color=INK)
    rows = [(7.6, "die 1", [("NC0  NC1", BLUE, "S2 (odd chips: S3)"), ("NC2  NC3", GREEN, "S3 (odd chips: S2)")]),
            (3.4, "die 0", [("NC4  NC5", BLUE, "S1 (odd chips: S0)"), ("NC6  NC7", GREEN, "S0 (odd chips: S1)")])]
    for y, die, pairs in rows:
        bx.add_patch(FancyBboxPatch((.3, y - 1.5), 5.6, 3.3, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#F4F6F8", ec=INK, lw=1.3))
        bx.text(.5, y + 1.5, die, fontsize=11.5, color=MUTED, fontweight="bold", va="center")
        for k, (lab, col, port) in enumerate(pairs):
            yy = y + .55 - 1.3 * k
            bx.add_patch(FancyBboxPatch((.8, yy - .42), 2.5, .84, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=INK, lw=1.4))
            bx.text(2.05, yy, lab, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
            bx.plot([3.3, 6.9], [yy, yy], color=col, lw=3.2 if col == GREEN else 2.4)
            bx.text(4.65, yy + .26, port, ha="center", va="bottom", fontsize=9.5, color=INK)
            bx.text(7.05, yy, "partner chip" if col == GREEN else "switch plane", va="center", fontsize=11.5, color=col, fontweight="bold")
    bx.plot([2.05, 2.05], [5.22, 6.08], color=GREY, lw=1.5, ls=":")
    bx.text(2.2, 5.45, "die to die (inside the package)", fontsize=9.5, color=MUTED, va="center")
    cp_ok = sorted(m for m, r in cp.items() if r["ok"])
    cp_bad = sorted(m for m, r in cp.items() if not r["ok"] and r["reason"])  # an empty reason = run not finished
    notes = [
        "Port names (CCdma_S0-S3) are the queue names in the profiler trace.",
        "A pair without the port in use borrows its same-die neighbour's.",
        "Partner chip: every core reaches every core (HBM collective_permute, ~16k);",
        "   the same core index also runs all_gather.",
        "Switch plane: a core reaches only the cores on its own die side",
        "   (NC0-3 to NC0-3, NC4-7 to NC4-7); across, the runtime says",
        "   \"not connected physically or over the network\".",
    ]
    for i, t in enumerate(notes):
        bx.text(.3, .9 - .42 * i, t, fontsize=10.5, color=INK, va="center")

    fig.suptitle(f"Trainium3 chip to chip inside a node: tray partner {kc(ag[1]['cyclesMedian'])} cycles, {g['1']} GB/s; "
                 f"same plane {both()}, {g['2']}; two hops {rng(odd)}, {g['3']}",
                 x=.02, ha="left", fontsize=15, fontweight="bold", color=INK, y=.965)
    fig.text(.02, .915, "Cycles per 512 B exchange at 1.2 GHz: HBM all_gather chains, same core on both chips, median over the core pairs. "
             "GB/s per chip, one direction, 64 MiB per core (ring; two hops on mesh).\n"
             f"Measured chip-index offsets (XOR): {', '.join(str(m) for m in sorted(ag))}. Each plane also reaches the same-parity chips "
             "of the rack's other node (not drawn). Schematic layout.", fontsize=10, color=MUTED, va="top")
    fig.text(.02, .055, "Red: other plane and not the tray partner = two hops, a switch then a tray link (the chip in between forwards). "
             f"Ring and collective_permute refuse these pairs (offsets {', '.join(map(str, cp_bad)) or 'none'}); the default mesh runs them.\n"
             f"Tray link and plane at once: {G['bothLinksGbps']} GB/s per chip. collective_permute runs on offsets "
             f"{', '.join(map(str, cp_ok))} at ~16k cycles.", fontsize=10, color=MUTED, va="top")
    fig.savefig(HERE / "figures" / "chip-topology.png", dpi=150)
    print("wrote figures/chip-topology.png")


if __name__ == "__main__":
    main()
