"""One trn3.48xlarge node as a hand-drawn block diagram, in the style of the SemiAnalysis Trainium3 article figures.

  python3 c2c/figure_chip_topology_sketch.py     # writes figures/chip-topology-sketch.png

What is measured or read off the box: 16 chips per node; PCI addresses come in 8 adjacent pairs; `neuron-ls` says
ultraserver-type Switch and lists, per chip, its partner chip and the 7 chips of the same parity as connected; the
cycle counts (results/core-xchip.json, HBM all_gather, 512 B, same core index).
What is taken from the article (NL32x2 Switched / Teton3 PDS) and NOT seen from the box: that a chip pair is one
JBOG tray, that the pair link is PCB, that each plane is 8 PCIe switches of 160 lanes, and the AECs to the other rack.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
BLUE_BG, BLUE = "#A5D8FF", "#1971C2"
GREEN_BG, GREEN = "#EBFBEE", "#2F9E44"
PURPLE_BG, PURPLE = "#EEBEFA", "#9C36B5"
YELLOW_BG, ORANGE = "#FFF9DB", "#F08C00"
RED, INK = "#E03131", "#1E1E1E"
FONT = ["Chalkboard SE", "Chalkboard", "Comic Sans MS", "Marker Felt"]


def kc(v):
    return f"{v / 1000:.1f}k"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    runs = {(r["kernel"], r["chipMask"]): r for r in json.loads((HERE / "results" / "core-xchip.json").read_text())["runs"]
            if r["coreMask"] == 0 and r["ok"]}
    med = lambda k, ms: float(np.median([runs[(k, m)]["cyclesMedian"] for m in ms if (k, m) in runs]))  # noqa: E731
    pcb, hop1, hop2 = med("ag_hbm", [1]), med("ag_hbm", range(2, 16, 2)), med("ag_hbm", range(3, 16, 2))
    p_pcb, p_hop1 = med("cp_hbm", [1]), med("cp_hbm", range(2, 16, 2))
    # rack to rack, same chip index (nodes 0-3 / 1-2), default mesh algorithm: one nccom-test op, not a chain slope
    xrack = [r["cyclesP50"] for r in json.loads((HERE / "results" / "core-nccom.json").read_text())["rows"]
             if r["scope"] == "xnode" and r["nodeMask"] == 3 and r["chipMask"] == 0 and r["algorithm"] == "Mesh"
             and r["kernel"] == "ag_hbm" and r["tileBytes"] == 512]
    st = json.loads((HERE / "results" / "chain-steady.json").read_text())   # 64 MiB per core, GB/s per chip and direction
    g = st["gbpsByChipMask"]
    xn = {c["nodeMask"]: c["gbps"] for c in st["crossNode"] if 0 in c["chipMasks"]}
    xrack = f"{xn[3]} GB/s"

    with plt.xkcd(scale=.7, length=120, randomness=2):
        plt.rcParams.update({"path.effects": [], "font.family": FONT})
        fig, ax = plt.subplots(figsize=(20, 14))
        ax.set_xlim(0, 200)
        ax.set_ylim(-27, 124)
        ax.axis("off")

        def box(x, y, w, h, fc, ec, lw=1.6, r=1.6, z=2):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

        def txt(x, y, s, c=INK, fs=13, **k):
            ax.text(x, y, s, color=c, fontsize=fs, ha=k.pop("ha", "center"), va=k.pop("va", "center"), zorder=k.pop("z", 6), **k)

        box(2, 13, 196, 104, BLUE_BG, BLUE, lw=1.8, r=2.5, z=1)
        txt(100, 120.5, "One trn3.48xlarge node: 16 Trainium3 (a quarter of the 64-chip UltraServer)", BLUE, 21)

        # the two switch planes: A above (even chips), B below (odd chips)
        SW = [(26, "PCIe\nSwitch 1"), (62, "PCIe\nSwitch 2"), (100, "o o o"), (138, "PCIe\nSwitch 7"), (174, "PCIe\nSwitch 8")]
        for y0, name in ((96, "Switch Plane A (shared with the rack's other node): even chips"),
                         (17, "Switch Plane B (shared with the rack's other node): odd chips")):
            box(8, y0, 188, 17, GREEN_BG, GREEN, z=2)
            up = 4 if y0 < 50 else 0   # plane B: switches at the top of the box, facing the chips
            txt(100, y0 + (2.2 if up else 14.6), name, GREEN, 15)
            for x, s in SW:
                if s == "o o o":
                    txt(x, y0 + 6 + up, s, GREEN, 16)
                else:
                    box(x - 13, y0 + 1.5 + up, 26, 10.5, GREEN_BG, GREEN, r=2.2, z=3)
                    txt(x, y0 + 6.8 + up, s, GREEN, 12.5)

        # 8 trays of two chips
        cx = [18 + 23.4 * j for j in range(8)]
        for j, x in enumerate(cx):
            box(x - 10, 43, 20, 43, YELLOW_BG, ORANGE, lw=1.3, z=2)
            txt(x - 7.6, 64.5, f"tray {j + 1}", ORANGE, 12, rotation=90)
            for c, y in ((2 * j, 73), (2 * j + 1, 47)):
                box(x - 5.5, y, 13, 9, PURPLE_BG, PURPLE, r=1.8, z=4)
                txt(x + 1, y + 4.5, f"Trn3 {c}", PURPLE, 14.5)
            for dx in (-1.2, 3.2):   # the pair link, on the tray
                ax.plot([x + dx, x + dx], [56, 73], color=GREEN, lw=2.2, zorder=3)
            # to the plane: one stub per chip; every chip reaches all 8 switches of its plane
            for dx in (-1.2, 3.2):   # 2 ports per chip to its plane (runtime debug log)
                ax.plot([x + dx, x + dx], [82, 96], color=BLUE, lw=2.2, zorder=3)
                ax.plot([x + dx, x + dx], [47, 34], color=BLUE, lw=2.2, zorder=3)
            # cross-rack cable: leaves at the tray edge and passes behind plane B
            for y, xc in ((77.5, 12.5), (51.5, 11.2)):   # one cable per chip
                ax.plot([x + xc, x + xc], [y, 9], color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1.5)
                ax.plot([x + 7.5, x + xc], [y, y], color=RED, lw=1.2, ls=(0, (4, 3)), zorder=5)
        txt(100, 6.5, f"One cable per chip to the same chip index in the other rack: {xrack} (AECs: from the article)", RED, 13)

        # the three measured cases, as callouts
        def call(x, y, w, head, body, c):
            box(x, y, w, 13, "white", c, lw=1.5, r=1.4, z=7)
            txt(x + w / 2, y + 9.9, head, c, 13.5, z=8)
            txt(x + w / 2, y + 4.3, body, INK, 12, z=8, linespacing=1.4)

        call(6, -10, 52, f"same tray (0-1): {kc(pcb)} cycles, {g['1']} GB/s", f"direct link (green); permute {kc(p_pcb)}", GREEN)
        call(61, -10, 64, f"same plane (0-2, 0-14, ...): {kc(hop1)} cycles, {g['2']} GB/s",
             f"through the switch plane (blue); permute {kc(p_hop1)}\nto the rack's other node: {xn[1]} GB/s", BLUE)
        call(128, -10, 67, f"other plane, other tray (0-3, 1-2): {kc(hop2)} cycles, {g['3']} GB/s",
             "two hops: switch, then tray link\nmesh only (mesh is 112 on any switched pair)", RED)
        txt(100, -14.5, f"Tray link and plane at once: {st['bothLinksGbps']} GB/s per chip", INK, 14)

        txt(100, -22, "Cycles per 512 B exchange at 1.2 GHz, HBM all_gather, same core index on both chips (median over pairs). "
            "GB/s per chip, one direction, 64 MiB per core (ring; two hops on mesh).\n"
            "Each chip: 2 ports to its tray partner, 2 to its plane. Connectivity as neuron-ls lists it and as measured; "
            "tray and switch counts from the article (NL32x2 Switched). Tray order not known.",
            "#495057", 11.5)
        fig.savefig(HERE / "figures" / "chip-topology-sketch.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote figures/chip-topology-sketch.png")


if __name__ == "__main__":
    main()
