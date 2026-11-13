"""The UltraServer as two racks of trays, hand-drawn, with the measured chip-to-chip costs on the links.

  python3 c2c/figure_rack_sketch.py     # writes figures/rack-topology-sketch.png

Physical layout (trays, copper backplane, switch trays, cross-rack cables) is the SemiAnalysis Trainium3 article's
NL32x2 Switched drawing; none of it is visible from the box. From the box: 4 nodes x 16 chips, chips in pairs, a
partner link, and one switch plane per chip parity that reaches both nodes of its rack (neuron-ls, and nccom-test
timings). The rack split is read from the driver, not measured.
Numbers: results/core-nccom.json (HBM all_gather, nccom-test p50, same core index), as in figures/node-allgather.png.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
from figure_scale import LARGE, LNAME, SMALL, efa_range, krange, lab, load, summary  # noqa: E402

GREEN, CYAN, ORANGE, PURPLE, RED, BLUE, INK = "#2F9E44", "#1098AD", "#F08C00", "#7048E8", "#E03131", "#1971C2", "#1E1E1E"
GREY = "#868E96"
EXCALIFONT = HERE / "common" / "fonts" / "Excalifont-Regular.ttf"   # Excalidraw's hand font (OFL-1.1), Latin subset
FONT = ["Excalifont", "Chalkboard SE", "Chalkboard", "Comic Sans MS", "Marker Felt"]
W, D = 46, 30                  # tray: width, depth
TH, PITCH = 3.0, 3.0           # tray height, and tray-to-tray spacing in a stack
KX, KY = .5, .34               # how depth shows on the page
# stacks, bottom to top: (base height, trays, kind)
STACKS = ((2, 8, "c"), (46, 4, "s"), (78, 8, "c"))
TOP = 127                      # top of the backplane
CAGE_X = (7, 35)               # cable cages along the tray front
OX1 = 104                      # where rack 1 starts
RACK_NODES = {0: [0, 1], 1: [3, 2]}   # driver: node id = server id in rack 0, 3 - server id in rack 1
_drawn = {}
GAP = (W + KX * D + OX1) / 2   # middle of the space between the racks
NODE_DZ = 7                      # node label height above the stack base (below the hanging cables)
SAG, SAG_X = 12, 9             # how far the rack-to-rack cables hang, and how fast they turn
PAIRS = {1: "0-1, 2-3", 2: "0-2, 1-3", 3: "0-3, 1-2"}   # which nodes a node mask pairs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
    from matplotlib.path import Path as MPath

    if EXCALIFONT.exists():
        font_manager.fontManager.addfont(str(EXCALIFONT))
    cyc = load()
    IN = summary(cyc, {0})
    permask = {nm: summary(cyc, {nm}).get("plane") for nm in (1, 2, 3)}

    def both(s):   # print only what was measured
        if not s or SMALL not in s:
            return "not measured"
        return f"{lab(s)} cycles" + (f", {lab(s, LARGE)}" if LARGE in s else "")

    with plt.xkcd(scale=.5, length=120, randomness=2):
        plt.rcParams.update({"path.effects": [], "font.family": FONT})
        fig, ax = plt.subplots(figsize=(17.7, 11.0))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(-79, 183)
        ax.set_ylim(-20, 143)
        ax.axis("off")

        def txt(x, y, s, c=INK, fs=13, **k):
            ax.text(x, y, s, color=c, fontsize=fs, ha=k.pop("ha", "center"), va=k.pop("va", "center"), zorder=k.pop("z", 9), **k)

        def P(ox, x, y, z):
            return ox + x + KX * y, z + KY * y

        def face(ox, pts, ec, fc="none", lw=1.6, z=3):
            ax.add_patch(Polygon([P(ox, *p) for p in pts], closed=True, fc=fc, ec=ec, lw=lw, zorder=z))

        def tray(ox, z, ec, fc, side, top=None):
            face(ox, [(0, 0, z), (W, 0, z), (W, 0, z + TH), (0, 0, z + TH)], ec, fc, 1.3)                  # front
            face(ox, [(W, 0, z), (W, D, z), (W, D, z + TH), (W, 0, z + TH)], ec, side, 1.3)                # side
            if top:
                face(ox, [(0, 0, z + TH), (W, 0, z + TH), (W, D, z + TH), (0, D, z + TH)], ec, top, 1.6)   # lid
            for x0 in (2.2, W - 5.2):                                                                      # handles
                ax.plot([ox + x0, ox + x0 + 3], [z + TH / 2] * 2, color=ec, lw=2.2, zorder=4, solid_capstyle="round")

        def pad(ox, x0, y0, w, d, z, ec, fc, zo=5, lw=1.5):
            face(ox, [(x0, y0, z), (x0 + w, y0, z), (x0 + w, y0 + d, z), (x0, y0 + d, z)], ec, fc, lw, zo)
            return P(ox, x0 + w / 2, y0 + d / 2, z)

        def dots(p, q, c, lw=1.6, z=6):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, ls=(0, (1.2, 2.2)), zorder=z)

        def bez(pts, t):
            a, b, c, d = pts
            f = lambda i: (1 - t) ** 3 * a[i] + 3 * (1 - t) ** 2 * t * b[i] + 3 * (1 - t) * t * t * c[i] + t ** 3 * d[i]  # noqa: E731
            return f(0), f(1)

        link, swpad, bpline, cables, chips = {}, {}, {}, [], {}
        for r, ox in enumerate((0, OX1)):
            # rack frame and the copper backplane, upright at the back
            for x, y in ((0, 0), (W, 0), (W, D)):
                ax.plot(*zip(P(ox, x, y, 0), P(ox, x, y, STACKS[2][0] + 7 * PITCH + TH)), color=GREY, lw=1.3, zorder=1.2)
            bx, by = P(ox, 17, D, 0)
            ax.add_patch(FancyBboxPatch((bx, by - 8), 21, TOP - by + 8, boxstyle="round,pad=0,rounding_size=1.5", fc="#FFF1D6",
                                        ec=ORANGE, lw=1.8, zorder=1))
            txt(bx + 10.5, TOP - 4.5, "copper\nbackplane", ORANGE, 14)
            conn_x = [P(ox, x0 + 3.5, 25, 0)[0] for x0 in (18, 29)]
            for x in conn_x:   # the backplane runs, seen in the gaps between the stacks
                dots((x, P(ox, 0, 25, STACKS[0][0])[1]), (x, P(ox, 0, 25, STACKS[2][0] + 8 * PITCH)[1]), GREEN, z=2)
            bpline[r] = conn_x

            for base, n, kind in STACKS:
                ec, fc, side, lid = (GREEN, "#EAF7EC", "#D3EFD8", "#F6FCF7") if kind == "c" else (CYAN, "#E3F6F9", "#C9EEF3", "#F1FBFC")
                for k in range(n):
                    tray(ox, base + k * PITCH, ec, fc, side, lid if k == n - 1 else None)
                    if kind == "c":   # two cable cages per tray, one per chip (article: OSFP-XD; place on the tray not stated)
                        for x0 in CAGE_X:
                            z0 = base + k * PITCH
                            live = k == n - 1
                            face(ox, [(x0, 0, z0 + .7), (x0 + 4, 0, z0 + .7), (x0 + 4, 0, z0 + 2.3), (x0, 0, z0 + 2.3)],
                                 RED if live else "#F5A3A3", "#FFA8A8" if live else "#FFF0F0", 1.3 if live else 1.0, 4.5)
                top = base + (n - 1) * PITCH + TH
                if kind == "c":   # one backplane connector per chip
                    conn = [pad(ox, x0, 23, 7, 4, top, PURPLE, "#E5DBFF") for x0 in (18, 29)]
                    cs = [pad(ox, x0, 4, 8, 8, top, BLUE, "#D0EBFF") for x0 in (5, 33)]
                    a, b = (cs[0][0] + 4.6, cs[0][1]), (cs[1][0] - 4.6, cs[1][1])
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=ORANGE, lw=3.2, zorder=6, solid_capstyle="round")   # tray link
                    for c, k in zip(cs, conn):
                        dots((c[0], c[1] + 1.6), k, GREEN)
                    link[(r, base)] = ((a[0] + b[0]) / 2, a[1])
                    # the cables plug into the cages in the tray's front opening (top tray drawn)
                    chips[(r, base)] = [P(ox, x0 + 2, 0, top - TH / 2) for x0 in CAGE_X]
                    k = _drawn.get(r, 0); _drawn[r] = k + 1
                    txt(ox + W + KX * D + 2.5, base + NODE_DZ, f"node {RACK_NODES[r][k]}", GREEN, 19, ha="left")
                else:
                    # 16 switches in 4 trays = 4 per tray (the article gives the totals, not the per-tray count);
                    # connectors per switch tray are not given, so one strip stands for them
                    pad(ox, 17, 23, 20, 4, top, PURPLE, "#E5DBFF")
                    sw = [pad(ox, x0, 6, 7.5, 7, top, CYAN, "#C5F6FA") for x0 in (3.5, 14, 24.5, 35)]
                    for j, s in enumerate(sw):
                        dots((s[0], s[1] + 1.4), P(ox, 20 + 4.7 * j, 23.4, top), CYAN)
                    swpad[r] = P(ox, 3.5, 9.5, top)
                    if r == 0:   # centred in the gap: it describes both racks
                        txt(GAP, base + 9, "per rack:\n4 switch trays,\ntwo planes: one for\neven chips, one for odd", CYAN, 15, linespacing=1.25)
            txt(ox + 30, -6, f"Rack {r}: nodes {RACK_NODES[r][0]} and {RACK_NODES[r][1]}", INK, 18)

        # what those cables connect: each chip to the chip with the same index in the same server slot of the other
        # rack, one link per chip and no switch (driver: inter-rack p2p; EFA byte counters, nccom/xpair.sh)
        # They hang in front of the rack, so they never pass over a lid; the two of a tray cross once, in the gap.
        for base in (STACKS[0][0], STACKS[2][0]):
            for j in (0, 1):
                a, b = chips[(0, base)][j], chips[(1, base)][j]
                near, far = (SAG_X, 104 - 3.2 * SAG_X)[::1 if j == 0 else -1]   # lowest point early for one, late for the other
                pts = [a, (a[0] + near, a[1] - SAG * 1.25), (b[0] - far, b[1] - SAG * 1.25), b]
                cables.append(pts)
                ax.add_patch(PathPatch(MPath(pts, [MPath.MOVETO] + [MPath.CURVE4] * 3), fc="none", ec="white", lw=5.5, zorder=6.5,
                                       capstyle="butt"))
                ax.add_patch(PathPatch(MPath(pts, [MPath.MOVETO] + [MPath.CURVE4] * 3), fc="none", ec=RED, lw=2.2, zorder=7))
                for q in (a, b):
                    ax.plot(*q, "o", color=RED, ms=5.5, zorder=7.5)
        txt(GAP, STACKS[2][0] + 7 * PITCH + TH + 10, "one PCIe cable (AEC)\nper chip: NeuronLink,\nchip to chip\n(top tray's drawn)", RED, 15, linespacing=1.25)

        # callouts: what each link is, and what an exchange over it costs; every arrow ends on the thing it names
        def call(y, h, head, lines, c, start_y, target, rad=0, ac=None):
            ax.add_patch(FancyBboxPatch((-76, y), 60, h, boxstyle="round,pad=0,rounding_size=1.6", fc=c, ec=c, lw=1.5, zorder=8))
            txt(-46, y + h - 4.6, head, "white", 19)
            txt(-46, y + (h - 8) / 2 + .5, lines, "white", 15, linespacing=1.35)
            ax.annotate("", xy=target, xytext=(-16, start_y), zorder=10,
                        arrowprops=dict(arrowstyle="-|>", color=ac or c, lw=2, mutation_scale=18, shrinkA=0, shrinkB=1,
                                        connectionstyle=f"arc3,rad={rad}"))

        def pick(nm, same):   # median over chip masks: the same chip index, or every other even (same-plane) one
            v = {}
            for (n_, cm, tile), c in cyc.items():
                if n_ == nm and cm % 2 == 0 and (cm == 0) == same:
                    v.setdefault(tile, []).append(c)
            return {k: sorted(x)[len(x) // 2] for k, x in v.items()}

        cx, cy = cables[2][0][0] - 1.0, cables[2][0][1] - .1
        efa = efa_range(cyc)   # mesh on nodes 0-2 / 1-3 and ring on nodes 0-3 / 1-2 off the same chip index
        call(94, 29, "Rack to rack cable", f"same chip index (nodes {PAIRS[3]}):\n{both(pick(3, True))}\n"
             f"other pairs, stock runtime over EFA:\nabout {krange(*efa[0])} cycles, {efa[1][0]:.0f}-{efa[1][1]:.0f} GB/s", RED, 96.8, (cx, cy), ac=INK)
        gy = (P(0, 0, D, STACKS[1][0] + 3 * PITCH + TH)[1] + STACKS[2][0]) / 2
        call(gy - 16.5, 33, "Backplane to switch trays", f"same plane, same node:\n{both(IN.get('plane'))}\n"
             f"same plane, other node: {lab(permask[1], LARGE)}\n"
             f"same node, other plane, not the tray\npartner (two hops, mesh): {lab(IN.get('other'), LARGE)}", GREEN, gy, (bpline[0][0], gy))
        lx, ly = link[(0, STACKS[0][0])]
        call(27, 20, "Tray link", f"the two chips of a tray:\n{both(IN.get('partner'))}", ORANGE, 39, (lx, ly + .5), rad=-.15)
        both_links = json.loads((HERE / "results" / "chain-steady.json").read_text())["bothLinksGbps"]
        ax.add_patch(FancyBboxPatch((-76, 4), 60, 16, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec=INK, lw=1.8, zorder=8))
        txt(-46, 12, f"Tray link and plane at once:\n{both_links} GB/s per chip", INK, 16, linespacing=1.35)

        mid = (-76 + OX1 + W + KX * D + 16) / 2
        txt(mid, 136, "Trainium3 UltraServer: 4 nodes, 64 chips, two racks", INK, 26)
        txt(mid, -15, f"Each node: 8 trays of 2 chips.  Cycles: a 512 B exchange at 1.2 GHz.  GB/s: per chip, one direction, {LNAME} per core, ring algorithm.", "#495057", 14)
        txt(mid, -19.5, "Layout after the SemiAnalysis Trainium3 article; rack split from the driver; costs measured.", GREY, 13)
        fig.savefig(HERE / "figures" / "rack-topology-sketch.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote figures/rack-topology-sketch.png")


if __name__ == "__main__":
    main()
