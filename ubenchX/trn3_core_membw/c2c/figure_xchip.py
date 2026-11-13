"""Core to core ACROSS chips on a Trainium3 node: the cross-chip counterpart of figures/c2c-stock/core-c2c.png.

  python3 p2p/analyze_xchip.py && python3 c2c/figure_xchip.py

figures/c2c-stock/core-xchip.png: one 8 x 8 matrix per chip distance (chip c to c^1, c^2, c^3). Row = core on the lower
  chip, column = core on the other chip. Each cell is the cheapest method that runs on that pair, named in the
  cell: all_gather (HBM), collective_permute (HBM), collective_permute_implicit (SBUF), all_gather (SBUF).
figures/c2c-stock/chip-c2c.png: the 16 x 16 chip matrix of a node, same core on both chips. Above the diagonal HBM
  all_gather, below HBM collective_permute.
Values: results/core-xchip.json (p2p/xchip.sh), cycles per 512 B exchange at 1.2 GHz, mean of the pair's two ranks.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
from figure_c2c import CLK_GHZ, kc  # noqa: E402

NAME = {"ag_hbm": "all_gather HBM", "cp_hbm": "permute HBM", "cpi_sbuf": "permute SBUF", "ag_sbuf": "all_gather SBUF"}
RED = "#C0392B"
# chip-index XOR offset -> (axis label, panel title); see figures/chip-topology.png for the links
LABEL = {1: ("the partner chip", "Tray partners (chips 0-1)"),
         2: ("the other chip", "Same switch plane (chips 0-2, 1-3)"),
         3: ("the other chip", "Other switch plane, not partners (chips 0-3, 1-2)")}


def load():
    doc = json.loads((HERE / "results" / "core-xchip.json").read_text())
    runs = doc["runs"]
    cell = defaultdict(lambda: defaultdict(list))  # (cm, kernel) -> (coreA, coreB) -> [cycles]
    chip = defaultdict(lambda: defaultdict(list))  # kernel -> (chipA, chipB) -> [cycles], same core
    why = {}
    for r in runs:
        if not r["ok"]:
            why[(r["chipMask"], r["coreMask"], r["kernel"])] = r["reason"]
            continue
        for p in r["pairs"]:
            cell[(r["chipMask"], r["kernel"])][(p["coreA"], p["coreB"])].append(p["cycles"])
    # the chip matrix: an all-16-chip launch (`xchip.sh full`) replaces the partial launch of the same offset
    full = {(r["kernel"], r["chipMask"]): r for r in doc.get("fullRuns", [])}
    chip, nopermute = defaultdict(lambda: defaultdict(list)), set()
    for r in runs:
        if r["coreMask"]:
            continue
        r = full.get((r["kernel"], r["chipMask"]), r)
        n = 16
        while n <= 8 * r["chipMask"]:
            n *= 2
        n = 16 if (r["kernel"], r["chipMask"]) in full else n // 8   # chips in the launch
        if r["ok"]:
            for p in r["pairs"]:
                chip[r["kernel"]][(p["chipA"], p["chipB"])].append(p["cycles"])
        elif r["kernel"] == "cp_hbm" and r["reason"]:
            nopermute |= {(c, c ^ r["chipMask"]) for c in range(n)}
    return runs, cell, chip, why, nopermute, full


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
    vir = plt.get_cmap("viridis")
    cmap = LinearSegmentedColormap.from_list("c2c", [vir(x) for x in np.linspace(.85, .02, 64)])
    runs, cell, chip, why, nopermute, full = load()
    ok = {(r["chipMask"], r["coreMask"], r["kernel"]): r for r in runs if r["ok"]}
    norm = Normalize(0, 19200)  # the colour scale of figures/c2c-stock/core-c2c.png (its two-hop relay is 19.2k)

    # ---------------- core x core, three chip distances
    cms = [cm for cm in (1, 2, 3) if any(k[0] == cm for k in cell)]
    fig, axes = plt.subplots(1, len(cms), figsize=(8.6 * len(cms) + 1.2, 10.4), squeeze=False)
    for ax, cm in zip(axes[0], cms):
        show = np.full((8, 8), np.nan)
        best = {}
        for a in range(8):
            for b in range(8):
                cand = {k: float(np.median(cell[(cm, k)][(a, b)])) for k in NAME if (a, b) in cell.get((cm, k), {})}
                if cand:
                    best[(a, b)] = min(cand, key=cand.get)
                    show[a, b] = cand[best[(a, b)]]
        im = ax.imshow(show, cmap=cmap, norm=norm)
        for a in range(8):
            for b in range(8):
                if np.isfinite(show[a, b]):
                    tc = "white" if norm(show[a, b]) > .45 else INK
                    ax.text(b, a - .12, kc(show[a, b]), ha="center", va="center", fontsize=13, color=tc, fontweight="bold")
                    ax.text(b, a + .24, NAME[best[(a, b)]], ha="center", va="center", fontsize=7.5, color=tc)
                else:
                    ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fc="#EEF1F3", ec="none"))
                    if not any((cm, a ^ b, k) in why for k in NAME):
                        continue  # not measured
                    ax.add_patch(Rectangle((b - .46, a - .46), .92, .92, fill=False, ec=RED, lw=1.2, ls=":"))
                    ax.text(b, a, "does not\nrun", ha="center", va="center", fontsize=9, color=RED)
        for t, lw in ((1.5, 1.2), (3.5, 3.5), (5.5, 1.2)):
            ax.axhline(t, color="white", lw=lw)
            ax.axvline(t, color="white", lw=lw)
        ax.set_xticks(range(8), [f"NC{i}" for i in range(8)])
        ax.set_yticks(range(8), [f"NC{i}" for i in range(8)])
        ax.tick_params(length=0, labelsize=13, top=True, labeltop=True, bottom=False, labelbottom=False)
        other, title = LABEL[cm]
        ax.set_xlabel(f"core on {other}", fontsize=13, color=INK, labelpad=8)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("core on the lower-numbered chip", fontsize=13, color=INK)
        ag = ok.get((cm, 0, "ag_hbm"))
        ax.set_title(title + (f": same core {kc(ag['cyclesMedian'])}" if ag else ""), fontsize=15,
                     fontweight="bold", color=INK, loc="left", pad=44)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=.018, pad=.015)
    cb.set_label(f"cycles per exchange (512 B, {CLK_GHZ} GHz NeuronCore clock)", fontsize=12)
    fig.suptitle("Trainium3 core to core across chips inside a node: the same core index is cheap, any other core costs a permute or does not run",
                 x=.06, y=1.06, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(.06, 1.005, "Each cell: the cheapest method that runs on that pair, named in the cell (HBM all_gather, HBM collective_permute, "
             "SBUF collective_permute_implicit, SBUF all_gather). Cycles per 512 B exchange; dependent chains of 1/9/33 two-way exchanges,\n"
             "every core of the chips in a launch as a rank, cost = slope against chain length, mean of the pair's two ranks. Outputs "
             "checked against distinct per-rank inputs. Red: every method fails at setup (the runtime's refusal, not a wiring\nmeasurement). "
             "Other-plane same-core cells are timed on the instruction queue span (no packets on the collectives queue). "
             "All three panels: node 0.\nColour scale close to "
             "figures/c2c-stock/core-c2c.png.", fontsize=11, color=MUTED, va="top")
    fig.savefig(HERE / "figures" / "c2c-stock" / "core-xchip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/c2c-stock/core-xchip.png")

    # ---------------- chip x chip, same core
    show = np.full((16, 16), np.nan)
    for (a, b), v in chip["ag_hbm"].items():
        show[min(a, b), max(a, b)] = np.median(v)   # above the diagonal: all_gather
    for (a, b), v in chip["cp_hbm"].items():
        show[max(a, b), min(a, b)] = np.median(v)   # below: permute
    fig, ax = plt.subplots(figsize=(15.5, 14.2))
    im = ax.imshow(show, cmap=cmap, norm=norm)
    for a in range(16):
        for b in range(16):
            if np.isfinite(show[a, b]):
                ax.text(b, a, kc(show[a, b]), ha="center", va="center", fontsize=10.5, fontweight="bold",
                        color="white" if norm(show[a, b]) > .45 else INK)
            else:
                ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fc="#EEF1F3", ec="none"))
                if a > b and (a, b) in nopermute:
                    ax.text(b, a, "no", ha="center", va="center", fontsize=9.5, color=RED)
    for t in range(1, 16):
        ax.axhline(t - .5, color="white", lw=2.6 if t % 2 == 0 else .8)
        ax.axvline(t - .5, color="white", lw=2.6 if t % 2 == 0 else .8)
    ax.plot([-.5, 15.5], [-.5, 15.5], color=INK, lw=.8, alpha=.5)
    ax.set_xticks(range(16), [str(i) for i in range(16)])
    ax.set_yticks(range(16), [str(i) for i in range(16)])
    ax.tick_params(length=0, labelsize=13)
    ax.set_xlabel("chip", fontsize=13, color=INK)
    ax.set_ylabel("chip", fontsize=13, color=INK)
    ax.set_title("Above the diagonal: HBM all_gather. Below: HBM collective_permute (\"no\" = setup fails).\n"
                 "Same core index on both chips; median over the pair's 8 cores.", fontsize=12, color=INK, loc="left", pad=10)
    cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.set_label(f"cycles per exchange (512 B, {CLK_GHZ} GHz NeuronCore clock)", fontsize=12)
    # the runs the cells are drawn from: the all-16-chip launch where there is one (as load() does for the matrix)
    drawn = {m: full.get(("ag_hbm", m), ok.get((m, 0, "ag_hbm"))) for m in range(1, 16)}
    g = lambda ms: kc(float(np.median([drawn[m]["cyclesMedian"] for m in ms if drawn[m] and drawn[m]["ok"]])))  # noqa: E731
    fig.suptitle(f"Trainium3 chip to chip inside a node (all_gather): tray partner {g([1])}, same switch plane {g(range(2, 16, 2))}, "
                 f"other plane {g(range(3, 16, 2))}", x=.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(.02, .945, "Dependent chains of 1/9/33 two-way 512 B exchanges between the same core of two chips; each launch pairs every chip c "
             "with chip c XOR one fixed offset, all 16 chips (128 ranks) at once.\nCost = slope against "
             "chain length. Other-plane pairs (opposite parity, not partners) put nothing on the collectives queue and are timed on\n"
             "the instruction queue span; the permute refuses them. all_gather: node 0; permute: node 1. "
             "Links: figures/chip-topology.png.", fontsize=10, color=MUTED, va="top")
    fig.subplots_adjust(left=.06, right=.93, top=.84, bottom=.05)
    fig.savefig(HERE / "figures" / "c2c-stock" / "chip-c2c.png", dpi=150)
    plt.close(fig)
    print("wrote figures/c2c-stock/chip-c2c.png")


if __name__ == "__main__":
    main()
