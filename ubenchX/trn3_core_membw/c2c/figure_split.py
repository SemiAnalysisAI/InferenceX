"""Three single-protocol 8 x 8 core-to-core matrices for a Trainium3 chip, 512 B per exchange, cycles at 1.2 GHz.

  python3 c2c/figure_split.py

figures/c2c-stock/core-sendrecv.png: nisa.sendrecv only. It compiles only for the two cores of an LNC pair, so 4 pairs
  are filled. Above the diagonal dma_engine=gpsimd_dma, below the default DMA engine. Per pair, chip 0
  (sendrecv/pairs.sh). Each step is a two-way exchange, so one cell covers both directions.
figures/c2c-stock/core-permute.png: the permutes only, on every pair the runtime routes. Above the diagonal HBM
  collective_permute, below SBUF collective_permute_implicit (results/core-p2p.json, p2p/sweep.sh). One
  measurement per route class (XOR mask of the two core ids); its 4 pairs run at once.
figures/c2c-stock/core-allgather.png: nki.collectives.all_gather only. Above the diagonal HBM (20 direct pairs, and the
  8 unrouted pairs as a two-hop relay of two all_gathers); below SBUF, which runs on the LNC pair and i+4 only
  (results/core-c2c-twohop.json, results/core-c2c.json). One measurement per route class.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
from figure_c2c import CLK_GHZ, SR_PAIR, SR_PAIR_DMA, ci, kc, load, shim  # noqa: E402

ROUTE = {1: "m1", 2: "m2", 3: "m3", 4: "m4", 5: "m5"}  # XOR mask -> p2p route; masks 6 and 7 have no route


def matrix(ax, plt, show, fmt, cmap, norm, INK, sub=None, noroute=None, blank=None):
    from matplotlib.patches import Rectangle
    RED = "#C0392B"
    im = ax.imshow(show, cmap=cmap, norm=norm)
    for a in range(8):
        for b in range(8):
            if np.isfinite(show[a, b]):
                tc = "white" if norm(show[a, b]) > .45 else INK
                st = sub(a, b) if sub else None
                ax.text(b, a - (.12 if st else 0), fmt(show[a, b]), ha="center", va="center", fontsize=13, color=tc,
                        fontweight="bold")
                if st:
                    ax.text(b, a + .24, st, ha="center", va="center", fontsize=8.5, color=tc)
                    if st.startswith("via"):  # two-hop relay
                        ax.add_patch(Rectangle((b - .46, a - .46), .92, .92, fill=False, ec=RED, lw=1.8))
                continue
            ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fc="#EEF1F3", ec="none"))
            if a != b and noroute is not None and noroute(a, b):
                ax.add_patch(Rectangle((b - .46, a - .46), .92, .92, fill=False, ec=RED, lw=1.2, ls=":"))
                ax.text(b, a, "stock:\nrefused", ha="center", va="center", fontsize=9.5, color=RED)
            elif a != b and blank is not None:
                ax.text(b, a, blank, ha="center", va="center", fontsize=9.5, color="#5E707A")
    for t, lw in ((1.5, 1.2), (3.5, 3.5), (5.5, 1.2)):
        ax.axhline(t, color="white", lw=lw)
        ax.axvline(t, color="white", lw=lw)
    ax.plot([-.5, 7.5], [-.5, 7.5], color=INK, lw=.8, alpha=.5)
    ax.set_xticks(range(8), [f"NC{i}" for i in range(8)])
    ax.set_yticks(range(8), [f"NC{i}" for i in range(8)])
    ax.tick_params(length=0, labelsize=14)
    for t in (1.5, 5.5):
        ax.text(-1.4, t, "die 1 (driver)" if t < 4 else "die 0 (driver)", ha="center", va="center", fontsize=12.5, color=INK,
                rotation=90, clip_on=False, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.set_label(f"cycles per exchange (512 B, {CLK_GHZ} GHz NeuronCore clock)", fontsize=12)
    cb.ax.tick_params(labelsize=11)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    sys.path.insert(0, str(HERE / "common"))
    try:
        from ubench_style import configure, INK, MUTED
        configure()
    except Exception:  # noqa: BLE001
        INK, MUTED = "#172F3B", "#5E707A"
    vir = plt.get_cmap("viridis")
    cmap = LinearSegmentedColormap.from_list("c2c", [vir(x) for x in np.linspace(.85, .02, 64)])
    (HERE / "figures" / "c2c-stock").mkdir(parents=True, exist_ok=True)

    # ---------------- sendrecv only
    show = np.full((8, 8), np.nan)
    for (a, b), v in SR_PAIR.items():
        show[a, b] = v  # above the diagonal: gpsimd_dma
    for (a, b), v in SR_PAIR_DMA.items():
        show[b, a] = v  # below: default DMA engine
    g, d = np.median(list(SR_PAIR.values())), np.median(list(SR_PAIR_DMA.values()))
    fig, ax = plt.subplots(figsize=(11.5, 10.6))
    matrix(ax, plt, show, ci, cmap, Normalize(0, 2600), INK, sub=lambda a, b: "gpsimd_dma" if a < b else "default DMA")
    ax.set_title("Above the diagonal: dma_engine=gpsimd_dma. Below: the default DMA engine.\n"
                 "Grey: stock sendrecv needs both cores in one LNC=2 logical core. With a core-map shim the default engine\n"
                 "crosses SEngines (NC0 to each of the other 7 cores of one chip, figures/core-sendrecv-xseng.png); GPSIMD does not.",
                 fontsize=12, color=INK, loc="left", pad=10)
    fig.suptitle(f"Trainium3 nisa.sendrecv, stock runtime: {ci(g)} cycles per exchange inside an LNC pair",
                 x=.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(.02, .935, "Chain of nisa.sendrecv exchanges between the two cores of an LNC pair, SBUF to SBUF; cost = slope of "
             "whole-NEFF time against chain length (1, 4, 16, 64).\nEach step is two-way: both cores send and receive, so a "
             "cell covers both directions. Measured per pair on chip 0. gpsimd_dma takes <= 1 KiB per partition\n"
             f"(128 KiB tiles); the default DMA engine ({ci(d)}) carries larger tiles. Outputs checked in the NKI "
             "simulator (NC0-NC1 later also on the device).", fontsize=10, color=MUTED, va="top")
    fig.subplots_adjust(left=.1, right=.93, top=.8, bottom=.05)
    fig.savefig(HERE / "figures" / "c2c-stock" / "core-sendrecv.png", dpi=150)
    plt.close(fig)
    print("wrote figures/c2c-stock/core-sendrecv.png")

    # ---------------- permutes only
    xcoll = shim()[1]  # shim runs on two diagonal pairs (results/core-xcoll.json): kernel -> (min, max) cycles
    xp = [x for k in ("cp_hbm", "cpi_sbuf") for x in xcoll[k]]
    rows = json.loads((HERE / "results" / "core-p2p.json").read_text())["rows"]
    cyc = {(r["method"], r["route"]): r["cycles"] for r in rows if r["tile"] == 512}
    show = np.full((8, 8), np.nan)
    for a in range(8):
        for b in range(8):
            if a != b and (a ^ b) in ROUTE:
                show[a, b] = cyc[("cp_hbm" if a < b else "cpi_sbuf", ROUTE[a ^ b])]
    v = show[np.isfinite(show)]
    fig, ax = plt.subplots(figsize=(11.5, 10.6))
    matrix(ax, plt, show, kc, cmap, Normalize(v.min() - 3000, v.max() + 500), INK, noroute=lambda a, b: (a ^ b) in (6, 7))
    ax.set_title("Above the diagonal: HBM collective_permute. Below: SBUF collective_permute_implicit.\n"
                 "One measurement per route class (its 4 pairs run at once), not one per pair.",
                 fontsize=12, color=INK, loc="left", pad=10)
    c = lambda m: kc(cyc[("cp_hbm", m)])  # noqa: E731
    fig.suptitle(f"Trainium3 core to core with one protocol (permute), stock runtime: LNC pair {c('m1')}, same die {c('m2')}, "
                 f"other die {c('m4')}", x=.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(.02, .935, "Dependent chains of 1/9/33 two-way exchanges with all 8 cores of a chip as ranks (4 pairs at once); "
             "cost = slope of the least-waiting rank's in-trace collectives\nspan against chain length. Outputs checked "
             "against distinct per-rank inputs. Cells are full step cost, mostly handshake waits (data packets < 0.6 us on the\n"
             "LNC and same-die routes). Refused: a stock-runtime limit (no_hier no_mesh) on {0,1}-{6,7} and {2,3}-{4,5}; with a "
             f"core-map shim both permutes ran on two\nof these pairs at {kc(min(xp))}-{kc(max(xp))}. "
             "Dies from the Neuron driver source.", fontsize=10, color=MUTED, va="top")
    fig.subplots_adjust(left=.1, right=.93, top=.8, bottom=.05)
    fig.savefig(HERE / "figures" / "c2c-stock" / "core-permute.png", dpi=150)
    plt.close(fig)
    print("wrote figures/c2c-stock/core-permute.png")

    # ---------------- all_gather only
    hbm, sbuf, via, cls, two, stalls, chips = load()
    xag = xcoll["ag_hbm"]
    show = np.full((8, 8), np.nan)
    for a in range(8):
        for b in range(8):
            if a != b:
                show[a, b] = hbm[a, b] if a < b else sbuf[a, b]
    v = show[np.isfinite(show)]
    fig, ax = plt.subplots(figsize=(11.5, 10.6))
    matrix(ax, plt, show, kc, cmap, Normalize(v.min() - 3500, v.max() + 500), INK,
           sub=lambda a, b: f"via {via[(a, b)]}" if a < b and (a, b) in via else None,
           noroute=lambda a, b: (a ^ b) in (6, 7), blank="HBM\nonly")
    ax.set_title("Above the diagonal: HBM all_gather; red box = pair the stock runtime refuses (no diagonal link), shown as "
                 "two all_gathers\nthrough a third core. "
"Below: SBUF all_gather, which runs on the LNC pair and i+4 only. One measurement per route class.",
                 fontsize=12, color=INK, loc="left", pad=10)
    fig.suptitle(f"Trainium3 core to core with all_gather: LNC pair {kc(cls['1'])}, i+4 {kc(cls['4'])}, "
                 f"same die {kc(min(cls['2'], cls['3']))}, other die other core {kc(cls['5'])}",
                 x=.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    rs = [stalls[k] for k in ("4-2", "2-4", "4-3", "3-4") if k in stalls]
    fig.text(.02, .935, "Dependent chains of two-way nki.collectives.all_gather exchanges with all 8 cores of a chip as ranks "
             "(4 pairs at once); cost = slope of the least-waiting rank's\nin-trace collectives span against chain length "
             f"(HBM: median over {chips[0]} chips; SBUF: {chips[1]} chips). Outputs checked against distinct per-rank inputs. "
             "Cells are full step cost.\nall_gather gets its fast handshake on the LNC pair and i+4 only, so i+4 (other die) "
             f"beats the same-die route here. Relays stall: {min(rs):.0%}-{max(rs):.0%} of spans exceed 1.5x the floor.\n"
             "The fast handshake is software, not a wire: NC0-NC4 and NC0-NC5 join the same SEngines. "
             f"With a core-map shim two refused pairs (one chip) ran as one all_gather at {kc(xag[0])}-{kc(xag[1])}.",
             fontsize=10, color=MUTED, va="top")
    fig.subplots_adjust(left=.1, right=.93, top=.8, bottom=.05)
    fig.savefig(HERE / "figures" / "c2c-stock" / "core-allgather.png", dpi=150)
    plt.close(fig)
    print("wrote figures/c2c-stock/core-allgather.png")


if __name__ == "__main__":
    main()
