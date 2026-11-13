"""One figure for core-to-core cost on a Trainium3 chip: the floorplan with every route, and the 8 x 8 matrix.

  python3 c2c/analyze_c2c.py && python3 c2c/analyze_relay.py && python3 c2c/figure_c2c.py

Every value uses one method and one estimator: a chain of nki.collectives.all_gather with all 8 cores of a chip as
ranks, cost per exchange = slope against chain length of the least-waiting rank's in-trace collectives span
(the "floor"). HBM values (all 28 pairs: 20 direct, 8 two-hop relays) come from results/core-c2c-twohop.json;
SBUF values (the 8 pairs SBUF can reach) from the nsFloor field of results/core-c2c.json. 512 B per exchange.
The nccom-test runtime permute is a different quantity (one runtime call per exchange) and is only footnoted.

Writes figures/c2c-stock/core-c2c.png.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CLK_GHZ = 1.2  # profiler Metadata.neuroncore_clock_freq
TWO = {"4-2": (4, 2), "2-4": (2, 4), "4-3": (4, 3), "3-4": (3, 4)}


# nisa.sendrecv per exchange at 512 B (sendrecv/run_sendrecv.py, results/core-p2p.json): the only raw
# core-to-core DMA NKI exposes, and only inside an LNC=2 pair (LNC degree must be 1 or 2; sendrecv needs >= 2).
def _sendrecv():
    try:
        rows = json.loads((HERE / "results" / "core-p2p.json").read_text())["rows"]
        get = lambda m: next(r["cycles"] for r in rows if r["method"] == m and r["tile"] == 512)  # noqa: E731
        return get("sendrecv"), get("sendrecv_gpsimd")
    except (OSError, StopIteration, KeyError):
        return 2255.0, 855.0


SR_DMA, SR_GPSIMD = _sendrecv()


def _sendrecv_pairs(kernel="xchg_gpsimd", fallback=None):
    """{(2k, 2k+1): cycles} from sendrecv/pairs.sh (logical core k of chip 0 at LNC=2 = NeuronCores 2k, 2k+1);
    pairs not measured there take SR_DMA (logical core 20 = chip 5 pair 0)."""
    out = {}
    for k in range(4):
        fs = glob.glob(str(HERE / "data" / "raw" / "*" / "runs" / "sendrecv-pairs" / f"lc{k}" / kernel / "4" /
                           "result.json"))
        r = json.loads(Path(fs[0]).read_text()) if fs else {}
        ok = r.get("ns_per_step") and r.get("sim_ok")
        out[(2 * k, 2 * k + 1)] = r["ns_per_step"] * CLK_GHZ if ok else fallback
    return out


SR_PAIR = _sendrecv_pairs("xchg_gpsimd", SR_GPSIMD)  # dma_engine=gpsimd_dma: the fastest at 512 B
SR_PAIR_DMA = _sendrecv_pairs("xchg_dma", SR_DMA)  # default dma_engine, for the legend
# SBUF collective_permute_implicit reaches every routable pair (p2p sweep); SBUF all_gather only LNC and i+4
CPI = {r["route"]: r["cycles"] for r in json.loads((HERE / "results" / "core-p2p.json").read_text())["rows"]
       if r["method"] == "cpi_sbuf" and r["tile"] == 512}
# HBM permute GB/s at 8 MiB per route (m1-m3 inside a die, m4-m5 across the dies), for the floorplan footnote
CP8 = {r["route"]: r["GBps"] for r in json.loads((HERE / "results" / "core-p2p.json").read_text())["rows"]
       if r["method"] == "cp_hbm" and r["tile"] == 8 << 20}


C_LNC, C_R, C_D2D = "#2EAD6B", "#2F7F8F", "#6A4C93"  # link kinds (wiring), not costs


def shim():
    """Costs reached only with the core-map shim (sendrecv/XSENG.md): nisa.sendrecv (default DMA, 512 B) per route
    kind from results/core-sendrecv-xseng.json, and the range of the collectives on the two diagonal pairs from
    results/core-xcoll.json (per kernel)."""
    rows = [r for r in json.loads((HERE / "results" / "core-sendrecv-xseng.json").read_text())["rows"]
            if r["kernel"] == "xchg_dma" and r["deviceOk"] and r["tileBytes"] == 512]
    by = {}
    for r in rows:
        if r["partner"] not in by or r.get("run") == "xseng-long":
            by[r["partner"]] = r["cycles"]
    sr = {"same": by[1], "R": (by[2] + by[3]) / 2, "D2D": (by[4] + by[5]) / 2, "diag": (by[6] + by[7]) / 2}
    coll = {}
    for run in json.loads((HERE / "results" / "core-xcoll.json").read_text())["runs"]:
        if run["variant"] == "diag" and run["outputsOk"]:
            v = [c for p in run["pairs"] if p["kind"].startswith("diagonal") for c in p["cyclesPerRank"]]
            if v:
                coll[run["kernel"]] = (min(v), max(v))
    return sr, coll


def load():
    rel = json.loads((HERE / "results" / "core-c2c-twohop.json").read_text())
    sb = json.loads((HERE / "results" / "core-c2c.json").read_text())
    acc = defaultdict(list)
    for r in rel["pairs"]:
        acc[(r["a"], r["b"], r["seq"])].append(r["nsPerExchange"] * CLK_GHZ)
    best = {}  # (a, b) -> (cycles, seq): best one- or two-hop sequence, median over chips
    for (a, b, s), v in acc.items():
        c = float(np.median(v))
        if (a, b) not in best or c < best[(a, b)][0]:
            best[(a, b)] = (c, s)
    hbm = np.full((8, 8), np.nan)
    via = {}
    for (a, b), (c, s) in best.items():
        hbm[a, b] = hbm[b, a] = c
        if s in TWO:
            via[(a, b)] = a ^ TWO[s][0]  # first hop of the lower rank: a -> a^m1 -> b
            via[(b, a)] = via[(a, b)]
    sacc = defaultdict(list)
    for r in sb["sbuf"]:
        if r["tileBytes"] == 512:
            sacc[(r["a"], r["b"])].append(r["cyclesFloor"])
    sbuf = np.full((8, 8), np.nan)
    for (a, b), v in sacc.items():
        sbuf[a, b] = sbuf[b, a] = float(np.median(v))
    cls = {s: v["median_ns"] * CLK_GHZ for s, v in rel["bySequence"].items()}
    two = min(c for (c, s) in best.values() if s in TWO), max(c for (c, s) in best.values() if s in TWO)
    two = kc(two[0]) if kc(two[0]) == kc(two[1]) else f"{kc(two[0])}-{kc(two[1])}"
    stalls = {k: v["over_1p5x_floor"] / v["spans"] for k, v in rel["stalls"].items()}
    chips = (len({r["chip"] for r in rel["pairs"]}), len({r["chip"] for r in sb["sbuf"]}))
    return hbm, sbuf, via, cls, two, stalls, chips


def kc(c):
    return f"{c / 1e3:.1f}k"


def ci(c):  # sendrecv: exact cycles, it is the one sub-1k number
    return f"{c:,.0f}"


def main():
    hbm, sbuf, via, cls, two, stalls, chips = load()
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
    GREY, RED = "#5E707A", "#C0392B"
    vir = plt.get_cmap("viridis")
    cmap = LinearSegmentedColormap.from_list("c2c", [vir(x) for x in np.linspace(.85, .02, 64)])  # no pale end
    vals = np.concatenate([hbm[np.isfinite(hbm)], sbuf[np.isfinite(sbuf)], list(SR_PAIR.values())])
    norm = Normalize(vals.min(), vals.max())
    col = lambda c: cmap(norm(c))  # noqa: E731

    fig = plt.figure(figsize=(22, 11.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=.06)

    # ---------------- right: floorplan; one line per route the runtime builds
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(.1, 13.2)
    ax.set_ylim(-5.45, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")
    X = [1.5, 3.9, 8.1, 10.5]
    YA, YB = 7.35, 2.45
    HW, HH = .78, .58  # NC box half width / height
    PAD = .2  # gap between the NC boxes and the LNC-pair outline
    HT = 1.8  # outer edge of the HBM stack bar, from the NC row
    YM = (YA + YB) / 2
    pos = {i: (X[i], YA) for i in range(4)} | {i + 4: (X[i], YB) for i in range(4)}
    for y0, lab, ly, va in ((YA - 1.25, "die 1 (driver)", YA + 2.4, "bottom"), (YB - 2.2, "die 0 (driver)", YB - 2.25, "top")):
        ax.add_patch(FancyBboxPatch((.25, y0), 11.9, 3.5, boxstyle="round,pad=0.05", fc="#F6F8F9", ec=INK, lw=1.4,
                                    zorder=0))
        ax.text(.3, ly, lab, fontsize=14, color=INK, va=va, fontweight="bold")
    stack = {}
    for p, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
        (xa, y), (xb, _) = pos[a], pos[b]
        up = p < 2
        hy = y + 1.4 if up else y - 1.8
        ax.add_patch(Rectangle((xa - HW, hy), xb - xa + 2 * HW, .4, fc="#DCE6EA", ec=GREY, lw=.8, zorder=1))
        ax.text((xa + xb) / 2, hy + .2, f"HBM stack {(2, 3, 1, 0)[p]} (odd chips: {(3, 2, 0, 1)[p]})", fontsize=13, color=INK, ha="center", va="center", zorder=2)
        stack[p] = (xa + xb) / 2, hy + .2
        for c in (a, b):
            ax.plot([pos[c][0]] * 2, [hy, y + HH] if up else [hy + .4, y - HH], color=GREY, lw=1, zorder=1)

    lnc, d2d, r_, lane = cls["1"], cls["4"], min(cls["2"], cls["3"]), cls["5"]
    sr, xc = shim()
    xlo, xhi = min(v[0] for v in xc.values()), max(v[1] for v in xc.values())
    edge = lambda p, q, c, lw=7: ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, solid_capstyle="butt", zorder=2)  # noqa: E731
    tag = lambda xy, txt, c, ha="center": ax.text(  # noqa: E731
        *xy, txt, fontsize=14, color=INK, ha=ha, va="center", fontweight="bold", zorder=6,
        bbox=dict(fc="white", ec=c, lw=1.8, boxstyle="round,pad=0.25"))
    s_lnc, s_d2d = np.nanmedian([sbuf[i, i ^ 1] for i in range(8)]), np.nanmedian([sbuf[i, i ^ 4] for i in range(8)])
    # LNC pair (SBUF + HBM): solid, horizontal inside each pair
    for a in (0, 2, 4, 6):
        edge((pos[a][0] + HW, pos[a][1]), (pos[a + 1][0] - HW, pos[a][1]), C_LNC, lw=6)
    # Links join SEngines (LNC pair + HBM stack), not single cores: with one protocol, and with sendrecv through
    # the core-map shim, either core of the far SEngine costs the same. The four SEngines form a ring.
    # D2D link (CCdma_D2D queues): {0,1}-{4,5} and {2,3}-{6,7}, across the dies. One line per SEngine pair; i+4
    # is the all_gather fast pairing on this link, not a wire of its own.
    for a in (0, 2):
        xm = (X[a] + X[a + 1]) / 2
        edge((xm, YA - HH - PAD), (xm, YB + HH + PAD), C_D2D)
    # R link (CCdma_R queues): the two SEngines of a die
    for a in (1, 5):
        edge((pos[a][0] + HW + PAD, pos[a][1]), (pos[a + 1][0] - HW - PAD, pos[a][1]), C_R)
    # outline each SEngine (the driver's SENG: 2 cores + 1 stack + 32 DMA engines)
    for a in (0, 2, 4, 6):
        lo, hi = ((-HH - PAD, HT + .12) if a < 4 else (-HT - .12, HH + PAD))  # the pair's HBM stack is inside
        ax.add_patch(FancyBboxPatch((pos[a][0] - HW - PAD, pos[a][1] + lo), pos[a + 1][0] - pos[a][0] + 2 * (HW + PAD),
                                    hi - lo, boxstyle="round,pad=0.02", fc="#E9EFF8", ec="#9DB0CF", lw=1.2, zorder=.5))
    for c, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - HW, y - HH), 2 * HW, 2 * HH, boxstyle="round,pad=0.02", fc="white", ec=INK,
                                    lw=1.3, zorder=4))
        ax.text(x, y, f"NC{c}", fontsize=20, color=INK, fontweight="bold", ha="center", va="center", zorder=5)
    tag(((X[0] + X[1]) / 2, YA + .85), f"LNC {kc(lnc)}  sendrecv {ci(np.median(list(SR_PAIR.values())))}", C_LNC)
    for a in (0, 2):
        tag(((X[a] + X[a + 1]) / 2, YM), f"D2D link\ni+4 {kc(d2d)}\nother {kc(lane)}", C_D2D)
    tag((6.0, YA), f"R link {kc(r_)}", C_R)
    tag((6.0, YB), f"R link {kc(r_)}", C_R)
    ax.text(2.75, YB - 2.3, "Dies, SEngines and HBM stacks from the Neuron driver source (NC0-3 = physical die 1, NC4-7 = die 0); "
            "positions schematic.\nLinks join SEngines, not cores: i+4 is cheap only because all_gather has a fast pairing "
            "there. With one protocol\n(HBM permute): LNC 13.1k < R link 14.0k < D2D link 17.0k, either core; at 8 MiB, "
            f"{min(CP8[m] for m in ('m1', 'm2', 'm3')):.0f}-{max(CP8[m] for m in ('m1', 'm2', 'm3')):.0f} GB/s inside a die, "
            f"{np.mean([CP8['m4'], CP8['m5']]):.0f} across.\nOne D2D line drawn per SEngine pair; whether it is one link or one lane per "
            "core is not known.\nOff chip: SEngines {2,3} and {6,7} port to the partner chip, {0,1} and {4,5} to the chip's switch plane.",
            fontsize=10.5, color=MUTED, ha="left", va="top", style="italic")
    ax.text(6.0, YM, f"no diagonal link\n{{0,1}}-{{6,7}}, {{2,3}}-{{4,5}}\nstock: refused, relay {two}\n"
            f"core-map shim: {kc(xlo)}-{kc(xhi)}",
            fontsize=12.5, color=RED, ha="center", va="center", fontweight="bold", zorder=6,
            bbox=dict(fc="white", ec=RED, lw=1.2, ls=":", boxstyle="round,pad=0.35"))
    ax.set_title("The SEngine ring and what each link costs (512 B per exchange, cycles at 1.2 GHz)", fontsize=15, color=INK,
                 loc="left", fontweight="bold", pad=18)
    # legend table
    rows = [("solid", "same SEngine: LNC pair (i, i^1)", "SBUF + HBM", f"HBM {kc(lnc)}   SBUF {kc(s_lnc)}", C_LNC),
            ("none", "   stock nisa.sendrecv, LNC pair only", "SBUF", f"{ci(np.median(list(SR_PAIR.values())))}"
             f" (default DMA {ci(SR_DMA)})", None),
            ("solid", "R link: same die, other SEngine", "SBUF + HBM", f"HBM {kc(r_)}   SBUF {kc(CPI['m2'])} (permute)", C_R),
            ("solid", "D2D link, same core index (i, i+4)", "SBUF + HBM", f"HBM {kc(d2d)}   SBUF {kc(s_d2d)}", C_D2D),
            ("solid", "D2D link, other core (i, i^5)", "SBUF + HBM", f"HBM {kc(lane)}   SBUF {kc(CPI['m5'])} (permute)", C_D2D),
            ("none", "no diagonal link: stock relay / shim", "SBUF + HBM", f"relay {two}   shim {kc(xlo)}-{kc(xhi)}", None),
            ("none", "   shim sendrecv: same/R/D2D/diagonal", "SBUF",
             " / ".join(ci(sr[k]) for k in ("same", "R", "D2D", "diag")) + " (default DMA)", None)]
    for k, (st, name, tr, v, c) in enumerate(rows):
        y = -2.02 - .52 * k
        if name.startswith("   "):
            pass
        elif st != "none":
            ax.plot([.4, 1.3], [y, y], color=c, lw=4, solid_capstyle="butt")
        else:
            ax.text(.85, y, "no line", fontsize=10.5, color=MUTED, ha="center", va="center", style="italic")
        ax.text(1.5, y, name, fontsize=13.5, color=INK, va="center")
        ax.text(7.6, y, tr, fontsize=13.5, color=MUTED, va="center")
        ax.text(9.6, y, v, fontsize=13.5, color=INK, va="center", fontweight="bold")

    # ---------------- left: 8 x 8, every cell filled
    ax = fig.add_subplot(gs[0, 0])
    # best of every method on each pair, mirrored: nisa.sendrecv, all_gather (HBM, SBUF), collective_permute (HBM),
    # collective_permute_implicit (SBUF); the 8 unrouted pairs take the two-hop all_gather relay
    p2p = {(r["method"], r["route"]): r["cycles"] for r in
           json.loads((HERE / "results" / "core-p2p.json").read_text())["rows"] if r["tile"] == 512}
    show = np.full((8, 8), np.nan)
    how = {}
    for a in range(8):
        for b in range(8):
            if a == b:
                continue
            lo, hi = min(a, b), max(a, b)
            if (lo, hi) in via:  # row = sender: its data goes through sender ^ (first hop mask)
                cand = {f"relay via {a ^ lo ^ via[(lo, hi)]}": hbm[a, b]}
            else:
                cand = {"all_gather HBM": hbm[a, b], "all_gather SBUF": sbuf[a, b], "sendrecv": SR_PAIR.get((lo, hi), np.nan),
                        "permute HBM": p2p.get(("cp_hbm", f"m{a ^ b}"), np.nan),
                        "permute SBUF": p2p.get(("cpi_sbuf", f"m{a ^ b}"), np.nan)}
            how[(a, b)] = min((k for k in cand if np.isfinite(cand[k])), key=cand.get)
            show[a, b] = cand[how[(a, b)]]
    im = ax.imshow(show, cmap=cmap, norm=norm)
    for a in range(8):
        for b in range(8):
            if a == b:
                ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fc="#EEF1F3", ec="none"))
                continue
            tc = "white" if norm(show[a, b]) > .45 else INK
            ax.text(b, a - .14, ci(show[a, b]) if show[a, b] < 1000 else kc(show[a, b]), ha="center", va="center",
                    fontsize=12, color=tc, fontweight="bold")
            ax.text(b, a + .24, how[(a, b)], ha="center", va="center", fontsize=7.5, color=tc)
            if how[(a, b)].startswith("relay"):
                ax.add_patch(Rectangle((b - .46, a - .46), .92, .92, fill=False, ec=RED, lw=1.8))
    for t, lw in ((1.5, 1.2), (3.5, 3.5), (5.5, 1.2)):
        ax.axhline(t, color="white", lw=lw)
        ax.axvline(t, color="white", lw=lw)
    ax.plot([-.5, 7.5], [-.5, 7.5], color=INK, lw=.8, alpha=.5)
    ax.set_xticks(range(8), [f"NC{i}" for i in range(8)])
    ax.set_yticks(range(8), [f"NC{i}" for i in range(8)])
    ax.tick_params(length=0, labelsize=15)
    for t in (1.5, 5.5):
        ax.text(-1.2, t, "die 1" if t < 4 else "die 0", ha="center", va="center", fontsize=12.5, color=INK,
                rotation=90, clip_on=False,
                fontweight="bold")
    ax.set_title("Stock runtime, every pair: the cheapest exchange of any method, named in the cell; mirrored (two-way).\n"
                 "Collective cells: one measurement per route class, not per pair. Red box: refused pair, 2-hop relay via core N.",
                 fontsize=12.5, color=INK, loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.set_label(f"cycles per exchange (512 B, {CLK_GHZ} GHz NeuronCore clock)", fontsize=12.5)
    cb.ax.tick_params(labelsize=11.5)
    rs = [stalls[s] for s in TWO if s in stalls]
    os_ = [stalls[s] for s in ("1", "2", "3", "4", "5") if s in stalls]
    fig.suptitle(f"Trainium3 core to core, stock runtime: {ci(np.median(list(SR_PAIR.values())))} cycles over sendrecv inside "
                 f"an SEngine, {kc(d2d)} to the same core across the die link (all_gather), 14k-19k to the other five",
                 x=.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(.01, .925,
             f"Candidates: nisa.sendrecv, all_gather, the two permutes. Collectives: all 8 cores of a chip as ranks; cost = slope of the least-waiting "
             f"rank's in-trace collectives span against chain length (HBM: median over {chips[0]} chips; SBUF: "
             f"{chips[1]} chips). Outputs checked against distinct per-rank inputs. Relays stall: "
             f"{min(rs):.0%}-{max(rs):.0%} of spans exceed 1.5x the floor.\n"
             "For scale only, not on this colour scale: an nccom-test runtime permute costs 10.1k-11.7k cycles per "
             "call by route class (8 B), because the per-call runtime cost hides the route. In a 512 B step the data "
             "packets take < 0.6 us on every route; the rest is handshake waits.\nsendrecv cells: "
             "nisa.sendrecv chain, dma_engine=gpsimd_dma (<= 1 KiB/partition; default DMA "
             f"{ci(SR_DMA)}), whole-NEFF time slope. Cells are full step cost, not data time.",
             fontsize=10, color=MUTED, va="top")
    fig.subplots_adjust(left=.06, right=.99, top=.845, bottom=.04)
    (HERE / "figures" / "c2c-stock").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figures" / "c2c-stock" / "core-c2c.png", dpi=150)
    print("wrote figures/c2c-stock/core-c2c.png")


if __name__ == "__main__":
    main()
