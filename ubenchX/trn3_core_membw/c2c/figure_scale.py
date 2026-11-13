"""The chip-to-chip counterpart of figures/core-sendrecv-xseng.png, at two scales: one node and the whole UltraServer.

  python3 c2c/figure_scale.py            # writes figures/node-allgather.png and figures/rack-allgather.png

nisa.sendrecv stops at the chip edge (a logical core spanning two chips faults, sendrecv/XSENG.md), so between chips
the exchange is HBM all_gather, timed by nccom-test (results/core-nccom.json, nccom/sweep.sh and nccom/xnode.sh):
2-rank groups (r, r ^ mask) on the same core index, every pair of the node or of the 4 nodes at once, median of 50
independent ops. Above the diagonal: 512 B per rank. Below: 2 MiB per rank, with GB/s per direction (tile / time).
One measurement per XOR mask (chip mask, and node mask at UltraServer scale), applied to every pair with that mask.
nccom-test times one op; the chain slopes of c2c/ and p2p/ (figures/c2c-stock/chip-c2c.png) sit about 1.7k cycles higher.
Red frame: pairs collective_permute refuses (other switch plane, not the tray partner); same rack only.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "c2c"))
from figure_c2c import CLK_GHZ, ci  # noqa: E402

SMALL, LARGE = 512, 2097152   # LARGE is raised below to the largest tile the tray partner and every same-plane mask have
SCOPES = ("xchip", "xnode", "xchip-big", "xnode-big")
RING_SCOPES = ("xchip-ring", "xnode-ring", "xnode-ring2")   # ring algorithm forced; used for the large tile only
CORES = 8                     # core pairs of one chip pair, all running at once
RED, GREEN, BLUE = "#C0392B", "#2EAD6B", "#2F5D8C"
k1 = lambda v: f"{v / 1000:.1f}k"  # noqa: E731


def klass(nm, cm):
    """Route class of chip (node n, chip c) to (n ^ nm, c ^ cm)."""
    if cm % 2 == 0:
        return "plane"                       # same parity: one switch plane
    return "partner" if (nm == 0 and cm == 1) else "other"


def _t(r):
    """Cycles of one op: the median at small tiles (latency); the mean at large ones, where ranks run at different fixed
    rates and 8 tiles / mean is what a host-clock chain of ops measures (nccom/chain.sh: 385 GB/s tray, 245 switch)."""
    return r["cyclesAvg"] if r["tileBytes"] > 2097152 else r["cyclesP50"]


def load():
    rows = [r for r in json.loads((HERE / "results" / "core-nccom.json").read_text())["rows"]
            if r["kernel"] == "ag_hbm" and r["coreMask"] == 0 and r["scope"] in SCOPES and r["checked"]]
    acc = {}
    for r in rows:
        acc.setdefault((r["nodeMask"], r["chipMask"], r["tileBytes"]), []).append(_t(r))
    cyc = {k: float(np.mean(v)) for k, v in acc.items()}
    # above 2 MiB: the forced ring where it ran (the default 2-rank mesh is rate-pinned, and its split T / 2T op times
    # make 8 tiles / median an overestimate: 201 GB/s on the 128 GB/s rack-to-rack link)
    ring = {}
    for r in json.loads((HERE / "results" / "core-nccom.json").read_text())["rows"]:
        if (r["kernel"] == "ag_hbm" and r["coreMask"] == 0 and r["scope"] in RING_SCOPES and r["checked"]
                and r["tileBytes"] > 2097152 and r.get("algorithm") == "Ring"):
            k = (r["nodeMask"], r["chipMask"], r["tileBytes"])
            ring[k] = min(ring.get(k, float("inf")), _t(r))
    cyc.update(ring)
    BEST_RING.update(ring)
    # inside a node: the steady-state rate from chained ops by host clock (nccom/chain.sh), where it was measured.
    # A 50-op nccom-test run reads 10-30% higher: its first ops run before every rank is busy.
    st = HERE / "results" / "chain-steady.json"
    if st.exists():
        d = json.loads(st.read_text())
        for cm, g in d["gbpsByChipMask"].items():
            cyc[(d["nodeMask"], int(cm), d["tileBytes"])] = CORES * d["tileBytes"] / (g * 1e9) * CLK_GHZ * 1e9
        for e in d.get("crossNode", []):   # 500-op runs across nodes
            for cm in e["chipMasks"]:
                cyc[(e["nodeMask"], cm, d["tileBytes"])] = CORES * d["tileBytes"] / (e["gbps"] * 1e9) * CLK_GHZ * 1e9
    return cyc


BEST_RING = set()


def size_name(b):
    for u, n in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if b >= u:
            return f"{b / u:g} {n}"
    return f"{b} B"


def _largest():
    cyc = load()
    full = [t for t in {k[2] for k in cyc} if all((0, cm, t) in cyc for cm in [1] + list(range(2, 16, 2)))]
    return max(full) if full else LARGE


LARGE = _largest()
LNAME = size_name(LARGE)


def gbps(cycles):
    """GB/s per chip pair and direction: 8 tiles in the mean op time."""
    return CORES * LARGE / (cycles / (CLK_GHZ * 1e9)) / 1e9


def summary(cyc, nms):
    """class -> {tile: median cycles over the masks of that class}, for the given node masks."""
    out = {}
    for (nm, cm, t), v in cyc.items():
        if nm in nms and (nm, cm) != (0, 0):
            out.setdefault(klass(nm, cm), {}).setdefault(t, []).append(v)
    return {c: {t: float(np.median(v)) for t, v in d.items()} for c, d in out.items()}


def _rows():
    return json.loads((HERE / "results" / "core-nccom.json").read_text())["rows"]


def ring_masks():
    """(node mask, chip mask) of the plotted rows that ran with the ring algorithm (the mesh asserts there)."""
    return {(r["nodeMask"], r["chipMask"]) for r in _rows() if r["kernel"] == "ag_hbm" and r["coreMask"] == 0
            and r["scope"] in SCOPES and r["checked"] and r.get("algorithm") == "Ring"}


def efa_range(cyc):
    """Every cross-rack pair that is not the same slot and chip index: (lo, hi) cycles at 512 B and (lo, hi) GB/s at the large tile."""
    pick = lambda t: [v for (nm, cm, tt), v in cyc.items() if tt == t and (nm == 2 or (nm == 3 and cm))]  # noqa: E731
    s, l = pick(SMALL), pick(LARGE)
    return (min(s), max(s)), (gbps(max(l)), gbps(min(l)))


def on_efa(nm, cm):
    """Two-rank groups the stock runtime carries over EFA at the large tile (EFA byte counters, nccom/xpair.sh and the
    ring runs): other-slot nodes, same-slot nodes off the same chip index, and same-rack pairs on the other plane."""
    return nm == 2 or (nm == 3 and cm != 0) or (nm == 1 and cm % 2 == 1)


def krange(lo, hi):
    return f"{lo / 1000:.0f}-{hi / 1000:.0f}k"


def node_split():
    """Which chip masks of the one-node matrix came from which node, as text ('' if one node measured them all)."""
    by = {}
    for r in _rows():
        if r["kernel"] == "ag_hbm" and r["coreMask"] == 0 and r["scope"] == "xchip" and r["checked"] and r["nodeMask"] == 0:
            by.setdefault(r["node"], set()).add(r["chipMask"])
    if len(by) < 2:
        return ""
    parts = [f"{min(m)}-{max(m)}" for m in sorted(by.values(), key=min)]
    return " Chip masks " + ", ".join(f"{q} on {'one node' if i == 0 else 'another'}" for i, q in enumerate(parts)) + "."


def ring_vs_mesh(cyc, nm=2, cm=0):
    """512 B cycles (ring, mesh) on one EFA pairing (nodes 0-2, same chip index), or None if the ring probe is absent."""
    ring = [r["cyclesP50"] for r in _rows() if r["scope"] == "xprobe" and r.get("algorithm") == "Ring" and r["checked"]
            and (r["nodeMask"], r["chipMask"], r["coreMask"], r["tileBytes"]) == (nm, cm, 0, SMALL)]
    mesh = cyc.get((nm, cm, SMALL))
    return (float(np.median(ring)), mesh) if ring and mesh and (nm, cm) not in ring_masks() else None


def heat(fig, ax, plt, cyc, n, INK, MUTED, text):
    """n x n chip matrix (n = 16 or 64); chip id = 16 * node + chip. Two colour scales, one per triangle."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    vir = plt.get_cmap("viridis")
    cmap = LinearSegmentedColormap.from_list("c2c", [vir(x) for x in np.linspace(.85, .12, 64)])
    up, lo = np.full((n, n), np.nan), np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            if a != b:
                m = a ^ b
                v = cyc.get((m >> 4, m & 15, SMALL if a < b else LARGE), np.nan)
                (up if a < b else lo)[a, b] = v if a < b or not np.isfinite(v) else gbps(v)
    nu, nl = Normalize(3500, 7000), Normalize(0, 512)
    ax.add_patch(Rectangle((-.5, -.5), n, n, fc="#EEF1F3", ec="none", zorder=0))
    iu = ax.imshow(np.ma.masked_invalid(up), cmap=cmap, norm=nu)
    il = ax.imshow(np.ma.masked_invalid(lo), cmap=cmap.reversed(), norm=nl)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            m = a ^ b
            v = (up if a < b else lo)[a, b]
            if text and np.isfinite(v):
                tc = "white" if (nu(v) if a < b else 1 - nl(v)) > .45 else INK
                ax.text(b, a - .13, ci(v) if a < b else f"{v:.0f}", ha="center", va="center", fontsize=9.5, color=tc,
                        fontweight="bold")
                ax.text(b, a + .24, {"partner": "partner", "plane": "plane", "other": "2 hops"}[klass(m >> 4, m & 15)]
                        if a < b else "GB/s", ha="center", va="center", fontsize=6.8, color=tc)
            if a > b and np.isfinite(v) and on_efa(m >> 4, m & 15):   # large tile: the stock runtime sent it over EFA
                ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="white", lw=0, hatch="////", alpha=.55))
            if (m >> 4) in (0, 1) and klass(m >> 4, m & 15) == "other" and np.isfinite(v):   # planes exist inside a rack only
                ax.add_patch(Rectangle((b - .46, a - .46), .92, .92, fill=False, ec=RED, lw=1.3 if text else .35))
    if n == 16:
        for t in np.arange(1.5, n - 1, 2):       # tray boundaries
            ax.axhline(t, color="white", lw=1.6)
            ax.axvline(t, color="white", lw=1.6)
        ax.set_xticks(range(n), [str(i) for i in range(n)])
        ax.set_yticks(range(n), [str(i) for i in range(n)])
        ax.set_xlabel("chip", fontsize=12)
        ax.set_ylabel("chip", fontsize=12)
    else:
        for t in (15.5, 31.5, 47.5):             # node boundaries
            ax.axhline(t, color="white", lw=3)
            ax.axvline(t, color="white", lw=3)
        ax.set_xticks([7.5 + 16 * i for i in range(4)], [f"node {i}\nchips 0-15" for i in range(4)])
        ax.set_yticks([7.5 + 16 * i for i in range(4)], [f"node {i}" for i in range(4)])
    ax.plot([-.5, n - .5], [-.5, n - .5], color=INK, lw=.8, alpha=.5)
    ax.tick_params(length=0, labelsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    p = ax.get_position()
    for im, y0, lab in ((iu, p.y0 + p.height * .53, f"above: cycles per 512 B message ({CLK_GHZ} GHz)"),
                        (il, p.y0, "below: GB/s per chip pair, one direction, " + LNAME + " tiles")):
        cax = fig.add_axes([p.x1 + .012, y0, .012, p.height * .47])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(lab, fontsize=10.5)
        cb.ax.tick_params(labelsize=9.5)
    return np.isfinite(up).sum() + np.isfinite(lo).sum()


def lab(s, t=SMALL):
    return "not measured" if s is None or t not in s else (ci(s[t]) if t == SMALL else f"{gbps(s[t]):.0f} GB/s")


def node_diagram(ax, S, INK, MUTED):
    """One node: 8 trays of an even and an odd chip; even chips on switch plane A, odd chips on plane B."""
    from matplotlib.patches import FancyBboxPatch
    W, H, GAP = 1.0, 1.0, .42
    xs = [.55 + i * (W + GAP) for i in range(8)]
    ye, yo = 6.2, 3.5
    for yb, name in ((9.0, "switch plane A: even chips"), (1.1, "switch plane B: odd chips")):
        ax.add_patch(FancyBboxPatch((.2, yb), 11.6, .9, boxstyle="round,pad=0,rounding_size=.12", fc="#EAF1F8", ec=BLUE, lw=1.3))
        ax.text(6.0, yb + .45, name, ha="center", va="center", fontsize=11, color=BLUE, fontweight="bold")
    for i, x in enumerate(xs):
        ax.add_patch(FancyBboxPatch((x - .14, yo - .3), W + .28, ye + H - yo + .6, boxstyle="round,pad=0,rounding_size=.12",
                                    fc="none", ec="#D08A1F", lw=1.1, ls=(0, (5, 3)), zorder=1))
        ax.plot([x + W / 2] * 2, [ye + H, 9.0], color=BLUE, lw=3, zorder=1)
        ax.plot([x + W / 2] * 2, [yo, 2.0], color=BLUE, lw=3, zorder=1)
        ax.plot([x + W / 2] * 2, [yo + H, ye], color=GREEN, lw=6, solid_capstyle="butt", zorder=1)
        for y, c in ((ye, 2 * i), (yo, 2 * i + 1)):
            ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=.08", fc="white", ec=INK, lw=1.4,
                                        zorder=3))
            ax.text(x + W / 2, y + H / 2, str(c), ha="center", va="center", fontsize=13, color=INK, fontweight="bold", zorder=4)
    # the pair with no direct link: chip 0 and chip 3
    ax.plot([xs[0] + W, xs[1]], [ye + .25, yo + H - .25], color=RED, lw=1.6, ls=(0, (4, 3)), zorder=2)
    box = dict(fc="white", ec="none", pad=2.5)
    ax.text(6.0, 8.45, f"same plane, any two chips: {lab(S.get('plane'))}, {lab(S.get('plane'), LARGE)}", ha="center",
            va="center", fontsize=11.5, color=BLUE, fontweight="bold", bbox=box, zorder=5)
    for i, (col, t) in enumerate((
            (GREEN, f"tray partner (green link, e.g. 0 and 1): {lab(S.get('partner'))}, {lab(S.get('partner'), LARGE)}"),
            (BLUE, f"same plane (e.g. 0 and 2): {lab(S.get('plane'))}, {lab(S.get('plane'), LARGE)}"),
            (RED, f"other plane, not the partner (dashed, e.g. 0 and 3), two hops: {lab(S.get('other'))}, "
                  f"{lab(S.get('other'), LARGE)} (mesh only)"),
            ("#B46F0C", "dashed orange outline: one tray = two chips"))):
        ax.text(.2, .45 - i * .5, t, fontsize=11.5 if i < 3 else 9.5, color=col, fontweight="bold" if i < 3 else "normal",
                va="center")
    ax.set_xlim(0, 12)
    ax.set_ylim(-2.6, 10.6)
    ax.set_aspect("equal")
    ax.axis("off")


def rack_diagram(ax, cyc, INK, MUTED):
    """The UltraServer: 4 nodes of 8 trays; two switch planes per rack, point-to-point links rack to rack."""
    from matplotlib.patches import FancyBboxPatch
    NW, GAP, c = 2.6, .3, .22
    ye, yo = 5.9, 4.2
    for yb, name in ((8.6, "plane A: the even chips"), (1.5, "plane B: the odd chips")):
        for r in range(2):   # the driver: the switch is inside a rack; rack to rack is point to point
            ax.add_patch(FancyBboxPatch((.2 + 5.9 * r, yb), 5.7, .9, boxstyle="round,pad=0,rounding_size=.12", fc="#EAF1F8", ec=BLUE, lw=1.3))
            ax.text(3.05 + 5.9 * r, yb + .45, f"rack {r}, switch {name}", ha="center", va="center", fontsize=10.5, color=BLUE, fontweight="bold")
    cx = [.35 + n * (NW + GAP) + NW / 2 for n in range(4)]
    for a, b, y in ((0, 3, 7.95), (1, 2, 7.45)):   # same slot of the other rack: direct links, no switch
        ax.plot([cx[a], cx[a], cx[b], cx[b]], [7.0, y, y, 7.0], color="white", lw=5, zorder=5, solid_capstyle="butt")
        ax.plot([cx[a], cx[a], cx[b], cx[b]], [7.0, y, y, 7.0], color=RED, lw=2.2, zorder=6, solid_capstyle="butt")
        ax.text(6.0, y, f"nodes {a}-{b}: cable, same chip index", ha="center", va="center", fontsize=8.5, color=RED, zorder=7,
                bbox=dict(fc="white", ec="none", pad=1.5))
    for n in range(4):
        x0 = .35 + n * (NW + GAP)
        ax.add_patch(FancyBboxPatch((x0, 3.6), NW, 3.4, boxstyle="round,pad=0,rounding_size=.12", fc="#FAFBFC", ec=INK, lw=1.5,
                                    zorder=2))
        ax.text(x0 + NW / 2, 6.72, f"node {n}", ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold", zorder=4)
        for i in range(8):
            x = x0 + .2 + i * (NW - .4 - c) / 7
            ax.plot([x + c / 2] * 2, [ye + c, 8.6], color=BLUE, lw=1.4, zorder=1)
            ax.plot([x + c / 2] * 2, [yo, 2.4], color=BLUE, lw=1.4, zorder=1)
            ax.plot([x + c / 2] * 2, [yo + c, ye], color=GREEN, lw=2.6, zorder=3)
            for y in (ye, yo):
                ax.add_patch(FancyBboxPatch((x, y), c, c, boxstyle="square,pad=0", fc="white", ec=INK, lw=1.1, zorder=4))
        ax.text(x0 + NW / 2, 3.85, "8 trays, 16 chips", ha="center", va="center", fontsize=9, color=MUTED, zorder=4)
    def med(nm, pick):   # {tile: median over the chip masks that pick() keeps}
        out = {}
        for (n_, cm, t), v in cyc.items():
            if n_ == nm and pick(cm):
                out.setdefault(t, []).append(v)
        return {t: float(np.median(v)) for t, v in out.items()}

    rows = [("same chip index", lambda cm: cm == 0, INK), ("same chip parity", lambda cm: cm and cm % 2 == 0, BLUE),
            ("other chip parity", lambda cm: cm % 2 == 1 and cm != 1, RED), ("chip index XOR 1", lambda cm: cm == 1, GREEN)]
    cols = [(0, "inside a node"), (1, "same rack\n(0-1, 2-3)"), (2, "other rack, other\nslot (0-2, 1-3)"), (3, "other rack, same\nslot (0-3, 1-2)")]
    ax.text(.2, .95, f"Message per core: 512 B for cycles, {LNAME} for GB/s (per chip pair, one direction)", fontsize=11.5, color=INK,
            fontweight="bold", va="center")
    ring = ring_masks()
    for j, (nm, head) in enumerate(cols):
        ax.text(3.0 + 2.35 * j, -.1, head, fontsize=10.5, color=INK, fontweight="bold", va="bottom")
    for i, (name, pick, col) in enumerate(rows):
        y = -.4 - i * .5
        ax.text(.2, y, name, fontsize=10.5, color=col, fontweight="bold", va="center")
        for j, (nm, _) in enumerate(cols):
            v = med(nm, pick)
            t = "the same chip" if (nm, i) == (0, 0) else "not measured" if SMALL not in v else \
                f"{lab(v)}   {lab(v, LARGE)}" if LARGE in v else lab(v)
            if any(n_ == nm and pick(cm) for n_, cm in ring):
                t += " *"
            ax.text(3.0 + 2.35 * j, y, t, fontsize=10.5,
                    color=INK if SMALL in v else "#868E96", va="center")
    ax.text(.2, -2.3, "* 512 B taken with the ring algorithm.\n"
            "Same parity = same switch plane (inside a rack). XOR 1 = tray partner (inside a node).\n"
            "Other rack, other slot, and same slot off the same chip index: the runtime used EFA.",
            fontsize=9.5, color=MUTED, va="top")
    ax.set_xlim(0, 12)
    ax.set_ylim(-3.4, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")


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
    cyc = load()
    IN, X = summary(cyc, {0}), summary(cyc, {1, 2, 3})
    method = ("nisa.sendrecv does not leave the chip, so the chip-to-chip exchange is HBM all_gather between the same core "
              "index of two chips, timed by nccom-test: every pair at one\nXOR mask at once, median of 50 independent ops, "
              "outputs checked. One measurement per mask, applied to every pair with that mask. GB/s: steady state (chained ops by host clock in a node, 500-op mean across NeuronLink nodes); EFA cells: mean of 50 ops.\nLarge tiles use the ring algorithm where it runs "
              "(the default 2-rank mesh holds a switched pair to about 112 GB/s). Article link budget: 512 GB/s tray link, 512 GB/s to the switch plane, 128 GB/s rack to rack.\nnccom-test times a single op; the dependent-chain figures elsewhere (5.8k / 6.9k / 7.9k) sit about "
              "1.7k cycles higher on the same routes.")

    # ---------------- one node
    fig = plt.figure(figsize=(21, 11))
    ax = fig.add_axes([.045, .06, .5, .74])
    heat(fig, ax, plt, cyc, 16, INK, MUTED, text=True)
    ax.set_title("Above the diagonal: cycles per 512 B exchange. Below: GB/s per chip pair at " + LNAME + " per core.\n"
                 "Red frame: other switch plane and not the tray partner (two hops); mesh only (the ring refuses them), buffer raised.",
                 fontsize=12, color=INK, loc="left", pad=10)
    fig.suptitle(f"Trainium3 chip to chip in one node (all_gather): {lab(IN['plane'])} over a switch plane, "
                 f"{lab(IN['partner'])} to the tray partner, {lab(IN['other'])} over two hops", x=.02, y=.975, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(.02, .93, method.replace("every pair with that mask.", "every pair with that mask." + node_split()), fontsize=10,
             color=MUTED, va="top")
    tx = fig.add_axes([.63, .08, .36, .72])
    node_diagram(tx, IN, INK, MUTED)
    tx.set_title("The same numbers on the node: 8 trays, two switch planes.\nCycles per 512 B exchange, and GB/s per "
                 "chip pair and direction at " + LNAME + ".", fontsize=12, color=INK, loc="left", pad=6)
    tx.text(.2, -1.5, "The switch hop is cheaper than the tray's direct link at 512 B; the direct link carries more at " + LNAME + ".\n"
            "Different plane, not partners: two hops, a switch and a tray link; the chip in between forwards (SemiAnalysis article).",
            fontsize=9.5, color=MUTED, va="top")
    fig.savefig(HERE / "figures" / "node-allgather.png", dpi=150)
    print("wrote figures/node-allgather.png")

    # ---------------- the UltraServer (4 nodes, 64 chips)
    fig = plt.figure(figsize=(21, 11))
    ax = fig.add_axes([.045, .07, .5, .73])
    got = heat(fig, ax, plt, cyc, 64, INK, MUTED, text=False)
    miss = 64 * 63 - got
    ax.set_title("Above the diagonal: cycles per 512 B message. Below: GB/s per chip pair at " + LNAME + " per core.\n"
                 "Hatched: runtime used EFA. Red frame: two hops (other plane, not the tray partner)."
                 + (f" Grey: not measured ({miss} cells)." if miss else ""), fontsize=12, color=INK, loc="left", pad=10)
    fig.suptitle("Trainium3 chip to chip across the UltraServer (all_gather)", x=.02, y=.975,
                 ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(.02, .93, "HBM all_gather between the same core of two chips, timed by nccom-test, outputs checked. One measurement per XOR mask.\n"
             "Link budget (article): 512 GB/s tray link, 512 GB/s to the switch plane, 128 GB/s rack to rack.", fontsize=10.5,
             color=MUTED, va="top")
    tx = fig.add_axes([.63, .08, .36, .72])
    rack_diagram(tx, cyc, INK, MUTED)
    fig.savefig(HERE / "figures" / "rack-allgather.png", dpi=150)
    print(f"wrote figures/rack-allgather.png ({miss} of {64 * 63} cells not measured)")


if __name__ == "__main__":
    main()
