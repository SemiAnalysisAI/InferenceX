"""Figures for the trn3 ubenchX pages, in the TPU7x port's style (common/ubench_style.py).

Called by analyze_ubenchx.py with the reduced results; writes figures/<name>.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
TPU = HERE.parents[1] / "tpu7x-ubench" / "ubenchx" / "results" / "mem-bw.json"
AWS, AWS_DARK, AWS_LIGHT = "#FF9900", "#B35C00", "#FFC266"
PEAK_STACK = 1225.0
TPU_BLUE = "#1A73E8"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE / "common"))
    try:
        from ubench_style import configure, INK, MUTED, GRID
        configure()
    except Exception:
        INK, MUTED, GRID = "#172F3B", "#5E707A", "#E1E8EB"
    return plt, INK, MUTED, GRID


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")


def size_ticks(ax, lo=8, hi=2**34, target=7):
    """The x axis of inferencex.semianalysis.com/ubx/mem-bw: the axis runs exactly from the smallest to the largest
    size, with a tick at the first size, then every ceil(octaves / 7) octaves, then the last size (which replaces
    the tick before it when the two are closer than half a step). 8 B to 16 GiB gives 8 B, 256 B, 8 KiB, 256 KiB,
    8 MiB, 256 MiB, 16 GiB."""
    import math
    i, b = math.ceil(math.log2(lo)), math.floor(math.log2(hi))
    step = max(1, math.ceil((b - i) / target))
    ticks = [lo] + [2**k for k in range(i + step, b + 1, step)]
    if ticks[-1] != hi:
        if len(ticks) > 1 and math.log2(hi / ticks[-1]) < step / 2:
            ticks.pop()
        ticks.append(hi)

    def label(t):
        for unit, k in (("GiB", 30), ("MiB", 20), ("KiB", 10)):
            if t >= 2**k:
                return f"{t / 2**k:g} {unit}"
        return f"{t:g} B"

    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [label(t) for t in ticks])
    ax.minorticks_off()
    pad = 2 ** ((math.log2(hi) - math.log2(lo)) * .012)   # room for the end markers, nothing more
    ax.set_xlim(lo / pad, hi * pad)


def mem_bw(mb, plt, INK, MUTED):
    ref = json.loads((HERE / "reference" / "mem-bw.json").read_text())
    VENDOR = {"h100": "#B5D99C", "h200": "#7CB342", "b200": "#43A047", "b300": "#2E7D32", "gb200": "#1B5E20",
              "mi300x": "#F4A3A3", "mi325x": "#E06666", "mi355x": "#B71C1C", "huawei": "#F2B705"}
    from matplotlib.lines import Line2D
    style = {"chip": (AWS_DARK, "v", 2.6), "stack": (AWS, "^", 1.9), "core": (AWS_LIGHT, "o", 1.9)}
    short = {"chip": "Full chip, 8 NeuronCores", "stack": "One HBM stack, LNC=2", "core": "One NeuronCore"}

    def running_median(v, k=2):
        return [float(np.median(v[max(0, i - k):i + k + 1])) for i in range(len(v))]

    def plateau(group):
        s = next((s for s in group if s["unit"] == "chip"), None)
        top = [r["bandwidthGbps"] for r in s["rows"] if r["bytes"] >= 2**30] if s else []
        return (float(np.median(top)), s["peakBandwidthGbps"]) if top else None

    title = "Trainium3 HBM copy bandwidth vs message size, against the ubenchX GPUs"
    hi, lo = plateau(mb["gpus"]), plateau(mb.get("slowChip", []))
    gap = ""
    if hi and lo:
        title = (f"Trainium3 HBM copy: a full chip plateaus at {lo[0]:,.0f} to {hi[0]:,.0f} GB/s, "
                 f"{100 * lo[0] / lo[1]:.0f} to {100 * hi[0] / hi[1]:.0f}% of the 4.9 TB/s spec")
        gap = f" about {100 * (1 - lo[0] / hi[0]):.0f}% apart"
    elif hi:
        title = (f"Trainium3 HBM copy: a full chip plateaus at {hi[0]:,.0f} GB/s, "
                 f"{100 * hi[0] / hi[1]:.0f}% of the 4.9 TB/s spec")
    # Speed bins over the chip survey (one per-stack GB/s per node and chip; a node's first run wins): a chip is
    # in the slower bin when it sits below the midpoint of the survey's range.
    survey = {}
    for run, chips in mb.get("chipSurvey", {}).items():
        for c, v in chips.items():
            survey.setdefault((" ".join(run.split()[:2]), c), v["medianGbps"])
    n_chips, n_slow = len(survey), 0
    if survey:
        mid = (min(survey.values()) + max(survey.values())) / 2
        n_slow = sum(v < mid for v in survey.values())
    bins = f", {n_slow} of {n_chips} chips" if survey else ""
    node = mb.get("metadata", {}).get("chips", {}).get("node")
    fast = mb["gpus"][0].get("chip", "?") if mb["gpus"] else "?"
    slow = mb["slowChip"][0].get("chip", "?") if mb.get("slowChip") else None
    for metric, ylabel, fname in (("mbuPercent", "MBU (% of peak)", "mem-bw-mbu"),
                                  ("bandwidthGbps", "Bandwidth (GB/s)", "mem-bw-gbps")):
        fig, ax = plt.subplots(figsize=(13, 6.2))
        gpu_lines, trn_lines = [], []
        for g in ref["gpus"]:
            key = g["gpu"].split()[0].lower()
            gpu_lines += ax.plot([r["bytes"] for r in g["rows"]], [r[metric] for r in g["rows"]],
                                 color=VENDOR.get(key, "#BBB"), lw=1.2, alpha=.8, label=g["gpu"])
        for s in mb.get("slowChip", []):
            color, _, lw = style[s["unit"]]
            ys = [r[metric] for r in s["rows"]]
            ax.plot([r["bytes"] for r in s["rows"]], ys if s["unit"] == "chip" else running_median(ys), color=color,
                    lw=lw * .65, ls=":", zorder=3)
        for s in mb["gpus"]:
            color, mk, lw = style[s["unit"]]
            rows = s["rows"]
            xs = [r["bytes"] for r in rows]
            ys = [r[metric] for r in rows]
            if s["unit"] == "chip":  # its size structure repeats on the second chip, so it is drawn as measured
                k = 100 / s["peakBandwidthGbps"] if metric == "mbuPercent" else 1
                ax.fill_between(xs, [k * r["bandwidthP10Gbps"] for r in rows], [k * r["bandwidthP90Gbps"] for r in rows],
                                color=color, alpha=.18, lw=0, zorder=3)
                line = ys
            else:  # one NEFF load per size; loads scatter, so the line is a running median over the dots
                ax.plot(xs, ys, ls="none", marker=".", ms=2.5, color=color, alpha=.45, zorder=3)
                line = running_median(ys)
            trn_lines += ax.plot(xs, line, color=color, lw=lw, label=short[s["unit"]], zorder=4)
            pw = [i for i, r in enumerate(rows) if r["bytes"] & (r["bytes"] - 1) == 0]
            ax.plot([xs[i] for i in pw], [line[i] for i in pw], ls="none", color=color, marker=mk, ms=4.5, zorder=5)
        if slow is not None:
            trn_lines.append(Line2D([], [], color=AWS_DARK, lw=1.6, ls=":", label=f"Same series on chip {slow} (slower bin{bins})"))
        size_ticks(ax)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Message size")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK)
        leg = ax.legend(handles=trn_lines, title=f"Trainium3, {node + ' ' if node else ''}chip {fast} (on-device time)", fontsize=8.5,
                        title_fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
        leg._legend_box.align = "left"
        ax.add_artist(leg)
        leg2 = ax.legend(handles=gpu_lines, title="ubenchX site data (host-timed)", fontsize=8.5,
                         title_fontsize=9, loc="upper left", bbox_to_anchor=(1.01, .62))
        leg2._legend_box.align = "left"
        fig.text(.01, .01, "Trainium3: NKI dma_copy HBM -> HBM (static descriptors), float32, up to 16 sizes per octave "
                 "(markers at powers of two), 2 x bytes / time, time on-device (profiler, whole NEFF). GPUs:\n"
                 "b.copy_(a) under do_bench (Huawei 950PR: its own graph protocol, float16), timed from the host, so "
                 "the latency-bound sizes (below about 1 MiB) are not like for like. Full chip = 4 LNC=2 logical\n"
                 "cores at once, time = the slowest core. Every Trainium point of a series is from one chip; chips "
                 f"come in two speed bins{gap}. One HBM stack: dots are the measurements (one NEFF load\n"
                 "per size; loads scatter about 10% at 2-16 MiB), line = 5-point running median. MBU peaks: chip "
                 "4,900 GB/s (AWS spec 4.9 TB/s); one HBM stack 1,225 GB/s (a quarter).", fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, .1, .88, 1))
        save(fig, fname)
        plt.close(fig)


def per_chip(pr_mode, key):
    """{chip: 8x8 matrix} for every chip whose 8 physical cores were all measured."""
    ids = pr_mode["coreIds"]
    M = np.array(pr_mode[key])
    pos = {c: i for i, c in enumerate(ids)}
    out = {}
    for chip in sorted({c // 8 for c in ids}):
        cores = [8 * chip + k for k in range(8)]
        if all(c in pos for c in cores):
            ix = [pos[c] for c in cores]
            out[chip] = M[np.ix_(ix, ix)]
    return out


def topology(ax, pr, INK, MUTED):
    """Schematic chip (after AWS's Trainium3 block diagram) with the measured per-core hop latency, drawn to
    show why the profile matrix is a checkerboard. Measured: which cores share a stack, the per-core means,
    the split of the offset. Inferred: where each core and its stack's memory port sit."""
    from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
    m = "solo" if "solo" in pr else "conc"
    ids, hop = pr[m]["coreIds"], pr[m]["meanHopNs"]
    by = {k: np.mean([h for c, h in zip(ids, hop) if c % 8 == k]) for k in range(8)}
    nchip = len({c // 8 for c in ids})
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 1.55), 9.7, 9.0, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="none", ec="#4A90D9", lw=1.4))
    ax.text(0.45, 10.15, "Trainium3 chip (schematic)", fontsize=11.5, fontweight="bold", color=INK, va="center")
    EVEN, ODD, HBMC, PORT = AWS_LIGHT, AWS_DARK, "#E8EEF4", "#2E7D32"
    # pair k -> (column, row) in AWS's drawing; which runtime pair sits where is not published.
    slots = {0: ("L", 1), 1: ("L", 0), 2: ("R", 1), 3: ("R", 0)}
    for pair, (side, row) in slots.items():
        y0 = 1.95 + row * 4.0
        hx = 0.35 if side == "L" else 8.95
        ax.add_patch(Rectangle((hx, y0), 0.65, 3.6, fc=HBMC, ec="#9AAABB", lw=.8))
        ax.text(hx + .325, y0 + 1.8, f"HBM stack\n(36 GB)", rotation=90, ha="center", va="center", fontsize=6.5,
                color=MUTED)
        cx = 1.5 if side == "L" else 5.3
        for j, core in enumerate((2 * pair, 2 * pair + 1)):  # even above odd (order assumed)
            cy = y0 + 1.9 - j * 1.8
            odd = core % 2 == 1
            ax.add_patch(FancyBboxPatch((cx, cy), 3.2, 1.55, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc=ODD if odd else EVEN, ec=INK, lw=.8))
            ax.text(cx + 1.6, cy + 1.02, f"NeuronCore {core}", ha="center", va="center", fontsize=9,
                    color="white" if odd else INK, fontweight="bold")
            ax.text(cx + 1.6, cy + 0.42, f"{by[core]:.0f} ns/hop", ha="center", va="center", fontsize=8.5,
                    color="white" if odd else INK)
        # memory port of this stack: beside the odd core (inferred)
        px = cx - 0.42 if side == "L" else cx + 3.22
        py = y0 + 0.62
        ax.add_patch(Rectangle((px, py), 0.2, 0.55, fc=PORT, ec="none"))
        hxe = hx + 0.65 if side == "L" else hx
        ax.plot([hxe, px + (0 if side == "L" else 0.2)], [py + .27, py + .27], color=PORT, lw=1.2)
        # odd core: one hop to the port; even core: through one more hop
        odd_y, even_y = y0 + 0.78, y0 + 2.67
        near = cx if side == "L" else cx + 3.2
        far_x = px + (0.2 if side == "L" else 0)
        ax.add_patch(FancyArrowPatch((near, odd_y + .12), (far_x, py + .4), arrowstyle="<|-|>", mutation_scale=7,
                                     color=PORT, lw=1.0))
        ax.add_patch(FancyArrowPatch((near, even_y), (far_x, py + .5), arrowstyle="<|-|>", mutation_scale=7,
                                     color="#C62828", lw=1.0, ls="--", connectionstyle="arc3,rad=0.35" if side == "L"
                                     else "arc3,rad=-0.35"))
        ax.text(cx + 1.6, y0 + 3.55, "LNC=2 pair: one HBM stack", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    for x, w, lab in ((0.45, 3.0, "DMA (x128)"), (3.6, 2.9, "CC-Core (x20)"), (6.65, 2.9, "Host PCIe")):
        ax.add_patch(FancyBboxPatch((x, 0.75 + .85), w, 0.25, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#F1F3F4", ec="#BBBBBB", lw=.6))
        ax.text(x + w / 2, 0.75 + .97, lab, ha="center", va="center", fontsize=6, color=MUTED)
    d = np.mean([by[k] for k in (0, 2, 4, 6)]) - np.mean([by[k] for k in (1, 3, 5, 7)])
    ax.text(0.2, 1.2, f"Green: path to the stack's memory port. Red dashed: the even core's path, {d:.1f} ns longer\n"
            "per dependent HBM read (+9 ns request out, +13 ns completion back; on-core stages identical).",
            fontsize=7.6, color=INK, va="center")
    ax.text(0.2, 0.42, f"Measured ({nchip} chips): pairs (2k, 2k+1) share a stack; every pair has the same split (+-0.4 ns),\n"
            "so it is not distance to a shared edge block. Inferred: port beside the odd core; core placement.",
            fontsize=7.6, color=MUTED, va="center")


def profiles(pr, plt, INK, MUTED):
    """8x8 per chip, one cell per physical NeuronCore (median over chips of each chip's matrix), with the
    topology schematic that accounts for the checkerboard."""
    if not pr:
        return
    modes = [m for m in ("solo", "conc") if m in pr]
    mats, nchips = {}, {}
    for m in modes:
        for k in ("matrix", "shapeMatrix"):
            chips = per_chip(pr[m], k)
            if chips:
                mats[(m, k)] = np.median(np.stack(list(chips.values())), 0)
                nchips[(m, k)] = len(chips)
    modes = [m for m in modes if (m, "matrix") in mats]
    if not modes:
        return
    vmax = max(float(v.max()) for v in mats.values())
    nr = len(modes)
    fig = plt.figure(figsize=(17, 5.4 * nr + .9))
    gs = fig.add_gridspec(nr, 4, width_ratios=[1, 1, .05, 1.75], wspace=.28, hspace=.42,
                          left=.04, right=.99, top=.88 if nr == 1 else .92, bottom=.12 if nr == 1 else .07)
    title = {("solo", "matrix"): "One active core", ("solo", "shapeMatrix"): "One active core, mean removed",
             ("conc", "matrix"): "Concurrent launch (8 per chip)", ("conc", "shapeMatrix"): "Concurrent, mean removed"}
    im = None
    for r, m in enumerate(modes):
        for c, k in enumerate(("matrix", "shapeMatrix")):
            ax = fig.add_subplot(gs[r, c])
            M = mats[(m, k)]
            im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=vmax, interpolation="nearest")
            for i in range(8):
                for j in range(8):
                    if i != j:
                        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                                color="white" if M[i, j] > .6 * vmax else INK)
            for b in (1.5, 3.5, 5.5):  # LNC=2 pairs (2k, 2k+1) share an HBM stack
                ax.axhline(b, color="white", lw=1.2)
                ax.axvline(b, color="white", lw=1.2)
            ax.set_xticks(range(8), range(8))
            ax.set_yticks(range(8), range(8))
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_color(AWS_DARK if int(t.get_text()) % 2 else "#C77700")
                t.set_fontweight("bold")
            ax.set_xlabel("NeuronCore within the chip")
            ax.set_title(f"{title[(m, k)]}\nmedian of {nchips[(m, k)]} chips", fontsize=10.5, color=INK)
    cax = fig.add_subplot(gs[:, 2])
    fig.colorbar(im, cax=cax, label="Mean absolute difference (ns)")
    tax = fig.add_subplot(gs[:, 3])
    topology(tax, pr, INK, MUTED)
    fig.suptitle("Trainium3 core memory profiles: HBM pointer-chase hop latency, profile differences", x=.01,
                 ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(.01, .015, "White lines separate the 4 HBM stacks (cores 2k and 2k+1 form an LNC=2 pair). 1,024 dependent "
             "4-byte reads over a 128 MiB chain, 6 passes averaged per hop, as the Ascend page. The checkerboard is the "
             "even/odd mean offset: it disappears when each core's mean is removed.", fontsize=8, color=MUTED)
    save(fig, "core-memory-profiles")
    plt.close(fig)

    m = "solo" if "solo" in pr else "conc"
    ids, hop = pr[m]["coreIds"], pr[m]["meanHopNs"]
    by_local = {k: [h for c, h in zip(ids, hop) if c % 8 == k] for k in range(8)}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, v in by_local.items():
        ax.scatter([k] * len(v), v, s=12, color=AWS if k % 2 == 0 else AWS_DARK, alpha=.6)
        if v:
            ax.plot([k - .3, k + .3], [np.median(v)] * 2, color=INK, lw=1.5)
    ax.set_xticks(range(8), range(8))
    ax.set_xlabel("NeuronCore within the chip (dots: one per chip)")
    ax.set_ylabel("Mean ns per dependent hop")
    ax.grid(True, axis="y")
    ax.set_title("Mean pointer-chase hop by core position (one active core)", loc="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "core-mean-hop")
    plt.close(fig)


def distance(di, plt, INK, MUTED):
    for name, pairs in di.items():
        if not name.startswith("intra") or len(pairs) < 28:
            continue
        cores = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
        idx = {c: i for i, c in enumerate(cores)}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, (size, key, lab, fmt) in zip(axes, ((8, "timeUs", "Exchange time at 8 B (us)", "{:.1f}"),
                                                     (67108864, "algbwGBps", "GB/s per direction at 64 MiB",
                                                      "{:.0f}"))):
            M = np.full((len(cores), len(cores)), np.nan)
            for p in pairs:
                r = [x for x in p["rows"] if x["bytes"] == size]
                if r:
                    M[idx[p["a"]], idx[p["b"]]] = M[idx[p["b"]], idx[p["a"]]] = r[0][key]
            im = ax.imshow(M, cmap="viridis_r" if key == "timeUs" else "viridis")
            for i in range(len(cores)):
                for j in range(len(cores)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7, color="white")
            ax.set_xticks(range(len(cores)), cores)
            ax.set_yticks(range(len(cores)), cores)
            ax.set_title(lab, fontsize=11, color=INK)
            fig.colorbar(im, ax=ax, shrink=.8)
        fig.suptitle(f"Trainium3 core-to-core exchange through the collectives runtime ({name})", x=.01, ha="left",
                     fontsize=13, fontweight="bold", color=INK)
        fig.tight_layout()
        save(fig, f"core-distance-{name}")
        plt.close(fig)


def main(mb, pr, di):
    plt, INK, MUTED, GRID = setup()
    mem_bw(mb, plt, INK, MUTED)
    # core-memory-profiles / core-mean-hop retired 2026-11-11: an HBM chase is core -> own stack, not
    # core -> core; see c2c/figure_c2c.py (figures/c2c-stock/core-c2c.png). profiles() kept for the PARITY.md numbers.
    distance(di, plt, INK, MUTED)
