"""figures/c2c-stock/core-p2p.png from results/core-p2p.json (p2p/analyze_p2p.py).

Left: cycles per 512 B exchange by route, one bar per method (missing bar = the method does not run there).
Right: GB/s per direction against tile size for the methods that move data fastest on each route class.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import figures as F  # noqa: E402
import analyze_ubenchx as A  # noqa: E402

ROUTES = [("m1", "LNC pair\n(i, i^1)"), ("m4", "other die\nsame position (i+4)"), ("m2", "same die\nother pair (i^2)"),
          ("m3", "same die\nother pair (i^3)"), ("m5", "other die\nother position (i^5)"),
          ("x8", "tray partner\nchip c to c^1"), ("x16", "same switch plane\nchip c to c^2")]
# legend names for the routes drawn on the right (the JSON route strings predate the tray / switch-plane naming)
RLAB = {"m4": "other die, same position (i+4)", "x8": "tray partner chip (c^1)", "x16": "same switch plane chip (c^2)"}
METHODS = [("sendrecv_gpsimd", "nisa.sendrecv, gpsimd_dma (SBUF)", "#1B5E20"), ("ag_sbuf", "all_gather (SBUF)", "#7CB342"),
           ("ag_hbm", "all_gather (HBM)", F.AWS_DARK), ("cpi_sbuf", "collective_permute_implicit (SBUF)", F.AWS),
           ("cp_hbm", "collective_permute (HBM)", F.AWS_LIGHT)]
# default DMA engine: not in the latency bars (gpsimd_dma is 2.6x faster at 512 B), but the only sendrecv above 128 KiB
EXTRA = [("sendrecv", "nisa.sendrecv, default DMA (SBUF)", "#1B5E20")]
CURVES = [("sendrecv_gpsimd", "m1", "-"), ("sendrecv", "m1", "--"), ("ag_hbm", "m1", "-"), ("ag_sbuf", "m1", "-"), ("ag_hbm", "m4", "--"),
          ("cpi_sbuf", "m2", ":"), ("ag_hbm", "x8", "-."), ("ag_hbm", "x16", (0, (1, 1))),
          ("ag_hbm", "xchip", "--"), ("cpi_sbuf", "xchip", ":")]


def main():
    d = json.loads((HERE / "results" / "core-p2p.json").read_text())
    idx = {(r["method"], r["route"], r["tile"]): r for r in d["rows"]}
    plt, INK, MUTED, _ = F.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(18, 6.8), gridspec_kw={"width_ratios": [1.45, 1]})
    w = .8 / len(METHODS)
    for i, (m, lab, col) in enumerate(METHODS):
        xs, ys = [], []
        for j, (rt, _) in enumerate(ROUTES):
            r = idx.get((m, rt, 512))
            if r:
                xs.append(j + (i - (len(METHODS) - 1) / 2) * w)
                ys.append(r["cycles"])
        bars = a.bar(xs, ys, w * .92, color=col, label=lab)
        for x, y in zip(xs, ys):
            a.text(x, y + 150, f"{y:.0f}" if y < 1000 else f"{y / 1000:.1f}k", ha="center", va="bottom", fontsize=7.5, rotation=90, color=INK)
    j = [k for k, _ in ROUTES].index("x8")
    xm = j + (1 - (len(METHODS) - 1) / 2) * w
    a.bar([xm], [22000 - 7700], w * .92, bottom=7700, fill=False, edgecolor="#7CB342", hatch="///", lw=1)
    a.annotate("cores 0,1,4,5 (not the\ntray-link port owners)\n~22k, both runs", (xm, 20500), (j - .55, 19000), fontsize=8.5,
               color=MUTED, ha="right", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .8})
    a.text(1.75, 22800, "no bar = does not run on that route (stock):\nsendrecv LNC pair only; SBUF all_gather\n"
           "LNC pair, i+4 and chip to chip only", ha="left", va="top", fontsize=8.5, color=MUTED)
    a.set_xticks(range(len(ROUTES)), [l for _, l in ROUTES], fontsize=10)
    a.set_ylabel("cycles per 512 B exchange (1.2 GHz)")
    a.set_ylim(0, 23500)
    a.grid(True, axis="y")
    a.legend(fontsize=9.5, loc="upper left", ncol=1, frameon=False)
    a.set_title("Latency: what each method costs on each route", loc="left", fontsize=14, fontweight="bold",
                color=INK)
    for (m, rt, ls) in CURVES:
        pts = sorted((t, r["GBps"]) for (mm, rr, t), r in idx.items() if mm == m and rr == rt and r["GBps"])
        if len(pts) < 2:
            continue
        col = dict((k, c) for k, _, c in METHODS + EXTRA)[m]
        b.plot([t for t, _ in pts], [g for _, g in pts], color=col, ls=ls, lw=2, marker="o", ms=4,
               label=f"{dict((k, l) for k, l, _ in METHODS + EXTRA)[m]}, {RLAB.get(rt, d['routes'][rt])}")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    ticks = [2 ** k for k in range(9, 24, 2)]
    b.set_xticks(ticks, [A.size_label(t) for t in ticks])
    b.set_xlim(300, 14e6)
    b.set_xlabel("Tile per exchange (bytes each way)")
    b.set_ylabel("GB/s per direction")
    b.grid(True, which="both", alpha=.4)
    b.legend(fontsize=8, loc="upper left", frameon=False)
    b.set_title("Bandwidth vs tile", loc="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(.01, .01, "Dependent chains of 1/9/33 exchanges, cost = slope of the least-waiting rank's collectives-queue "
             "span (median over profiled executions; the chip-topology figures use the median over pairs, which is higher); "
             "outputs checked against per-rank inputs.\nnisa.sendrecv: whole-NEFF slope, outputs checked in the NKI simulator, "
             "LNC pair only on the stock runtime (a core-map shim crosses SEngines: core-sendrecv-xseng.png); gpsimd_dma takes "
             "<= 1 KiB/partition (128 KiB tiles), the default DMA engine carries larger tiles.\nInside a chip 4 pairs run at "
             "once; chip to chip, 8 pairs per chip pair, so large-tile figures there are the best rank under contention. A single pair "
             "run alone matches the concurrent figures within 1% at 512 B and 8 MiB (LNC and same-die pairs).\nA lone 2-rank "
             "cross-chip pair with different core indices (chip 6 core 7 to chip 7 core 0) runs only the runtime collective_permute "
             "(15.6k); the in-kernel collectives fail to set up (no_hier no_mesh).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .1, 1, 1))
    (F.FIG / "c2c-stock").mkdir(parents=True, exist_ok=True)
    F.save(fig, "c2c-stock/core-p2p")


if __name__ == "__main__":
    main()
