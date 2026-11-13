"""Core-to-core latency and bandwidth by method and route, from p2p/sweep.sh (and sendrecv/ for the LNC pair).

  python3 p2p/analyze_p2p.py     # after cluster/fetch.sh

Per (method, route, tile size): for each NSTEP, the per-rank median over profiled executions of the span of
the rank's `collectives` DMA queue, then the minimum over ranks (the floor estimator of c2c/analyze_c2c.py);
cost per exchange = least-squares slope against NSTEP. Every exchange moves one tile in each direction at
once, so GB/s = tile bytes / cost is per direction. Only points whose output check passed are used.
nisa.sendrecv (LNC pair only) comes from sendrecv/run_sendrecv.py: slope of whole-NEFF time (single launch).
Writes results/core-p2p.json.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "data" / "raw"
CLK_GHZ = 1.2
ROUTE = {"m1": "LNC pair (i, i^1)", "m2": "same die, other pair (i^2)", "m3": "same die, other pair (i^3)",
         "m4": "other die, same lane (i+4)", "m5": "other die, other lane (i^5)",
         "x8": "chip c to c^1 (same core)", "x16": "chip c to c^2 (same core)",
         "lnc": "ONE pair alone: LNC (cores 0-1)", "die": "ONE pair alone: same die, other pair (cores 1-2)",
         "xchip": "ONE pair alone: chip c core 7 to c+1 core 0",
         "x8r": "chip c to c^1, repeat on chips 2-3"}
SUBDIRS = {"sweep": {}, "single": {}, "sweep-repeat": {"x8": "x8r"}}  # runs/p2p/<subdir>, route renames
METHOD = {"sendrecv": "nisa.sendrecv (SBUF, raw DMA)", "sendrecv_gpsimd": "nisa.sendrecv gpsimd_dma (SBUF)",
          "ag_sbuf": "all_gather (SBUF)", "cpi_sbuf": "collective_permute_implicit (SBUF)",
          "ag_hbm": "all_gather (HBM)", "cp_hbm": "collective_permute (HBM)"}


def load_sweep():
    pts = defaultdict(lambda: defaultdict(list))  # (method, route, tile) -> nstep -> [per-rank medians]
    fails = []
    files = [(f, ren) for sub, ren in SUBDIRS.items()
             for f in glob.glob(str(RAW / "*" / "runs" / "p2p" / sub / "*" / "f*" / "n*" / "result.json"))]
    for f, ren in files:
        p = Path(f)
        name, fb, n = p.parts[-4], int(p.parts[-3][1:]), int(p.parts[-2][1:])
        method, route = name.rsplit("-", 1)
        route = ren.get(route, route)
        r = json.loads(p.read_text())
        if not r.get("check"):
            fails.append({"method": method, "route": route, "tile": fb * 128, "nstep": n,
                          "why": r.get("check_note") or (r.get("error") or "")[-200:]})
            continue
        by = defaultdict(list)
        for row in r["rows"]:
            q = row["queues"].get("collectives")
            if q:
                by[row["rank"]].append(q["span_ns"])
        pts[(method, route, fb * 128)][n] = [float(np.median(v)) for v in by.values()]
    out = []
    for (method, route, tile), byn in pts.items():
        if len(byn) < 2:
            continue
        ns = sorted(byn)
        slope = float(np.polyfit(ns, [min(byn[n]) for n in ns], 1)[0])
        out.append({"method": method, "route": route, "tile": tile, "ns": slope, "cycles": slope * CLK_GHZ,
                    "GBps": tile / slope if slope > 0 else None, "nsteps": ns})
    return out, fails


def load_sendrecv():
    out = []
    for f in glob.glob(str(RAW / "*" / "runs" / "sendrecv" / "*" / "*" / "result.json")):
        r = json.loads(Path(f).read_text())
        if r.get("ns_per_step") and r.get("sim_ok") and r["kernel"] in ("xchg_dma", "xchg_gpsimd"):
            m = "sendrecv" if r["kernel"] == "xchg_dma" else "sendrecv_gpsimd"
            out.append({"method": m, "route": "m1", "tile": r["tile_bytes"], "ns": r["ns_per_step"],
                        "cycles": r["ns_per_step"] * CLK_GHZ, "GBps": r["tile_bytes"] / r["ns_per_step"]})
    return out


def main():
    rows, fails = load_sweep()
    rows += load_sendrecv()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "core-p2p.json").write_text(json.dumps(
        {"clockGHz": CLK_GHZ, "routes": ROUTE, "methods": METHOD, "rows": rows, "failures": fails}, indent=1))
    tiles = sorted({r["tile"] for r in rows})
    idx = {(r["method"], r["route"], r["tile"]): r for r in rows}
    hdr = "".join(f"{(str(t // 1024) + ' KiB') if t >= 1024 else str(t) + ' B':>16}" for t in tiles)
    print(f"{'method':<38}{'route':<30}{hdr}")
    for m in METHOD:
        for rt in ROUTE:
            if not any((m, rt, t) in idx for t in tiles):
                continue
            cells = ""
            for t in tiles:
                r = idx.get((m, rt, t))
                cells += f"{'-':>16}" if r is None else f"{r['cycles']:>8.0f}c {r['GBps']:5.1f}G"[-16:].rjust(16)
            print(f"{METHOD[m]:<38}{ROUTE[rt]:<30}{cells}")
    print("failed points:", len(fails))
    for f in sorted(fails, key=lambda x: (x["method"], x["route"], x["tile"])):
        print("  ", f["method"], f["route"], f["tile"], f["nstep"], f["why"][-120:])


if __name__ == "__main__":
    main()
