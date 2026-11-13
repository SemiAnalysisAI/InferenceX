"""Collectives on physically unrouted core pairs (p2p/xcoll.sh) -> results/core-xcoll.json.

  cluster/fetch.sh && python3 p2p/analyze_xcoll.py

Per rank: slope of its median in-trace collectives span against chain length (1, 9, 33), in cycles at
1.2 GHz, for the stock labelling and for the shimmed one (map entries 1 and 6 swapped). Ranks pair as
(r, r^1). Physical cores come from the driver's stock map of chip 0 (logical 0..7 -> nc 4 5 6 7 2 3 0 1).
"""
import collections
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CLK = 1.2
STOCK = [4, 5, 6, 7, 2, 3, 0, 1]                   # logical NC -> physical nc on an even chip
DIAG = list(STOCK); DIAG[1], DIAG[6] = DIAG[6], DIAG[1]
SENGINE = lambda nc: nc // 2                       # noqa: E731
UNROUTED = {frozenset((2, 0)), frozenset((3, 1))}   # SEngine pairs with no collective route: {4,5}-{0,1}, {6,7}-{2,3}


def main():
    out = []
    for d in sorted(glob.glob(str(HERE / "data/raw/*/runs/xcoll/*-*"))):
        variant, kernel = Path(d).name.split("-", 1)
        phys = STOCK if variant == "stock" else DIAG
        spans, checks = collections.defaultdict(dict), []
        for f in sorted(glob.glob(d + "/f*/n*/result.json")):
            r = json.load(open(f))
            checks.append(bool(r.get("check")))
            per = collections.defaultdict(list)
            for row in r.get("rows", []):
                if "collectives" in row["queues"]:
                    per[row["rank"]].append(row["queues"]["collectives"]["span_ns"])
            for rank, v in per.items():
                spans[rank][r["nstep"]] = float(np.median(v))
        entry = {"variant": variant, "kernel": kernel, "outputsOk": bool(checks) and all(checks), "points": len(checks),
                 "error": None, "pairs": []}
        if not entry["outputsOk"]:
            r = json.load(open(sorted(glob.glob(d + "/f*/n*/result.json"))[0]))
            entry["error"] = (r.get("error") or r.get("check_note") or "")[-160:].strip()
        for a in range(0, 8, 2):
            cyc = []
            for rank in (a, a + 1):
                if len(spans.get(rank, {})) >= 2:
                    ns = sorted(spans[rank])
                    cyc.append(float(np.polyfit(ns, [spans[rank][n] for n in ns], 1)[0]) * CLK)
            se = frozenset((SENGINE(phys[a]), SENGINE(phys[a + 1])))
            kind = "same SEngine" if len(se) == 1 else ("diagonal (stock: no route)" if se in UNROUTED else "other SEngine")
            entry["pairs"].append({"ranks": [a, a + 1], "physical": [phys[a], phys[a + 1]], "kind": kind,
                                   "cyclesPerRank": cyc})
        out.append(entry)
    (HERE / "results" / "core-xcoll.json").write_text(json.dumps({"clockGHz": CLK, "runs": out}, indent=1))
    print("wrote results/core-xcoll.json")
    for e in out:
        print(f"\n{e['variant']:5} {e['kernel']:9} outputs ok: {e['outputsOk']} ({e['points']} points)" +
              (f"  error: {e['error']}" if e["error"] else ""))
        for p in e["pairs"]:
            c = " / ".join(f"{v / 1000:.1f}k" for v in p["cyclesPerRank"]) or "-"
            print(f"   ranks {p['ranks']} = nc{p['physical'][0]}-nc{p['physical'][1]}  {p['kind']:28} {c}")


if __name__ == "__main__":
    main()
