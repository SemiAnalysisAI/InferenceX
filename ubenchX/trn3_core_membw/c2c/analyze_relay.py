"""Two-hop core-to-core routes on Trainium3, from c2c/run_relay.py (HBM collectives, 8 ranks per chip).

  python3 c2c/analyze_relay.py     # after cluster/fetch.sh; the figure is c2c/figure_c2c.py

Per (chip, sequence): each rank's cost per collective = slope of its in-trace collectives span against the
number of collectives (median over profiled executions per point). A one-hop exchange is one collective; a
two-hop relay is two. Pair value = mean over the pair's two ranks. Writes results/core-c2c-twohop.json.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "data" / "raw"
ONE = {"1": "LNC pair (LOCAL)", "2": "same die, other pair (R)", "3": "same die, other pair (R)",
       "4": "other die, same lane (D2D)", "5": "other die, other lane (D2D)"}
CLK_GHZ = 1.2  # profiler Metadata.neuroncore_clock_freq
TWO = {"4-2": (4, 2), "2-4": (2, 4), "4-3": (4, 3), "3-4": (3, 4)}


def load():
    pts = defaultdict(lambda: defaultdict(dict))  # (chip, seq) -> nstep -> rank -> median span
    raw = defaultdict(lambda: defaultdict(list))  # (chip, seq) -> nstep -> every rank/exec span
    fails = []
    for f in glob.glob(str(RAW / "*" / "runs" / "relay" / "chip*" / "s*" / "n*" / "result.json")):
        r = json.loads(Path(f).read_text())
        if not r.get("check"):
            fails.append({k: r.get(k) for k in ("chip", "seq", "nstep", "check_note", "error")})
            continue
        by = defaultdict(list)
        for row in r["rows"]:
            if row["span_ns"]:
                by[row["rank"]].append(row["span_ns"])
        pts[(r["chip"], r["seq"])][r["nstep"]] = {k: float(np.median(v)) for k, v in by.items()}
        raw[(r["chip"], r["seq"])][r["nstep"]] += [x for v in by.values() for x in v]
    # Floor estimator. Every collective of a chain involves all 8 ranks, and in alternating (relay) chains a delay on
    # any rank propagates to all of them, so a relay's cost is a chip-wide quantity. Per (chip, seq, nstep):
    # median over executions per rank, then the minimum over ranks (the rank that waits least). Per-collective
    # cost = slope of that against nstep. Stalls are counted separately against the fitted floor.
    rows, stalls = [], defaultdict(lambda: [0, 0])
    for (chip, seq), byn in pts.items():
        if len(byn) < 2:
            continue
        ns = np.array(sorted(byn))
        floor = np.array([min(byn[n].values()) for n in ns])
        slope, icpt = np.polyfit(ns, floor, 1)
        for n in ns:
            for v in raw[(chip, seq)][n]:
                stalls[seq][0] += v > 1.5 * (slope * n + icpt)
                stalls[seq][1] += 1
        if seq in ONE:
            m, hops = int(seq), 1
        else:
            m, hops = TWO[seq][0] ^ TWO[seq][1], 2
        for lo in range(8):
            hi = lo ^ m
            if lo < hi:
                rows.append({"chip": chip, "seq": seq, "a": lo, "b": hi, "hops": hops,
                             "via": lo ^ TWO[seq][0] if hops == 2 else None, "nsPerExchange": hops * float(slope)})
    load.stalls = {k: {"over_1p5x_floor": int(a), "spans": int(b)} for k, (a, b) in stalls.items()}
    return rows, fails


def main():
    rows, fails = load()
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r["nsPerExchange"])
    summ = {s: {"median_ns": float(np.median(v)), "min_ns": float(np.min(v)), "max_ns": float(np.max(v)), "n": len(v)}
            for s, v in by_seq.items()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "core-c2c-twohop.json").write_text(json.dumps(
        {"clockGHz": CLK_GHZ, "pairs": rows, "bySequence": summ, "failures": fails, "stalls": load.stalls,
         "method": "HBM nki.collectives.all_gather chains, 8 ranks per chip, replica group cycling through XOR "
                   "matchings; cost per collective = slope of in-trace collectives span vs collectives; a relay "
                   "is two collectives."}, indent=1))
    for s in list(ONE) + list(TWO):
        if s in summ:
            v = summ[s]
            print(f"{s:>4} {'1 hop' if s in ONE else '2 hops':7} median {v['median_ns'] / 1e3:6.2f} us = {v['median_ns'] * CLK_GHZ:6.0f} cyc "
                  f"[{v['min_ns'] / 1e3:.2f}, {v['max_ns'] / 1e3:.2f}] n={v['n']}")
    print("failures:", len(fails))
    for k, v in load.stalls.items():
        print(f"stalls {k:>4}: {v['over_1p5x_floor']}/{v['spans']} rank-exec spans > 1.5x floor")


if __name__ == "__main__":
    main()
