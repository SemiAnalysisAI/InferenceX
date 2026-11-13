"""Core to core across chips, from p2p/xchip.sh (512 B per exchange).

  python3 p2p/analyze_xchip.py     # after cluster/fetch.sh; the figure is c2c/figure_xchip.py

A run pairs rank r with r ^ (8*CM + KM): core i of chip c with core i^KM of chip c^CM, every pair of the launch at
once. Per rank: median `collectives` DMA queue span over the profiled executions at each chain length (1, 9, 33),
then the least-squares slope against chain length = that rank's cost per exchange. Other-plane routes put no packets on
`collectives`; there the `instruction` queue span is used. The whole-NEFF slope is kept for every run (`cyclesMedianNeff`). Per pair: the mean of its two
ranks. Per run ("floor", the estimator of p2p/analyze_p2p.py): the slope of the least-waiting rank at each length.
Only points whose output check passed are used; a run whose every point failed is recorded with the runtime's reason.
Runs are kept per node; `runs` holds one per (kernel, chip mask, core mask), node 0 first (PREFER), `allRuns` all of them.
Writes results/core-xchip.json (`runs`; `fullRuns` = the all-16-chip launches of `xchip.sh full`).
"""
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CLK_GHZ = 1.2
REASONS = ("not connected physically", "no_hier no_mesh", "sender receiver table", "failed to init ENC comm", "Failed to build and load collectives")


NODES = {f"node{i}": i for i in range(4)}   # data/raw/<dir> -> UltraServer index; TRN3_HOSTS (4 ips in index order) adds address-named dirs
NODES.update({h: i for i, h in enumerate(os.environ.get("TRN3_HOSTS", "").split())})
PREFER = [(0, "xchip-1n"), (0, "xchip"), (3, "xchip"), (1, "xchip")]   # which launch a figure cell is drawn from


def collect(subs):
    """Every launch under runs/p2p/<sub> for sub in subs, one record per (node, sub, kernel, chip mask, core mask)."""
    runs = defaultdict(lambda: {"q": {}, "neff": {}, "errors": []})
    for sub in subs:
        for f in glob.glob(str(HERE / "data" / "raw" / "*" / "runs" / "p2p" / sub / "*" / "f4" / "n*" / "result.json")):
            p = Path(f)
            m = re.fullmatch(r"(\w+)-c(\d+)k(\d+)", p.parts[-4])
            key, n = (NODES[p.parts[-8]], sub, m[1], int(m[2]), int(m[3])), int(p.parts[-2][1:])
            r = json.loads(p.read_text())
            if not r.get("check"):
                runs[key]["errors"].append(r.get("error") or r.get("check_note") or "")
                continue
            by, ex = defaultdict(lambda: defaultdict(list)), defaultdict(list)
            for row in r["rows"]:
                ex[row["rank"]].append(row["exec_ns"])
                for q in ("collectives", "instruction"):
                    if q in row["queues"]:
                        by[q][row["rank"]].append(row["queues"][q]["span_ns"])
            runs[key]["q"][n] = {q: {k: float(np.median(v)) for k, v in d.items()} for q, d in by.items()}
            runs[key]["neff"][n] = {k: float(np.median(v)) for k, v in ex.items()}
    out = []
    for (node, sub, kernel, cm, km), d in sorted(runs.items()):
        rec = {"node": node, "launch": sub, "kernel": kernel, "chipMask": cm, "coreMask": km}
        ns = sorted(d["neff"])
        if len(ns) < 2:
            err = " ".join(d["errors"])
            rec.update(ok=False, reason=next((x for x in REASONS if x in err), err[-160:]))
            out.append(rec)
            continue
        # In-trace clock: the `collectives` DMA queue span where the exchange is on it. Other-plane routes put nothing
        # there; their exchange is on the `instruction` queue, whose span is then the in-trace clock. The whole-NEFF
        # slope is kept beside it for every run as a cross-check (`cyclesMedianNeff`).
        if all("collectives" in d["q"][n] for n in ns):
            clock, byn = "collectives span", {n: d["q"][n]["collectives"] for n in ns}
        elif all("instruction" in d["q"][n] for n in ns):
            clock, byn = "instruction span", {n: d["q"][n]["instruction"] for n in ns}
        else:
            clock, byn = "whole NEFF", d["neff"]
        mask = 8 * cm + km

        def fit(byn):
            ranks = sorted(set.intersection(*(set(byn[n]) for n in ns)))
            slope = {r: float(np.polyfit(ns, [byn[n][r] for n in ns], 1)[0]) * CLK_GHZ for r in ranks}
            floor = float(np.polyfit(ns, [min(byn[n].values()) for n in ns], 1)[0]) * CLK_GHZ
            pairs = [{"chipA": r // 8, "coreA": r % 8, "chipB": (r ^ mask) // 8, "coreB": (r ^ mask) % 8,
                      "cyclesA": slope[r], "cyclesB": slope[r ^ mask], "cycles": (slope[r] + slope[r ^ mask]) / 2}
                     for r in ranks if r < r ^ mask and (r ^ mask) in slope]
            return ranks, floor, pairs

        ranks, floor, pairs = fit(byn)
        v = [p["cycles"] for p in pairs]
        neff = float(np.median([p["cycles"] for p in fit(d["neff"])[2]]))
        rec.update(ok=True, clock=clock, ranks=len(ranks), nsteps=ns, cyclesFloor=floor, cyclesMedian=float(np.median(v)),
                   cyclesMin=min(v), cyclesMax=max(v), cyclesMedianNeff=neff, failedPoints=len(d["errors"]), pairs=pairs)
        out.append(rec)
    return out


def pick(recs):
    """One record per (kernel, chip mask, core mask): the first launch in PREFER that has it."""
    best = {}
    for r in recs:
        k, o = (r["kernel"], r["chipMask"], r["coreMask"]), (r["node"], r["launch"])
        rank = PREFER.index(o) if o in PREFER else len(PREFER)
        if k not in best or rank < best[k][0]:
            best[k] = (rank, r)
    return [best[k][1] for k in sorted(best)]


def main():
    # `runs`: the launch each figure cell is drawn from (PREFER: node 0 first); `allRuns`: every launch on every node.
    # full: `xchip.sh full`, offsets 1..7 with all 16 chips as ranks.
    every, fullall = collect(["xchip", "xchip-1n"]), collect(["xchip-full"])
    out = pick(every)
    f0 = [r for r in fullall if r["node"] == 0]
    full = pick(f0) + [r for r in pick(fullall) if (r["kernel"], r["chipMask"]) not in {(x["kernel"], x["chipMask"]) for x in f0}]
    (HERE / "results" / "core-xchip.json").write_text(json.dumps(
        {"clockGHz": CLK_GHZ, "tileBytes": 512, "runs": out, "fullRuns": full, "allRuns": every, "allFullRuns": fullall}, indent=1))
    for r in out + full:
        tag = f"n{r['node']} {r['launch']:<10} {r['kernel']:<9} chip^{r['chipMask']:<2} core^{r['coreMask']}"
        if r["ok"]:
            print(f"{tag}  floor {r['cyclesFloor']:7.0f}  pairs median {r['cyclesMedian']:7.0f}  neff {r['cyclesMedianNeff']:7.0f}  "
                  f"({r['cyclesMin']:.0f}-{r['cyclesMax']:.0f}, {len(r['pairs'])} pairs, {r['failedPoints']} failed points, {r['clock']})")
        else:
            print(f"{tag}  FAILED: {r['reason']}")
    used = sorted({(r["node"], r["launch"]) for r in out + full})
    print("launches drawn from:", used)


if __name__ == "__main__":
    main()
