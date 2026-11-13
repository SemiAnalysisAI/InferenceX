"""nisa.sendrecv across SEngines (sendrecv/xseng.sh, sendrecv/ncmap_shim.c) -> results/core-sendrecv-xseng.json.

  cluster/fetch.sh && python3 sendrecv/analyze_xseng.py

Logical NC 0 exchanges with logical NC b; b = 1 is the stock LNC pair. Cost per exchange = slope of whole-NEFF
device time against chain length, in cycles at 1.2 GHz. A row counts only if the device output matched at
every chain length that ran, and `physicalCores` is what the driver's per-core success counters say executed.
"""
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CLK = 1.2
ROUTE = {1: "LNC pair (same SEngine)", 2: "same die, other SEngine", 3: "same die, other SEngine",
         4: "other die (i+4)", 5: "other die, other lane", 6: "diagonal (unrouted for collectives)",
         7: "diagonal (unrouted for collectives)"}


def main():
    rows = []
    for f in sorted(glob.glob(str(HERE / "data/raw/*/runs/xseng*/b*/*/*/result.json"))):
        r = json.load(open(f))
        b = int(re.search(r"/b(\d+)/", f)[1])
        if "xseng-chips" in f:
            continue
        cores = Path(f).parents[2] / "physical_cores.txt"
        ok = r.get("device_ok") or {}
        rows.append({"partner": b, "route": ROUTE[b], "kernel": r["kernel"], "tileBytes": r["tile_bytes"],
                     "cycles": r["ns_per_step"] * CLK if r.get("ns_per_step") else None,
                     "GBpsPerDirection": r.get("GBps_per_direction"), "deviceOk": bool(ok) and all(ok.values()),
                     "chainLengths": sorted(int(k) for k in ok), "swap": r.get("ncmap_swap"),
                     "physicalCores": cores.read_text().split() if cores.exists() else None,
                     "node": Path(f).parents[6].name, "run": Path(f).parents[3].name,
                     "chip": int(r["ncmap_swap"].split(":")[0]) // 8 if r.get("ncmap_swap") else 0})
    seen = {r["partner"]: r["physicalCores"] for r in rows if r["physicalCores"]}   # counters, per partner
    for r in rows:
        r["physicalCores"] = seen.get(r["partner"])
    best = {}
    # One row per (partner, kernel, tile). Prefer a verified row from the long-chain run (xseng-long: every
    # partner on the same chip with the same chain lengths 64-256), so partners are compared like for like;
    # the first run mixed chain lengths and measured partner 3 on chip 1.
    rank = lambda r: (r["deviceOk"] and bool(r["cycles"]), r["run"] == "xseng-long", r["chip"] == 0, len(r["chainLengths"]))  # noqa: E731
    for r in rows:
        key = (r["partner"], r["kernel"], r["tileBytes"])
        if key not in best or rank(r) > rank(best[key]):
            best[key] = r
    rows = [best[k] for k in sorted(best)]
    (HERE / "results" / "core-sendrecv-xseng.json").write_text(json.dumps(
        {"clockGHz": CLK, "method": "LD_PRELOAD shim swaps logical-to-physical NC map entries 1 and b; stock LNC=2 "
         "sendrecv NEFF; slope of device time against chain length", "rows": rows}, indent=1))
    print("wrote results/core-sendrecv-xseng.json")
    tiles = sorted({r["tileBytes"] for r in rows})
    for kernel in sorted({r["kernel"] for r in rows}):
        print(f"\n{kernel}: cycles per exchange (GB/s per direction)\n{'b':>2} {'physical':>10} {'route':38}" +
              "".join(f"{t:>22}" for t in tiles))
        for b in sorted({r["partner"] for r in rows}):
            cells, phys = "", ""
            for t in tiles:
                r = best.get((b, kernel, t))
                phys = "+".join(r["physicalCores"]) if r and r["physicalCores"] else phys
                cells += f"{'-' if not r else ('FAIL' if not r['deviceOk'] or not r['cycles'] else f'{r['cycles']:,.0f} ({r['GBpsPerDirection']:.0f})'):>22}"
            print(f"{b:>2} {phys:>10} {ROUTE[b]:38}{cells}")


if __name__ == "__main__":
    main()
