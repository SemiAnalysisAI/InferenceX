"""Compare raw ISA exchange (nisa.sendrecv / sendrecv_compute) with the runtime collective on one LNC pair.

  python3 sendrecv/analyze_sendrecv.py      # after cluster/fetch.sh; reads data/raw/*/runs/{sendrecv,distance}

Writes results/sendrecv.json (no figure). Both sides are per-exchange costs:
the ISA side is the slope of device time against chain length (fixed per-NEFF cost removed); nccom-test
reports per-op time, which does not change with ops per NEFF (-I 1..64 measured equal on cores 16-17).
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "data" / "raw"
LAB = {"xchg_dma": "nisa.sendrecv (DMA engine)", "xchg_gpsimd": "nisa.sendrecv (gpsimd_dma, <= 1 KiB/partition)",
       "xchg_reduce": "nisa.sendrecv_compute (add in DMA)", "nccom": "nccom-test permute (runtime collective)"}


def load():
    out = {k: [] for k in LAB}
    for f in glob.glob(str(RAW / "*" / "runs" / "sendrecv" / "*" / "*" / "result.json")):
        r = json.loads(Path(f).read_text())
        if r.get("ns_per_step"):
            out[r["kernel"]].append({"tileBytes": r["tile_bytes"], "nsPerStep": r["ns_per_step"],
                                     "fixedNs": r["fixed_ns"], "GBpsPerDir": r["GBps_per_direction"],
                                     "simOk": r["sim_ok"], "medians": r["median_ns"]})
    for f in glob.glob(str(RAW / "*" / "runs" / "distance" / "sendrecv-pair" / "*.json")):
        r = json.loads(Path(f).read_text())
        for x in r["rows"]:
            if "time_us" in x:
                out["nccom"].append({"tileBytes": x["bytes"], "nsPerStep": x["time_us"] * 1e3,
                                     "GBpsPerDir": x["algbw_GBps"], "pair": f"{r['src_core']}-{r['dst_core']}"})
    for k in out:
        out[k].sort(key=lambda r: r["tileBytes"])
    return out


def main():
    d = load()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sendrecv.json").write_text(json.dumps(d, indent=1))
    for k, rows in d.items():
        print(f"{LAB[k]:<48}", "  ".join(f"{r['tileBytes']}B:{r['nsPerStep'] / 1e3:.2f}us/{r['GBpsPerDir'] or 0:.0f}GB/s"
                                          for r in rows))


if __name__ == "__main__":
    main()
