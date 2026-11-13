"""SBUF -> SBUF core-to-core latency matrix (8 x 8 per chip) from c2c/run_c2c.py, next to the HBM-routed
runtime permute (nccom-test, distance/run_pairs.py) and the LNC-pair ISA exchange (sendrecv/).

  python3 c2c/analyze_c2c.py     # after cluster/fetch.sh; then c2c/analyze_relay.py, c2c/figure_c2c.py

Per (chip, pair, tile size): each rank's per-exchange cost is the least-squares slope of its in-trace
collectives span against NSTEP (median over profiled executions per NSTEP); the pair's value is the mean
of its two ranks. Only points whose output check passed are used.

Writes results/core-c2c.json (the figure is c2c/figure_c2c.py).
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "data" / "raw"
LAT_FB = 4  # bytes per partition for the latency matrix (512 B tile)
# NeuronCore clock from the profiler's Metadata table (neuroncore_clock_freq; GpSimd, Vector and Scalar engines
# the same, TensorEngine 2.4). A configured value recorded in the trace, not a measured live clock.
CLK_GHZ = 1.2


def load_sbuf():
    pts = defaultdict(dict)  # (chip, mask, fb) -> {nstep: {rank: median span}}
    meta, fails = {}, []
    for f in glob.glob(str(RAW / "*" / "runs" / "c2c" / "chip*" / "m*" / "f*" / "n*" / "result.json")):
        r = json.loads(Path(f).read_text())
        if not r.get("check"):
            fails.append({k: r.get(k) for k in ("chip", "mask", "bytes_per_partition", "nstep", "check_note", "error")})
            continue
        by = defaultdict(list)
        for row in r["rows"]:
            if row["span_ns"]:
                by[row["rank"]].append(row["span_ns"])
        key = (r["chip"], r["mask"], r["bytes_per_partition"])
        pts[key][r["nstep"]] = {k: float(np.median(v)) for k, v in by.items()}
        meta[key] = r["pairs"]
    out = []
    for key, byn in pts.items():
        if len(byn) < 2:
            continue
        chip, mask, fb = key
        ns = np.array(sorted(byn))
        # Floor estimator, the same as c2c/analyze_relay.py: per NSTEP, the minimum over the chip's 8 ranks of the
        # per-rank median span; slope against NSTEP. One chip-wide value per (mask, size).
        floor = float(np.polyfit(ns, [min(byn[n].values()) for n in ns], 1)[0])
        for lo, hi in meta[key]:
            slopes = []
            for rk in (lo, hi):
                ts = [byn[n].get(rk) for n in ns]
                if None not in ts:
                    slopes.append(np.polyfit(ns, ts, 1)[0])
            if slopes:
                out.append({"chip": chip, "a": lo, "b": hi, "cls": "lnc-pair" if mask == 1 else "cross-half (i, i+4)",
                            "bytesPerPartition": fb, "tileBytes": fb * 128, "nsPerExchange": float(np.mean(slopes)),
                            "cyclesPerExchange": float(np.mean(slopes)) * CLK_GHZ,
                            "nsFloor": floor, "cyclesFloor": floor * CLK_GHZ,
                            "rankSlopesNs": [float(s) for s in slopes], "nsteps": ns.tolist()})
    return out, fails


def load_hbm():
    out = []
    for f in glob.glob(str(RAW / "*" / "runs" / "distance" / "intra-*" / "*.json")):
        r = json.loads(Path(f).read_text())
        a, b = int(r["src_core"]), int(r["dst_core"])
        for x in r["rows"]:
            if "time_us" in x:
                out.append({"chip": a // 8, "a": a % 8, "b": b % 8, "bytes": x["bytes"], "us": x["time_us"],
                            "GBps": x.get("algbw_GBps")})
        if not any("time_us" in x for x in r["rows"]):
            out.append({"chip": a // 8, "a": a % 8, "b": b % 8, "bytes": None, "us": None, "unroutable": True})
    return out


def main():
    sb, fails = load_sbuf()
    hbm = load_hbm()
    sizes = sorted({r["bytesPerPartition"] for r in sb})
    summary = {}
    for fb in sizes:
        for cls in ("lnc-pair", "cross-half (i, i+4)"):
            v = [r["nsPerExchange"] for r in sb if r["bytesPerPartition"] == fb and r["cls"] == cls]
            fl = [r["nsFloor"] for r in sb if r["bytesPerPartition"] == fb and r["cls"] == cls]
            if v:
                summary[f"{cls} {fb * 128}B"] = {"median_ns": float(np.median(v)), "min_ns": float(np.min(v)),
                                                 "max_ns": float(np.max(v)), "n": len(v),
                                                 "floor_median_ns": float(np.median(fl)),
                                                 "floor_min_ns": float(np.min(fl)), "floor_max_ns": float(np.max(fl))}
    unrout = sorted({(r["a"], r["b"]) for r in hbm if r.get("unroutable")})
    res = {"clockGHz": CLK_GHZ, "clockSource": "profiler Metadata.neuroncore_clock_freq", "sbuf": sb, "sbufSummary": summary, "sbufFailures": fails, "hbmPermute": hbm,
           "unroutablePairs": unrout,
           "method": "SBUF: chain of 2-rank nki.collectives.all_gather on SBUF tiles, all 8 cores of a chip launched "
                     "as 8 ranks (4 disjoint pairs at once), per-exchange = slope of in-trace collectives span vs "
                     "chain length. HBM: nccom-test permute exchange, per-op time."}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "core-c2c.json").write_text(json.dumps(res, indent=1))
    for k, v in summary.items():
        print(f"{k:<34} median {v['median_ns'] / 1e3:7.2f} us = {v['median_ns'] * CLK_GHZ:6.0f} cycles  "
              f"[{v['min_ns'] * CLK_GHZ:.0f}, {v['max_ns'] * CLK_GHZ:.0f}]  n={v['n']}  floor {v['floor_median_ns'] * CLK_GHZ:6.0f}")
    print("unroutable pairs (runtime, HBM permute):", unrout, " SBUF failures:", len(fails))


if __name__ == "__main__":
    main()
