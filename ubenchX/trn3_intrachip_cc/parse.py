"""Collect nccom-test JSON reports from run_placement.sh into one CSV (needs pandas).

  python ubenchX/trn3_intrachip_cc/parse.py runs/sweep_r0 --csv cc.csv

One row per (placement, op, buffer, size). nccom-test sizes: all_gather = output bytes, reduce_scatter =
input bytes, all_reduce / all_to_all = buffer bytes per rank. time_us is the average collective time
over the timed iterations; cycles at the 1.2 GHz NeuronCore clock. algbw = size / time; busbw = algbw x
the op's traffic factor (nccom-test's: (n-1)/n for all_gather, reduce_scatter, all_to_all; 2(n-1)/n for
all_reduce, n = ranks per replica group).
"""
import argparse
import glob
import json
import os

import pandas as pd

CLOCK_GHZ = 1.2
OPS = {"allg_sbuf": ("all_gather", "SBUF"), "allr_sbuf": ("all_reduce", "SBUF"), "redsct_sbuf": ("reduce_scatter", "SBUF"), "all_gather_hbm": ("all_gather", "HBM"),
       "all_reduce_hbm": ("all_reduce", "HBM"), "reduce_scatter_hbm": ("reduce_scatter", "HBM"),
       "alltoall_hbm": ("all_to_all", "HBM")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for pdir in sorted(glob.glob(os.path.join(a.sweep_dir, "*"))):
        info = open(os.path.join(pdir, "placement.txt")).read().split() if os.path.exists(
            os.path.join(pdir, "placement.txt")) else [os.path.basename(pdir)]
        kv = dict(x.split("=") for x in info[1:])
        for name, (op, buf) in OPS.items():
            f = os.path.join(pdir, f"{name}.json")
            if not os.path.exists(f):
                continue
            for r in json.load(open(f)).get("results", []):
                if "size(B)" not in r:  # a failed size: {"error_message": ..., "error_log": ...}
                    continue
                t = r.get("time:avg(us)")
                rows.append({"placement": os.path.basename(pdir), "lnc": int(kv.get("lnc", 0)), "ranks": int(kv.get("ranks", 0)),
                             "group": int(kv.get("group", 0)), "op": op, "buffer": buf, "bytes": int(r["size(B)"]),
                             "time_us": t, "cycles": None if t is None else t * 1e3 * CLOCK_GHZ,
                             "algbw_GBps": r.get("algbw(GB/s)"), "busbw_GBps": r.get("busbw(GB/s)"),
                             "algorithm": r.get("algorithm")})
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 600):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
