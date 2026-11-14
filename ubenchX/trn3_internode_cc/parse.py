"""Collect trn3_internode_cc runs into one CSV (needs pandas).

  python ubenchX/trn3_internode_cc/parse.py ubenchX/trn3_internode_cc/runs/std_r0 --csv cc.csv

One row per (topology, op, buffer, size), failed sizes included with status=failed. nccom-test sizes: all_gather = output
bytes, reduce_scatter = input bytes, all_reduce / alltoall = buffer bytes per rank. busbw is nccom-test's:
algbw x (n-1)/n for all_gather, reduce_scatter, alltoall and x 2(n-1)/n for all_reduce, n = ranks.
efa_tx_GB is the largest per-node EFA transmit delta over the whole run, all sizes together: near zero means
the run stayed on NeuronLink.
"""
import argparse
import glob
import json
import os

import pandas as pd


def first_size(op: str, ranks: int, buf: str) -> int:
    # run.sh's start: HBM 8 B, raised for ops that split the buffer to one bf16 element per rank; SBUF 512 B,
    # all_gather two bf16 per partition per rank
    if buf == "sbuf":
        return 512 * ranks if op == "all_gather" else 512
    size = 8
    while op != "all_reduce" and size < 2 * ranks:
        size *= 2
    return size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(a.run_dir, "*.txt"))):
        meta = dict(kv.split("=", 1) for kv in open(meta_path).read().split())
        report = meta_path[:-4] + ".json"
        if not os.path.exists(report):
            continue
        ranks = int(meta["ranks"])
        before = [int(x) for x in meta["efa_tx_before"].split(",")]
        after = [int(x) for x in meta["efa_tx_after"].split(",")]
        efa_gb = max(b - a_ for a_, b in zip(before, after)) / 1e9
        buf = meta.get("buf", "hbm")
        size = first_size(meta["op"], ranks, buf)
        for r in json.load(open(report)).get("results", []):
            ok = "size(B)" in r
            rows.append({
                "topology": meta["name"].split("_lnc")[0], "nodes": len(meta["nodes"].split(",")),
                "lnc": int(meta["lnc"]), "ranks": ranks, "op": meta["op"], "buffer": buf.upper(),
                "bytes": int(r["size(B)"]) if ok else size, "status": "ok" if ok else "failed",
                "time_us": r.get("time:avg(us)"), "algbw_GBps": r.get("algbw(GB/s)"),
                "busbw_GBps": r.get("busbw(GB/s)"), "algorithm": r.get("algorithm"),
                "efa_tx_GB": round(efa_gb, 3),
            })
            size *= 2
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 2000):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
