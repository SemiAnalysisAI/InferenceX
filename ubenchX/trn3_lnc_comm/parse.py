"""Per-operation latency and bandwidth for the intra-LNC communication kernels (needs pandas + pyarrow).

  python ubenchX/trn3_lnc_comm/parse.py RUN_DIR... --csv lnc.csv   # one row per (trace, physical core)

Each kernel issues num_ops operations per core. An operation's data moves as DMA packets on the
loop's queue family (the non-Q0, non-host family with the most bytes: qSyncSpillReload0 for
sendrecv / sendrecv_compute, Q1 for dma_copy); every queue carries the same number of packets per
operation, so the k-th operation is packets [k m, (k+1) m) of each queue in time order. Per
operation:

  start / end   first packet start / last packet end over all queues (for sr_add the end is the
                k-th VectorE TENSOR_TENSOR end if later; for sr_gpsimd, whose GpSimd-internal DMA
                makes no packets, the k-th GpSimd instruction of the loop)
  xfer          end - start: the operation's data movement alone (first packet start to last end)
  op            median interval between consecutive operation ends: the time one operation takes,
                including the cross-core synchronization before its transfer starts. Operations do not
                overlap (op > xfer in every kernel), so this is the end-to-end operation latency.

Over operations SKIP..num_ops-1. bw_GBps = bytes per rank / op (data each core receives per
operation per unit time; both directions together move twice that). Cycles at the 1.2 GHz
NeuronCore clock from the trace Metadata.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SKIP = 8


def read_table(d, name):
    files = sorted(glob.glob(os.path.join(d, "**", f"{name}.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def op_windows_dma(dp, n_ops):
    fam = dp.queue_name.str.replace(r"-E\d+$", "", regex=True)
    loop = dp[~fam.str.startswith(("Q0", "Host-device"))]
    fam = fam[loop.index]
    # candidate families: every queue carries a whole number of packets per operation (this skips small
    # housekeeping queues such as Q14, which can out-weigh the loop's bytes at the smallest sizes)
    per_q = loop.groupby([fam, loop.queue_name]).size()
    ok = per_q.groupby(level=0).apply(lambda s: bool((s % n_ops == 0).all() and (s >= n_ops).all()))
    sizes = loop.groupby(fam).transfer_bytes.sum()
    best = sizes[ok[ok].index].idxmax()
    loop = loop[fam == best]
    starts, ends = np.full(n_ops, np.inf), np.full(n_ops, -np.inf)
    for _, q in loop.groupby("queue_name"):
        q = q.sort_values("start_ts")
        if len(q) % n_ops:
            raise RuntimeError(f"queue {q.queue_name.iloc[0]}: {len(q)} packets not a multiple of {n_ops} ops")
        m = len(q) // n_ops
        k = np.arange(len(q)) // m
        np.minimum.at(starts, k, q.start_ts.to_numpy())
        np.maximum.at(ends, k, (q.start_ts + q.duration_ns).to_numpy())
    return starts, ends, best


def parse_core(meta, ins, dp, core, clock):
    n = meta["num_ops"]
    ins = ins[ins.pcore_idx == core]
    if meta["kernel"] == "sr_gpsimd":
        g = ins[(ins.engine.astype(str) == "GpSimd") & (ins.opcode.astype(str) == "UNKNOWN")].sort_values("start_ts")
        if len(g) != n:
            raise RuntimeError(f"expected {n} GpSimd swap instructions, got {len(g)}")
        starts, ends, fam = g.start_ts.to_numpy(), (g.start_ts + g.duration_ns).to_numpy(), "gpsimd"
    else:
        starts, ends, fam = op_windows_dma(dp[dp.pcore_idx == core], n)
        if meta["kernel"] == "sr_add":
            v = ins[(ins.engine.astype(str) == "Vector") & (ins.opcode.astype(str) == "TENSOR_TENSOR")]
            v = v.sort_values("start_ts")
            if len(v) != n:
                raise RuntimeError(f"expected {n} TENSOR_TENSOR, got {len(v)}")
            ends = np.maximum(ends, (v.start_ts + v.duration_ns).to_numpy())
    xfer = float(np.median((ends - starts)[SKIP:]))
    op = float(np.median(np.diff(ends[SKIP:])))
    return {"core": core, "queue_family": fam, "op_ns": op, "op_cycles": op * clock, "xfer_ns": xfer,
            "xfer_cycles": xfer * clock, "bw_GBps": meta["bytes"] / op}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for d in a.runs:
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        for t in meta.get("traces", []):
            p = os.path.join(d, t)
            ins, dp = read_table(p, "Instruction"), read_table(p, "DmaPacket")
            clock = float(read_table(p, "Metadata").neuroncore_clock_freq.iloc[0])
            for core in sorted(ins.pcore_idx.unique()):
                r = {k: meta[k] for k in ("kernel", "dtype", "bytes", "P", "F", "num_ops", "lcore")}
                r.update({"run": os.path.basename(os.path.normpath(d)), "trace": t})
                r.update(parse_core(meta, ins, dp, core, clock))
                rows.append(r)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.drop(columns=["run", "num_ops", "F"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
