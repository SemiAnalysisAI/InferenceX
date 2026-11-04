"""Compute DMA bandwidth from run_point.py outputs (needs pandas + pyarrow).

  python ubenchX/trn3_dma_bw/parse.py RUN_DIR... --csv single.csv   # one row per profiled execution

Bandwidth = bytes moved / (last packet end - first packet start) within one profiled execution,
from the profiler's per-packet DmaPacket table (the DMA queue type that carries the burst):
  load_*   HBM -> SBUF packets
  store_*  SBUF -> HBM packets
The window is an envelope: any idle gap inside the burst counts against the number.
bytes_ok checks that every byte of the burst appears in the profile (if not, GBps covers only the
captured part of the burst). kernel_GBps_lb = all bytes / whole kernel execution time, a
profiler-independent lower bound. GBps_agg_table repeats the calculation on the per-transfer
DmaPacketAggregated table when the profiler exports it.
"""

import argparse
import glob
import json
import os

import pandas as pd


def read_table(pq_dir, name):
    files = sorted(glob.glob(os.path.join(pq_dir, "**", f"{name}.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def parse_trace(meta, pq_dir):
    load = meta["kernel"].startswith("load")
    pk = read_table(pq_dir, "DmaPacket")
    qt = pk.queue_type.astype(str)
    # The burst is the queue type carrying the most bytes, ignoring instruction fetches (static DMA:
    # "input", HW DGE: "hardware_dynamic", SW DGE: "software_dynamic").
    moved = pk[qt != "instruction"].groupby(qt[qt != "instruction"]).transfer_bytes.sum()
    burst = pk[qt == moved.idxmax()]
    data = burst[burst.transfer_bytes > 64]  # 4-byte packets are DMA completion (semaphore) writes
    # Static DMA folds the 4-byte completion write into the last packet of a transfer (e.g.
    # 16384 + 4 B); count payload only.
    payload = data.transfer_bytes - data.transfer_bytes % 64
    # With SW DGE the load kernels' final output store rides the same queue type as the burst. It
    # depends on the reduce over every ring slot, so it starts only after every burst packet has
    # ended: keep packets in start order up to the burst's byte count, and check the cut is clean.
    trimmed_bytes = 0
    if payload.sum() > meta["total_bytes"]:
        order = data.start_ts.sort_values(kind="stable").index
        data, payload = data.loc[order], payload.loc[order]
        keep = (payload.cumsum() <= meta["total_bytes"]).to_numpy()
        rest = data[~keep]
        trimmed_bytes = int(payload[~keep].sum())
        data, payload = data[keep], payload[keep]
        if len(rest) and rest.start_ts.min() < data.end_ts.max():
            raise RuntimeError(f"{pq_dir}: packets past the burst's bytes overlap the burst; cannot separate them")
    start, end = data.start_ts.min(), data.end_ts.max()
    nbytes = int(payload.sum())
    sizes = payload.value_counts()
    row = {
        "GBps": nbytes / (end - start),
        "window_us": (end - start) / 1e3,
        "bytes_ok": nbytes == meta["total_bytes"],
        "pkt_bytes": int(sizes.idxmax()),
        "pkt_sizes": "/".join(str(int(s)) for s in sorted(sizes.index)),
        "packets": len(data),
        "trimmed_bytes": trimmed_bytes,  # same-queue bytes after the burst (SW DGE loads' output store)
        "dma_engines": int(data.engine_idx.nunique()),
        "pkt_GBps_med": float((payload / data.duration_ns).median()),
    }
    # Profiler-independent lower bound: all bytes over the whole kernel execution (includes setup,
    # the load kernels' final reduce/store, and any DMA ramp).
    ex = read_table(pq_dir, "ExecutionInfo")
    row["kernel_GBps_lb"] = meta["total_bytes"] / float(ex.execution_end_ts.iloc[0] - ex.execution_start_ts.iloc[0])
    # Cross-check against the per-transfer table when the profiler exports it.
    agg = read_table(pq_dir, "DmaPacketAggregated")
    if agg is not None:
        src, dst, col = ("INPUT", "SB", "read_bytes") if load else ("SB", "OUTPUT", "write_bytes")
        agg = agg[agg.source.astype(str).str.contains(src, regex=False)
                  & agg.dest.astype(str).str.contains(dst, regex=False)]
        a0, a1 = agg.start_ts.min(), (agg.start_ts + agg.duration_ns.fillna(0)).max()
        row["GBps_agg_table"] = agg[col].fillna(0).sum() / (a1 - a0)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run_point.py --out directories")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for d in a.runs:
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        for k in meta.get("profiled_execs", []):
            r = {key: meta[key] for key in ("kernel", "pdim", "fdim", "row_bytes", "transfer_bytes",
                                            "num_transfers", "total_bytes", "core")}
            r.update({"run": os.path.basename(os.path.normpath(d)), "exec": k})
            r.update(parse_trace(meta, os.path.join(d, f"parquet_exec_{k}")))
            rows.append(r)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        print(df.drop(columns=["total_bytes"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
