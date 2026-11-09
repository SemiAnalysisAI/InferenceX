"""Compute PSUM -> SBUF copy bandwidth from run_point.py outputs (needs pandas + pyarrow).

  python ubenchX/trn3_psum_bw/parse.py RUN_DIR... --csv psum.csv   # one row per profiled execution

From the profiler's per-instruction Instruction table, the burst is every instruction that reads
PSUM (the tensor_copy instructions on the chosen engine). Bandwidth = PSUM bytes read / (last copy
end - first copy start) within one profiled execution, so idle gaps count against it; dst_GBps uses
the SBUF bytes written instead (equal to GBps for fp32 -> fp32). bytes_ok checks that every
copy of the burst is in the trace. ns_per_copy is the window over the number of copies, dur_ns the
median instruction duration (copies overlap in the engine pipeline, so it can exceed ns_per_copy)
and wait_ns the median semaphore wait (about 0 when copies issue back to back).

When PSUM is read by two engines (dual_alternate), GBps is the aggregate over the combined window
and each engine also gets <Engine>_GBps (its bytes over its own window), <Engine>_ns_per_copy,
<Engine>_ii_ns (start-to-start interval) and <Engine>_wait_ns; overlap is the fraction of the
combined window covered by both engines' windows.
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
    ins = read_table(pq_dir, "Instruction")
    burst = ins[ins.psum_read_bytes.fillna(0) > 0].sort_values("start_ts")
    engines = sorted(burst.engine.astype(str).unique(), key=lambda e: e != "Vector")
    start, end = burst.start_ts.min(), burst.end_ts.max()
    src = int(burst.psum_read_bytes.sum())
    row = {
        "engine_seen": "+".join(engines),
        "copies_seen": len(burst),
        "GBps": src / (end - start),
        "dst_GBps": burst.sbuf_write_bytes.sum() / (end - start),
        "window_us": (end - start) / 1e3,
        "ns_per_copy": (end - start) / len(burst),
        "dur_ns": float(burst.duration_ns.median()),
        "wait_ns": float(burst.evt_wait_time_ns.median()),
        "bytes_ok": src == meta["total_src_bytes"] and len(burst) == meta["num_copies"],
    }
    if len(engines) > 1:
        lo, hi = [], []
        for e in engines:
            b = burst[burst.engine.astype(str) == e]
            s0, s1 = b.start_ts.min(), b.end_ts.max()
            lo.append(s0)
            hi.append(s1)
            row[f"{e}_copies"] = len(b)
            row[f"{e}_GBps"] = b.psum_read_bytes.sum() / (s1 - s0)
            row[f"{e}_ns_per_copy"] = (s1 - s0) / len(b)
            row[f"{e}_ii_ns"] = (b.start_ts.max() - s0) / (len(b) - 1)
            row[f"{e}_wait_ns"] = float(b.evt_wait_time_ns.median())
        row["overlap"] = max(0, min(hi) - max(lo)) / (end - start)
    ex = read_table(pq_dir, "ExecutionInfo")
    row["kernel_GBps_lb"] = meta["total_src_bytes"] / float(ex.execution_end_ts.iloc[0] - ex.execution_start_ts.iloc[0])
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
            r = {key: meta[key] for key in ("kernel", "engine", "pdim", "fdim", "src_row_bytes",
                                            "copy_src_bytes", "num_copies", "core")}
            r.update({"run": os.path.relpath(os.path.normpath(d), os.path.dirname(os.path.dirname(os.path.normpath(d)))),
                      "exec": k})
            r.update(parse_trace(meta, os.path.join(d, f"parquet_exec_{k}")))
            rows.append(r)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
