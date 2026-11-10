"""Compute Tensor Engine matmul II and latency from run_point.py outputs (needs pandas + pyarrow).

  python ubenchX/trn3_matmul/parse.py RUN_DIR... --csv mm.csv   # one row per profiled execution

From the profiler's per-instruction Instruction table, the burst is the num_mm MATMUL instructions on
the Tensor engine. TensorE starts a burst throttled (activity throttle 4/8, sometimes lower) and is
released to 8/8 after about 1-3 us; the profiler's Throttle table records the release. The steady
window is the matmuls that start after the first 8/8 event, or, when the trace has no such event,
all but the first SKIP matmuls. Over the steady window:

  II        (last start - first start) / (count - 1): start-to-start interval, i.e. throughput
  dur       median start-to-end of one MATMUL: latency of one instruction
  ld_*      the same for LDWEIGHTS (LoadStationary) when the kernel issues more than one
  pair_lat  median LDWEIGHTS start -> its MATMUL end, when every matmul has its own load

Cycles use the tensor_engine_clock_freq from the trace's Metadata (2.4 GHz). eff = N / II_cycles
(one moving column per cycle is the peak); TFLOPS = 2 K M N / II.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SKIP = 16


def read_table(pq_dir, name):
    files = sorted(glob.glob(os.path.join(pq_dir, "**", f"{name}.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def window(d, release):
    d = d.sort_values("start_ts")
    return d[d.start_ts > release] if release is not None else d.iloc[SKIP:]


def parse_trace(meta, pq_dir):
    ins = read_table(pq_dir, "Instruction")
    te = ins[ins.engine.astype(str) == "Tensor"]
    mm = te[te.opcode.astype(str) == "MATMUL"].sort_values("start_ts")
    ld = te[te.opcode.astype(str) == "LDWEIGHTS"]
    if len(mm) != meta["num_mm"]:
        raise RuntimeError(f"{pq_dir}: expected {meta['num_mm']} MATMUL, got {len(mm)}")
    clock = float(read_table(pq_dir, "Metadata").tensor_engine_clock_freq.iloc[0])
    th = read_table(pq_dir, "Throttle")
    release = None
    if th is not None and len(th):
        full = th[(th.k == th.n) & (th.timestamp >= mm.start_ts.iloc[0])]
        if len(full):
            release = int(full.timestamp.min())
    st = window(mm, release)
    ii = (st.start_ts.iloc[-1] - st.start_ts.iloc[0]) / (len(st) - 1)
    K, M, N = meta["K"], meta["M"], meta["N"]
    r = {
        "throttle_release_ns": None if release is None else release - int(mm.start_ts.iloc[0]),
        "steady_mm": len(st),
        "clock_ghz": clock,
        "II_ns": ii,
        "II_cycles": ii * clock,
        "gap_med_cycles": float(np.median(np.diff(st.start_ts.to_numpy()))) * clock,
        "dur_ns": float(st.duration_ns.median()),
        "dur_cycles": float(st.duration_ns.median()) * clock,
        "in_flight": float(st.duration_ns.median()) / ii,
        "eff": N / (ii * clock),
        "TFLOPS": 2 * K * M * N / ii / 1e3,
        "wait_ns": float(st.evt_wait_time_ns.median()),
        "ld_count": len(ld),
    }
    if len(ld) > 1:
        sl = window(ld, release)
        r["ld_dur_cycles"] = float(sl.duration_ns.median()) * clock
        r["ld_II_cycles"] = (sl.start_ts.iloc[-1] - sl.start_ts.iloc[0]) / (len(sl) - 1) * clock
    if len(ld) == len(mm):
        # pair i = the i-th LDWEIGHTS and the i-th MATMUL; pair latency = load start -> matmul end
        lds = ld.sort_values("start_ts").start_ts.to_numpy()
        ends = (mm.start_ts + mm.duration_ns).to_numpy()
        lat = (ends - lds)[mm.start_ts.to_numpy() > (release if release is not None else mm.start_ts.iloc[SKIP - 1])]
        r["pair_lat_cycles"] = float(np.median(lat)) * clock
    return r


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
            r = {key: meta[key] for key in ("kernel", "K", "M", "N", "num_mm", "core")}
            r.update({"run": os.path.basename(os.path.normpath(d)), "exec": k})
            r.update(parse_trace(meta, os.path.join(d, f"parquet_exec_{k}")))
            rows.append(r)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 40):
        print(df.drop(columns=["run", "core", "num_mm"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
