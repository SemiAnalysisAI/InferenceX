"""Compute exp / activation throughput from run_point.py outputs (needs pandas + pyarrow).

  python ubenchX/trn3_exp_bw/parse.py RUN_DIR... --csv exp.csv   # one row per profiled execution

From the profiler's per-instruction Instruction table, the burst is the instructions on the kernel's
engine that read one [P, F] fp32 source tile (plus, for the softmax kernels, the [P, 1] fp32 max or
bias tile) and share the one opcode that occurs exactly num_ops times among them (the closing
reduce can read similar bytes, but only once). Other instructions on that engine
inside the burst window (such as a separate reduce_res read-out) are counted in extra_instrs. II_ns is the initiation interval, (last op start - first op start) / (num_ops - 1);
II_cycles uses the engine clock from the trace's Metadata. elem_per_cycle = fdim / II_cycles per
partition. GBps = fp32 input bytes processed per second. report.py fits
II_cycles = issue + fdim * cycles_per_element per kernel.
"""

import argparse
import glob
import json
import os

import pandas as pd

CLOCK_COL = {"vector": "vector_engine_clock_freq", "scalar": "scalar_engine_clock_freq",
             "gpsimd": "gpsimd_engine_clock_freq"}


def read_table(pq_dir, name):
    files = sorted(glob.glob(os.path.join(pq_dir, "**", f"{name}.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def parse_trace(meta, pq_dir):
    ins = read_table(pq_dir, "Instruction")
    n = meta["num_ops"]
    engine = meta["engine"].capitalize()
    on_engine = ins[ins.engine.astype(str) == engine]
    extra = (on_engine.sbuf_read_bytes.fillna(0) + on_engine.psum_read_bytes.fillna(0)) - meta["op_bytes"]
    reads_tile = on_engine[extra.isin([0, meta["pdim"] * 4])]
    counts = reads_tile.opcode.astype(str).value_counts()
    if (counts == n).sum() != 1:
        raise RuntimeError(f"{pq_dir}: expected one opcode with {n} {engine} instructions reading one tile, got "
                           f"{counts.to_dict()}")
    opcode = counts[counts == n].index[0]
    burst = reads_tile[reads_tile.opcode.astype(str) == opcode].sort_values("start_ts")
    window = on_engine[(on_engine.start_ts >= burst.start_ts.iloc[0]) & (on_engine.start_ts <= burst.start_ts.iloc[-1])]
    clock = float(read_table(pq_dir, "Metadata")[CLOCK_COL[meta["engine"]]].iloc[0])
    ii = (burst.start_ts.iloc[-1] - burst.start_ts.iloc[0]) / (n - 1)
    return {
        "engine_seen": engine,
        "opcode": opcode,
        "clock_ghz": clock,
        "II_ns": ii,
        "II_cycles": ii * clock,
        "elem_per_cycle": meta["fdim"] / (ii * clock),
        "GBps": meta["op_bytes"] / ii,
        "Gelem_per_s": meta["op_elements"] / ii,
        "dur_ns": float(burst.duration_ns.median()),
        "wait_ns": float(burst.evt_wait_time_ns.median()),
        "extra_instrs": len(window) - n,
        "dst_bytes_per_op": float(burst.sbuf_write_bytes.median()),
    }


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
            r = {key: meta[key] for key in ("kernel", "engine", "dst_dtype", "pdim", "fdim", "op_elements", "op_bytes",
                                            "num_ops", "core")}
            r.update({"run": os.path.relpath(os.path.normpath(d), os.path.dirname(os.path.dirname(os.path.normpath(d)))),
                      "exec": k})
            r.update(parse_trace(meta, os.path.join(d, f"parquet_exec_{k}")))
            rows.append(r)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
