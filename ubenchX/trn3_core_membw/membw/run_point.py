"""Compile, profile and parse one Trainium3 mem-bw point.

  python run_point.py --kernel copy_h2h_static --bytes 1048576 --out runs/x [--core 0] [--lnc 1]

Pipeline adapted from ubenchX/trn3_dma_bw/run_point.py:
NKI -> NEFF, `neuron-explorer capture` of --num-exec executions profiling every --profile-every'th,
each profiled execution exported to parquet and reduced to one row in result.json:

  exec_ns      ExecutionInfo end - start: the whole NEFF on the device, the analogue of the TPU
               port's XLA-module device time (includes the fixed per-execution cost)
  dma_ns       first data packet start -> last data packet end (the copy itself)
  read_bytes / write_bytes   payload bytes of HBM-reading / HBM-writing data packets; both must equal
               --bytes or the row is marked bytes_ok = False (the profile proves the copy happened)

Parquet directories are deleted after parsing unless --keep.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORER = "neuron-explorer"


def run(cmd, env=None, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise RuntimeError(f"failed (rc={r.returncode}): {' '.join(cmd)}")
    return r


def compile_neff(kernel_name, in_tensor, out_dir, target, lnc):
    sys.path.insert(0, HERE)
    from kernels import KERNELS
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

    kernel = KERNELS[kernel_name]
    build = os.path.join(out_dir, "build")
    shutil.rmtree(build, ignore_errors=True)
    os.makedirs(build)
    neff = os.path.join(out_dir, "kernel.neff")
    opts = CompileOptions(target=target, lnc=lnc, artifacts_dir=build, output_path=neff)
    bir, result = compile_to_bir(kernel, frontend=ParserFrontend(), inputs={"in_tensor": in_tensor},
                                 compile_opts=opts)
    compile_bir_to_neff(opts, bir, input_arrays=[in_tensor], argument_names=["in_tensor"],
                        output_arg_names=getattr(result, "output_names", []))
    if not os.path.exists(neff):
        raise RuntimeError(f"compile produced no NEFF at {neff}")
    shutil.rmtree(build, ignore_errors=True)
    return neff


def read_table(pq_dir, name):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(pq_dir, "**", f"{name}.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def parse_trace(pq_dir, nbytes):
    ex = read_table(pq_dir, "ExecutionInfo")
    row = {"exec_ns": float(ex.execution_end_ts.iloc[0] - ex.execution_start_ts.iloc[0])}
    pk = read_table(pq_dir, "DmaPacket")
    if pk is None or not len(pk):
        return {**row, "dma_ns": None, "read_bytes": 0, "write_bytes": 0, "bytes_ok": False}
    qt = pk.queue_type.astype(object).fillna("none").astype(str)
    data = pk[(qt != "instruction") & (pk.transfer_bytes > 64)]
    payload = data.transfer_bytes - data.transfer_bytes % 64
    row["queue_types"] = "/".join(sorted(set(qt[data.index])))
    row["dma_engines"] = int(data.engine_idx.nunique()) if len(data) else 0
    row["packets"] = int(len(data))
    if len(data):
        row["dma_ns"] = float(data.end_ts.max() - data.start_ts.min())
    # Byte accounting from the per-transfer table: HBM-sourced reads and HBM-destined writes.
    agg = read_table(pq_dir, "DmaPacketAggregated")
    if agg is not None and len(agg):
        src = agg.source.astype(object).fillna("").astype(str)
        dst = agg.dest.astype(object).fillna("").astype(str)
        row["read_bytes"] = int(agg.read_bytes[src.str.contains("INPUT")].fillna(0).sum())
        row["write_bytes"] = int(agg.write_bytes[dst.str.contains("OUTPUT")].fillna(0).sum())
        row["agg_pairs"] = "/".join(sorted(set(src + ">" + dst)))
    else:
        row["read_bytes"] = row["write_bytes"] = int(payload.sum())  # fallback: cannot split
    row["bytes_ok"] = row["read_bytes"] == nbytes and row["write_bytes"] == nbytes
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--bytes", type=int, required=True)
    ap.add_argument("--max-chunk", type=int, default=4 * 2**20)
    ap.add_argument("--target", default="trn3")
    ap.add_argument("--lnc", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core", default=None, help="NEURON_RT_VISIBLE_CORES value")
    ap.add_argument("--num-exec", type=int, default=12)
    ap.add_argument("--profile-every", type=int, default=2)
    ap.add_argument("--neff", default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, HERE)
    from kernels import layout
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    spmd = a.kernel.endswith("_spmd")  # LNC=2: each physical core copies half, input [2C, P, F]
    if spmd and a.lnc != 2:
        raise SystemExit("_spmd kernels need --lnc 2")
    c, p, f = layout(a.bytes // 2 if spmd else a.bytes, a.max_chunk)
    if spmd:
        c *= 2
    meta = {"kernel": a.kernel, "bytes": a.bytes, "C": c, "P": p, "F": f, "lnc": a.lnc, "target": a.target,
            "core": a.core, "num_exec": a.num_exec, "profile_every": a.profile_every}
    inp = os.path.abspath(a.input) if a.input else os.path.join(out, "input.npy")
    if not a.input:
        np.save(inp, np.arange(c * p * f, dtype=np.float32).reshape(c, p, f))
    if a.neff:
        neff = os.path.abspath(a.neff)
    else:
        t0 = time.time()
        neff = compile_neff(a.kernel, np.load(inp, mmap_mode="r"), out, a.target, a.lnc)
        meta["compile_s"] = round(time.time() - t0, 1)
    meta["neff"] = neff
    if a.compile_only:
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        return

    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG=str(a.lnc))
    if a.core is not None:
        env["NEURON_RT_VISIBLE_CORES"] = str(a.core)
    for fn in os.listdir(out):
        if fn.endswith(".ntff"):
            os.remove(os.path.join(out, fn))
    meta["capture_start"] = time.time()
    run([EXPLORER, "capture", "-n", neff, "-s", os.path.join(out, "profile.ntff"), f"--num-exec={a.num_exec}",
         f"--profile-nth-exec={a.profile_every}", "--ignore-exec-errors", "in_tensor", inp], env=env, cwd=out)
    meta["capture_end"] = time.time()
    traces = sorted((fn for fn in os.listdir(out) if fn.endswith(".ntff") and "_exec_" in fn),
                    key=lambda fn: int(fn.rsplit("_exec_", 1)[1].split(".")[0]))
    rows = []
    for fn in traces:
        k = int(fn.rsplit("_exec_", 1)[1].split(".")[0])
        pq = os.path.join(out, f"parquet_exec_{k}")
        shutil.rmtree(pq, ignore_errors=True)
        os.makedirs(pq)
        run([EXPLORER, "view", "-n", neff, "-s", os.path.join(out, fn), "--ingest-only",
             "--output-format=parquet", f"--output-file={pq}/"])
        os.remove(os.path.join(out, fn))
        rows.append({"exec": k, **parse_trace(pq, a.bytes)})
        if not a.keep:
            shutil.rmtree(pq, ignore_errors=True)
    if not a.input and not a.keep:
        os.remove(inp)
    meta["rows"] = rows
    json.dump(meta, open(os.path.join(out, "result.json"), "w"), indent=1)
    ok = all(r["bytes_ok"] for r in rows)
    med = sorted(r["exec_ns"] for r in rows)[len(rows) // 2] if rows else float("nan")
    print(f"done {a.kernel} {a.bytes} B C={c} P={p} F={f} traces={len(rows)} exec_med={med:.0f} ns "
          f"GB/s={2 * a.bytes / med:.1f} bytes_ok={ok}")


if __name__ == "__main__":
    main()
