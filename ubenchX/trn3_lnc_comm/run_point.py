"""Compile and profile one Trainium3 intra-LNC communication point at LNC=2 (both cores of an HBM stack).

  python ubenchX/trn3_lnc_comm/run_point.py --target trn3 --kernel sr_dma --bytes 65536 --dtype bf16 \
    --ops 64 --lcore 0 --out ubenchX/trn3_lnc_comm/runs/x

--bytes is the message per rank per operation: the tile is [P, bytes / P / dtype size], P = --partitions
(default 128). --lcore is
the LNC=2 logical core (chip c holds logical cores 4c .. 4c+3). Writes kernel.neff, input.npy, one
parquet export per profiled execution (parquet_exec_<n>/) and meta.json into --out.
"""

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import time

import ml_dtypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORER = "neuron-explorer"
NP_DTYPES = {"bf16": ml_dtypes.bfloat16, "fp32": np.float32}
SIZES = {"bf16": 2, "fp32": 4}


def compile_neff(kernel, in_tensor, out_dir, target, lnc):
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

    build = os.path.join(out_dir, "build")
    shutil.rmtree(build, ignore_errors=True)
    os.makedirs(build)
    neff = os.path.join(out_dir, "kernel.neff")
    opts = CompileOptions(target=target, lnc=lnc, artifacts_dir=build, output_path=neff)
    param = list(inspect.signature(kernel.func).parameters)[0]
    bir, result = compile_to_bir(kernel, frontend=ParserFrontend(), inputs={param: in_tensor}, compile_opts=opts)
    compile_bir_to_neff(opts, bir, input_arrays=[in_tensor], argument_names=[param],
                        output_arg_names=getattr(result, "output_names", []))
    if not os.path.exists(neff):
        raise RuntimeError(f"compile produced no NEFF at {neff}")
    return neff, param


def run(cmd, env=None, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise RuntimeError(f"failed (rc={r.returncode}): {' '.join(cmd)}")
    return r


def main():
    sys.path.insert(0, HERE)
    from kernels import KERNELS, get_kernel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    ap.add_argument("--target", required=True, choices=["trn3", "trn3pre"])
    ap.add_argument("--bytes", type=int, required=True, help="message bytes per rank per operation")
    ap.add_argument("--dtype", default="bf16", choices=list(NP_DTYPES))
    ap.add_argument("--ops", type=int, default=64, help="operations per execution")
    ap.add_argument("--partitions", type=int, default=128, help="tile partitions P (tile is [P, bytes / P / dtype size])")
    ap.add_argument("--lcore", type=int, default=0, help="LNC=2 logical core (NEURON_RT_VISIBLE_CORES)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-exec", type=int, default=12)
    ap.add_argument("--profile-every", type=int, default=4)
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()

    P = a.partitions
    F = a.bytes // (P * SIZES[a.dtype])
    if F * P * SIZES[a.dtype] != a.bytes or F < 1:
        raise SystemExit(f"--bytes {a.bytes} is not a multiple of {P * SIZES[a.dtype]}")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    meta = {"kernel": a.kernel, "bytes": a.bytes, "dtype": a.dtype, "P": P, "F": F, "num_ops": a.ops, "lnc": 2,
            "lcore": a.lcore, "num_exec": a.num_exec, "profile_every": a.profile_every, "target": a.target}
    inp = os.path.join(out, "input.npy")
    arr = np.random.default_rng(0).standard_normal((2, P, F)).astype(NP_DTYPES[a.dtype])
    np.save(inp, arr)  # bf16 reloads as raw |V2, so compile from the in-memory array
    t0 = time.time()
    neff, param = compile_neff(get_kernel(a.kernel, a.ops, a.dtype), arr, out, a.target, lnc=2)
    meta["compile_s"] = round(time.time() - t0, 1)
    if a.compile_only:
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"compiled {a.kernel} {a.dtype} bytes={a.bytes} -> {neff}")
        return

    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="2", NEURON_RT_VISIBLE_CORES=str(a.lcore))
    for f in os.listdir(out):
        if f.endswith(".ntff"):
            os.remove(os.path.join(out, f))
    run([EXPLORER, "capture", "-n", neff, "-s", os.path.join(out, "profile.ntff"), f"--num-exec={a.num_exec}",
         f"--profile-nth-exec={a.profile_every}", "--ignore-exec-errors", param, inp], env=env, cwd=out)
    traces = sorted(f for f in os.listdir(out) if f.endswith(".ntff"))
    meta["traces"] = []
    for f in traces:
        pq = os.path.join(out, "parquet_" + f[:-5])
        shutil.rmtree(pq, ignore_errors=True)
        os.makedirs(pq)
        run([EXPLORER, "view", "-n", neff, "-s", os.path.join(out, f), "--ingest-only", "--output-format=parquet",
             f"--output-file={pq}/"])
        os.remove(os.path.join(out, f))
        meta["traces"].append(os.path.basename(pq))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"done {a.kernel} {a.dtype} bytes={a.bytes} lcore={a.lcore} traces={len(traces)}")


if __name__ == "__main__":
    main()
