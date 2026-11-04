"""Compile and profile one Trainium3 DMA bandwidth point on one NeuronCore (LNC=1).

  python ubenchX/trn3_dma_bw/run_point.py --target trn3pre --kernel load_static --fdim 8192 \
    --core 0 --out ubenchX/trn3_dma_bw/runs/x

Writes into --out: kernel.neff, input.npy, one parquet export per profiled execution
(parquet_exec_<n>/), and meta.json. parse.py computes bandwidth from them.

--neff/--input reuse an existing NEFF and input (allcore.sh runs one NEFF on every core).
--core pins the run to one NeuronCore via NEURON_RT_VISIBLE_CORES.
"""

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = np.float16
EXPLORER = "neuron-explorer"
KERNEL_NAMES = [f"{d}_{m}" for d in ("load", "store") for m in ("static", "hwdge", "swdge")]


def compile_neff(kernel_name, in_tensor, out_dir, target):
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
    opts = CompileOptions(target=target, lnc=1, artifacts_dir=build, output_path=neff)
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, choices=KERNEL_NAMES)
    # trn3 is Trainium3 silicon revision v2 and trn3pre is revision v1; the runtime refuses a NEFF
    # built for a newer revision than the chip.
    ap.add_argument("--target", required=True, choices=["trn3", "trn3pre"])
    ap.add_argument("--pdim", type=int, default=128, help="partitions per transfer (max 128)")
    ap.add_argument("--fdim", type=int, required=True, help="fp16 elements per partition row per transfer")
    ap.add_argument("--total-mib", type=int, default=256, help="bytes moved per execution, MiB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--core", type=int, default=None, help="NeuronCore to run on (NEURON_RT_VISIBLE_CORES)")
    ap.add_argument("--num-exec", type=int, default=11, help="executions per capture (first ones are warm-up)")
    ap.add_argument("--profile-every", type=int, default=2, help="profile every Nth execution")
    ap.add_argument("--neff", default=None, help="reuse a compiled NEFF instead of compiling")
    ap.add_argument("--input", default=None, help="reuse an input .npy")
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    xfer = a.pdim * a.fdim * np.dtype(DTYPE).itemsize
    total = a.total_mib * 2**20
    if total % xfer:
        raise SystemExit(f"--total-mib must be a multiple of one transfer ({xfer} B)")
    n = total // xfer
    meta = {"kernel": a.kernel, "pdim": a.pdim, "fdim": a.fdim, "row_bytes": a.fdim * 2, "transfer_bytes": xfer,
            "num_transfers": n, "total_bytes": total, "core": a.core, "num_exec": a.num_exec,
            "profile_every": a.profile_every, "lnc": 1, "target": a.target}

    inp = os.path.abspath(a.input) if a.input else os.path.join(out, "input.npy")
    if not a.input:
        np.save(inp, np.random.default_rng(42).random((n, a.pdim, a.fdim), dtype=np.float32).astype(DTYPE))
    if a.neff:
        neff, param = os.path.abspath(a.neff), "in_tensor" if a.kernel.startswith("load") else "in_shape_tensor"
    else:
        t0 = time.time()
        neff, param = compile_neff(a.kernel, np.load(inp, mmap_mode="r"), out, a.target)
        meta["compile_s"] = round(time.time() - t0, 1)
    meta["neff"] = neff
    if a.compile_only:
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        return

    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1")
    if a.core is not None:
        env["NEURON_RT_VISIBLE_CORES"] = str(a.core)
    for f in os.listdir(out):
        if f.endswith(".ntff"):
            os.remove(os.path.join(out, f))
    meta["capture_start"] = time.time()
    run([EXPLORER, "capture", "-n", neff, "-s", os.path.join(out, "profile.ntff"), f"--num-exec={a.num_exec}",
         f"--profile-nth-exec={a.profile_every}", "--ignore-exec-errors", param, inp], env=env, cwd=out)
    meta["capture_end"] = time.time()

    traces = sorted((f for f in os.listdir(out) if f.endswith(".ntff") and "_exec_" in f),
                    key=lambda f: int(f.rsplit("_exec_", 1)[1].split(".")[0]))
    meta["profiled_execs"] = []
    for f in traces:
        k = int(f.rsplit("_exec_", 1)[1].split(".")[0])
        pq = os.path.join(out, f"parquet_exec_{k}")
        shutil.rmtree(pq, ignore_errors=True)
        os.makedirs(pq)
        run([EXPLORER, "view", "-n", neff, "-s", os.path.join(out, f), "--ingest-only", "--output-format=parquet",
             f"--output-file={pq}/"])
        os.remove(os.path.join(out, f))
        meta["profiled_execs"].append(k)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"done {a.kernel} pdim={a.pdim} fdim={a.fdim} total={a.total_mib}MiB core={a.core} traces={len(traces)}")


if __name__ == "__main__":
    main()
