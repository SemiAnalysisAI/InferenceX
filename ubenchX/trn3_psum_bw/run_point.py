"""Compile and profile one Trainium3 PSUM -> SBUF fp32 tensor_copy bandwidth point on one NeuronCore (LNC=1).

  python ubenchX/trn3_psum_bw/run_point.py --target trn3 --kernel copy_vector --fdim 512 \
    --copies 1024 --core 0 --out ubenchX/trn3_psum_bw/runs/x

Writes into --out: kernel.neff, input.npy, one parquet export per profiled execution
(parquet_exec_<n>/), and meta.json. parse.py computes bandwidth from them.
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
EXPLORER = "neuron-explorer"


def compile_neff(kernel, in_tensor, out_dir, target):
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

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
    sys.path.insert(0, HERE)
    from kernels import ENGINES, KERNELS, get_kernel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    # trn3 is Trainium3 silicon revision v2 and trn3pre is revision v1.
    ap.add_argument("--target", required=True, choices=["trn3", "trn3pre"])
    ap.add_argument("--pdim", type=int, default=128, help="partitions per copy (max 128)")
    ap.add_argument("--fdim", type=int, required=True, help="fp32 PSUM elements per partition per copy")
    ap.add_argument("--copies", type=int, default=1024, help="tensor_copy instructions per engine per execution")
    ap.add_argument("--out", required=True)
    ap.add_argument("--core", type=int, default=0, help="NeuronCore to run on (NEURON_RT_VISIBLE_CORES)")
    ap.add_argument("--num-exec", type=int, default=12, help="executions per capture (first ones are warm-up)")
    ap.add_argument("--profile-every", type=int, default=4, help="profile every Nth execution")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    n_eng = ENGINES.get(a.kernel, 1)
    meta = {"kernel": a.kernel, "engine": "vector+scalar" if n_eng == 2 else a.kernel.split("_")[1],
            "pdim": a.pdim, "fdim": a.fdim, "src_row_bytes": a.fdim * 4, "copy_src_bytes": a.pdim * a.fdim * 4,
            "num_copies": a.copies * n_eng, "total_src_bytes": a.copies * n_eng * a.pdim * a.fdim * 4,
            "core": a.core, "num_exec": a.num_exec,
            "profile_every": a.profile_every, "lnc": 1, "target": a.target}

    inp = os.path.join(out, "input.npy")
    np.save(inp, np.zeros((a.pdim, a.fdim), dtype=np.float32))  # shape carrier only
    t0 = time.time()
    neff, param = compile_neff(get_kernel(a.kernel, a.copies), np.load(inp), out, a.target)
    meta["compile_s"] = round(time.time() - t0, 1)

    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=str(a.core))
    for f in os.listdir(out):
        if f.endswith(".ntff"):
            os.remove(os.path.join(out, f))
    run([EXPLORER, "capture", "-n", neff, "-s", os.path.join(out, "profile.ntff"), f"--num-exec={a.num_exec}",
         f"--profile-nth-exec={a.profile_every}", "--ignore-exec-errors", param, inp], env=env, cwd=out)
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
    print(f"done {a.kernel} pdim={a.pdim} fdim={a.fdim} copies={a.copies} core={a.core} traces={len(traces)}")


if __name__ == "__main__":
    main()
