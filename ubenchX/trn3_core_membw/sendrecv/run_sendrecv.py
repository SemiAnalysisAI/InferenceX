"""Time the LNC=2 exchange chains of kernels.py and fit the per-step cost.

  python run_sendrecv.py --kernels xchg_dma,xchg_gpsimd,xchg_reduce --core 20 --out /w/runs/sendrecv

For every kernel and per-partition size (P = 128 partitions, F float32 per partition), compile NSTEP in
--steps, check the NSTEP=--check-steps output in NKI's simulator against kernels.expected(), capture each
NEFF with neuron-explorer on one LNC=2 logical core, and record the median ExecutionInfo time. The
per-step cost is the least-squares slope of device time against NSTEP.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernels as K  # noqa: E402

GPSIMD_MAX_BYTES = 1024  # per partition, fp32 (nisa.sendrecv docs)


def compile_neff(name, x, nstep, out):
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    K.NSTEP = nstep
    os.makedirs(out + "/build", exist_ok=True)
    opts = CompileOptions(target="trn3", lnc=2, artifacts_dir=out + "/build", output_path=out + "/kernel.neff")
    bir, res = compile_to_bir(K.KERNELS[name], frontend=ParserFrontend(), inputs={"in_tensor": x}, compile_opts=opts)
    compile_bir_to_neff(opts, bir, input_arrays=[x], argument_names=["in_tensor"],
                        output_arg_names=getattr(res, "output_names", []))
    shutil.rmtree(out + "/build", ignore_errors=True)
    np.save(out + "/input.npy", x)


def simulate_ok(name, x, nstep):
    import nki
    K.NSTEP = nstep
    try:
        got = np.asarray(nki.simulate(K.KERNELS[name])(x))
    except Exception as e:  # simulator may not support LNC=2 collectives
        return None, f"simulate failed: {str(e)[-300:]}"
    want = K.expected(name, x, nstep)
    return bool(np.allclose(got, want, rtol=1e-5, atol=0)), f"max abs err {float(np.abs(got - want).max()):.3g}"


def capture(out, core, num_exec=12, every=2):
    """Run the NEFF; returns per-execution device times. Also saves the last execution's output
    (out/device_output.npy) so the caller can check what the device produced, not only the simulator."""
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="2", NEURON_RT_VISIBLE_CORES=str(core))
    for f in glob.glob(out + "/*.ntff") + glob.glob(out + "/output_0*.npy"):
        os.remove(f)
    r = subprocess.run(["neuron-explorer", "capture", "-n", out + "/kernel.neff", "-s", out + "/p.ntff",
                        f"--num-exec={num_exec}", f"--profile-nth-exec={every}", "--ignore-exec-errors",
                        "--save-output-npy", "--save-nth-output", str(num_exec),
                        "in_tensor", out + "/input.npy"], capture_output=True, text=True, env=env, cwd=out)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    saved = sorted(glob.glob(out + "/output_0*.npy"))
    if saved:
        os.replace(saved[-1], out + "/device_output.npy")
    import pandas as pd
    ns = []
    for f in sorted(glob.glob(out + "/p_exec_*.ntff")):
        pq = f[:-5] + "_pq"
        os.makedirs(pq, exist_ok=True)
        subprocess.run(["neuron-explorer", "view", "-n", out + "/kernel.neff", "-s", f, "--ingest-only",
                        "--output-format=parquet", f"--output-file={pq}/"], capture_output=True, check=True)
        ex = pd.read_parquet(glob.glob(pq + "/**/ExecutionInfo.parquet", recursive=True)[0])
        ns.append(float(ex.execution_end_ts.iloc[0] - ex.execution_start_ts.iloc[0]))
        os.remove(f)
        shutil.rmtree(pq, ignore_errors=True)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="xchg_dma,xchg_gpsimd,xchg_reduce")
    ap.add_argument("--fbytes", default="4,64,256,1024,4096,16384,65536", help="bytes per partition")
    ap.add_argument("--steps", default="1,4,16,64")
    ap.add_argument("--check-steps", type=int, default=5)
    ap.add_argument("--core", default="20", help="NEURON_RT_VISIBLE_CORES (logical cores at LNC=2); the first one runs the kernel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")]
    for name in a.kernels.split(","):
        for fb in (int(s) for s in a.fbytes.split(",")):
            if name == "xchg_gpsimd" and fb > GPSIMD_MAX_BYTES:
                continue
            d = f"{a.out}/{name}/{fb}"
            if os.path.exists(d + "/result.json"):
                continue
            F = fb // 4
            rng = np.random.default_rng(fb)
            x = rng.random((2, 128, F), dtype=np.float32) + np.array([1, 2], np.float32)[:, None, None]
            ok, note = simulate_ok(name, x, a.check_steps)
            pts, device_ok = {}, {}
            for n in steps:
                nd = f"{d}/n{n}"
                try:
                    if not os.path.exists(nd + "/kernel.neff"):   # a pre-seeded tree reuses the NEFFs
                        compile_neff(name, x, n, nd)
                    pts[n] = capture(nd, a.core)
                    got = np.load(nd + "/device_output.npy").reshape(x.shape)
                    device_ok[n] = bool(np.allclose(got, K.expected(name, np.load(nd + "/input.npy"), n), rtol=1e-5, atol=0))
                except Exception as e:
                    pts[n] = str(e)[-400:]
            good = {n: float(np.median(v)) for n, v in pts.items() if isinstance(v, list) and v}
            res = {"kernel": name, "bytes_per_partition": fb, "partitions": 128, "tile_bytes": fb * 128,
                   "sim_ok": ok, "sim_note": note, "exec_ns": pts, "median_ns": good,
                   "device_ok": device_ok, "ncmap_swap": os.environ.get("NCMAP_SWAP")}
            if len(good) >= 2:
                ns_, ts = np.array(sorted(good)), np.array([good[n] for n in sorted(good)])
                slope, icpt = np.polyfit(ns_, ts, 1)
                res.update({"ns_per_step": float(slope), "fixed_ns": float(icpt),
                            "GBps_per_direction": fb * 128 / slope if slope > 0 else None})
            json.dump(res, open(d + "/result.json", "w"), indent=1)
            print(f"{name:12s} {fb:6d} B/part tile {fb * 128:>9d} B  sim_ok={ok} ({note}) device_ok={device_ok}  "
                  f"step {res.get('ns_per_step', float('nan')):9.1f} ns  fixed {res.get('fixed_ns', float('nan')):8.0f} ns  "
                  f"{res.get('GBps_per_direction') or 0:7.1f} GB/s/dir  medians {good}", flush=True)


if __name__ == "__main__":
    main()
