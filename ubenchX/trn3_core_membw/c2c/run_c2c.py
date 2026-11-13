"""SBUF -> SBUF latency between every pair of NeuronCores on one chip (8 x 8 matrix, 28 pairs).

  python c2c/run_c2c.py --chip 2 --out /w/runs/c2c        # LNC=1, physical cores 8*chip .. 8*chip+7

For each XOR matching M = 1..7 (pairs r, r^M: together the 7 matchings cover all 28 pairs, 4 pairs at a
time), each tile size and each NSTEP, compile kernels.xchg_ag and launch it on all 8 cores as 8 collective
ranks (neuron-explorer capture -r 8, one input file per rank). Every rank is profiled.

Timing, per rank and execution: span of the rank's `collectives` DMA queue packets (first packet of step 1
to last packet of step N). Ranks launch up to tens of us apart, so whole-NEFF time is mostly launch skew;
the in-trace span starts when this rank's first exchange starts. The per-exchange cost is the least-squares
slope of the span against NSTEP, which also drops the first step's wait for a late peer only on average,
so the median over executions is used per point. Whole-NEFF time slope is kept as a cross-check.

Check: rank r gets input x_r (distinct per rank). After N >= 2 steps both ranks of a pair must hold
[x_hi | x_hi], x_hi being the higher rank's tile, which the lower rank never had (N = 1: [x_lo | x_hi]).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def pairs_for(m):
    return [[r, r ^ m] for r in range(8) if r < r ^ m]


def rank_input(r, F):
    return ((np.arange(128 * F, dtype=np.float32).reshape(128, F) * 7 + 1000 * (r + 1)) % 65521).astype(np.float32)


def compile_one(job):
    m, fb, nstep, d = job
    if os.path.exists(d + "/kernel.neff"):
        return d, None
    import kernels as K
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    K.NSTEP, K.REPLICA = nstep, pairs_for(m)
    F = fb // 4
    x = rank_input(0, F)
    os.makedirs(d + "/build", exist_ok=True)
    try:
        opts = CompileOptions(target="trn3", lnc=1, artifacts_dir=d + "/build", output_path=d + "/kernel.tmp.neff")
        bir, res = compile_to_bir(K.xchg_ag, frontend=ParserFrontend(), inputs={"in_tensor": x}, compile_opts=opts)
        compile_bir_to_neff(opts, bir, input_arrays=[x], argument_names=["in_tensor"],
                            output_arg_names=getattr(res, "output_names", []))
        os.replace(d + "/kernel.tmp.neff", d + "/kernel.neff")
    except Exception as e:  # noqa: BLE001
        return d, str(e)[-500:]
    finally:
        shutil.rmtree(d + "/build", ignore_errors=True)
    for r in range(8):
        np.save(f"{d}/in_r{r}.npy", rank_input(r, F))
    with open(d + "/inputs.txt", "w") as f:
        f.writelines(f"in_tensor {d}/in_r{r}.npy\n" for r in range(8))
    return d, None


def parse_one(args):
    d, f = args
    import pandas as pd
    pq = f[:-5] + "_pq"
    os.makedirs(pq, exist_ok=True)
    subprocess.run(["neuron-explorer", "view", "-n", d + "/kernel.neff", "-s", f, "--ingest-only",
                    "--output-format=parquet", f"--output-file={pq}/"], capture_output=True, check=True)
    ex = pd.read_parquet(glob.glob(pq + "/**/ExecutionInfo.parquet", recursive=True)[0])
    p = pd.read_parquet(glob.glob(pq + "/**/DmaPacket.parquet", recursive=True)[0])
    c = p[p.queue_type.astype(str) == "collectives"]
    shutil.rmtree(pq, ignore_errors=True)
    base = os.path.basename(f)  # p_rank_<r>_exec_<k>.ntff
    r, k = int(base.split("_")[2]), int(base.split("_")[4].split(".")[0])
    return {"rank": r, "exec": k, "exec_ns": float(ex.execution_end_ts.iloc[0] - ex.execution_start_ts.iloc[0]),
            "span_ns": float(c.end_ts.max() - c.start_ts.min()) if len(c) else None,
            "cc_packets": int(len(c)), "cc_bytes": int(c.transfer_bytes.sum())}


def capture(d, cores, num_exec, every):
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=cores)
    for f in glob.glob(d + "/*.ntff") + glob.glob(d + "/out*/**", recursive=True):
        if os.path.isfile(f):
            os.remove(f)
    cmd = ["neuron-explorer", "capture", "-n", d + "/kernel.neff", "-s", d + "/p.ntff", "-r", "8",
           "-m", d + "/inputs.txt", f"--num-exec={num_exec}", f"--profile-nth-exec={every}",
           "--save-output-npy", "--save-nth-output", str(num_exec)] + [f"--save-output-rank={r}" for r in range(8)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=d, timeout=900)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-1500:])


def check_outputs(d, m, nstep, F):
    """Return (ok, note). Output files are found by search: their naming is per runtime version."""
    outs = sorted(glob.glob(d + "/**/*.npy", recursive=True))
    outs = [o for o in outs if not os.path.basename(o).startswith("in_r")]
    if len(outs) < 8:
        return None, f"found {len(outs)} output files: {outs[:4]}"
    got = {}
    for o in outs:
        rr = [int(t[4:]) for t in o.replace("/", "_").split("_") if t.startswith("rank") and t[4:].isdigit()]
        if not rr:
            rr = [int(t) for t in os.path.basename(os.path.dirname(o)).split("_") if t.isdigit()][-1:]
        if rr:
            got[rr[0]] = np.load(o)
    if len(got) < 8:
        return None, f"could not map outputs to ranks: {outs[:4]}"
    bad = []
    for lo, hi in pairs_for(m):
        xl, xh = rank_input(lo, F), rank_input(hi, F)
        want = np.concatenate([xl, xh], 1) if nstep == 1 else np.concatenate([xh, xh], 1)
        for r in (lo, hi):
            if got[r].shape != want.shape or not np.array_equal(got[r], want):
                bad.append(r)
    return not bad, f"bad ranks {bad}" if bad else "all ranks match"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--masks", default="1,2,3,4,5,6,7")
    ap.add_argument("--fbytes", default="4,256,4096", help="bytes per partition (tile = 128 x this)")
    ap.add_argument("--steps", default="2,8,32")
    ap.add_argument("--num-exec", type=int, default=7)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    masks = [int(s) for s in a.masks.split(",")]
    fbs = [int(s) for s in a.fbytes.split(",")]
    steps = [int(s) for s in a.steps.split(",")]
    cores = f"{8 * a.chip}-{8 * a.chip + 7}"
    root = f"{a.out}/chip{a.chip}"
    jobs = [(m, fb, n, f"{root}/m{m}/f{fb}/n{n}") for m in masks for fb in fbs for n in steps]
    with ProcessPoolExecutor(a.jobs) as ex:
        for d, err in ex.map(compile_one, jobs):
            if err:
                print("compile failed", d, err, flush=True)
    order = list(range(len(jobs)))
    np.random.default_rng(a.chip).shuffle(order)  # spread drift across masks and sizes
    for i in order:
        m, fb, n, d = jobs[i]
        if os.path.exists(d + "/result.json") or not os.path.exists(d + "/kernel.neff"):
            continue
        res = {"chip": a.chip, "cores": cores, "mask": m, "pairs": pairs_for(m), "bytes_per_partition": fb,
               "tile_bytes": fb * 128, "nstep": n}
        try:
            capture(d, cores, a.num_exec, a.every)
            ok, note = check_outputs(d, m, n, fb // 4)
            files = sorted(glob.glob(d + "/p_rank_*_exec_*.ntff"))
            with ProcessPoolExecutor(a.jobs) as ex:
                rows = list(ex.map(parse_one, [(d, f) for f in files]))
            for f in files:
                os.remove(f)
            res.update({"check": ok, "check_note": note, "rows": rows})
        except Exception as e:  # noqa: BLE001
            res.update({"error": str(e)[-800:]})
        json.dump(res, open(d + "/result.json", "w"), indent=1)
        sp = [r["span_ns"] for r in res.get("rows", []) if r["span_ns"]]
        print(f"chip {a.chip} m{m} f{fb} n{n}: check={res.get('check')} ({res.get('check_note', res.get('error', '')[-200:])}) "
              f"span med {np.median(sp) / 1e3 if sp else float('nan'):.2f} us", flush=True)


if __name__ == "__main__":
    main()
