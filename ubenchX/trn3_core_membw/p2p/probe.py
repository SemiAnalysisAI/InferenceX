"""Compile, launch, check and time one point-to-point candidate from p2p/kernels.py.

  python p2p/probe.py --kernel cpi_sbuf --ranks 8 --cores 16-23 --mask 1 --out /w/runs/p2p/x
  python p2p/probe.py --kernel sr_lncn --lnc 4 --ranks 1 --cores 16-19 --mask 2 --out ...

--ranks R > 1 launches R ranks, one per NeuronCore of --cores (at --lnc), with distinct inputs; pairs are
(r, r ^ mask). --ranks 1 runs one logical core of degree --lnc whose physical cores exchange (sr_lncn).
After N steps rank r must hold its own tile if N is even, its peer's if N is odd; odd step counts make
the check discriminating. Timing per rank and execution: whole-NEFF time, and the span of every DMA
queue type present (first packet start to last packet end). Per-step cost = slope against NSTEP.
Writes <out>/f<bytes>/n<N>/result.json and <out>/summary.json.
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


def tile(r, P, F):
    return ((np.arange(P * F, dtype=np.float32).reshape(P, F) * 7 + 1000 * (r + 1)) % 65521).astype(np.float32)


def setup(a, nstep):
    import kernels as K
    K.NSTEP, K.MASK, K.PEER, K.GPSIMD = nstep, a.mask, a.peer, a.gpsimd
    K.PAIRS = [[r, r ^ a.mask] for r in range(a.ranks) if r < r ^ a.mask]
    K.ST_PAIRS = [(x, y) for x, y in K.PAIRS] + [(y, x) for x, y in K.PAIRS]
    return K


def inputs(a, F):
    if a.ranks == 1:
        return [np.stack([tile(k, 128, F) for k in range(a.lnc)])]
    return [tile(r, 128, F) for r in range(a.ranks)]


def expected(a, nstep, F):
    odd = nstep % 2 == 1
    if a.kernel.startswith("ag_"):
        ax = 1 if a.kernel == "ag_sbuf" else 0
        want = []
        for r in range(a.ranks):
            lo, hi = sorted((r, r ^ a.mask))
            want.append(np.concatenate([tile(lo if nstep == 1 else hi, 128, F), tile(hi, 128, F)], ax))
        return want
    if a.ranks == 1:
        return [np.stack([tile(k ^ a.mask if odd else k, 128, F) for k in range(a.lnc)])]
    return [tile(r ^ a.mask if odd else r, 128, F) for r in range(a.ranks)]


def compile_neff(a, nstep, F, d):
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    K = setup(a, nstep)
    x = inputs(a, F)[0]
    os.makedirs(d + "/build", exist_ok=True)
    try:
        opts = CompileOptions(target="trn3", lnc=a.lnc, artifacts_dir=d + "/build", output_path=d + "/kernel.neff")
        bir, res = compile_to_bir(K.KERNELS[a.kernel], frontend=ParserFrontend(), inputs={"in_tensor": x},
                                  compile_opts=opts)
        compile_bir_to_neff(opts, bir, input_arrays=[x], argument_names=["in_tensor"],
                            output_arg_names=getattr(res, "output_names", []))
    finally:
        shutil.rmtree(d + "/build", ignore_errors=True)
    with open(d + "/inputs.txt", "w") as f:
        for r, x in enumerate(inputs(a, F)):
            np.save(f"{d}/in_r{r}.npy", x)
            f.write(f"in_tensor {d}/in_r{r}.npy\n")


def capture(a, d):
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG=str(a.lnc), NEURON_RT_VISIBLE_CORES=a.cores)
    cmd = ["neuron-explorer", "capture", "-n", d + "/kernel.neff", "-s", d + "/p.ntff",
           f"--num-exec={a.num_exec}", f"--profile-nth-exec={a.every}", "--save-output-npy",
           "--save-nth-output", str(a.num_exec)]
    if a.ranks > 1:
        cmd += ["-r", str(a.ranks), "-m", d + "/inputs.txt"] + [f"--save-output-rank={r}" for r in range(a.ranks)]
    else:
        cmd += ["in_tensor", d + "/in_r0.npy"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=d, timeout=900)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-2000:])
    return (r.stdout + r.stderr)[-600:]


def outputs(a, d):
    outs = [o for o in sorted(glob.glob(d + "/**/*.npy", recursive=True)) if not os.path.basename(o).startswith("in_r")]
    if a.ranks == 1:
        return {0: np.load(outs[-1])} if outs else {}
    got = {}
    for o in outs:
        rr = [int(t[4:]) for t in o.replace("/", "_").split("_") if t.startswith("rank") and t[4:].isdigit()]
        if not rr:
            rr = [int(t) for t in os.path.basename(os.path.dirname(o)).split("_") if t.isdigit()][-1:]
        if rr:
            got[rr[0]] = np.load(o)
    return got


def parse(d, f):
    import pandas as pd
    pq = f[:-5] + "_pq"
    os.makedirs(pq, exist_ok=True)
    subprocess.run(["neuron-explorer", "view", "-n", d + "/kernel.neff", "-s", f, "--ingest-only",
                    "--output-format=parquet", f"--output-file={pq}/"], capture_output=True, check=True)
    ex = pd.read_parquet(glob.glob(pq + "/**/ExecutionInfo.parquet", recursive=True)[0])
    p = pd.read_parquet(glob.glob(pq + "/**/DmaPacket.parquet", recursive=True)[0])
    shutil.rmtree(pq, ignore_errors=True)
    spans = {}
    for q, g in p.groupby(p.queue_type.astype(str)):
        spans[q] = {"span_ns": float(g.end_ts.max() - g.start_ts.min()), "packets": int(len(g)),
                    "bytes": int(g.transfer_bytes.sum())}
    base = os.path.basename(f).replace(".ntff", "").split("_")
    rank = int(base[base.index("rank") + 1]) if "rank" in base else 0
    return {"rank": rank, "exec_ns": float(ex.execution_end_ts.iloc[0] - ex.execution_start_ts.iloc[0]),
            "queues": spans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--lnc", type=int, default=1)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--mask", type=int, default=1)
    ap.add_argument("--peer", type=int, default=1)
    ap.add_argument("--gpsimd", action="store_true")
    ap.add_argument("--fbytes", default="4")
    ap.add_argument("--steps", default="1,9,33")
    ap.add_argument("--num-exec", type=int, default=7)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summ = {"args": vars(a), "points": {}}
    for fb in (int(s) for s in a.fbytes.split(",")):
        F = fb // 4
        for n in (int(s) for s in a.steps.split(",")):
            d = f"{a.out}/f{fb}/n{n}"
            if os.path.exists(d + "/result.json"):
                res = json.load(open(d + "/result.json"))
            else:
                os.makedirs(d, exist_ok=True)
                for f in glob.glob(d + "/**/*", recursive=True):
                    if os.path.isfile(f) and not f.endswith((".neff", ".npy", "inputs.txt")):
                        os.remove(f)
                res = {"bytes_per_partition": fb, "tile_bytes": fb * 128, "nstep": n}
                try:
                    if not os.path.exists(d + "/kernel.neff"):
                        compile_neff(a, n, F, d)
                    res["log"] = capture(a, d)
                    got, want = outputs(a, d), expected(a, n, F)
                    bad = [r for r in range(len(want)) if r not in got or got[r].shape != want[r].shape
                           or not np.array_equal(got[r], want[r])]
                    res["check"] = len(got) == len(want) and not bad
                    res["check_note"] = f"{len(got)} outputs, bad ranks {bad}"
                    if bad and got:
                        r0 = bad[0]
                        res["check_detail"] = {"rank": r0, "got_first": got[r0].ravel()[:4].tolist() if r0 in got else None,
                                               "want_first": want[r0].ravel()[:4].tolist(),
                                               "own_first": tile(r0, 128, F).ravel()[:4].tolist()}
                    files = sorted(glob.glob(d + "/p_*exec_*.ntff"))
                    res["rows"] = [parse(d, f) for f in files]
                    for f in files:
                        os.remove(f)
                except Exception as e:  # noqa: BLE001
                    res["error"] = str(e)[-2000:]
                json.dump(res, open(d + "/result.json", "w"), indent=1)
            summ["points"][f"{fb}/{n}"] = {k: res.get(k) for k in ("check", "check_note", "check_detail", "error")}
            rows = res.get("rows", [])
            ex = np.median([r["exec_ns"] for r in rows]) if rows else float("nan")
            qs = sorted({q for r in rows for q in r["queues"]})
            qmed = {q: float(np.median([r["queues"][q]["span_ns"] for r in rows if q in r["queues"]])) for q in qs}
            summ["points"][f"{fb}/{n}"].update({"exec_med_ns": ex, "queue_span_med_ns": qmed})
            print(f"{a.kernel} lnc{a.lnc} r{a.ranks} m{a.mask} f{fb} n{n}: check={res.get('check')} "
                  f"({res.get('check_note') or (res.get('error') or '')[-700:]}) exec {ex / 1e3:.2f} us "
                  f"queues {{{', '.join(f'{q}: {v / 1e3:.2f}' for q, v in qmed.items())}}}", flush=True)
    json.dump(summ, open(a.out + "/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
