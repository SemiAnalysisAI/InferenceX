"""One-hop and two-hop core-to-core exchange over HBM collectives (all 8 cores of a chip as 8 ranks).

  python c2c/run_relay.py --chip 0 --out /w/runs/relay

Sequences of XOR matchings (r, r^M), cycled one per collective:
  [1] [2] [3] [4] [5]  one hop: LNC pair, other pair same die (R), other die (D2D)
  [4,2] [2,4]          two hops to r^6 (0 -> 4 -> 6, 0 -> 2 -> 6)
  [4,3] [3,4]          two hops to r^7 (0 -> 4 -> 7, 0 -> 3 -> 7)
Per-collective cost = slope of the in-trace collectives span against the number of collectives; a two-hop
relay costs two collectives. Check: every rank's output equals a numpy model of the same all_gather chain
(and is not equal to the model of the chain with the sequence reversed, so the oracle tells orders apart).
"""
import argparse, glob, json, os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_c2c as R  # noqa: E402

SEQS = {"1": [1], "2": [2], "3": [3], "4": [4], "5": [5], "4-2": [4, 2], "2-4": [2, 4], "4-3": [4, 3], "3-4": [3, 4]}


def model(seq, nstep, F):
    P = 128
    x = [R.rank_input(r, F) for r in range(8)]
    a = [np.zeros((2 * P, F), np.float32) for _ in range(8)]
    b = [np.zeros((2 * P, F), np.float32) for _ in range(8)]
    for r in range(8):
        a[r][:P] = x[r]
    for i in range(nstep):
        m = seq[i % len(seq)]
        src = [a[r][:P] if i % 2 == 0 else b[r][P:] for r in range(8)]
        dst = b if i % 2 == 0 else a
        new = {}
        for lo, hi in R.pairs_for(m):
            new[lo] = new[hi] = np.concatenate([src[lo], src[hi]], 0)
        for r in range(8):
            dst[r] = new[r].copy()
    return a if nstep % 2 == 0 else b


def compile_one(job):
    name, nstep, fb, d = job
    if os.path.exists(d + "/kernel.neff"):
        return d, None
    import kernels as K
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    K.NSTEP, K.SEQ = nstep, [R.pairs_for(m) for m in SEQS[name]]
    F = fb // 4
    x = R.rank_input(0, F)
    os.makedirs(d + "/build", exist_ok=True)
    try:
        opts = CompileOptions(target="trn3", lnc=1, artifacts_dir=d + "/build", output_path=d + "/kernel.tmp.neff")
        bir, res = compile_to_bir(K.relay_hbm, frontend=ParserFrontend(), inputs={"in_tensor": x}, compile_opts=opts)
        compile_bir_to_neff(opts, bir, input_arrays=[x], argument_names=["in_tensor"],
                            output_arg_names=getattr(res, "output_names", []))
        os.replace(d + "/kernel.tmp.neff", d + "/kernel.neff")
    except Exception as e:  # noqa: BLE001
        return d, str(e)[-400:]
    finally:
        shutil.rmtree(d + "/build", ignore_errors=True)
    for r in range(8):
        np.save(f"{d}/in_r{r}.npy", R.rank_input(r, F))
    with open(d + "/inputs.txt", "w") as f:
        f.writelines(f"in_tensor {d}/in_r{r}.npy\n" for r in range(8))
    return d, None


def check(d, seq, nstep, F):
    want = model(seq, nstep, F)
    alt = model(seq[::-1], nstep, F) if len(seq) > 1 else None
    bad, same_as_alt = [], 0
    for r in range(8):
        o = glob.glob(f"{d}/output_worker_{r}/*.npy")
        if not o:
            return None, "missing outputs"
        g = np.load(o[0])
        if not np.array_equal(g, want[r]):
            bad.append(r)
        if alt is not None and np.array_equal(g, alt[r]):
            same_as_alt += 1
    return not bad, f"bad {bad}; ranks also matching reversed order: {same_as_alt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--seqs", default=",".join(SEQS))
    ap.add_argument("--fbytes", type=int, default=4)
    ap.add_argument("--steps", default="4,16,48")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cores = f"{8 * a.chip}-{8 * a.chip + 7}"
    steps = [int(s) for s in a.steps.split(",")]
    jobs = [(n, k, a.fbytes, f"{a.out}/chip{a.chip}/s{n}/n{k}") for n in a.seqs.split(",") for k in steps]
    with ProcessPoolExecutor(a.jobs) as ex:
        for d, err in ex.map(compile_one, jobs):
            if err:
                print("compile failed", d, err, flush=True)
    for name, k, fb, d in jobs:
        if os.path.exists(d + "/result.json") or not os.path.exists(d + "/kernel.neff"):
            continue
        res = {"chip": a.chip, "seq": name, "masks": SEQS[name], "nstep": k, "bytes_per_partition": fb}
        try:
            R.capture(d, cores, 7, 2)
            ok, note = check(d, SEQS[name], k, fb // 4)
            files = sorted(glob.glob(d + "/p_rank_*_exec_*.ntff"))
            with ProcessPoolExecutor(a.jobs) as ex:
                rows = list(ex.map(R.parse_one, [(d, f) for f in files]))
            for f in files:
                os.remove(f)
            res.update({"check": ok, "check_note": note, "rows": rows})
        except Exception as e:  # noqa: BLE001
            res["error"] = str(e)[-600:]
        json.dump(res, open(d + "/result.json", "w"), indent=1)
        sp = [r["span_ns"] for r in res.get("rows", []) if r["span_ns"]]
        print(f"chip {a.chip} seq {name} n{k}: check={res.get('check')} ({res.get('check_note', res.get('error', ''))[:120]}) "
              f"span med {np.median(sp) / 1e3 if sp else float('nan'):.1f} us", flush=True)


if __name__ == "__main__":
    main()
