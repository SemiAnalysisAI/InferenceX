"""Per-NeuronCore HBM pointer-chase latency profile (core-memory-profiles analogue).

  python run_profile.py compile --out /w/runs/profiles/neff [--slots 65536 --stride 4096]
  python run_profile.py capture --neff-dir /w/runs/profiles/neff --core 5 --out /w/runs/profiles/solo/c5

Every core runs the same NEFF and the same chain (one seeded random cyclic permutation of --slots slots,
--stride bytes apart, so the footprint is slots*stride). The chase was checked on hardware: 513 dependent
hops from slot 0 land on numpy's answer (hw_check.py).

Per-hop latency = start-to-start gap between consecutive hops' first DMA packet in the profile (each
hop is one hwdge indirect DMA of 4 B). The profile is hop_ns[pass, hop]; analyze.py takes the mean over
passes per hop, then the mean absolute difference between two cores' profiles, as the GPU page does
per SM.
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
SEED = 2026111101


def make_chain(slots, stride):
    w = stride // 4
    perm = np.random.default_rng(SEED).permutation(slots)
    chain = np.zeros((slots, w), np.int32)
    chain[perm, 0] = np.roll(perm, -1)
    return chain


def run(cmd, env=None, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise RuntimeError(f"failed (rc={r.returncode}): {' '.join(cmd)}")
    return r


def do_compile(a):
    sys.path.insert(0, HERE)
    import kernels as K
    K.HOPS = a.hops
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    os.makedirs(a.out, exist_ok=True)
    chain = make_chain(a.slots, a.stride)
    np.save(os.path.join(a.out, "chain.npy"), chain)
    build = os.path.join(a.out, "build")
    os.makedirs(build, exist_ok=True)
    neff = os.path.join(a.out, "kernel.neff")
    opts = CompileOptions(target="trn3", lnc=1, artifacts_dir=build, output_path=neff)
    kern = K.KERNELS[a.kernel]
    bir, res = compile_to_bir(kern, frontend=ParserFrontend(), inputs={"chain": chain}, compile_opts=opts)
    compile_bir_to_neff(opts, bir, input_arrays=[chain], argument_names=["chain"],
                        output_arg_names=getattr(res, "output_names", []))
    shutil.rmtree(build, ignore_errors=True)
    json.dump({"kernel": a.kernel, "hops": a.hops, "slots": a.slots, "stride": a.stride, "seed": SEED},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("compiled", neff)


def hop_gaps(pq_dir, hops):
    import pandas as pd
    files = glob.glob(os.path.join(pq_dir, "**", "DmaPacket.parquet"), recursive=True)
    pk = pd.concat([pd.read_parquet(f) for f in files])
    qt = pk.queue_type.astype(object).fillna("none").astype(str)
    d = pk[qt == "hardware_dynamic"].sort_values("start_ts")
    starts = d.start_ts.to_numpy()
    per_hop = len(starts) // hops
    if per_hop * hops != len(starts) or per_hop == 0:
        raise RuntimeError(f"{len(starts)} dynamic packets for {hops} hops")
    first = starts[::per_hop]
    files = glob.glob(os.path.join(pq_dir, "**", "ExecutionInfo.parquet"), recursive=True)
    ex = pd.read_parquet(files[0])
    return np.diff(first), float(ex.execution_end_ts.iloc[0] - ex.execution_start_ts.iloc[0]), per_hop


def do_capture(a):
    meta = json.load(open(os.path.join(a.neff_dir, "meta.json")))
    os.makedirs(a.out, exist_ok=True)
    neff = os.path.join(a.neff_dir, "kernel.neff")
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=str(a.core))
    for f in glob.glob(os.path.join(a.out, "*.ntff")):
        os.remove(f)
    t0 = time.time()
    run(["neuron-explorer", "capture", "-n", neff, "-s", os.path.join(a.out, "profile.ntff"),
         f"--num-exec={a.num_exec}", f"--profile-nth-exec={a.profile_every}", "--ignore-exec-errors",
         "chain", os.path.join(a.neff_dir, "chain.npy")], env=env, cwd=a.out)
    t1 = time.time()
    gaps, execs = [], []
    for f in sorted(glob.glob(os.path.join(a.out, "profile_exec_*.ntff"))):
        pq = f[:-5] + "_pq"
        shutil.rmtree(pq, ignore_errors=True)
        os.makedirs(pq)
        run(["neuron-explorer", "view", "-n", neff, "-s", f, "--ingest-only", "--output-format=parquet",
             f"--output-file={pq}/"])
        g, e, per_hop = hop_gaps(pq, meta["hops"])
        gaps.append(g)
        execs.append(e)
        os.remove(f)
        shutil.rmtree(pq, ignore_errors=True)
    np.save(os.path.join(a.out, "hop_ns.npy"), np.array(gaps))
    json.dump({**meta, "core": a.core, "exec_ns": execs, "capture_start": t0, "capture_end": t1,
               "packets_per_hop": per_hop}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    g = np.array(gaps)
    print(f"core {a.core}: passes={len(gaps)} hop median {np.median(g):.0f} ns mean {g.mean():.0f} "
          f"exec {np.median(execs) / 1e3:.0f} us")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--out", required=True)
    c.add_argument("--kernel", default="chase_hwdge")
    c.add_argument("--hops", type=int, default=1024)
    c.add_argument("--slots", type=int, default=32768)  # x 4 KiB = 128 MiB, the Ascend page's footprint
    c.add_argument("--stride", type=int, default=4096)
    p = sub.add_parser("capture")
    p.add_argument("--neff-dir", required=True)
    p.add_argument("--core", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-exec", type=int, default=13)  # profiled execs 2,4,..,12: 6 passes, as the Ascend page
    p.add_argument("--profile-every", type=int, default=2)
    a = ap.parse_args()
    do_compile(a) if a.cmd == "compile" else do_capture(a)


if __name__ == "__main__":
    main()
