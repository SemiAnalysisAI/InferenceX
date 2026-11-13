"""Parity test: both physical cores of one LNC=2 logical core chase the SAME chain (same HBM addresses).

Compiles kernels.chase_hwdge at lnc=2 (both cores run it on the one input tensor), captures on one logical
core, and splits the profile by pcore_idx. If the even/odd hop gap of the LNC=1 runs survives here, it
belongs to the core (its path to the DMA engine and HBM), not to where each core's buffer was placed.

  python lnc2_shared.py --core 8 --out /w/runs/parity/lnc2
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import kernels as K
    from run_profile import make_chain
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    import pandas as pd
    K.HOPS = 1024
    os.makedirs(a.out + "/build", exist_ok=True)
    chain = make_chain(32768, 4096)
    np.save(a.out + "/chain.npy", chain)
    opts = CompileOptions(target="trn3", lnc=2, artifacts_dir=a.out + "/build", output_path=a.out + "/kernel.neff")
    bir, res = compile_to_bir(K.chase_hwdge, frontend=ParserFrontend(), inputs={"chain": chain}, compile_opts=opts)
    compile_bir_to_neff(opts, bir, input_arrays=[chain], argument_names=["chain"],
                        output_arg_names=getattr(res, "output_names", []))
    shutil.rmtree(a.out + "/build", ignore_errors=True)
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="2", NEURON_RT_VISIBLE_CORES=str(a.core))
    subprocess.run(["neuron-explorer", "capture", "-n", a.out + "/kernel.neff", "-s", a.out + "/p.ntff",
                    "--num-exec=13", "--profile-nth-exec=2", "--ignore-exec-errors", "chain", a.out + "/chain.npy"],
                   check=True, capture_output=True, env=env, cwd=a.out)
    per = {0: [], 1: []}
    for f in sorted(glob.glob(a.out + "/p_exec_*.ntff")):
        pq = f[:-5] + "_pq"
        os.makedirs(pq, exist_ok=True)
        subprocess.run(["neuron-explorer", "view", "-n", a.out + "/kernel.neff", "-s", f, "--ingest-only",
                        "--output-format=parquet", f"--output-file={pq}/"], check=True, capture_output=True)
        pk = pd.concat([pd.read_parquet(x) for x in glob.glob(pq + "/**/DmaPacket.parquet", recursive=True)])
        d = pk[pk.queue_type.astype(str) == "hardware_dynamic"]
        for p in (0, 1):
            s = d[d.pcore_idx == p].sort_values("start_ts").start_ts.to_numpy()
            if len(s) >= 4:
                per[p].append(np.diff(s[::2]))
    for p in (0, 1):
        if per[p]:
            g = np.concatenate(per[p])
            print(f"pcore {p}: passes={len(per[p])} hop mean {g.mean():.1f} ns median {np.median(g):.0f}")
    if per[0] and per[1]:
        m0, m1 = np.mean([x.mean() for x in per[0]]), np.mean([x.mean() for x in per[1]])
        print(f"pcore0 - pcore1 = {m0 - m1:+.1f} ns per hop (LNC=1 runs: even - odd = +23 ns)")
        P0, P1 = np.mean(per[0], 0), np.mean(per[1], 0)
        print(f"profile corr {np.corrcoef(P0, P1)[0, 1]:.3f}, shape MAD {np.abs((P0 - P0.mean()) - (P1 - P1.mean())).mean():.1f} ns")


if __name__ == "__main__":
    main()
