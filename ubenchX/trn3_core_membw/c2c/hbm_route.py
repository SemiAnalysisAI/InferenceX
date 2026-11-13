"""Compile xchg_ag_hbm for a matching, capture once on 8 ranks, report per-rank engines/queues and data bytes."""
import os, sys, glob, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as K, run_c2c as R
from nki.compiler.driver import compile_to_bir
from nki.compiler.frontend import ParserFrontend
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
out, cores, F = sys.argv[1], sys.argv[2], 16
for m in [int(x) for x in sys.argv[3].split(",")]:
    d = f"{out}/m{m}"; os.makedirs(d + "/build", exist_ok=True)
    K.NSTEP, K.REPLICA = 8, R.pairs_for(m)
    x = R.rank_input(0, F)
    try:
        opts = CompileOptions(target="trn3", lnc=1, artifacts_dir=d + "/build", output_path=d + "/kernel.neff")
        bir, res = compile_to_bir(K.xchg_ag_hbm, frontend=ParserFrontend(), inputs={"in_tensor": x}, compile_opts=opts)
        compile_bir_to_neff(opts, bir, input_arrays=[x], argument_names=["in_tensor"], output_arg_names=getattr(res, "output_names", []))
    except Exception as e:
        print(f"mask {m}: compile failed: {str(e).strip().splitlines()[-1][:200]}"); continue
    for r in range(8): np.save(f"{d}/in_r{r}.npy", R.rank_input(r, F))
    open(d + "/inputs.txt", "w").writelines(f"in_tensor {d}/in_r{r}.npy\n" for r in range(8))
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=cores)
    p = subprocess.run(["neuron-explorer", "capture", "-n", d + "/kernel.neff", "-s", d + "/p.ntff", "-r", "8", "-m", d + "/inputs.txt",
                        "--num-exec=3", "--profile-nth-exec=3", "--save-output-npy", "--save-nth-output", "3"] +
                       [f"--save-output-rank={r}" for r in range(8)], capture_output=True, text=True, env=env, cwd=d)
    if p.returncode:
        rs = [l.split("reason:")[1][:90] for l in (p.stdout + p.stderr).splitlines() if "reason:" in l]
        print(f"mask {m} pairs {R.pairs_for(m)}: RUN FAILED {rs[:1]}"); continue
    # oracle: 2P x F gather, N=8 even -> both ranks hold [x_hi; x_hi]
    bad = []
    for lo, hi in R.pairs_for(m):
        want = np.concatenate([R.rank_input(hi, F)] * 2, 0)
        for r in (lo, hi):
            o = glob.glob(f"{d}/output_worker_{r}/*.npy")
            if not o or not np.array_equal(np.load(o[0]), want): bad.append(r)
    print(f"mask {m} pairs {R.pairs_for(m)}: check {'ok' if not bad else 'BAD ' + str(bad)}")
    for f in sorted(glob.glob(d + "/p_rank_*_exec_3.ntff")):
        pq = f[:-5] + "_pq"; os.makedirs(pq, exist_ok=True)
        subprocess.run(["neuron-explorer", "view", "-n", d + "/kernel.neff", "-s", f, "--ingest-only", "--output-format=parquet",
                        f"--output-file={pq}/"], capture_output=True, check=True)
        pk = pd.read_parquet(glob.glob(pq + "/**/DmaPacket.parquet", recursive=True)[0])
        c = pk[pk.queue_type.astype(str) == "collectives"]
        big = c[c.transfer_bytes > 64]
        q = big.groupby(big.queue_name.str.replace(r"-Q\d+-E\d+$", "", regex=True)).agg(
            eng=("engine_idx", lambda s: f"E{s.min()}-{s.max()}"), KiB=("transfer_bytes", lambda s: round(s.sum() / 1024)))
        span = (c.end_ts.max() - c.start_ts.min()) / 1e3 if len(c) else float("nan")
        print(f"  rank {f.split('_')[-3]}: span {span:.1f} us  " + "  ".join(f"{k}:{v.eng}:{v.KiB}KiB" for k, v in q.iterrows()))
