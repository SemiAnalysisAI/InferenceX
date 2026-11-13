"""Packet-level traces of one exchange chain: where a step's time goes (data burst vs everything else).

  python p2p/dissect.py /w/runs/p2p/sweep/ag_hbm-m2/f4/n9 /w/runs/p2p/dissect/ag_hbm-m2 16-23 [ranks=8]

Re-captures an existing probe point (kernel.neff + inputs.txt) and keeps every rank's DmaPacket table as
<out>/DmaPacket_r<k>.parquet (rank 0 also as DmaPacket.parquet). Summaries: p2p/dissect_sum.py.
"""
import glob, os, shutil, subprocess, sys
import pandas as pd
src, out, cores = sys.argv[1:4]
ranks = int(sys.argv[4]) if len(sys.argv) > 4 else 8
os.makedirs(out, exist_ok=True)
env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=cores)
r = subprocess.run(["neuron-explorer", "capture", "-n", src + "/kernel.neff", "-s", out + "/p.ntff", "-r", str(ranks), "-m",
                    src + "/inputs.txt", "--num-exec=4", "--profile-nth-exec=4"], capture_output=True, text=True, env=env, cwd=out)
if r.returncode:
    sys.exit((r.stdout + r.stderr)[-1500:])
for k in range(ranks):
    f = sorted(glob.glob(f"{out}/p_rank_{k}_exec_*.ntff"))[-1]
    pq = f"{out}/pq{k}"
    subprocess.run(["neuron-explorer", "view", "-n", src + "/kernel.neff", "-s", f, "--ingest-only", "--output-format=parquet",
                    f"--output-file={pq}/"], capture_output=True, check=True)
    p = pd.read_parquet(glob.glob(pq + "/**/DmaPacket.parquet", recursive=True)[0])
    p.to_parquet(f"{out}/DmaPacket_r{k}.parquet")
    if k == 0:
        p.to_parquet(out + "/DmaPacket.parquet")
    shutil.rmtree(pq, ignore_errors=True)
    c = p[p.queue_type.astype(str) == "collectives"]
    print(f"rank {k}: {len(c)} collectives packets, span {(c.end_ts.max() - c.start_ts.min()) / 1e3:.1f} us", flush=True)
