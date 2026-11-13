"""Per rank: which DMA engines/queues carry its collective traffic (one capture, NSTEP=8, 8 KiB/partition)."""
import glob, os, subprocess, sys
import pandas as pd
d = sys.argv[1]
env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=sys.argv[2])
for f in glob.glob(d + "/*.ntff"): os.remove(f)
subprocess.run(["neuron-explorer", "capture", "-n", d + "/kernel.neff", "-s", d + "/p.ntff", "-r", "8", "-m", d + "/inputs.txt",
                "--num-exec=3", "--profile-nth-exec=3"], capture_output=True, env=env, cwd=d, check=True)
for f in sorted(glob.glob(d + "/p_rank_*_exec_3.ntff")):
    pq = f[:-5] + "_pq"; os.makedirs(pq, exist_ok=True)
    subprocess.run(["neuron-explorer", "view", "-n", d + "/kernel.neff", "-s", f, "--ingest-only", "--output-format=parquet",
                    f"--output-file={pq}/"], capture_output=True, check=True)
    p = pd.read_parquet(glob.glob(pq + "/**/DmaPacket.parquet", recursive=True)[0])
    c = p[p.queue_type.astype(str) == "collectives"]
    big = c[c.transfer_bytes > 64]
    g = big.groupby("queue_name").transfer_bytes.sum()
    eng = sorted(c.engine_idx.unique())
    print(f"{os.path.basename(f)[:9]}: engines {eng[0]}-{eng[-1]} ({len(eng)})  data by queue (KiB):",
          {k: round(v / 1024) for k, v in g.items()}, "pcore", sorted(c.pcore_idx.unique()) if "pcore_idx" in c else "")
