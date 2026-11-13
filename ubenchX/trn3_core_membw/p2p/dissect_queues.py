"""Queue families per rank for any p2p/dissect.py trace (also when nothing is on the collectives queue).
Usage: dissect_queues.py DIR [RANKS_TO_PRINT=all]"""
import glob, re, sys
import pandas as pd
d = sys.argv[1]
for f in sorted(glob.glob(d + "/DmaPacket_r*.parquet"), key=lambda s: int(re.search(r"_r(\d+)", s).group(1))):
    k = int(re.search(r"_r(\d+)", f).group(1))
    p = pd.read_parquet(f)
    q = p.queue_name.astype(str).str.replace(r"(-\d+)?-Q\d+-E\d+$", "", regex=True)
    g = p.assign(q=q, qt=p.queue_type.astype(str)).groupby(["qt", "q"]).agg(n=("start_ts", "size"), b=("transfer_bytes", "sum"))
    print(f"rank {k:3d} E{p.engine_idx.min()}-{p.engine_idx.max()} | " + "  ".join(f"{qt}/{qn}:{r.n}p,{r.b}B" for (qt, qn), r in g.iterrows()))
