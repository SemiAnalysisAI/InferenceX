"""Per-rank summary of p2p/dissect.py traces: step time, queues, data-packet timing. Usage: dissect_ranks.py DIR [NSTEP=9]"""
import glob, re, sys
import pandas as pd
d = sys.argv[1]; nstep = int(sys.argv[2]) if len(sys.argv) > 2 else 9
for f in sorted(glob.glob(d + "/DmaPacket_r*.parquet"), key=lambda s: int(re.search(r"_r(\d+)", s).group(1))):
    k = int(re.search(r"_r(\d+)", f).group(1))
    p = pd.read_parquet(f)
    c = p[p.queue_type.astype(str) == "collectives"].sort_values("start_ts")
    c = c.assign(q=c.queue_name.str.replace(r"(-\d+)?-Q\d+-E\d+$", "", regex=True), t=(c.start_ts - c.start_ts.min()) / 1e3, dur=c.end_ts - c.start_ts)
    big = c[c.transfer_bytes >= c.transfer_bytes.max()]
    st = big.t[big.t.diff().fillna(99) > 1.0]  # burst starts
    step = st.diff().median()
    qs = " ".join(f"{q}:{n}" for q, n in c.q.value_counts().sort_index().items())
    dq = " ".join(f"{q}:{n}" for q, n in big.q.value_counts().sort_index().items())
    eng = f"E{c.engine_idx.min()}-{c.engine_idx.max()}"
    print(f"rank {k:2d} pcore {int(p.pcore_idx.iloc[0]):3d} {eng:9s} span {c.t.max():7.1f} us  bursts {len(st):3d} step {step:6.2f} us  "
          f"data {int(big.transfer_bytes.iloc[0])} B x{len(big)} dur med {big.dur.median():5.0f} max {big.dur.max():6.0f} ns | all [{qs}] data [{dq}]")
