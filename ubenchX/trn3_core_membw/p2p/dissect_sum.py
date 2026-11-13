"""Summarise saved DmaPacket tables from p2p/dissect.py: data bursts vs the rest of each step."""
import sys, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 80)
for d in sys.argv[1:]:
    p = pd.read_parquet(d + "/DmaPacket.parquet")
    print("\n=====", d)
    if d == sys.argv[1]:
        print(p.dtypes.to_string()); print(p.iloc[len(p) // 2].to_string())
    print("queue types:", p.groupby(p.queue_type.astype(str)).agg(n=("start_ts", "size"), bytes=("transfer_bytes", "sum")).to_dict("index"))
    c = p[p.queue_type.astype(str) == "collectives"].sort_values("start_ts")
    t0 = c.start_ts.min(); span = c.end_ts.max() - t0
    c = c.assign(q=c.queue_name.str.replace(r"-Q\d+-E\d+$", "", regex=True), t=(c.start_ts - t0) / 1e3, dur=c.end_ts - c.start_ts)
    print(f"span {span / 1e3:.1f} us, {len(c)} packets")
    print(c.groupby(["q", "transfer_bytes"]).agg(n=("t", "size"), dur_med=("dur", "median"), dur_max=("dur", "max")).to_string())
    big = c[c.transfer_bytes >= c.transfer_bytes.max()]
    # bursts: data packets separated by > 1 us
    g = (big.t.diff() > 1).cumsum()
    b = big.groupby(g).agg(start=("t", "min"), end=("t", "max"), n=("t", "size"), last_dur=("dur", "max"), qs=("q", lambda s: ",".join(sorted(set(s)))))
    b["width_us"] = b.end - b.start + b.last_dur / 1e3; b["gap_us"] = b.start.diff()
    print(b[["start", "width_us", "gap_us", "n", "qs"]].round(2).to_string())
    busy = c.dur.sum() / 1e3
    print(f"sum of all packet durations {busy:.2f} us of span {span / 1e3:.1f} us; data-burst widths total {b.width_us.sum():.2f} us")
