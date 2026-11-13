"""Split each pointer-chase hop into stages from a kept profile (parquet dir), for one core.

Stages per hop, matched in order (hop i = i-th Sync DMA_DIRECT2D, i-th pair of hardware_dynamic packets,
i-th Sync TENSOR_LOAD after it):
  issue      DMA_DIRECT2D start -> end (trigger instruction on the Sync engine)
  to_data    DMA_DIRECT2D end -> first packet start (descriptor generation, queue, the HBM read)
  data       first packet start -> last packet end
  to_engine  last packet end -> Sync TENSOR_LOAD start (completion semaphore reaching the engine)
  reg_load   Sync TENSOR_LOAD duration (index from SBUF into the engine register)
  to_next    TENSOR_LOAD end -> next DMA_DIRECT2D start (ALU ops, semaphores, MOVE)
"""
import glob
import sys

import numpy as np
import pandas as pd


def stages(pq):
    rd = lambda t: pd.concat([pd.read_parquet(f) for f in glob.glob(f"{pq}/**/{t}.parquet", recursive=True)])  # noqa: E731
    ins = rd("Instruction")
    ins = ins[ins.engine.astype(str) == "Sync"].sort_values("start_ts")
    trig = ins[ins.opcode.astype(str) == "DMA_DIRECT2D"].reset_index(drop=True)
    tl = ins[ins.opcode.astype(str) == "TENSOR_LOAD"].sort_values("start_ts")
    pk = rd("DmaPacket")
    d = pk[pk.queue_type.astype(str) == "hardware_dynamic"].sort_values("start_ts").reset_index(drop=True)
    a, b = d.iloc[0::2].reset_index(drop=True), d.iloc[1::2].reset_index(drop=True)
    n = min(len(trig), len(a)) - 1
    rows = []
    tl_start = tl.start_ts.to_numpy()
    tl_end = tl.end_ts.to_numpy()
    for i in range(n):
        pend = max(a.end_ts[i], b.end_ts[i])
        j = np.searchsorted(tl_start, pend)
        if j >= len(tl_start):
            break
        rows.append({"issue": trig.end_ts[i] - trig.start_ts[i], "to_data": a.start_ts[i] - trig.end_ts[i],
                     "data": pend - a.start_ts[i], "to_engine": tl_start[j] - pend,
                     "reg_load": tl_end[j] - tl_start[j], "to_next": trig.start_ts[i + 1] - tl_end[j],
                     "hop": trig.start_ts[i + 1] - trig.start_ts[i]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for pq in sys.argv[1:]:
        s = stages(pq)
        print(pq, len(s), "hops")
        print("   median:", {k: float(s[k].median()) for k in s.columns})
        print("   mean:  ", {k: round(float(s[k].mean()), 1) for k in s.columns})
