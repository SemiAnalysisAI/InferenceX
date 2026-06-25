#!/usr/bin/env python3
"""Generate canonical serialized workloads (goal Part 1). Runs build_workload (needs torch) for
each (routing, global_tokens) in a ladder and writes <workload_id>.npz + .manifest.json into a
dir that runs then consume via `run_ep.py --workload-dir`. One trace per global-token count
because the generator is not prefix-consistent across sizes.

  python3 tests/make_workloads.py --out-dir /data/sa-shared/cx_workloads \\
      --routing uniform --ep 8 --hidden 7168 --topk 8 --experts 256 --seed 67 \\
      --tokens-ladder "1 2 4 8 16 32 64 128 256 512"

Generate every routing the suites need by running once per --routing. Idempotent (same id => same
file). The dir is the cross-hardware artifact: copy it to each cluster so all consume identical bytes.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload as wl   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate canonical CollectiveX workloads")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--routing", required=True)
    ap.add_argument("--ep", type=int, required=True, help="ep_size (global_tokens = T * ep)")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--tokens-ladder", default="1 2 4 8 16 32 64 128 256 512")
    a = ap.parse_args()
    epr = a.experts // a.ep
    ladder = sorted({int(t) for t in a.tokens_ladder.replace(",", " ").split() if int(t) > 0})
    os.makedirs(a.out_dir, exist_ok=True)
    made = []
    for T in ladder:
        gt = T * a.ep
        idx, w, man = wl.build_workload(a.hidden, a.topk, a.experts, a.routing, gt, a.seed, epr)
        wid = wl.save_workload(a.out_dir, idx, w, man)
        made.append((T, gt, wid))
        print(f"  T={T:<5} gt={gt:<6} routing={a.routing} -> {wid}  "
              f"(trace sha {man['checksums']['trace'][:12]})")
    print(f"wrote {len(made)} canonical workloads to {a.out_dir} (routing={a.routing}, ep={a.ep})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
