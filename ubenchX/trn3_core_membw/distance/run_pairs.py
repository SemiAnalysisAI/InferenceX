"""Core-to-core one-way latency and bandwidth through the Neuron collectives runtime (nccom-test permute).

The TPU port's distance page measured every TensorCore pair with a chain of ppermutes. On trn3, NKI's
raw core-to-core DMA (nisa.sendrecv) only reaches the other core sharing an HBM stack, so every other
path is measured through the runtime's own collective: a permute in which the two ranks exchange
buffers (a -> b and b -> a at once; the runtime rejects a one-way permute: "Rank 0 appears as source
but never as target"), with output verification (--check), as the trn3 counterpart of ppermute.
time_us is per exchange; algbw is per direction.

  python run_pairs.py --pairs 0-1,0-2,0-8 --out /w/runs/distance/intra [--sizes 8,4096,...]

A pair "a-b" sets NEURON_RT_VISIBLE_CORES=lo-hi at LNC=1: the runtime only accepts a contiguous core
range ("selected NeuronCores are not contiguous"). The two end ranks exchange; every rank in between
permutes to itself (the runtime fails with idle ranks). Caveat: at large sizes such a self-copy is local
HBM traffic, and a middle core can share an HBM stack with an endpoint. Each pair writes <out>/<a>-<b>.json with nccom-test's own JSON report plus the parsed table.
"""

import argparse
import json
import os
import re
import subprocess
import time

ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def one_pair(a, b, args):
    path = os.path.join(args.out, f"{a}-{b}.json")
    if os.path.exists(path):
        return "skip"
    lo, hi = sorted((a, b))
    n = hi - lo + 1
    pairs = os.path.join(args.out, f".pairs-{a}-{b}.json")
    with open(pairs, "w") as f:
        json.dump({"src_target_pairs": [[[0, n - 1], [n - 1, 0]] + [[k, k] for k in range(1, n - 1)]]}, f)
    rows, raw = [], {}
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=f"{lo}-{hi}")
    t0 = time.time()
    for size in args.sizes:
        rep = os.path.join(args.out, f".rep-{a}-{b}-{size}.json")
        cmd = ["nccom-test", "permute", "-r", str(n), "-b", str(size), "-e", str(size), "-n", str(args.iters),
               "-w", str(args.warmup), "--custom-src-target-pairs", pairs, "--check", "--non-interactive",
               "--report-to-json-file", rep]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        ok = False
        for line in r.stdout.splitlines():
            m = ROW.match(line)
            if m:
                rows.append({"bytes": int(m[1]), "time_us": float(m[4]), "algbw_GBps": float(m[5])})
                ok = True
        if os.path.exists(rep):
            raw[size] = json.load(open(rep))
            os.remove(rep)
        if not ok:
            rows.append({"bytes": size, "error": (r.stdout + r.stderr)[-600:]})
    os.remove(pairs)
    json.dump({"src_core": a, "dst_core": b, "ranks": n, "rows": rows, "report": raw, "t0": t0, "t1": time.time()},
              open(path, "w"), indent=1)
    good = [r for r in rows if "time_us" in r]
    return f"{a}<->{b}: " + " ".join(f"{r['bytes']}B={r['time_us']}us" for r in good[:1] + good[-1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="comma list of a-b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="8,4096,65536,1048576,16777216,67108864")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    args.sizes = [int(s) for s in args.sizes.split(",")]
    os.makedirs(args.out, exist_ok=True)
    for p in args.pairs.split(","):
        a, b = (int(x) for x in p.split("-"))
        print(one_pair(a, b, args), flush=True)


if __name__ == "__main__":
    main()
