"""Run a mem-bw sweep: compile every point in parallel (CPU only), then capture them one at a time.

  python sweep.py --tag single --kernels copy_h2h_static --sizes fine --core 0 [--max-bytes 8G]
  python sweep.py --tag ladder --kernels copy_h2h_static,copy_stage_static --sizes ladder --core 0

Each point lands in /w/runs/<tag>/<kernel>/<bytes>/result.json; finished points are skipped, so a
sweep resumes after interruption. --order shuffle randomises the capture order (seeded) so slow drift
in the machine does not line up with size.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 2**30


def fine_sizes(lo=8, hi=16 * GIB, per_octave=16):
    """Same grid as the TPU7x port (ubenchx_tpu.py fine_sizes): per_octave log-spaced sizes, each
    rounded to a multiple of the largest power of two <= size/64 (8 B minimum)."""
    out, k = [], 0
    while True:
        x = lo * 2 ** (k / per_octave)
        if x > hi * (1 + 1e-9):
            return out
        g = max(8, 2 ** int(math.floor(math.log2(max(x / 64, 1)))))
        s = int(round(x / g)) * g
        if not out or s != out[-1]:
            out.append(s)
        k += 1


def parse_bytes(s):
    mult = {"K": 2**10, "M": 2**20, "G": 2**30}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def size_list(name, max_bytes):
    if name == "fine":
        sizes = fine_sizes()
    elif name == "pow2":
        sizes = [2**k for k in range(3, 35)]
    elif name == "ladder":
        sizes = [8, 1024, 65536, 2**20, 4 * 2**20, 16 * 2**20, 256 * 2**20, GIB]
    else:
        sizes = [parse_bytes(x) for x in name.split(",")]
    return [s for s in sizes if s <= max_bytes]


def min_bytes(kernel):
    return 8 if kernel.endswith("_spmd") else 4


def point(tag, kernel, size, args, compile_only):
    out = f"/w/runs/{tag}/{kernel}/{size}"
    if os.path.exists(os.path.join(out, "result.json")):
        return "skip"
    cmd = [sys.executable, os.path.join(HERE, "run_point.py"), "--kernel", kernel, "--bytes", str(size),
           "--out", out, "--lnc", str(args.lnc),
           "--num-exec", str(args.small_num_exec if size < args.small_bytes else args.num_exec),
           "--profile-every", str(args.profile_every), "--max-chunk", str(args.max_chunk)]
    if compile_only:
        if os.path.exists(os.path.join(out, "kernel.neff")):
            return "skip"
        cmd += ["--compile-only"]
    else:
        cmd += ["--neff", os.path.join(out, "kernel.neff"), "--core", args.core]
        cmd += ["--input", os.path.join(out, "input.npy")]
    os.makedirs(out, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(out, "compile.log" if compile_only else "capture.log"), "w") as f:
        f.write(r.stdout + r.stderr)
    if not compile_only and r.returncode == 0 and os.path.exists(os.path.join(out, "input.npy")):
        os.remove(os.path.join(out, "input.npy"))  # inputs reach 8 GiB; the NEFF stays
    if r.returncode != 0:
        return f"FAIL rc={r.returncode}"
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--sizes", default="fine")
    ap.add_argument("--max-bytes", default="8G")
    ap.add_argument("--core", default="0")
    ap.add_argument("--lnc", type=int, default=1)
    ap.add_argument("--num-exec", type=int, default=12)
    ap.add_argument("--profile-every", type=int, default=2)
    ap.add_argument("--small-bytes", type=int, default=64 * 2**20, help="below this, use --small-num-exec")
    ap.add_argument("--small-num-exec", type=int, default=36)
    ap.add_argument("--max-chunk", type=int, default=4 * 2**20)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--order", choices=["shuffle", "ascending"], default="shuffle")
    ap.add_argument("--shard", default="0/1", help="k/n: this worker takes every n-th size starting at k")
    args = ap.parse_args()
    kernels = args.kernels.split(",")
    sizes = size_list(args.sizes, parse_bytes(args.max_bytes))
    k_, n_ = (int(v) for v in args.shard.split("/"))
    sizes = sizes[k_::n_]
    pts = [(k, s) for k in kernels for s in sizes]
    print(f"{len(pts)} points, {len(sizes)} sizes {sizes[0]}..{sizes[-1]}", flush=True)
    # Compile big sizes first so the long compiles overlap the most work.
    with ThreadPoolExecutor(args.jobs) as ex:
        for (k, s), res in zip(sorted(pts, key=lambda p: -p[1]),
                               ex.map(lambda p: point(args.tag, p[0], p[1], args, True),
                                      sorted(pts, key=lambda p: -p[1]))):
            if res.startswith("FAIL"):
                print(f"compile {k} {s}: {res}", flush=True)
    order = list(pts)
    if args.order == "shuffle":
        random.Random(2026111100).shuffle(order)
    for i, (k, s) in enumerate(order):
        if not os.path.exists(f"/w/runs/{args.tag}/{k}/{s}/kernel.neff"):
            continue
        print(f"[{i + 1}/{len(order)}] {point(args.tag, k, s, args, False)}", flush=True)


if __name__ == "__main__":
    main()
