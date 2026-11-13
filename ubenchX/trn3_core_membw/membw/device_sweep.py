"""Full-device mem-bw: all 4 logical cores (LNC=2, one per HBM stack) copy a quarter each, concurrently.

The TPU port's full chip ran two single-chiplet programs at once and divided total bytes by the slower
chiplet's device time. Here: one capture process per logical core (4 per device), all started together;
each runs long enough (>= --min-run-s of executions) that the profiled executions, taken from the middle
of the run, overlap the other three cores' runs. Device bandwidth = 2 * total bytes / max over cores of
that core's median exec time. result.json per core plus device.json with the wall-clock windows, so the
analysis can check that all four ran at once.

  python device_sweep.py --tag device --device 4 --sizes fine --max-bytes 16G
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep import parse_bytes, size_list  # noqa: E402

KERNEL = "copy_h2h_static_spmd"


def est_exec_s(per_core_bytes):
    return 12e-6 + 2 * per_core_bytes / 0.9e12


def run_point(cmd, log):
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(log, "w").write(r.stdout + r.stderr)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="device")
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--sizes", default="fine")
    ap.add_argument("--max-bytes", default="16G")
    ap.add_argument("--min-run-s", type=float, default=8.0)
    ap.add_argument("--sizes-min", default="0")
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--shard", default="0/1", help="k/n: this worker takes every n-th size starting at k")
    a = ap.parse_args()
    cores = [4 * a.device + k for k in range(4)]
    sizes = [s for s in size_list(a.sizes, parse_bytes(a.max_bytes)) if s % 32 == 0 and s >= parse_bytes(a.sizes_min)]
    k_, n_ = (int(v) for v in a.shard.split("/"))
    sizes = sizes[k_::n_]
    base = f"/w/runs/{a.tag}"

    def cdir(s):
        return f"{base}/{KERNEL}/{s}"

    def compile_one(s):
        if os.path.exists(cdir(s) + "/neff/kernel.neff"):
            return 0
        os.makedirs(cdir(s), exist_ok=True)
        return run_point([sys.executable, f"{HERE}/run_point.py", "--kernel", KERNEL, "--bytes", str(s // 4),
                          "--lnc", "2", "--out", cdir(s) + "/neff", "--compile-only"], cdir(s) + "/compile.log")

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(compile_one, sorted(sizes, reverse=True)))
    order = list(sizes)
    random.Random(2026111102).shuffle(order)
    for i, s in enumerate(order):
        if os.path.exists(cdir(s) + "/device.json") or not os.path.exists(cdir(s) + "/neff/kernel.neff"):
            continue
        n = min(20000, max(41, int(a.min_run_s / est_exec_s(s // 4))))
        every = max(2, n // 14)  # ~14 profiled executions per core, spread over the run
        procs, t0 = [], time.time()
        for c in cores:
            cmd = [sys.executable, f"{HERE}/run_point.py", "--kernel", KERNEL, "--bytes", str(s // 4), "--lnc", "2",
                   "--neff", cdir(s) + "/neff/kernel.neff", "--input", cdir(s) + "/neff/input.npy",
                   "--core", str(c), "--num-exec", str(n), "--profile-every", str(every), "--out", f"{cdir(s)}/c{c}"]
            procs.append(subprocess.Popen(cmd, stdout=open(f"{cdir(s)}/c{c}.log", "w"), stderr=subprocess.STDOUT))
        rcs = [p.wait() for p in procs]
        res = {}
        for c in cores:
            try:
                res[c] = json.load(open(f"{cdir(s)}/c{c}/result.json"))
            except (OSError, ValueError):
                res[c] = None
        ok = all(r is not None for r in res.values()) and not any(rcs)
        dev = {"bytes": s, "per_core_bytes": s // 4, "cores": cores, "num_exec": n, "profile_every": every,
               "returncodes": rcs, "ok": ok, "t0": t0, "t1": time.time()}
        if ok:
            med = {c: sorted(x["exec_ns"] for x in r["rows"])[len(r["rows"]) // 2] for c, r in res.items()}
            dev.update({"exec_med_ns": med, "windows": {c: [r["capture_start"], r["capture_end"]] for c, r in res.items()},
                        "bytes_ok": all(x["bytes_ok"] for r in res.values() for x in r["rows"]),
                        "GBps": 2 * s / max(med.values())})
        json.dump(dev, open(cdir(s) + "/device.json", "w"), indent=1)
        if ok and os.path.exists(cdir(s) + "/neff/input.npy"):
            os.remove(cdir(s) + "/neff/input.npy")
        print(f"[{i + 1}/{len(order)}] {s} B n={n} ok={ok} GB/s={dev.get('GBps', 0):.1f} "
              f"bytes_ok={dev.get('bytes_ok')}", flush=True)


if __name__ == "__main__":
    main()
