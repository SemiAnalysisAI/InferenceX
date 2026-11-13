"""The GPU mem-bw page's method on trn3: eager PyTorch b.copy_(a), host-side timing.

Port of InferenceX ubenchX/mem_bw/bench.py with device "neuron" (native torch-neuronx 2.14). The GPU
script times with triton.testing.do_bench (CUDA events around each call, 25 ms warmup, 100 ms of repeats,
mean). Triton does not target Neuron, so do_bench_like() reproduces that loop with torch.neuron.Event,
minus the L2 flush (trn3 has no cache between HBM and the DMA engines). Two numbers per size:

  event_us  mean per-call time between Neuron events (do_bench's number)
  wall_us   N back-to-back async copies, synchronize, wall time / N (the TPU port's dashed line)

bandwidth = 2 * bytes / time (one read and one write per byte), as on the GPU page. A torch device is
one logical NeuronCore: at the default LNC=2 that is two physical cores sharing one HBM stack, a quarter
of the chip. The output is checked once per size (torch.equal on the device).

  NEURON_RT_VISIBLE_CORES=20 python torch_copy.py --out /w/runs/torch/lnc2.json
"""

import argparse
import json
import math
import time

import torch

GIB = 2**30


def fine_sizes(lo=8, hi=16 * GIB, per_octave=16):
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


def do_bench_like(fn, warmup_ms=25, rep_ms=100):
    m = torch.neuron
    fn()
    m.synchronize()
    s, e = m.Event(enable_timing=True), m.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        fn()
    e.record()
    m.synchronize()
    est = max(s.elapsed_time(e) / 5, 1e-4)
    for _ in range(max(1, int(warmup_ms / est))):
        fn()
    n = max(1, int(rep_ms / est))
    starts = [m.Event(enable_timing=True) for _ in range(n)]
    ends = [m.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        fn()
        ends[i].record()
    m.synchronize()
    ts = [a.elapsed_time(b) for a, b in zip(starts, ends)]
    return sum(ts) / len(ts) * 1e3, n


def wall_like(fn, n):
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bytes", type=int, default=8 * GIB)
    ap.add_argument("--grid", choices=["fine", "pow2"], default="fine")
    a = ap.parse_args()
    sizes = fine_sizes() if a.grid == "fine" else [1 << p for p in range(3, 35)]
    sizes = [s for s in sizes if s <= a.max_bytes]
    dev = torch.device("neuron")
    # empty + fill_, not randn: randn of 16 GiB compiles to an op with 56 GB of scratch, over one logical
    # core's 36 GB, and the failure surfaces asynchronously. --max-bytes keeps 2 x size inside the stack.
    fit = sizes[-1]
    x = torch.empty(fit // 4, device=dev)
    x.copy_(torch.arange(fit // 4, dtype=torch.float32) % 1000003)
    y = torch.empty_like(x)
    y.zero_()
    torch.neuron.synchronize()
    rows = []
    for size in sizes:
        if size > fit:
            rows.append({"bytes": size, "skipped": "source and destination do not fit"})
            continue
        n = size // 4
        src, dst = x[:n], y[:n]
        fn = lambda: dst.copy_(src)  # noqa: E731
        ev, reps = do_bench_like(fn)
        wall = wall_like(fn, max(10, min(reps, 2000)))
        ok = bool(torch.equal(src, dst))
        rows.append({"bytes": size, "event_us": ev, "wall_us": wall, "reps": reps, "check": ok,
                     "event_GBps": 2 * size / ev / 1e3, "wall_GBps": 2 * size / wall / 1e3})
        print(f"{size:>14} B  event {ev:10.2f} us {2 * size / ev / 1e3:9.1f} GB/s   wall {wall:10.2f} us "
              f"{2 * size / wall / 1e3:9.1f} GB/s  check={ok}", flush=True)
    json.dump({"device": torch.neuron.get_device_name(), "rows": rows, "fit_bytes": fit}, open(a.out, "w"),
              indent=1)


if __name__ == "__main__":
    main()
