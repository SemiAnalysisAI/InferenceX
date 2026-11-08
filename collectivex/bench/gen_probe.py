"""One-off probe: why do short EP kernels cost more on Blackwell than on Hopper?

Separates (a) per-kernel cost on one GPU, (b) per-SM instruction rate, (c) small-copy cost on one
GPU, (d) DMA peer copies, and (e) kernel-to-kernel signalling across NVLink (the DeepEP pattern).
Not part of the benchmark; lives on a probe branch only.
"""

import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
import traceback

PING_SRC = r"""
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("{\"error\": \"%s line %d\"}\n", cudaGetErrorString(e_), __LINE__); exit(0); } } while (0)

// One side of a ping-pong: wait for my flag to reach i (written by the other side), then write
// i+1 into the other side's flag. `first` starts the exchange.
__global__ void pingpong(volatile int *mine, volatile int *theirs, int rounds, int first) {
  for (int i = 0; i < rounds; ++i) {
    if (first) { *theirs = 2 * i + 1; __threadfence_system(); while (*mine != 2 * i + 2) {} }
    else { while (*mine != 2 * i + 1) {} *theirs = 2 * i + 2; __threadfence_system(); }
  }
}
// Write n bytes into dst (local or peer), one thread per 16 bytes like a tiny dispatch.
__global__ void put(unsigned char *dst, int n) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
  for (int k = 0; k < 16 && i + k < n; ++k) dst[i + k] = (unsigned char)(i + k);
}
__global__ void empty() {}

static double now_us() {
  return std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// Round-trip signalling time between a kernel on devA and one on devB (devA == devB: same GPU).
static double ping(int devA, int devB, int rounds) {
  int *fa, *fb; cudaStream_t sa, sb;
  CK(cudaSetDevice(devA)); CK(cudaMalloc(&fa, 4096)); CK(cudaMemset(fa, 0, 4096)); CK(cudaStreamCreate(&sa));
  CK(cudaSetDevice(devB)); CK(cudaMalloc(&fb, 4096)); CK(cudaMemset(fb, 0, 4096)); CK(cudaStreamCreate(&sb));
  std::vector<double> s;
  for (int rep = 0; rep < 7; ++rep) {
    CK(cudaSetDevice(devA)); CK(cudaMemset(fa, 0, 4096)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(devB)); CK(cudaMemset(fb, 0, 4096)); CK(cudaDeviceSynchronize());
    double t0 = now_us();
    CK(cudaSetDevice(devB)); pingpong<<<1, 1, 0, sb>>>(fb, fa, rounds, 0);
    CK(cudaSetDevice(devA)); pingpong<<<1, 1, 0, sa>>>(fa, fb, rounds, 1);
    CK(cudaStreamSynchronize(sa)); CK(cudaSetDevice(devB)); CK(cudaStreamSynchronize(sb));
    s.push_back((now_us() - t0) / rounds);
  }
  return med(s);
}
// GPU-timed cost of one kernel writing n bytes from dev 0 into memory on `target`.
static double putcost(int target, int n, int chain) {
  unsigned char *dst; CK(cudaSetDevice(target)); CK(cudaMalloc(&dst, 1 << 26));
  CK(cudaSetDevice(0));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int threads = (n + 15) / 16, blocks = (threads + 255) / 256;
  std::vector<double> s;
  for (int rep = 0; rep < 60; ++rep) {
    CK(cudaEventRecord(a));
    for (int k = 0; k < chain; ++k) put<<<blocks, 256>>>(dst, n);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); s.push_back(ms * 1000.0 / chain);
  }
  return med(s);
}
static double emptycost(int blocks, int chain) {
  CK(cudaSetDevice(0));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<double> s;
  for (int rep = 0; rep < 60; ++rep) {
    CK(cudaEventRecord(a));
    for (int k = 0; k < chain; ++k) empty<<<blocks, 256>>>();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); s.push_back(ms * 1000.0 / chain);
  }
  return med(s);
}

int main() {
  int n; CK(cudaGetDeviceCount(&n));
  CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
  printf("{\"gpus\": %d", n);
  printf(", \"empty_1block_us\": %.3f, \"empty_64blocks_us\": %.3f, \"empty_4096blocks_us\": %.3f",
         emptycost(1, 256), emptycost(64, 256), emptycost(4096, 64));
  int sizes[] = {64, 14336, 1 << 20, 1 << 24};
  for (int i = 0; i < 4; ++i) printf(", \"put_local_%d_us\": %.3f", sizes[i], putcost(0, sizes[i], 64));
  printf(", \"ping_same_gpu_us\": %.4f", ping(0, 0, 20000));
  if (n > 1) {
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
    CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));
    for (int i = 0; i < 4; ++i) printf(", \"put_peer_%d_us\": %.3f", sizes[i], putcost(1, sizes[i], 64));
    printf(", \"ping_peer_us\": %.4f", ping(0, 1, 20000));
  }
  printf("}\n");
  return 0;
}
"""


def _med(values):
    return statistics.median(values)


def _torch_part(out):
    import torch

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    out["device"] = torch.cuda.get_device_name(0)
    out["gpus"] = torch.cuda.device_count()
    out["torch"] = str(torch.__version__)

    def graph_of(op, count):
        op()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(count):
                op()
        return graph

    def gpu_us(fn, reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gpu, wall = [], []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start.record()
            fn()
            end.record()
            end.synchronize()
            wall.append((time.perf_counter() - t0) * 1e6)
            gpu.append(start.elapsed_time(end) * 1000.0)
        return _med(gpu), _med(wall)

    def slope(name, op, reps=80, counts=(8, 256)):
        """Per-kernel GPU time inside a replayed graph: host launch cost cancels in the slope."""
        try:
            times = {}
            for count in counts:
                graph = graph_of(op, count)
                for _ in range(10):
                    graph.replay()
                times[count] = gpu_us(graph.replay, reps)
            lo, hi = counts
            out[name] = {
                "per_kernel_us": (times[hi][0] - times[lo][0]) / (hi - lo),
                "graph_fixed_us": times[lo][0] - lo * (times[hi][0] - times[lo][0]) / (hi - lo),
                "wall_fixed_us": times[lo][1] - lo * (times[hi][1] - times[lo][1]) / (hi - lo),
            }
        except Exception:
            out[name] = {"error": traceback.format_exc(limit=2)}

    for count in (1, 4096, 65536, 1 << 20, 1 << 24):
        x = torch.zeros(count, device=dev, dtype=torch.float32)
        slope(f"add_f32_{count}", lambda x=x: x.add_(1.0), counts=(8, 256) if count < (1 << 24) else (4, 32))
    for nbytes in (14336, 1 << 20, 1 << 26):
        a = torch.zeros(nbytes, device=dev, dtype=torch.uint8)
        b = torch.zeros(nbytes, device=dev, dtype=torch.uint8)
        slope(f"copy_local_{nbytes}", lambda a=a, b=b: b.copy_(a), counts=(8, 256) if nbytes < (1 << 26) else (4, 32))
    for cycles in (1000, 100000):
        slope(f"spin_{cycles}_cycles", lambda c=cycles: torch.cuda._sleep(c), counts=(4, 32))

    # Eager launch + synchronize of one tiny kernel: host + GPU, for reference.
    x = torch.zeros(1, device=dev)
    wall = []
    for _ in range(300):
        t0 = time.perf_counter()
        x.add_(1.0)
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1e6)
    out["eager_tiny_kernel_and_sync_wall_us"] = _med(wall)

    if torch.cuda.device_count() > 1:
        for nbytes in (14336, 1 << 20, 1 << 26):
            a = torch.zeros(nbytes, device="cuda:0", dtype=torch.uint8)
            b = torch.zeros(nbytes, device="cuda:1", dtype=torch.uint8)
            b.copy_(a)
            wall = []
            for _ in range(100):
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                t0 = time.perf_counter()
                b.copy_(a)
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                wall.append((time.perf_counter() - t0) * 1e6)
            out[f"dma_peer_copy_{nbytes}_wall_us"] = _med(wall)


def _native_part(out):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not os.path.exists(nvcc):
        out["native"] = {"error": "no nvcc"}
        return
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = os.path.join(tmp, "ping.cu"), os.path.join(tmp, "ping")
        with open(src, "w") as handle:
            handle.write(PING_SRC)
        build = subprocess.run([nvcc, "-O2", "-o", exe, src], capture_output=True, text=True, timeout=300)
        if build.returncode:
            out["native"] = {"error": "build: " + build.stderr[-1500:]}
            return
        run = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        try:
            out["native"] = json.loads(run.stdout.strip().splitlines()[-1])
        except Exception:
            out["native"] = {"error": f"rc={run.returncode} out={run.stdout[-800:]} err={run.stderr[-800:]}"}


def run(out_dir):
    out = {"host": platform.node(), "machine": platform.machine()}
    for key, cmd in (
        ("driver", ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.draw", "--format=csv,noheader", "-i", "0"]),
        ("cpu", ["sh", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|MHz' | tr -s ' ' | tr '\\n' ';'"]),
        ("nvlink", ["sh", "-c", "nvidia-smi nvlink -s -i 0 2>&1 | head -6 | tr '\\n' ';'"]),
    ):
        try:
            out[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:
            out[key] = f"error: {exc}"
    for part in (_torch_part, _native_part):
        try:
            part(out)
        except Exception:
            out[part.__name__] = {"error": traceback.format_exc(limit=3)}
    text = json.dumps(out)
    print("GENPROBE " + text, flush=True)
    try:
        with open(os.path.join(out_dir, "gen_probe.json"), "w") as handle:
            handle.write(text)
    except Exception:
        pass
