"""One-off probe (v3): decompose why a kernel-to-kernel memory handoff is slower on Blackwell.

Splits the handoff into fence cost, load latency per cache level, store->load, atomics, same-SM vs
cross-SM, concurrency, and whether a busy GPU (clock / power state) changes it. Probe branch only.
"""

import datetime
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time

SRC = r"""
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unistd.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("\"error\": \"%s line %d\"}\n", cudaGetErrorString(e_), __LINE__); fflush(stdout); exit(0); } } while (0)
typedef unsigned long long u64;

__device__ __forceinline__ void fence_do(int f) {
  if (f == 1) __threadfence_block(); else if (f == 2) __threadfence(); else if (f == 3) __threadfence_system();
}
__global__ void empty() {}
__global__ void setflag(volatile int *stop) { *stop = 1; __threadfence_system(); }

// Ping-pong between (block 0, thread 0) and (block kb, thread kt); everyone else returns.
__global__ void ping(volatile int *a, volatile int *b, int rounds, int kb, int kt, int fence) {
  int first;
  if (blockIdx.x == 0 && threadIdx.x == 0) first = 1;
  else if ((int)blockIdx.x == kb && (int)threadIdx.x == kt) first = 0;
  else return;
  volatile int *mine = first ? a : b, *theirs = first ? b : a;
  for (int i = 0; i < rounds; ++i) {
    if (first) { *theirs = 2 * i + 1; fence_do(fence); while (*mine != 2 * i + 2) {} }
    else { while (*mine != 2 * i + 1) {} *theirs = 2 * i + 2; fence_do(fence); }
  }
}
// P pairs at once, blocks (2p, 2p+1); each counts round trips until the host raises stop.
__global__ void pairs(volatile int *flags, volatile int *stop, u64 *out, int max_rounds) {
  int b = blockIdx.x, p = b / 2, first = !(b & 1);
  volatile int *mine = flags + (p * 2 + (first ? 0 : 1)) * 64;
  volatile int *theirs = flags + (p * 2 + (first ? 1 : 0)) * 64;
  int i = 0; bool live = true; unsigned spins = 0;
  for (; live && i < max_rounds; ++i) {
    if ((i & 127) == 0 && *stop) break;
    if (first) {
      *theirs = 2 * i + 1; __threadfence_system();
      while (*mine != 2 * i + 2) { if ((++spins & 1023) == 0 && *stop) { live = false; break; } }
    } else {
      while (*mine != 2 * i + 1) { if ((++spins & 1023) == 0 && *stop) { live = false; break; } }
      if (live) { *theirs = 2 * i + 2; __threadfence_system(); }
    }
  }
  if (first) out[p] = i;
}
// Dependent loads: each address depends on the value just loaded, so time/step = load latency.
__global__ void chase(const unsigned *buf, u64 slot_mask, int steps, unsigned *out, int vol) {
  u64 idx = 1; unsigned acc = 0;
  for (int i = 0; i < steps; ++i) {
    const unsigned *p = buf + idx * 16;
    unsigned v = vol ? *(volatile const unsigned *)p : *p;
    acc += v;
    idx = (((idx + v) * 6364136223846793005ULL + 1442695040888963407ULL) >> 17) & slot_mask;
  }
  *out = acc + (unsigned)idx;
}
__global__ void storeload(volatile int *p, int rounds, int fence, int *out) {
  int acc = 0;
  for (int i = 0; i < rounds; ++i) { *p = i + acc; fence_do(fence); acc = *p - i; }
  *out = acc;
}
__global__ void storeonly(volatile int *p, int rounds, int fence) {
  for (int i = 0; i < rounds; ++i) { *p = i; fence_do(fence); }
}
__global__ void fenceonly(int rounds, int fence, int *out) {
  int acc = 0;
  for (int i = 0; i < rounds; ++i) { fence_do(fence); acc += i; }
  *out = acc;
}
__global__ void atomics(int *p, int rounds, int *out) {
  int acc = 0;
  for (int i = 0; i < rounds; ++i) acc += atomicAdd(p, 1);
  *out = acc;
}
__global__ void burn_spin(unsigned *out, volatile int *stop, u64 iters) {
  unsigned x = threadIdx.x + blockIdx.x * 977u + 1u;
  for (u64 i = 0; i < iters; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    if ((i & 4095) == 0 && *stop) break;
  }
  if (threadIdx.x == 0) out[blockIdx.x] = x;
}
__global__ void burn_mem(unsigned *buf, u64 words, volatile int *stop, int passes) {
  u64 base = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * words;
  for (int p = 0; p < passes; ++p) {
    for (u64 i = 0; i < words; i += 16) buf[base + i] += 1;
    if (*stop) break;
  }
}

static double now_us() {
  return std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double epoch_ms() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
static bool g_first = true;
static void key(const char *name) { printf("%s\"%s\": ", g_first ? "" : ", ", name); g_first = false; }
static void emit(const char *name, double v) { key(name); printf("%.4f", v); fflush(stdout); }

template <class F> static double timed(F f, int reps = 3) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    CK(cudaDeviceSynchronize());
    double t0 = now_us(); f(); CK(cudaDeviceSynchronize());
    best = std::min(best, now_us() - t0);
  }
  return best;
}

static int *g_flags;
static double ping_us(int B, int T, int kb, int kt, int fence, int rounds, cudaStream_t s) {
  CK(cudaMemsetAsync(g_flags, 0, 1 << 16, s)); CK(cudaStreamSynchronize(s));
  double t0 = now_us();
  ping<<<B, T, 0, s>>>(g_flags, g_flags + 4096, rounds, kb, kt, fence);
  CK(cudaStreamSynchronize(s));
  return (now_us() - t0) / rounds;
}

int main() {
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int sms = prop.multiProcessorCount;
  printf("{");
  emit("sms", sms);
  cudaStream_t s1, s2, s3; CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2)); CK(cudaStreamCreate(&s3));
  CK(cudaMalloc(&g_flags, 1 << 20)); CK(cudaMemset(g_flags, 0, 1 << 20));
  int *stop, *out; CK(cudaMalloc(&stop, 4096)); CK(cudaMalloc(&out, 1 << 20));
  double base = timed([&] { empty<<<1, 1>>>(); }, 20);
  emit("launch_sync_us", base);

  // 1. Light phase, long enough for nvidia-smi to sample clocks while one pair pings.
  key("phase_light_ms"); printf("[%.0f", epoch_ms());
  std::vector<double> light;
  for (double t0 = now_us(); now_us() - t0 < 3.0e6;) light.push_back(ping_us(8, 1, 4, 0, 3, 20000, s1));
  printf(", %.0f]", epoch_ms());
  std::sort(light.begin(), light.end());
  emit("ping_cross_sm_fence_system", light[light.size() / 2]);

  // 2. Fence kind, and same-SM (two warps of one block) vs cross-SM.
  const char *fn[] = {"none", "block", "device", "system"};
  char name[96];
  for (int f = 0; f < 4; ++f) {
    snprintf(name, sizeof name, "ping_cross_sm_fence_%s", fn[f]);
    if (f != 3) emit(name, ping_us(8, 1, 4, 0, f, 20000, s1));
    snprintf(name, sizeof name, "ping_same_sm_fence_%s", fn[f]);
    emit(name, ping_us(1, 64, 0, 32, f, 20000, s1));
  }

  // 3. Single-thread costs.
  for (int f = 0; f < 4; ++f) {
    snprintf(name, sizeof name, "fence_only_%s_ns", fn[f]);
    emit(name, (timed([&] { fenceonly<<<1, 1>>>(2000000, f, out); }) - base) / 2000.0);
    snprintf(name, sizeof name, "store_fence_%s_ns", fn[f]);
    emit(name, (timed([&] { storeonly<<<1, 1>>>(g_flags, 2000000, f); }) - base) / 2000.0);
    snprintf(name, sizeof name, "store_fence_%s_load_ns", fn[f]);
    emit(name, (timed([&] { storeload<<<1, 1>>>(g_flags, 200000, f, out); }) - base) / 200.0);
  }
  emit("atomic_add_ns", (timed([&] { atomics<<<1, 1>>>(g_flags, 200000, out); }) - base) / 200.0);

  // 4. Dependent-load latency by footprint (L1 -> L2 -> HBM), plain and volatile loads.
  {
    const u64 big = 1ull << 32; unsigned *buf; CK(cudaMalloc(&buf, big)); CK(cudaMemset(buf, 0, big));
    u64 sizes[] = {1ull << 12, 1ull << 16, 1ull << 20, 1ull << 24, 1ull << 26, 1ull << 28, 1ull << 30, 1ull << 32};
    for (int i = 0; i < 8; ++i) for (int vol = 0; vol < 2; ++vol) {
      u64 mask = sizes[i] / 64 - 1; int steps = 200000;
      snprintf(name, sizeof name, "chase_%s_%lluKB_ns", vol ? "volatile" : "plain", sizes[i] >> 10);
      emit(name, (timed([&] { chase<<<1, 1>>>(buf, mask, steps, (unsigned *)out, vol); }) - base) * 1000.0 / steps);
    }
    CK(cudaFree(buf));
  }

  // 5. Concurrency: P pairs at once.
  int counts[] = {1, 2, 3, 4, 6, 8, 16, 32};
  for (int c = 0; c < 8; ++c) {
    int P = counts[c]; u64 *res; CK(cudaMalloc(&res, 8 * 4096));
    CK(cudaMemset(res, 0, 8 * 4096)); CK(cudaMemset(stop, 0, 4096)); CK(cudaMemset(g_flags, 0, 1 << 20));
    CK(cudaDeviceSynchronize());
    double t0 = now_us();
    pairs<<<2 * P, 1, 0, s1>>>(g_flags, stop, res, 4000000);
    while (now_us() - t0 < 60000.0) {}
    setflag<<<1, 1, 0, s2>>>(stop); CK(cudaStreamSynchronize(s2));
    double window = now_us() - t0; CK(cudaStreamSynchronize(s1));
    std::vector<u64> host(P); CK(cudaMemcpy(host.data(), res, 8 * P, cudaMemcpyDeviceToHost));
    snprintf(name, sizeof name, "pairs_%d", P); key(name); printf("[");
    for (int p = 0; p < P; ++p) printf("%s%.3f", p ? ", " : "", host[p] ? window / (double)host[p] : -1.0);
    printf("]"); fflush(stdout); CK(cudaFree(res));
  }

  // 6. Does a busy GPU change the handoff? Ping while other SMs spin or move memory.
  unsigned *mem; const u64 words = 16384; const int threads = 256;
  for (int mode = 0; mode < 4; ++mode) {
    int blocks = (mode & 1) ? sms / 2 : sms - 8;
    const char *label = mode == 0 ? "spin_most_sms" : mode == 1 ? "spin_half_sms" : mode == 2 ? "mem_most_sms" : "mem_half_sms";
    if (mode >= 2) { CK(cudaMalloc(&mem, (u64)blocks * threads * words * 4)); CK(cudaMemset(mem, 0, (u64)blocks * threads * words * 4)); }
    CK(cudaMemset(stop, 0, 4096)); CK(cudaDeviceSynchronize());
    double t0 = now_us(), e0 = epoch_ms();
    if (mode < 2) burn_spin<<<blocks, threads, 0, s2>>>((unsigned *)out, stop, 4000000000ull);
    else burn_mem<<<blocks, threads, 0, s2>>>(mem, words, stop, 100000);
    usleep(100000);
    std::vector<double> got;
    while (now_us() - t0 < 2.0e6) got.push_back(ping_us(8, 1, 4, 0, 3, 5000, s1));
    int overlap = cudaStreamQuery(s2) == cudaErrorNotReady;
    double e1 = epoch_ms();
    setflag<<<1, 1, 0, s3>>>(stop); CK(cudaStreamSynchronize(s3)); CK(cudaStreamSynchronize(s2));
    std::sort(got.begin(), got.end());
    snprintf(name, sizeof name, "ping_while_%s", label); key(name);
    printf("{\"min\": %.3f, \"med\": %.3f, \"max\": %.3f, \"n\": %d, \"overlap\": %d, \"burn_ms\": %.0f, \"phase_ms\": [%.0f, %.0f]}",
           got.front(), got[got.size() / 2], got.back(), (int)got.size(), overlap, (now_us() - t0) / 1000.0, e0, e1);
    fflush(stdout);
    if (mode >= 2) CK(cudaFree(mem));
  }
  emit("ping_after_burn", ping_us(8, 1, 4, 0, 3, 20000, s1));
  printf("}\n");
  return 0;
}
"""

SMI = "timestamp,clocks.sm,clocks.mem,pstate,power.draw,utilization.gpu,utilization.memory,temperature.gpu"


def _samples(path):
    rows = []
    try:
        for line in open(path):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            try:
                stamp = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
                rows.append((time.mktime(stamp.timetuple()) * 1000 + stamp.microsecond / 1000, parts[1:]))
            except ValueError:
                continue
    except OSError:
        pass
    return rows


def _phase(rows, span):
    inside = [vals for stamp, vals in rows if span[0] <= stamp <= span[1]]
    if not inside:
        return {"n": 0}
    return {"n": len(inside), "sm_clock": sorted({v[0] for v in inside}), "mem_clock": sorted({v[1] for v in inside}),
            "pstate": sorted({v[2] for v in inside}), "power": [inside[0][3], inside[-1][3]],
            "util": sorted({v[4] for v in inside})[-3:]}


def run(out_dir):
    out = {}
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    with tempfile.TemporaryDirectory() as tmp:
        src, exe, log = os.path.join(tmp, "die.cu"), os.path.join(tmp, "die"), os.path.join(tmp, "smi.csv")
        with open(src, "w") as handle:
            handle.write(SRC)
        build = subprocess.run([nvcc, "-O2", "-o", exe, src], capture_output=True, text=True, timeout=300)
        if build.returncode:
            out = {"error": "build: " + build.stderr[-1500:]}
        else:
            idle = subprocess.run(["nvidia-smi", f"--query-gpu={SMI}", "--format=csv,noheader", "-i", "0"],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
            sampler = subprocess.Popen(["nvidia-smi", f"--query-gpu={SMI}", "--format=csv,noheader", "-i", "0",
                                        "-lms", "100", "-f", log])
            try:
                proc = subprocess.run([exe], capture_output=True, text=True, timeout=420)
                stdout, note = proc.stdout, f"rc={proc.returncode} err={proc.stderr[-400:]}"
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                note = "timeout"
            sampler.terminate()
            try:
                out = json.loads(stdout.strip().splitlines()[-1])
            except Exception:
                out = {"error": note, "partial": stdout[-8000:]}
            rows = _samples(log)
            out["smi_idle_before"] = idle
            out["smi_samples"] = len(rows)
            clocks = {}
            if isinstance(out.get("phase_light_ms"), list):
                clocks["light_ping"] = _phase(rows, out["phase_light_ms"])
            for name, value in out.items():
                if isinstance(value, dict) and "phase_ms" in value:
                    clocks[name] = _phase(rows, value["phase_ms"])
            out["smi_by_phase"] = clocks
    text = json.dumps(out)
    print("DIEPROBE " + text, flush=True)
    with open(os.path.join(out_dir, "die_probe.json"), "w") as handle:
        handle.write(text)
