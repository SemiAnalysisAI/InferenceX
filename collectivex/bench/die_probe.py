"""One-off probe: is Blackwell's slow kernel-to-kernel signalling tied to WHERE the two kernels'
SMs sit (two dies) or WHERE the memory sits? Looks for a bimodal split that Hopper lacks."""

import json
import os
import shutil
import subprocess
import tempfile

SRC = r"""
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("{\"error\": \"%s line %d\"}\n", cudaGetErrorString(e_), __LINE__); exit(0); } } while (0)
typedef unsigned long long u64;

__global__ void setflag(volatile int *stop) { *stop = 1; __threadfence_system(); }
// Every block is one side of a ping-pong pair and counts the round trips it finishes before the
// host raises `stop`. near: blocks (2p, 2p+1); far: (p, p + B/2).
__global__ void pairs(volatile int *flags, volatile int *stop, u64 *out, int max_rounds, int far, int B) {
  int b = blockIdx.x, half = B / 2, p, first;
  if (far) { p = b % half; first = b < half; } else { p = b / 2; first = !(b & 1); }
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
__global__ void sweep(volatile int *a, volatile int *b, int rounds, int k) {
  int first;
  if (blockIdx.x == 0) first = 1; else if ((int)blockIdx.x == k) first = 0; else return;
  volatile int *mine = first ? a : b, *theirs = first ? b : a;
  for (int i = 0; i < rounds; ++i) {
    if (first) { *theirs = 2 * i + 1; __threadfence_system(); while (*mine != 2 * i + 2) {} }
    else { while (*mine != 2 * i + 1) {} *theirs = 2 * i + 2; __threadfence_system(); }
  }
}
__global__ void pingpong(volatile int *mine, volatile int *theirs, int rounds, int first) {
  for (int i = 0; i < rounds; ++i) {
    if (first) { *theirs = 2 * i + 1; __threadfence_system(); while (*mine != 2 * i + 2) {} }
    else { while (*mine != 2 * i + 1) {} *theirs = 2 * i + 2; __threadfence_system(); }
  }
}
static double now_us() {
  return std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Per-pair round-trip time (us): round trips finished inside a host-timed window.
static void pair_test(const char *name, int B, int far) {
  int *flags, *stop; u64 *out; cudaStream_t sp, ss;
  CK(cudaMalloc(&flags, 1 << 20)); CK(cudaMalloc(&stop, 4096)); CK(cudaMalloc(&out, 8 * 4096));
  CK(cudaMemset(flags, 0, 1 << 20)); CK(cudaMemset(stop, 0, 4096)); CK(cudaMemset(out, 0, 8 * 4096));
  CK(cudaStreamCreate(&sp)); CK(cudaStreamCreate(&ss));
  CK(cudaDeviceSynchronize());
  double t0 = now_us();
  pairs<<<B, 1, 0, sp>>>(flags, stop, out, 400000, far, B);
  while (now_us() - t0 < 60000.0) {}
  setflag<<<1, 1, 0, ss>>>(stop);
  CK(cudaStreamSynchronize(ss));
  double window = now_us() - t0;
  CK(cudaStreamSynchronize(sp));
  double total = now_us() - t0;
  std::vector<u64> host(B / 2); CK(cudaMemcpy(host.data(), out, 8 * (B / 2), cudaMemcpyDeviceToHost));
  printf(", \"%s\": {\"window_us\": %.0f, \"total_us\": %.0f, \"per_round_us\": [", name, window, total);
  for (int p = 0; p < B / 2; ++p) printf("%s%.3f", p ? ", " : "", host[p] ? window / (double)host[p] : -1.0);
  printf("]}"); fflush(stdout);
  CK(cudaFree(flags)); CK(cudaFree(stop)); CK(cudaFree(out));
}

static double ping_at(int *fa, int *fb, int rounds) {
  cudaStream_t sa, sb; CK(cudaStreamCreate(&sa)); CK(cudaStreamCreate(&sb));
  std::vector<double> s;
  for (int rep = 0; rep < 3; ++rep) {
    CK(cudaMemset(fa, 0, 64)); CK(cudaMemset(fb, 0, 64)); CK(cudaDeviceSynchronize());
    double t0 = now_us();
    pingpong<<<1, 1, 0, sb>>>(fb, fa, rounds, 0);
    pingpong<<<1, 1, 0, sa>>>(fa, fb, rounds, 1);
    CK(cudaStreamSynchronize(sa)); CK(cudaStreamSynchronize(sb));
    s.push_back((now_us() - t0) / rounds);
  }
  std::sort(s.begin(), s.end());
  CK(cudaStreamDestroy(sa)); CK(cudaStreamDestroy(sb));
  return s[s.size() / 2];
}

// Round-trip time between block 0 and block k of one launch that fills the GPU's SMs.
static void sweep_test(const char *name, int B, int threads) {
  int *flags; CK(cudaMalloc(&flags, 1 << 20));
  const int rounds = 10000;
  printf(", \"%s\": [", name);
  for (int k = 1; k < B; ++k) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      CK(cudaMemset(flags, 0, 1 << 20)); CK(cudaDeviceSynchronize());
      double t0 = now_us();
      sweep<<<B, threads>>>(flags, flags + 4096, rounds, k);
      CK(cudaDeviceSynchronize());
      double t = (now_us() - t0) / rounds; if (t < best) best = t;
    }
    printf("%s%.3f", k > 1 ? ", " : "", best); fflush(stdout);
  }
  printf("]"); fflush(stdout);
  CK(cudaFree(flags));
}

int main() {
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int sms = prop.multiProcessorCount;
  size_t freeb, totalb; CK(cudaMemGetInfo(&freeb, &totalb));
  printf("{\"sms\": %d, \"mem_gb\": %.1f", sms, totalb / 1e9); fflush(stdout);
  int B = (sms - 2) & ~1;
  sweep_test("sweep_partner_block", sms, 1);
  sweep_test("sweep_partner_block_2x", 2 * sms, 1);
  pair_test("near_all", B, 0);
  pair_test("far_all", B, 1);
  pair_test("near_all_repeat", B, 0);
  pair_test("near_8pairs", 16, 0);
  pair_test("near_1pair", 2, 0);
  // Memory placement: one pair, flags placed in 16 chunks spread over ~85% of memory.
  const int chunks = 16; size_t chunk = (size_t)(freeb * 0.85 / chunks) & ~((size_t)4095);
  std::vector<int *> ptr(chunks, nullptr);
  int got = 0;
  for (int i = 0; i < chunks; ++i) { if (cudaMalloc(&ptr[i], chunk) != cudaSuccess) break; ++got; }
  printf(", \"chunk_gb\": %.2f, \"ping_by_chunk_us\": [", chunk / 1e9);
  for (int i = 0; i < got; ++i) printf("%s%.3f", i ? ", " : "", ping_at(ptr[i], ptr[i] + 2048, 5000)); fflush(stdout);
  printf("], \"ping_cross_chunk_us\": [");
  for (int i = 1; i < got; ++i) printf("%s%.3f", i > 1 ? ", " : "", ping_at(ptr[0], ptr[i], 5000)); fflush(stdout);
  printf("]}\n");
  return 0;
}
"""


def run(out_dir):
    out = {}
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = os.path.join(tmp, "die.cu"), os.path.join(tmp, "die")
        with open(src, "w") as handle:
            handle.write(SRC)
        build = subprocess.run([nvcc, "-O2", "-o", exe, src], capture_output=True, text=True, timeout=300)
        if build.returncode:
            out = {"error": "build: " + build.stderr[-1500:]}
        else:
            try:
                proc = subprocess.run([exe], capture_output=True, text=True, timeout=240)
                stdout, note = proc.stdout, f"rc={proc.returncode} err={proc.stderr[-400:]}"
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                note = "timeout"
            try:
                out = json.loads(stdout.strip().splitlines()[-1])
            except Exception:
                out = {"error": note, "partial": stdout[-6000:]}
    text = json.dumps(out)
    print("DIEPROBE " + text, flush=True)
    with open(os.path.join(out_dir, "die_probe.json"), "w") as handle:
        handle.write(text)
