"""One-off probe (v3, extended through v8): decompose why a kernel-to-kernel memory handoff is slower on Blackwell.

Splits the handoff into fence cost, load latency per cache level, store->load, atomics, same-SM vs
cross-SM, concurrency, and whether a busy GPU (clock / power state) changes it. Probe branch only.

v8 asks why one system departs from the rest: every result carries the SM clock it ran at, every
visible GPU is swept, the per-op cost is checked for stalls, the round trip to host memory is
measured, and the fences are re-timed while the PCIe link is kept busy and while the host sleeps.
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
#include <algorithm>
#include <vector>
#include <thread>
#include <cstring>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("\"error\": \"%s line %d\"}\n", cudaGetErrorString(e_), __LINE__); fflush(stdout); exit(0); } } while (0)

// Exact PTX memory-ordering ops the EP libraries use, by scope.
#define OP_FENCE(n, txt) __device__ __forceinline__ void n() { asm volatile(txt ::: "memory"); }
OP_FENCE(f_acq_cta, "fence.acq_rel.cta;") OP_FENCE(f_acq_gpu, "fence.acq_rel.gpu;") OP_FENCE(f_acq_sys, "fence.acq_rel.sys;")
OP_FENCE(f_sc_cta, "fence.sc.cta;") OP_FENCE(f_sc_gpu, "fence.sc.gpu;") OP_FENCE(f_sc_sys, "fence.sc.sys;")
__device__ __forceinline__ void st_rel_gpu(int *p, int v) { asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rel_sys(int *p, int v) { asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rlx_gpu(int *p, int v) { asm volatile("st.relaxed.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rlx_sys(int *p, int v) { asm volatile("st.relaxed.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ int ld_acq_gpu(int *p) { int r; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(r) : "l"(p) : "memory"); return r; }
__device__ __forceinline__ int ld_acq_sys(int *p) { int r; asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(r) : "l"(p) : "memory"); return r; }
__device__ __forceinline__ int ld_rlx_gpu(int *p) { int r; asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(r) : "l"(p) : "memory"); return r; }
__device__ __forceinline__ int ld_rlx_sys(int *p) { int r; asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(r) : "l"(p) : "memory"); return r; }
__device__ __forceinline__ int at_gpu(int *p, int v) { int r; asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ int at_sys(int *p, int v) { int r; asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ void red_gpu(int *p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_sys(int *p, int v) { asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

__global__ void empty() {}
// Write n bytes into dst (local or peer), one thread per 16 bytes like a tiny dispatch.
__global__ void put(unsigned char *dst, int n) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
  for (int k = 0; k < 16 && i + k < n; ++k) dst[i + k] = (unsigned char)(i + k);
}
// N threads each run the same ordering op in a loop, one per block or one per warp of a block.
__global__ void fencers(int *p, int rounds, int op, int nactive) {
  if (threadIdx.x % 32 != 0) return;
  int warps = (blockDim.x + 31) / 32, id = blockIdx.x * warps + threadIdx.x / 32;
  if (id >= nactive) return;
  int *slot = p + id * 64, acc = 0;
  for (int i = 0; i < rounds; ++i) {
    switch (op) {
      case 0: f_acq_sys(); break;
      case 1: st_rel_sys(slot, i); break;
      case 2: f_acq_gpu(); break;
      case 3: st_rel_gpu(slot, i); break;
      case 4: acc += ld_rlx_gpu(slot + (acc & 1)); break;
    }
  }
  if (acc == 12345) slot[1] = acc;
}
// One op per iteration; the running value feeds the next iteration so ops cannot overlap.
__global__ void single(int *p, int rounds, int op, int *out) {
  int acc = 0;
  for (int i = 0; i < rounds; ++i) {
    switch (op) {
      case 0: acc += i; break;
      case 1: f_acq_cta(); break;  case 2: f_acq_gpu(); break;  case 3: f_acq_sys(); break;
      case 4: f_sc_cta(); break;   case 5: f_sc_gpu(); break;   case 6: f_sc_sys(); break;
      case 7: st_rlx_gpu(p, i + acc); break;  case 8: st_rlx_sys(p, i + acc); break;
      case 9: st_rel_gpu(p, i + acc); break;  case 10: st_rel_sys(p, i + acc); break;
      case 11: acc += ld_rlx_gpu(p + (acc & 1)); break; case 12: acc += ld_rlx_sys(p + (acc & 1)); break;
      case 13: acc += ld_acq_gpu(p + (acc & 1)); break; case 14: acc += ld_acq_sys(p + (acc & 1)); break;
      case 15: acc += at_gpu(p, 1) & 1; break; case 16: acc += at_sys(p, 1) & 1; break;
      case 17: red_gpu(p, 1); break; case 18: red_sys(p, 1); break;
      case 19: st_rel_gpu(p, i + acc); acc = ld_acq_gpu(p) - i; break;
      case 20: st_rel_sys(p, i + acc); acc = ld_acq_sys(p) - i; break;
      case 21: __threadfence(); break; case 22: __threadfence_system(); break;
    }
  }
  *out = acc;
}
// Ping-pong with a chosen store/load flavour; block 0 vs block kb.
__global__ void ping(int *a, int *b, int rounds, int kb, int mode) {
  int first;
  if (blockIdx.x == 0) first = 1; else if ((int)blockIdx.x == kb) first = 0; else return;
  int *mine = first ? a : b, *theirs = first ? b : a;
  for (int i = 0; i < rounds; ++i) {
    int want = first ? 2 * i + 2 : 2 * i + 1, put = first ? 2 * i + 1 : 2 * i + 2;
    if (!first) {
      if (mode == 0) while (ld_rlx_gpu(mine) != want) {} else if (mode == 1) while (ld_acq_gpu(mine) != want) {}
      else if (mode == 2) while (ld_acq_sys(mine) != want) {} else while (ld_acq_sys(mine) != want) {}
    }
    if (mode == 0) st_rlx_gpu(theirs, put); else if (mode == 1) st_rel_gpu(theirs, put);
    else if (mode == 2) st_rel_sys(theirs, put); else { st_rlx_sys(theirs, put); f_acq_sys(); }
    if (first) {
      if (mode == 0) while (ld_rlx_gpu(mine) != want) {} else if (mode == 1) while (ld_acq_gpu(mine) != want) {}
      else if (mode == 2) while (ld_acq_sys(mine) != want) {} else while (ld_acq_sys(mine) != want) {}
    }
  }
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
// v8: one op per call, chosen at compile time so the loop adds only a few cycles.
template <int OP> __device__ __forceinline__ void do_op(int *p, int i, int &acc) {
  if (OP == 0) asm volatile("add.s32 %0, %0, %1;" : "+r"(acc) : "r"(i));
  else if (OP == 2) f_acq_gpu();
  else if (OP == 3) f_acq_sys();
  else if (OP == 10) st_rel_sys(p, i + acc);
  else if (OP == 11) acc += ld_rlx_gpu(p + (acc & 1));
  else if (OP == 12) acc += ld_rlx_sys(p + (acc & 1));
  else if (OP == 23) { st_rlx_sys(p, i + acc); f_acq_sys(); }
  else if (OP == 24) { st_rlx_gpu(p, i + acc); f_acq_gpu(); }
}
// The op loop bracketed by SM cycles and the global nanosecond timer: each result carries the
// clock it ran at and excludes launch cost.
template <int OP> __global__ void top(int *p, int rounds, long long *out) {
  int acc = 0;
  long long c0 = clock64(); unsigned long long t0 = gtimer();
  for (int i = 0; i < rounds; ++i) do_op<OP>(p, i, acc);
  long long c1 = clock64(); unsigned long long t1 = gtimer();
  out[0] = c1 - c0; out[1] = (long long)(t1 - t0); out[2] = acc; out[3] = (long long)t0; out[4] = (long long)t1;
}
// Cycles per chunk of `per` consecutive ops, kept in shared memory so recording adds no global
// store for the next fence to wait on. A fixed cost gives one narrow peak; stalls give a tail.
template <int OP> __global__ void chunks(int *p, int nchunks, int per, unsigned *out) {
  __shared__ unsigned buf[2048];
  int acc = 0;
  for (int c = 0; c < nchunks; ++c) {
    long long c0 = clock64();
    for (int k = 0; k < per; ++k) do_op<OP>(p, c * per + k, acc);
    buf[c] = (unsigned)(clock64() - c0);
  }
  for (int c = 0; c < nchunks; ++c) out[c] = buf[c];
  if (acc == 12345) out[0] = 0;
}
// Keeps one SM busy for `ns`: mode 0 spins on registers, mode 1 keeps re-reading `src` (local,
// peer or host memory), which keeps the link to that memory active.
__global__ void keep_busy(int *src, long long ns, int mode, long long *out) {
  unsigned long long t0 = gtimer(), t = t0; long long n = 0; int acc = 0;
  while ((long long)(t - t0) < ns) {
    if (mode) acc += ld_rlx_sys(src + (acc & 1)); else asm volatile("add.s32 %0, %0, 1;" : "+r"(acc));
    ++n; t = gtimer();
  }
  out[0] = n; out[1] = acc; out[2] = (long long)t0; out[3] = (long long)t;
}
static double now_us() {
  return std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class F> static double timed(F f, int reps = 3) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    CK(cudaDeviceSynchronize());
    double t0 = now_us(); f(); CK(cudaDeviceSynchronize());
    best = std::min(best, now_us() - t0);
  }
  return best;
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
// GPU-timed cost of one kernel in a chain of 64 writing n bytes from GPU 0 into memory on `target`.
static double putcost(int target, int n) {
  unsigned char *dst; CK(cudaSetDevice(target)); CK(cudaMalloc(&dst, 1 << 26)); CK(cudaMemset(dst, 0, 1 << 26));
  CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int threads = (n + 15) / 16, blocks = (threads + 255) / 256;
  std::vector<double> s;
  for (int rep = 0; rep < 60; ++rep) {
    CK(cudaEventRecord(a));
    for (int k = 0; k < 64; ++k) put<<<blocks, 256>>>(dst, n);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); s.push_back(ms * 1000.0 / 64);
  }
  return med(s);
}
static double emptycost(int blocks) {
  CK(cudaSetDevice(0));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<double> s;
  for (int rep = 0; rep < 60; ++rep) {
    CK(cudaEventRecord(a));
    for (int k = 0; k < 64; ++k) empty<<<blocks, 256>>>();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); s.push_back(ms * 1000.0 / 64);
  }
  return med(s);
}
static const int PUT_SIZES[] = {64, 14336, 86016, 1 << 20, 1 << 24};

static void launch_top(int op, int *p, int rounds, long long *out, cudaStream_t s) {
  switch (op) {
    case 0: top<0><<<1, 1, 0, s>>>(p, rounds, out); break;
    case 2: top<2><<<1, 1, 0, s>>>(p, rounds, out); break;
    case 3: top<3><<<1, 1, 0, s>>>(p, rounds, out); break;
    case 10: top<10><<<1, 1, 0, s>>>(p, rounds, out); break;
    case 11: top<11><<<1, 1, 0, s>>>(p, rounds, out); break;
    case 12: top<12><<<1, 1, 0, s>>>(p, rounds, out); break;
    case 23: top<23><<<1, 1, 0, s>>>(p, rounds, out); break;
    case 24: top<24><<<1, 1, 0, s>>>(p, rounds, out); break;
  }
  CK(cudaGetLastError());
}
struct Res { double ns, mhz; long long t0, t1; };
// Device-timed ns per op and the SM clock during it; best of `reps` by ns.
static Res measure(int op, int *p, int rounds, long long *dout, int reps = 3, cudaStream_t s = 0) {
  Res best{1e30, 0, 0, 0};
  for (int r = 0; r < reps; ++r) {
    launch_top(op, p, rounds, dout, s); CK(cudaStreamSynchronize(s));
    long long h[5]; CK(cudaMemcpy(h, dout, sizeof h, cudaMemcpyDeviceToHost));
    double ns = (double)h[1] / rounds, mhz = h[1] ? 1000.0 * (double)h[0] / (double)h[1] : 0;
    if (ns < best.ns) best = {ns, mhz, h[3], h[4]};
  }
  return best;
}
static void print_res(const char *key, Res r, bool first = false) {
  printf("%s\"%s\": {\"ns\": %.2f, \"mhz\": %.0f}", first ? "" : ", ", key, r.ns, r.mhz);
}
// Per-op cycles from 2048 chunks of 8 ops: percentiles, and the share of chunks above 1.5x the median.
static void print_chunks(const char *key, int op, int *p, unsigned *dbuf) {
  const int nchunks = 2048, per = 8;
  if (op == 2) chunks<2><<<1, 1>>>(p, nchunks, per, dbuf); else if (op == 3) chunks<3><<<1, 1>>>(p, nchunks, per, dbuf);
  else chunks<10><<<1, 1>>>(p, nchunks, per, dbuf);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<unsigned> h(nchunks); CK(cudaMemcpy(h.data(), dbuf, nchunks * sizeof(unsigned), cudaMemcpyDeviceToHost));
  std::vector<double> v; for (int i = 1; i < nchunks; ++i) v.push_back(h[i] / (double)per);
  std::sort(v.begin(), v.end());
  double q[] = {0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0};
  printf(", \"%s\": {\"cycles_per_op_q\": [", key);
  for (int i = 0; i < 7; ++i) printf("%s%.0f", i ? ", " : "", v[std::min((size_t)(q[i] * (v.size() - 1)), v.size() - 1)]);
  double med = v[v.size() / 2]; int slow = 0; for (double x : v) slow += x > 1.5 * med;
  printf("], \"share_above_1p5x_median\": %.4f}", slow / (double)v.size());
}

static int core(bool blocking);
int main(int argc, char **argv) {
  int n; CK(cudaGetDeviceCount(&n));
  if (argc > 1 && !strcmp(argv[1], "blocking")) return core(true);
  if (argc > 1 && !strcmp(argv[1], "core")) return core(false);
  CK(cudaSetDevice(0));
  int *flags, *out; CK(cudaMalloc(&flags, 1 << 20)); CK(cudaMemset(flags, 0, 1 << 20)); CK(cudaMalloc(&out, 4096));
  double base = timed([&] { empty<<<1, 1>>>(); }, 20);
  printf("{\"gpus\": %d", n);
  const char *names[] = {"loop_only", "fence_acq_rel_cta", "fence_acq_rel_gpu", "fence_acq_rel_sys", "fence_sc_cta",
    "fence_sc_gpu", "fence_sc_sys", "st_relaxed_gpu", "st_relaxed_sys", "st_release_gpu", "st_release_sys",
    "ld_relaxed_gpu", "ld_relaxed_sys", "ld_acquire_gpu", "ld_acquire_sys", "atom_add_release_gpu",
    "atom_add_release_sys", "red_add_release_gpu", "red_add_release_sys", "st_release_ld_acquire_gpu",
    "st_release_ld_acquire_sys", "threadfence", "threadfence_system"};
  for (int op = 0; op < 23; ++op) {
    int rounds = 200000;
    printf(", \"%s_ns\": %.2f", names[op], (timed([&] { single<<<1, 1>>>(flags, rounds, op, out); }) - base) * 1000.0 / rounds);
    fflush(stdout);
  }
  const char *modes[] = {"relaxed_gpu", "release_acquire_gpu", "release_acquire_sys", "relaxed_sys_plus_fence_sys"};
  for (int m = 0; m < 4; ++m) {
    CK(cudaMemset(flags, 0, 1 << 20));
    printf(", \"ping_local_%s_us\": %.4f", modes[m], (timed([&] { ping<<<8, 1>>>(flags, flags + 4096, 20000, 4, m); }) - base) / 20000.0);
    fflush(stdout);
  }
  printf(", \"empty_kernel_1block_us\": %.3f, \"empty_kernel_4blocks_us\": %.3f", emptycost(1), emptycost(4)); fflush(stdout);
  for (int i = 0; i < 5; ++i) { printf(", \"put_local_%d_us\": %.3f", PUT_SIZES[i], putcost(0, PUT_SIZES[i])); fflush(stdout); }
  if (n > 1) {
    // Same ops against memory on the peer GPU, and a ping whose two flags both live on the peer.
    int *peer; CK(cudaSetDevice(1)); CK(cudaMalloc(&peer, 1 << 20)); CK(cudaMemset(peer, 0, 1 << 20)); CK(cudaDeviceSynchronize());
    CK(cudaDeviceEnablePeerAccess(0, 0));
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
    for (int i = 0; i < 5; ++i) { printf(", \"put_peer_%d_us\": %.3f", PUT_SIZES[i], putcost(1, PUT_SIZES[i])); fflush(stdout); }
    int peer_ops[] = {8, 10, 12, 14, 16, 18, 20};
    for (int i = 0; i < 7; ++i) {
      int op = peer_ops[i], rounds = 50000;
      printf(", \"peer_%s_ns\": %.2f", names[op], (timed([&] { single<<<1, 1>>>(peer, rounds, op, out); }) - base) * 1000.0 / rounds);
      fflush(stdout);
    }
    for (int m = 2; m < 4; ++m) {
      CK(cudaSetDevice(1)); CK(cudaMemset(peer, 0, 1 << 20)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0));
      printf(", \"ping_flags_on_peer_%s_us\": %.4f", modes[m], (timed([&] { ping<<<8, 1>>>(peer, peer + 4096, 5000, 4, m); }, 2) - base) / 5000.0);
      fflush(stdout);
    }
  }
  {
    const char *ops[] = {"fence_sys", "st_release_sys", "fence_gpu", "st_release_gpu", "ld_relaxed"};
    int nb[] = {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128}, nw[] = {1, 2, 4, 8, 16, 32};
    const int rounds = 3000;
    for (int op = 0; op < 5; ++op) {
      printf(", \"conc_blocks_%s_ns\": [", ops[op]);
      for (int i = 0; i < 13; ++i) {
        int N = nb[i];
        printf("%s%.1f", i ? ", " : "", (timed([&] { fencers<<<N, 1>>>(flags, rounds, op, N); }, 2) - base) * 1000.0 / rounds);
        fflush(stdout);
      }
      printf("], \"conc_warps_%s_ns\": [", ops[op]);
      for (int i = 0; i < 6; ++i) {
        int N = nw[i];
        printf("%s%.1f", i ? ", " : "", (timed([&] { fencers<<<1, N * 32>>>(flags, rounds, op, N); }, 2) - base) * 1000.0 / rounds);
        fflush(stdout);
      }
      printf("]"); fflush(stdout);
    }
  }
  // ---------------- v8 ----------------
  {
    CK(cudaSetDevice(0));
    long long *dout; CK(cudaMalloc(&dout, 256));
    int *hp; CK(cudaHostAlloc((void **)&hp, 1 << 16, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(hp, 0, 1 << 16);
    int *hd; CK(cudaHostGetDevicePointer((void **)&hd, hp, 0));
    printf(", \"v8_bus_ids\": [");
    for (int d = 0; d < n; ++d) { char bus[32]; CK(cudaDeviceGetPCIBusId(bus, 32, d)); printf("%s\"%s\"", d ? ", " : "", bus); }
    printf("]"); fflush(stdout);
    // 1. Device-timed costs on GPU 0, each with the SM clock it ran at.
    printf(", \"v8_dev0\": {");
    print_res("loop", measure(0, flags, 200000, dout), true);
    print_res("fence_gpu", measure(2, flags, 100000, dout));
    print_res("fence_sys", measure(3, flags, 50000, dout));
    print_res("st_release_sys", measure(10, flags, 50000, dout));
    print_res("ld_relaxed_gpu", measure(11, flags, 100000, dout));
    print_res("st_relaxed_gpu_then_fence_gpu", measure(24, flags, 100000, dout));
    print_res("st_relaxed_sys_then_fence_sys", measure(23, flags, 50000, dout));
    print_res("host_ld_relaxed_sys", measure(12, hd, 20000, dout));
    print_res("host_st_release_sys", measure(10, hd, 20000, dout));
    print_res("host_st_relaxed_sys_then_fence_sys", measure(23, hd, 20000, dout));
    printf("}"); fflush(stdout);
    // 2. Every visible GPU.
    printf(", \"v8_per_device\": [");
    for (int d = 0; d < n; ++d) {
      CK(cudaSetDevice(d));
      int *f; long long *o; CK(cudaMalloc(&f, 1 << 16)); CK(cudaMemset(f, 0, 1 << 16)); CK(cudaMalloc(&o, 256));
      char bus[32]; CK(cudaDeviceGetPCIBusId(bus, 32, d));
      printf("%s{\"bus\": \"%s\"", d ? ", " : "", bus);
      print_res("loop", measure(0, f, 200000, o));
      print_res("fence_gpu", measure(2, f, 100000, o));
      print_res("fence_sys", measure(3, f, 30000, o));
      print_res("st_release_sys", measure(10, f, 30000, o));
      print_res("ld_relaxed_gpu", measure(11, f, 100000, o));
      print_res("host_ld_relaxed_sys", measure(12, hd, 10000, o));
      printf("}"); fflush(stdout);
      CK(cudaFree(f)); CK(cudaFree(o));
    }
    printf("]"); fflush(stdout);
    CK(cudaSetDevice(0));
    // 3. Forty back-to-back trials: does the cost hold steady or switch between levels?
    {
      const int ops[] = {3, 2}; const char *keys[] = {"fence_sys", "fence_gpu"}; const int rounds[] = {10000, 40000};
      printf(", \"v8_series\": {");
      for (int k = 0; k < 2; ++k) {
        std::vector<Res> rs; for (int t = 0; t < 40; ++t) rs.push_back(measure(ops[k], flags, rounds[k], dout, 1));
        printf("%s\"%s_ns\": [", k ? ", " : "", keys[k]);
        for (int t = 0; t < 40; ++t) printf("%s%.1f", t ? ", " : "", rs[t].ns);
        printf("], \"%s_mhz\": [", keys[k]);
        for (int t = 0; t < 40; ++t) printf("%s%.0f", t ? ", " : "", rs[t].mhz);
        printf("]");
      }
      printf("}"); fflush(stdout);
    }
    // 4. Per-op distribution.
    {
      unsigned *dbuf; CK(cudaMalloc(&dbuf, 2048 * sizeof(unsigned)));
      printf(", \"v8_chunks\": {\"_\": 0");
      print_chunks("fence_gpu", 2, flags, dbuf);
      print_chunks("fence_sys", 3, flags, dbuf);
      print_chunks("st_release_sys", 10, flags, dbuf);
      printf("}"); fflush(stdout);
      CK(cudaFree(dbuf));
    }
    // 5. Fences re-timed while another SM keeps a link busy (none, registers, local HBM, host, peer).
    {
      cudaStream_t sa, sb; CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
      long long *bout; CK(cudaMalloc(&bout, 64));
      int *peerbuf = nullptr;
      if (n > 1) {
        CK(cudaSetDevice(1)); CK(cudaMalloc(&peerbuf, 1 << 16)); CK(cudaMemset(peerbuf, 0, 1 << 16)); CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(0));
        cudaError_t e = cudaDeviceEnablePeerAccess(1, 0); if (e != cudaErrorPeerAccessAlreadyEnabled) CK(e); cudaGetLastError();
      }
      const char *names[] = {"none", "spin", "local", "host", "peer"};
      int *srcs[] = {nullptr, flags + 8192, flags + 8192, hd + 64, peerbuf};
      int modes[] = {-1, 0, 1, 1, 1};
      printf(", \"v8_busy\": {\"_\": 0");
      for (int b = 0; b < 5; ++b) {
        if (b == 4 && !peerbuf) continue;
        printf(", \"%s\": {", names[b]);
        const int ops[] = {3, 2}; const char *keys[] = {"fence_sys", "fence_gpu"}; const int rounds[] = {40000, 100000};
        for (int k = 0; k < 2; ++k) {
          if (modes[b] >= 0) { keep_busy<<<1, 1, 0, sb>>>(srcs[b], 400000000LL, modes[b], bout); CK(cudaGetLastError()); }
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          Res r = measure(ops[k], flags, rounds[k], dout, 1, sa);
          long long hb[4] = {0, 0, 0, 0};
          if (modes[b] >= 0) { CK(cudaStreamSynchronize(sb)); CK(cudaMemcpy(hb, bout, sizeof hb, cudaMemcpyDeviceToHost)); }
          bool overlap = modes[b] < 0 || (hb[2] <= r.t0 && r.t1 <= hb[3]);
          double busy_ns = hb[0] ? (double)(hb[3] - hb[2]) / hb[0] : 0;
          printf("%s\"%s\": {\"ns\": %.2f, \"mhz\": %.0f, \"overlap\": %s, \"busy_ns_per_iter\": %.1f}",
                 k ? ", " : "", keys[k], r.ns, r.mhz, overlap ? "true" : "false", busy_ns);
        }
        printf("}"); fflush(stdout);
      }
      printf("}"); fflush(stdout);
    }
  }
  printf("}\n");
  return 0;
}

// Core costs in a fresh process: with `blocking` the host thread sleeps while the GPU works, so
// the CPU can idle; without it this reruns the core set (used for the ahead-of-time build).
static int core(bool blocking) {
  if (blocking) CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));  // before GPU 0's context exists
  CK(cudaSetDevice(0));
  unsigned flags_now = 0; CK(cudaGetDeviceFlags(&flags_now));
  int *flags; long long *dout; CK(cudaMalloc(&flags, 1 << 16)); CK(cudaMemset(flags, 0, 1 << 16)); CK(cudaMalloc(&dout, 256));
  int *hp; CK(cudaHostAlloc((void **)&hp, 1 << 16, cudaHostAllocMapped)); memset(hp, 0, 1 << 16);
  int *hd; CK(cudaHostGetDevicePointer((void **)&hd, hp, 0));
  printf("{\"device_flags\": %u", flags_now);
  print_res("loop", measure(0, flags, 200000, dout));
  print_res("fence_gpu", measure(2, flags, 100000, dout));
  print_res("fence_sys", measure(3, flags, 50000, dout));
  print_res("st_release_sys", measure(10, flags, 50000, dout));
  print_res("ld_relaxed_gpu", measure(11, flags, 100000, dout));
  print_res("host_ld_relaxed_sys", measure(12, hd, 20000, dout));
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
                proc = subprocess.run([exe], capture_output=True, text=True, timeout=900)
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
            try:  # v8: same core costs with the host thread asleep instead of spinning
                blk = subprocess.run([exe, "blocking"], capture_output=True, text=True, timeout=300)
                out["v8_blocking"] = json.loads(blk.stdout.strip().splitlines()[-1])
            except Exception as exc:
                out["v8_blocking"] = {"error": repr(exc)[:300]}
            # v8: the default build is PTX that the installed driver compiles at load time, so two
            # drivers can run different machine code. Rebuild ahead of time for this GPU, rerun
            # the core set, and keep the fence instructions the compiler emitted.
            try:
                cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", "0"],
                                     capture_output=True, text=True, timeout=30).stdout.strip().replace(".", "")
                native = os.path.join(tmp, "die_native")
                nb = subprocess.run([nvcc, "-O2", f"-gencode=arch=compute_{cap},code=sm_{cap}", "-o", native, src],
                                    capture_output=True, text=True, timeout=300)
                if nb.returncode:
                    out["v8_native"] = {"error": "build: " + nb.stderr[-800:]}
                else:
                    nat = subprocess.run([native, "core"], capture_output=True, text=True, timeout=300)
                    out["v8_native"] = json.loads(nat.stdout.strip().splitlines()[-1])
                    out["v8_native"]["arch"] = f"sm_{cap}"
                    cuobjdump = os.path.join(os.path.dirname(nvcc), "cuobjdump")
                    sass = subprocess.run([cuobjdump, "-sass", native], capture_output=True, text=True, timeout=120).stdout
                    keep, fn = [], ""
                    for line in sass.splitlines():
                        if "Function :" in line:
                            fn = line.split("Function :")[-1].strip()
                        elif fn.startswith("_Z3topILi") and any(t in line for t in ("MEMBAR", "FENCE", "ERRBAR", "CCTL", "CGAERRBAR", "ST.E", "LD.E", "RED.E")):
                            ins = line.split("*/")[1].split("/*")[0] if "*/" in line else line
                            keep.append(fn[:16] + " " + " ".join(ins.split()))
                    out["v8_native_sass"] = keep[:80]
            except Exception as exc:
                out["v8_native"] = {"error": repr(exc)[:300]}
    info = {}
    for key, cmd in (
        ("driver", "nvidia-smi --query-gpu=name,driver_version,vbios_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,ecc.mode.current,compute_mode,persistence_mode --format=csv,noheader -i 0"),
        ("smi_q", "nvidia-smi -q -i 0 | grep -iE 'Virtualization|GPU Operation|Addressing|Host VGPU|MIG|Accounting|Product Architecture|Firmware|GSP|Drain|Fabric|IBMNPU|Bridge' | tr -s ' ' | tr '\\n' ';'"),
        ("conf_compute", "nvidia-smi conf-compute -f 2>&1 | head -3 | tr '\\n' ';'"),
        ("nvlink", "nvidia-smi nvlink -s -i 0 2>&1 | head -4 | tr '\\n' ';'"),
        ("nvlink_cap", "nvidia-smi nvlink -c -i 0 2>&1 | sed -n '2,10p' | tr '\\n' ';'"),
        ("proc_version", "cat /proc/driver/nvidia/version 2>&1 | head -2 | tr '\\n' ';'"),
        ("proc_params", "cat /proc/driver/nvidia/params 2>&1 | tr '\\n' ';'"),
        ("uvm_params", "for f in /sys/module/nvidia_uvm/parameters/*; do echo \"$(basename $f)=$(cat $f 2>/dev/null)\"; done 2>&1 | tr '\\n' ';'"),
        ("cmdline", "cat /proc/cmdline"),
        ("kernel", "uname -srm"),
        ("cpu", "lscpu | grep -E 'Model name|Hypervisor|Virtualization|^CPU\\(s\\)' | tr -s ' ' | tr '\\n' ';'"),
        ("dmi", "cat /sys/class/dmi/id/sys_vendor /sys/class/dmi/id/product_name 2>&1 | tr '\\n' ';'"),
        ("iommu", "ls /sys/kernel/iommu_groups 2>/dev/null | wc -l"),
    ):
        try:
            info[key] = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, timeout=30).stdout.strip()[:6000]
        except Exception as exc:
            info[key] = f"error: {exc}"
    bus = ""
    if isinstance(out.get("v8_bus_ids"), list) and out["v8_bus_ids"]:
        bus = out["v8_bus_ids"][0].lower()
        if len(bus.split(":")[0]) > 4:
            bus = bus[-12:]
    dev = f"/sys/bus/pci/devices/{bus}" if bus else "/nonexistent"
    for key, cmd in (
        ("hostname", "hostname"),
        ("topo", "nvidia-smi topo -m 2>&1 | head -16 | tr -s ' \t' ' ' | tr '\n' ';'"),
        ("clocks", "nvidia-smi -q -i 0 -d CLOCK,PERFORMANCE 2>&1 | tr -s ' ' | tr '\n' ';'"),
        ("mig", "nvidia-smi -q -i 0 2>&1 | grep -A2 -i 'MIG Mode' | tr -s ' ' | tr '\n' ';'"),
        ("pci_path", f"readlink -f {dev}"),
        ("pci_link", f"cd {dev} && echo speed=$(cat current_link_speed) width=$(cat current_link_width) max=$(cat max_link_speed) x$(cat max_link_width) numa=$(cat numa_node); for f in link/*; do echo \"$f=$(cat $f 2>/dev/null)\"; done 2>/dev/null | tr '\n' ' '"),
        ("aspm_policy", "cat /sys/module/pcie_aspm/parameters/policy 2>&1"),
        ("cpuidle", "cat /sys/devices/system/cpu/cpuidle/current_driver /sys/devices/system/cpu/cpuidle/current_governor_ro 2>/dev/null | tr '\n' ' '; for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do echo \"$(cat $s/name):lat=$(cat $s/latency):dis=$(cat $s/disable)\"; done 2>/dev/null | tr '\n' ' '"),
        ("cpufreq", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1 | tr '\n' ' '"),
        ("lspci_tree", "lspci -tv 2>&1 | head -80 | tr '\n' ';'"),
    ):
        try:
            info[key] = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, timeout=30).stdout.strip()[:6000]
        except Exception as exc:
            info[key] = f"error: {exc}"
    out["sysinfo"] = info
    text = json.dumps(out)
    print("DIEPROBE " + text, flush=True)
    with open(os.path.join(out_dir, "die_probe.json"), "w") as handle:
        handle.write(text)
