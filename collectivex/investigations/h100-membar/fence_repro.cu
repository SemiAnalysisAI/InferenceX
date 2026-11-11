// fence_repro.cu: cost of a system-scope memory fence (fence.acq_rel.sys) on NVIDIA GPUs.
//
// Build:  nvcc -O2 -arch=sm_90 -o fence_repro fence_repro.cu      (H100/H200; sm_100 for B200)
// Run:    ./fence_repro            one JSON line of per-op costs on GPU 0, device-timed
//         ./fence_repro spin 8     issue fence.acq_rel.sys back to back on GPU 0 for ~8 s
//                                  (watch `nvidia-smi dmon -i 0 -s t` meanwhile)
//
// Every timing brackets a single-thread loop with %clock64 and %globaltimer, so it excludes launch
// cost and reports the SM clock it ran at. The loop itself adds ~1 ns per iteration.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ int ld_sys(const int *p) {
  int v; asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_sys(int *p, int v) {
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// OP 0: fence.acq_rel.sys   1: fence.acq_rel.gpu   2: dependent load from p (pinned host memory)
// OP 3: store to p (pinned host memory), then fence.acq_rel.sys
template <int OP> __global__ void timed(int *p, int n, long long *out) {
  int acc = 0;
  long long c0 = clock64(); unsigned long long t0 = gtimer();
  for (int i = 0; i < n; ++i) {
    if (OP == 0) fence_sys();
    else if (OP == 1) fence_gpu();
    else if (OP == 2) acc += ld_sys(p + (acc & 1));
    else { st_sys(p, i); fence_sys(); }
  }
  long long c1 = clock64(); unsigned long long t1 = gtimer();
  out[0] = c1 - c0; out[1] = (long long)(t1 - t0); out[2] = acc;
}

// One thread on another SM stores to pinned host memory every gap_ns, for dur_ns.
__global__ void host_writer(int *p, long long dur_ns, long long gap_ns) {
  unsigned long long t0 = gtimer(), t = t0, next = t0; int n = 0;
  while ((long long)(t - t0) < dur_ns) {
    if (t >= next) { st_sys(p + (n & 63) * 32, n); ++n; next = t + gap_ns; }
    t = gtimer();
  }
}

// ns per op (and the SM clock in MHz during the loop) for n iterations of OP on stream s.
template <int OP> static double run(int *p, int n, long long *d, double *mhz = nullptr, cudaStream_t s = 0) {
  timed<OP><<<1, 1, 0, s>>>(p, n, d); CK(cudaGetLastError()); CK(cudaStreamSynchronize(s));
  long long h[3]; CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  if (mhz) *mhz = 1000.0 * (double)h[0] / (double)h[1];
  return (double)h[1] / n;
}

int main(int argc, char **argv) {
  CK(cudaSetDevice(0));
  int *hp, *hd, *dev; long long *d;
  CK(cudaHostAlloc((void **)&hp, 1 << 16, cudaHostAllocMapped)); memset(hp, 0, 1 << 16);
  CK(cudaHostGetDevicePointer((void **)&hd, hp, 0));
  CK(cudaMalloc(&dev, 1 << 16)); CK(cudaMemset(dev, 0, 1 << 16)); CK(cudaMalloc(&d, 64));

  if (argc > 2 && !strcmp(argv[1], "spin")) {
    double secs = atof(argv[2]), ns = run<0>(dev, 100000, d);
    int n = (int)(0.5e9 / ns);  // about half a second per launch
    auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(secs);
    long long launches = 0;
    while (std::chrono::steady_clock::now() < end) { run<0>(dev, n, d); ++launches; }
    printf("{\"spin_seconds\": %.1f, \"fence_sys_ns\": %.1f, \"fences\": %lld}\n", secs, ns, launches * (long long)n);
    return 0;
  }

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  double mhz, sys = run<0>(dev, 200000, d, &mhz);
  double gpu = run<1>(dev, 1000000, d);
  double host_load = run<2>(hd, 50000, d);
  double store_fence = run<3>(hd, 50000, d);
  // fence.acq_rel.sys while one thread on another SM writes host memory once per microsecond
  cudaStream_t sa, sb;
  CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
  host_writer<<<1, 1, 0, sb>>>(hd + 4096, 400000000LL, 1000); CK(cudaGetLastError());
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  double sys_writer = run<0>(dev, 50000, d, nullptr, sa);
  CK(cudaStreamSynchronize(sb));
  printf("{\"gpu\": \"%s\", \"sm_mhz\": %.0f, \"fence_sys_ns\": %.1f, \"fence_gpu_ns\": %.1f, "
         "\"host_load_ns\": %.1f, \"host_store_then_fence_sys_ns\": %.1f, \"fence_sys_with_host_writer_ns\": %.1f}\n",
         prop.name, mhz, sys, gpu, host_load, store_fence, sys_writer);
  return 0;
}
