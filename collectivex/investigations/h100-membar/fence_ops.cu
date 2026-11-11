// fence_ops.cu: per-op cost of fences and release operations (one thread, device-timed with
// %globaltimer), one kernel per op so a profiler can attribute memory-system traffic to each.
#include <cuda_runtime.h>
#include <cstdio>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
__device__ __forceinline__ unsigned long long gt() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define OP(name, body) __global__ void name(int *p, int n, long long *out) { \
  unsigned long long t0 = gt(); for (int i = 0; i < n; ++i) { body; } out[0] = (long long)(gt() - t0); }
OP(op_fence_acq_rel_sys, asm volatile("fence.acq_rel.sys;" ::: "memory"))
OP(op_fence_sc_sys,      asm volatile("fence.sc.sys;" ::: "memory"))
OP(op_st_release_sys,    asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(p), "r"(i) : "memory"))
OP(op_red_release_sys,   asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(p), "r"(1) : "memory"))
OP(op_fence_acq_rel_gpu, asm volatile("fence.acq_rel.gpu;" ::: "memory"))
OP(op_fence_sc_gpu,      asm volatile("fence.sc.gpu;" ::: "memory"))
OP(op_st_release_gpu,    asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(i) : "memory"))
OP(op_fence_acq_rel_cta, asm volatile("fence.acq_rel.cta;" ::: "memory"))
typedef void (*K)(int *, int, long long *);
int main() {
  struct { const char *name; K k; int n; } ops[] = {
    {"op_fence_acq_rel_sys", op_fence_acq_rel_sys, 50000}, {"op_fence_sc_sys", op_fence_sc_sys, 50000},
    {"op_st_release_sys", op_st_release_sys, 50000}, {"op_red_release_sys", op_red_release_sys, 50000},
    {"op_fence_acq_rel_gpu", op_fence_acq_rel_gpu, 200000}, {"op_fence_sc_gpu", op_fence_sc_gpu, 200000},
    {"op_st_release_gpu", op_st_release_gpu, 200000}, {"op_fence_acq_rel_cta", op_fence_acq_rel_cta, 200000}};
  int *p; long long *d, h;
  CK(cudaSetDevice(0)); CK(cudaMalloc(&p, 4096)); CK(cudaMemset(p, 0, 4096)); CK(cudaMalloc(&d, 64));
  for (auto &o : ops) {
    o.k<<<1, 1>>>(p, 1000, d); CK(cudaDeviceSynchronize());  // warm-up
    o.k<<<1, 1>>>(p, o.n, d); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost));
    printf("%-22s n=%-7d %7.1f ns/op\n", o.name, o.n, (double)h / o.n);
  }
  return 0;
}
