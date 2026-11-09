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
#include <algorithm>
#include <vector>
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

int main() {
  int n; CK(cudaGetDeviceCount(&n));
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
    out["sysinfo"] = info
    text = json.dumps(out)
    print("DIEPROBE " + text, flush=True)
    with open(os.path.join(out_dir, "die_probe.json"), "w") as handle:
        handle.write(text)
