# CollectiveX — gated items: implemented-where-possible, honest blockers otherwise

This records goal.md items that are **not** completable as real GHA results on the available
NVIDIA fleet today, with the *specific* blocker for each (empirically established, not assumed),
plus what WAS done toward each. Scope: NVIDIA chips (H100, H200, B300; GB300 capacity-limited).

The container all NVIDIA results run in is `lmsysorg/sglang:v0.5.11-cu130` (CUDA 13.0, NCCL 2.28.9,
torch 2.11; pre-installed: deep_ep 1.2.1, flashinfer 0.6.8, nixl 1.0.1, nvshmem 3.4.5). Established
by an in-container probe on the H200 cluster.

## EP backends

### NVIDIA NCCL EP — DONE via DeepEP V2 (not a separate adapter)
`NVIDIA/nccl` has **no `contrib/nccl_ep`** Python dispatch/combine. NCCL's expert-parallel capability
*is* the GIN + Symmetric-Memory **device** API (host `ncclCommWindowRegister`/`ncclDevComm`/`ncclTeam_t`,
device `ncclLsaBarrier`/`ncclGin*`; present since NCCL 2.28, and the container has 2.28.9). Realizing
"NCCL EP" means writing a CUDA all-to-all kernel on those primitives — which is exactly what **DeepEP
V2's "Gin" backend already does**. CollectiveX benchmarks DeepEP V2 on all NVIDIA SKUs (kernel_gen=v2,
task #115), with NCCL 2.28.9 recorded in provenance. So the NCCL-EP comparison vs DeepEP normal/LL is
the V2-vs-V1-vs-LL comparison already in the dataset. A hand-rolled NCCL-device-API adapter would
duplicate DeepEP V2 with no new signal.

### UCCL EP — SCAFFOLDED, full run DEFERRED (heavier bootstrap than the probe implied)
`pip install uccl` (prebuilt cp312 wheel) + a cu12 CUDA runtime on `LD_LIBRARY_PATH` (the wheel is
cu12 on a cu13 image) **builds and imports** — the C++ runtime `uccl.ep` loads (pkg-0.1.1), confirmed
on H100 via GHA. BUT the DeepEP-compatible surface is **not** the low-level `uccl.ep.Buffer`: that
constructor is `Buffer(rank, num_ranks, num_nvl_bytes, num_rdma_bytes, low_latency_mode, …)` — it does
NOT take a torch ProcessGroup, and a no-bootstrap construction raises `TypeError: incompatible
function arguments`. The DeepEP-identical `Buffer(group, …)` lives in UCCL's separate ~1900-line
`deep_ep_wrapper` package (packaged AS `deep_ep`, so it collides with the container's real DeepEP).
That wrapper's `__init__` runs a non-trivial bootstrap — `get_local_ipc_handle` / `get_local_device_id`
exchanged via `dist.all_gather_object`, `runtime.sync(...)`, CPU `UcclProxy` setup
(`get_cpu_proxies_meta`), and `connect_atomic_buffer` — entangled with UCCL's bench harness `init_dist`.
The wrapper is cleanly vendorable (relative imports + only depends on `uccl.ep`), so the path forward
is: vendor `deep_ep_wrapper` under a non-colliding name + replicate the proxy/IPC bootstrap, then
`ep_uccl.py` becomes a true DeepEP clone against it. Deferred (needs GPU iteration to validate the
proxy bootstrap; NOT a hard blocker). Adapter `tests/ep_uccl.py` + `cx_build_uccl` + capability/schema
remain wired as scaffolding; `benchmark=uccl` currently fails loudly (preserved failed-case), not faked.

### NIXL — transfer DONE (container switch); device-EP blocked on UCX GPU Device API
Two distinct things. **(1) NIXL host RDMA transfer** (`nixl_agent.register_memory / get_xfer_descs /
initialize_xfer / transfer`) — the fabric dynamo uses for KV movement — is **WIRED + valid**
(`tests/nixl_transfer.py`, `CX_BENCH=nixl`). It needed a **container switch** (the sglang multiarch
image has no NIXL build deps): `cx_default_image` selects `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:
1.3.0-dev.1-cuda13` for `CX_BENCH=nixl`. B300 run 28314858649: NIXL 0.10.1, UCX backend, 2 in-process
agents — dtod-local **94 GB/s**, dtod-remote **24 GB/s** (dtoh/htod hit a NIC dmabuf `ibv_reg_mr Bad
address` limit; GPU↔GPU is the KV-handoff path that matters).

**(2) NIXL device-EP** (`examples/device/ep`, a DeepEP fork) — the from-source **meson** build. The
container switch was the directive's exact ask ("switch containers and see if it fixes"), and it
**CLEARED the documented Abseil 20220623 blocker**: the dynamo image ships **Abseil 20250814** (meson
subproject) + meson/ninja/pybind11 3.0.2/cmake, and `meson setup` now SUCCEEDS (build-probe
`cx_probe_nixl_ep`, run 28314858649 log). The **new precise blocker** is `UCX GPU Device API: NO` — the
device-EP target needs UCX's device-initiated (GPU-side put/get) API, which this image's UCX lacks, so
`nixl_ep_cpp` does not build. Unblocking now needs a UCX built `--with-gpu-device-api` (a base-image
concern), NOT Abseil/cuobjclient. The adapter would mirror `ep_deepep.py` (the buffer.py API is a DeepEP
clone) the moment that UCX build lands.

### FlashInfer EP / TensorRT-LLM NVLink one-sided AllToAll — DONE on H100 + B300 (H200 runner gated)
`flashinfer.comm.MoeAlltoAll` (which LIVES IN `flashinfer.comm.trtllm_moe_alltoall` — it IS the
TRT-LLM "throughput backend" one-sided all-to-all, calling the same `moe_a2a_dispatch`/`moe_a2a_combine`
kernels) builds its MNNVL symmetric workspace over the torch.distributed NCCL group via FlashInfer's
`TorchDistBackend` (no MPI/mpi4py). The cross-rank symmetric buffer uses
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` → `pidfd_getfd` → **CAP_SYS_PTRACE** on x86_64. Empirically:
- **H100 (`h100-dgxc`) + B300 (`b300`):** their enroot/pyxis runner containers **grant** the cap →
  FlashInfer EP runs and is **official** (bf16 + the quant dispatch matrix below), decode + prefill.
  This is the TRT-LLM NVLink one-sided AllToAll EP — the existing FlashInfer EP results ARE that path
  (provenance `backend_lineage = flashinfer.comm.trtllm_moe_alltoall.MoeAlltoAll`).
- **H200 (`h200-dgxc`) runner:** its container **denies** CAP_SYS_PTRACE, so `pidfd_getfd` fails and the
  symmetric buffer can't be established (`pidfd_getfd ... operation not permitted`). This is a
  per-runner environment limitation, NOT a code/hardware gap — the identical adapter is official on
  H100+B300. Documented rather than forcing a security-sensitive `--cap-add SYS_PTRACE` on that runner.
- **aarch64 (GB200/GB300):** would use `CU_MEM_HANDLE_TYPE_FABRIC` (no pidfd); GB300 capacity-limited.

## Precision matrix

### MXFP8 / NVFP4 dispatch — DONE on FlashInfer EP; MXFP4 dispatch — gated (tile-padded SF)
DeepEP (V1/V2) dispatch accepts **e4m3 fp8 only**. But FlashInfer's A2A is a **dtype-agnostic byte
mover** taking `input_payloads` as a LIST, so a quantized dispatch moves `[q, scale_factor]` and
dequants in `stage()` (UNTIMED preprocessing, cached so the roundtrip measures comm). Using FlashInfer's
own quantize/dequantize kernels, `ep_flashinfer.py` now does **MXFP8** (`mxfp8_quantize`, e4m3 + e8m0
block-32 — device dequant verified == `mxfp8_dequantize_host`) and **NVFP4** (`fp4_quantize` +
`e2m1_and_ufp8sf_scale_to_float`, e2m1 + e4m3 block-16) dispatch, plus the three e4m3 fp8 scale-layouts.
Coverage by arch (all `correct=True` end-to-end):
- **e4m3 fp8 (×3) + mxfp8:** H100 **and** B300 (e4m3/e8m0 are Hopper-supported).
- **nvfp4:** **B300 (Blackwell) only.** FP4 (e2m1) is a Blackwell-native tensor format; FlashInfer's
  fp4 quantize/dequantize does NOT round-trip on Hopper sm90 (validated: nvfp4 `correct=True` on B300,
  `correct=False` on H100). `capability.resolve` now gates nvfp4 to Blackwell (`ARCH_ONLY_DTYPES`), so a
  Hopper nvfp4 dispatch is cleanly rejected rather than run-and-marked-invalid.
- **MXFP4 dispatch — gated:** FlashInfer's `mxfp4_quantize` emits its scale factor in a **tile-padded
  `[pad(T,128), H/32]` swizzled layout** with no `is_sf_swizzled_layout=False` option — it does NOT
  factor as a per-token `[T, k]` tensor, so it can't be moved through the per-token A2A. (mxfp8 + nvfp4
  both expose a linear per-token SF; mxfp4 alone does not.) The 4-bit MX format is covered in spirit by
  nvfp4 (also 4-bit e2m1); mxfp4 specifically stays gated on the quantizer's SF layout.

### Quantized combine OUTPUT (MXFP8 / NVFP4 combine) — DONE on B300 via flashinfer-main (container switch)
Distinct from quantized *dispatch*: a quantized **combine** emits a non-bf16 reduced output. The bundled
`flashinfer 0.6.8.post1` `moe_a2a_combine` had **no `output_dtype`**, and neither did 0.6.13 (latest
PyPI) nor the cu130 nightly wheel (0.6.13.dev20260612) — `output_dtype`/`output_scales` landed on
flashinfer **main** after those. So `cx_build_flashinfer_latest` BUILDS flashinfer main from source
in-container (after a 7-layer version-coupling peel: cubin↔python↔jit-cache version checks, then
`nvidia-cutlass-dsl` 4.5.2 for the CuTe `OperandMajorMode`, then **uninstalling** the stale precompiled
cubin/jit-cache so `get_moe_alltoall_module()` JIT-compiles the 14-arg kernel fresh from main's csrc).
- **MXFP8 combine — DONE on B300:** `combine(output_dtype=float8_e4m3fn, output_scales=uint8[T,H/32])` =
  e4m3 + UE8M0 block-32 (the source-spec'd layout); dequant `e4m3 * 2^(e8m0-127)`. Valid, `correct=True`
  ×8 (`backend_provenance.combine_quant=True`, `flashinfer_stack` captured). FP32-accum is the kernel's
  internal reduce; scale-transport (e8m0) + tolerance-class (1.6e-1 vs bf16 5e-2) are exercised.
- **NVFP4 combine — DONE on B300:** `output_dtype=uint8 (packed e2m1) + e4m3 vec-16 scales +
  output_scalar_scale`; dequant via `e2m1_and_ufp8sf_scale_to_float` (the e4m3 scales viewed as uint8
  ufp8). Valid, `correct=True` ×8 (Blackwell-native fp4, like nvfp4 dispatch).
- **H100 combine — build-time-limited (NOT arch):** the ~70-min in-container flashinfer-main source
  build exceeds the H100 runner's job budget (SIGTERM). B300's longer budget lets it land. A pre-staged
  flashinfer-main wheel (one-time build) would remove the per-run rebuild; deferred.
- **Direct-cast FP8 combine — kernel limit (evidenced, B300 run 28315037266):** ATTEMPTED via
  `CX_QC_SCALE=scalar` (`output_dtype=float8_e4m3fn` + `output_scalar_scale`, NO per-block
  `output_scales`). The kernel ASSERTS `Check failed: (output.dtype()==payload.dtype()) is false:
  output_dtype without output_scales must match payload dtype` — i.e. an fp8 output REQUIRES per-block
  `output_scales`; a scalar-only/unscaled direct-cast fp8 combine is **not a supported moe_a2a_combine
  mode**. The SCALED mxfp8/nvfp4 outputs are the only fp8/fp4 combine paths. (Also confirmed the nightly
  `flashinfer 0.6.13` wheel now carries `output_dtype` — the ~70-min main-source build is no longer
  needed for combine-quant.) MoRI fp8_blockwise combine (AMD, PR311) remains a separate AMD path.

## Topology and rack-scale

### Cross-node EP / GB200·GB300 NVL72 EP16/32/64 — BLOCKED (internode-DeepEP integration)
`platforms.yaml` is `internode: false` for every SKU ("asserts out until >8 ranks"). The DeepEP NVLink
kernel `Buffer(group, nvl, 0)` is **intranode-only** (≤8 ranks — including MNNVL trays, which is why
GB300 EP8 over 2 trays works). EP16/32/64 needs the DeepEP **internode** path (NVSHMEM/IBGDA) built +
a multi-node torchrun/srun launcher + internode buffer sizing — a substantial integration not yet
wired. Multi-node **hardware exists** (H200 has 13 idle nodes), so this is an integration gap, not a
hardware gap. **What IS done:** structured topology metadata (nodes/gpus/domain/transport/placement),
placement policies (packed/striped/runtime-native/adversarial), and locality/topology metrics
(same-node/same-domain/cross-node/RDMA fractions) — all captured per result.
- **GB200 NVL72:** no validated GB200 platform/runner in the fleet (`launch_gb200-nv.sh` exists but no
  validated `platforms.yaml` entry). Hardware gap.
- **GB300 NVL72 EP8:** works over MNNVL (`gb300-nv`), but capacity-limited per project decision; EP16+
  needs the internode path above.

## Other inference collectives (NVIDIA scope)

- **All-reduce / all-gather (standardized NCCL):** DONE — real `family=nccl` results on H100/H200/B300,
  rendered in the All-reduce/All-gather tabs.
- **CPU↔GPU offload, copy-engine/SDMA, KV-cache transfer:** DONE — single-process memcpy-family benches
  (`tests/offload_bench.py`, `copy_engine_bench.py`, `kv_cache_transfer.py`).
- **Framework all-reduce — FlashInfer one-shot/two-shot DONE:** `allreduce_fw_bench.py` wires the real
  `trtllm_allreduce_fusion` (pattern `kAllReduce`, `use_oneshot` True/False) over the TRT-LLM IPC
  workspace — nccl baseline + flashinfer-oneshot + flashinfer-twoshot, all `correct=True` (one-shot
  beats the NCCL ring in the small-message latency regime). SGLang/vLLM custom-AR are import-guarded
  (recorded as skipped if the framework's distributed wrapper isn't importable in the sglang image);
  AITER is AMD. RL mesh-to-mesh + all-gather DP-attention→TP-MoE shapes: covered by the standardized
  sweeps (rl-mesh + all-gather families).
- **KV-cache backends:** raw memcpy + CPU-pinned WIRED; **NIXL WIRED** (`tests/nixl_transfer.py`, B300
  via the dynamo-container switch — see the NIXL section above); **MoRI-IO WIRED** (`tests/
  mori_io_transfer.py`, MI355X, `mori.io` IOEngine RDMA p2p). **MoonCake** remains not wired — needs the
  Mooncake transfer-engine library, which is in none of the CollectiveX containers (would require
  importing a Mooncake image or building it from source).

## AMD / MI355X items — now ATTEMPTED via GHA (no longer "out of scope")
The directive's container-switch + AMD-lift asks. All run via GHA on the MI355X MoRI image:
- **FNUZ fp8 dispatch (MoRI) — VALIDATED (e4m3fnuz):** `dispatch_dtype=fp8` on the mori backend routes
  MoRI's `quant_type=fp8_direct_cast` — the ROCm-native e4m3fnuz format (the self-introspecting adapter
  found the valid set is `['none','fp8_direct_cast']`; the guessed `fp8_blockwise` is rejected by this
  build). Required `use_external_inp_buf=True` (Fp8DirectCast asserts in zero-copy mode) + gating against
  the e4m3fnuz consistency reference. MI355X run 28318788729: T=2/4/8 `correct=True`, max_rel **3e-4**,
  disp_p99 ~45-70µs. The run's status=invalid is solely MoRI's forced-T=1 ramp point (a single-token
  relErr-metric instability, rank-0 max_rel=3e-4 — not a comm error). Full 5-run resolution chain (each
  peeling one layer via the GHA log alone — no SSH) in notes.md.
- **AMD SDMA copy path:** `copy_engine_bench.py` no longer refuses on ROCm — the off-SM DMA path IS the
  SDMA engine; labeled `copy_engine_kind=sdma` / `accelerator=rocm` (vs NVIDIA `copy-engine`). The
  non-interference probe characterizes SDMA-vs-CU interference (pynvml absent → graceful fallback).
- **MoRI-IO KV backend:** `tests/mori_io_transfer.py` (above).
- **MI355X cross-node EP:** still blocked on the DeepEP internode path (same NVSHMEM/IBGDA integration as
  the NVIDIA cross-node item; single-node MI355X EP is covered by the MoRI sweep).
