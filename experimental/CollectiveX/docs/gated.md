# CollectiveX — gated items: implemented-where-possible, honest blockers otherwise

This records goal.md items that are **not** completable as real GHA results on the available
NVIDIA fleet today, with the *specific* blocker for each (empirically established, not assumed),
plus what WAS done toward each. Scope: NVIDIA chips (H100, H200, B300; GB300 capacity-limited).

The container all NVIDIA results run in is `lmsysorg/sglang:v0.5.11-cu130` (CUDA 13.0, NCCL 2.28.9,
torch 2.11; pre-installed: deep_ep 1.2.1, flashinfer 0.6.8, nixl 1.0.1, nvshmem 3.4.5). Established
by an in-container probe on the H200 cluster.

## EP backends

### NVIDIA NCCL EP — NOT represented by DeepEP V2; needs its own adapter
Upstream `NVIDIA/nccl` now has a real `contrib/nccl_ep` implementation. It is an NCCL API extension for
MoE dispatch/combine built on NCCL Device API LSA/GIN, and should be treated as its own backend surface,
not as a synonym for DeepEP V2.

CollectiveX currently keeps these surfaces separate:
- **DeepEP V2**: `backend=deepep`, `shape.kernel_gen=v2`, `deepep_version=2.0.0+...`; this is DeepEP's
  ElasticBuffer/dispatch/combine implementation using the NCCL Gin backend.
- **`nccl-ep` baseline in this harness**: a portable token-shuffle implementation using
  `torch.distributed.all_to_all_single` over NCCL/RCCL. This is useful as a host-orchestrated baseline,
  especially cross-node, but it is **not** upstream `contrib/nccl_ep`.
- **Upstream NCCL EP**: still needs a dedicated adapter/provenance label before CollectiveX can claim
  native NCCL EP results. When wired, it must not overwrite either DeepEP V2 or the current
  all-to-all baseline identity.

So the correct comparison is not "NCCL EP = DeepEP V2". DeepEP V2 remains a relevant NCCL-Gin-backed
comparison point, but native NCCL EP needs its own line in the backend/version matrix.

### UCCL EP — DONE via vendored deep_ep_wrapper (was deferred; the bootstrap is now wired)
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
The wrapper is cleanly vendorable (relative imports + only depends on `uccl.ep`), and that is now
DONE: `cx_build_uccl` git-clones `uccl-project/uccl` at the wheel-matched tag and vendors
`deep_ep_wrapper` under the non-colliding name `uccl_deepep`; `ep_uccl.py` imports its
`Buffer(group, …)` and runs genuine UCCL dispatch/combine. **Validated: 507 valid docs, `correct=True`,
`uccl_version=0.1.1`, intranode NVLink on h100/h200/b300/b200** (normal bf16+fp8 + LL). If the wrapper
is ever absent the import falls back to the low-level `uccl.ep.Buffer`, which fails loudly (preserved
failed-case) — never faked. Remaining gap: aarch64 GB200/GB300 (the from-source/proxy bootstrap doesn't
come up there — see the aarch64 wall below); uccl is x86-single-node so far.

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
`cx_probe_nixl_ep`, run 28314858649 log). The next blocker is `UCX GPU Device API: NO` (the device-EP
needs UCX's device-initiated GPU put/get API via `<ucp/api/device/ucp_device_impl.h>`). **Build attempt
made:** `cx_probe_nixl_ep` now BUILDS UCX from source with `--with-cuda` and points pkg-config at it —
but `meson setup` STILL reports `UCX GPU Device API : NO` (run 28320702204). So it is NOT a missing
build flag: UCX's device API compiles in only with GPUDirect-Async / device-initiated-comm **driver +
hardware** support (IBGDA/GDAKI), a base-platform capability absent here — not a container/build fix.
`nixl_ep_cpp` therefore does not build; the adapter (mirroring `ep_deepep.py`) waits on a platform with
that device-comm support. Evidenced terminal wall.

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

### NVL72 rack-scale EP — DONE up to EP64 via FlashInfer-MNNVL; cross-node-over-IB DONE via nccl-ep
**Within an NVL72 NVLink domain, EP8/16/32/64 are DONE.** The key: DeepEP's NVLink `Buffer(group,nvl,0)`
is intranode-only (≤8 ranks, incl. MNNVL trays → GB300/GB200 EP8 over 2 trays via deepep), BUT
**FlashInfer's MoeAlltoAll MNNVL symmetric workspace SPANS the whole NVL72 NVLink domain** — so
`benchmark=flashinfer nodes=4/8/16` runs EP16/32/64 across 4/8/16 trays. Validated correct=True:
GB300 EP8 (28319504164) + EP16 (28319809968); GB200 EP8 (28319793439, after porting the GB300 EP
multi-srun path into launch_gb200-nv.sh — was nccl-only) + EP16 (28319971335) + EP64 (28319975631,
ep_size=64/world=64). EP32 (both SKUs) re-dispatched after a workflow concurrency-group collision
(the group omitted inputs.nodes — fixed). Bounded only by NVL72 tray CAPACITY, not the method.
- **Cross-node over InfiniBand (H100/H200, goal 182) — DONE via nccl-ep.** Two layers had to fall:
  (1) **Rendezvous:** torch's `env://` TCPStore *and* torchrun's elastic-agent store advertise the
  rank-0 management-subnet NodeAddr, which is NOT reachable from a peer rank's enroot container net
  namespace (900s connect timeout; runs 28325250919 / 28326334616). Solved with a shared-mount
  **FileStore** (`CX_RDZV_FILE`) + a **local NGPUS-process spawn** (no torchrun elastic agent) — the PG
  bootstraps through the shared file and NCCL then connects peers over IB. (2) **Data path:** the custom
  one-sided RDMA backends do NOT survive cross-node — UCCL's `ibv_reg_mr` fails EINVAL → `free():
  corrupted unsorted chunks` → SIGSEGV (run 28326528672, *after* the rendezvous now forms), DeepEP
  normal-internode asserts out — because they need GPUDirect-RDMA peer-memory registration the cluster's
  IB HCAs / container don't expose. The portable fix is a transport that host-stages gracefully:
  **nccl-ep** (`tests/ep_nccl.py`), the NCCL `all_to_all_single` token-shuffle EP baseline. H200
  nodes=2 / **world=16 over IB**, run 28327088942: **correct=True at every T(1→128)**, disp_p50
  547–808µs, status=comparable-experimental (single-node world=8 validated first, run 28327013318). The
  same nccl-ep path covers H100. (IBGDA/internode-DeepEP would be a faster one-sided path but needs the
  driver capability — gated; nccl-ep is the validated, portable cross-node EP.)
- **Cross-node MI355X (goal 183, "if available") — via nccl-ep on RCCL.** MoRI's RDMA registration also
  aborts cross-node (SIGABRT, run 28325251742, *after* the rendezvous master is correctly resolved) —
  the AMD analogue of UCCL's GPUDirect-RDMA wall. nccl-ep runs on RCCL (identical `all_to_all_single`
  API) over a 2-node MI355X allocation with the same FileStore rendezvous (the MI355X multi-srun gained
  `CX_RDZV_FILE`; nccl-ep uses a pure rccl PG, sidestepping the gloo `connectFullMesh` 127.0.1.1 alias
  too — and `nccl-ep` had to be added to the MI355X launcher's AMD-bench allowlist, else it silently
  fell back to MoRI). **DONE:** MI355X nodes=2 / **world=16 over RoCE/IB**, run 28328718973,
  **correct=True** T=1→8, disp_p50 345–431µs, status=comparable-experimental.
- **UCCL + DeepEP-hybrid on aarch64 GB200/GB300 — WALL (backend-specific, not the launcher).** The
  combined `backend=all` sweep confirmed these two fail ENTIRELY on the Grace-Blackwell SKUs: 0 valid
  docs at BOTH EP4 (single-tray) and EP8 (2-tray MNNVL) — uccl gb200 5/5 EP4 + 6/6 EP8 failed; deepep-
  hybrid gb200/gb300 same. This is NOT the rack launcher (the positive control is decisive: on the SAME
  gb200/gb300 clusters, **flashinfer lands 104/68 rack EP8 docs, nccl-ep 98/16, deepep (bundled V1) 175/174**),
  and NOT cross-node (it's intra-NVL72). Both backends work on x86 single-node (uccl b300=126/b200=124
  valid; deepep-hybrid h100=84/b300=36). Cause: their FROM-SOURCE in-container builds were probe-confirmed
  on x86 B300 only — uccl's `ibv`/proxy RDMA bootstrap and deepep-hybrid's TMA+NVSHMEM build don't come up
  on aarch64 Grace-Blackwell. deepep (bundled V1), flashinfer (bundled), and the nccl-ep
  `all_to_all_single` baseline all run there, so rack-scale coverage is complete via those three
  surfaces. Native upstream NCCL EP remains separate until a real `contrib/nccl_ep` adapter lands.
- **DeepEP V2 (from-source `kernel_gen=v2`): DONE on x86 + aarch64, EP4 AND rack EP8.** Genuine V2
  (`deepep_version=2.0.0+af9a040`) builds on h100/h200/b300/b200 AND on aarch64 Grace-Blackwell — gb300
  EP4 (run 28429220764) produced `kernel_gen=v2`/`2.0.0`, log "built deep_ep 2.0.0 … V2 ready". So aarch64
  V2 is NOT a wall: wherever the EP4/single-node path runs (it calls `cx_build_deepep_v2` once in
  `run_in_container`), V2 builds and runs. **Rack EP8 (gb200/gb300, 2 trays) — now DONE too**, after two
  fixes the earlier "deferred" note anticipated only the first of: (1) the EP8 multi-srun launcher ran
  `run_ep.py` over 8 ephemeral per-rank containers, BYPASSING `cx_build_deepep_v2` (so `deepep_v2=true`
  silently ran bundled V1 and the doc `kernel_gen` was honestly `v1`). Fixed with `CX_BUILD_ONLY` +
  a setup-srun that builds V2 ONCE PER NODE into a persistent `--container-name` every case-srun reuses.
  (2) With V2 actually installed, EP8 then crashed `cudaErrorIllegalAddress` at `csrc/legacy/buffer.hpp`
  across trays — NOT a hardware wall (bundled V1 runs 180 correct cross-tray EP8 docs, `ws8/nodes2/mnnvl`).
  Upstream V2's `Buffer` ADDED `allow_mnnvl` (default **False**); when off, DeepEP itself sets
  `NVSHMEM_DISABLE_MNNVL=1` and the legacy buffer falls onto the intranode-only CUDA-IPC peer path, which
  faults across NVL72 trays. `tests/ep_deepep.py` now passes `allow_mnnvl=True` on both Buffer ctors when
  `CX_ALLOW_MNNVL=1` (gated on `inspect` finding the param, so bundled-V1 + x86 single-node are unchanged);
  the gb300 launcher exports it for the deepep EP8 case. **Validated:** gb300 EP8 run 28434764062 →
  `kernel_gen=v2 / ws8 / nodes2 / transport=mnnvl / allow_mnnvl=True / mode=normal / correct=8/8`, roundtrip
  p50 158→227µs (T=8→1024). `sweep_matrix` re-enables v2 at gb200/gb300 EP8. (gb200 launcher inherits the
  same build-once + `CX_ALLOW_MNNVL` fix; pending a gb200 allocation to re-confirm.)

## Other inference collectives (NVIDIA scope)

- **All-reduce / all-gather (standardized NCCL):** DONE — real `family=nccl` results on H100/H200/B300,
  rendered in the All-reduce/All-gather tabs.
- **CPU↔GPU offload, copy-engine/SDMA, KV-cache transfer:** DONE — single-process memcpy-family benches
  (`tests/offload_bench.py`, `copy_engine_bench.py`, `kv_cache_transfer.py`).
- **Framework all-reduce — FlashInfer one-shot/two-shot DONE:** `allreduce_fw_bench.py` wires the real
  `trtllm_allreduce_fusion` (pattern `kAllReduce`, `use_oneshot` True/False) over the TRT-LLM IPC
  workspace — nccl baseline + flashinfer-oneshot + flashinfer-twoshot, all `correct=True` (one-shot
  beats the NCCL ring in the small-message latency regime). **SGLang/vLLM/AITER custom-AR — now DONE**
  by REPLICATING the framework's serving distributed-init (init_distributed_environment +
  initialize_model_parallel) on the torchrun group and using the TP GroupCoordinator's
  ca_comm.custom_all_reduce (the wrapper builds ca_comm only inside that init — a bare ctor skipped):
  sglang H200 175 GB/s correct=True (run 28320404895); AITER MI355X 367.8 GB/s correct=True (run
  28320579741, aiter.dist.parallel_state, ca_comm under device_communicator); vLLM via the
  allreduce-fw-vllm CONTAINER SWITCH to vllm/vllm-openai + entering set_current_vllm_config(VllmConfig())
  (its CustomAllreduce is a CustomOp asserting an active config), H200 correct=True (run 28320699661).
  RL mesh-to-mesh + all-gather DP-attention→TP-MoE shapes: covered by the standardized sweeps.
- **KV-cache backends:** raw memcpy + CPU-pinned WIRED; **NIXL WIRED** (`tests/nixl_transfer.py`, B300
  via the dynamo-container switch — see the NIXL section above); **MoRI-IO WIRED** (`tests/
  mori_io_transfer.py`, MI355X, `mori.io` IOEngine RDMA p2p). **MoonCake WIRED on NVIDIA** (`tests/
  mooncake_transfer.py`, run_mooncake_suite pip-installs the engine; B300 35.4 GB/s via
  `transfer_write_on_cuda`). **MoonCake on MI355X = ROCm wall (evidenced):** the engine initializes on
  ROCm (`MOONCAKE_INIT … on rdma device rdma0`) but the pip wheel exposes NO `transfer_write_on_hip`
  method (only the CUDA one) — `0 groups, status=invalid`, run 28342781762. A HIP transfer path would
  need an upstream Mooncake ROCm build, not a container/flag fix.

- **MI355X primitives (rccl-tests) tab:** the All-reduce/All-gather tabs render `family=nccl`; the AMD
  equivalent is `rccl` (`CX_BENCH=nccl` → rccl-tests on the MI355X launcher). Repeated dispatches
  (28340951946, 28342780904) failed in the runner *checkout/setup* step (exit 2/3, `EACCES` on a shared
  `LOGS/agentic` dir + missing workspace) — the MI355X GHA runners are shared with the agentic
  benchmark fleet, so the CollectiveX checkout collides intermittently. This is a runner-contention
  infra flake, NOT an rccl-tests limitation; it lands when it gets a clean runner.

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
- **MI355X cross-node EP (goal 183):** the custom-RDMA MoRI path aborts cross-node (SIGABRT, GPUDirect-
  RDMA wall) — same class as UCCL on NVIDIA — so cross-node MI355X EP runs via **nccl-ep on RCCL**
  (NCCL/RCCL `all_to_all_single`, host-staged over IB) with the shared-mount FileStore rendezvous. See
  the rack-scale section above; single-node MI355X EP is covered by the MoRI sweep.
