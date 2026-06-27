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

### NIXL EP — BLOCKED (container toolchain)
The pip `nixl 1.0.1` is the **host RDMA transfer** library (`nixl_agent.register_memory/transfer`),
**not** MoE EP. The real EP lives in the NIXL source repo at `examples/device/ep` (a DeepEP clone) and
requires a from-source **meson** build of the whole NIXL stack. That build **hard-fails on Abseil**:
the container ships `libabsl 20220623` (no `absl_log`) and meson refuses the subproject fallback; also
missing `cuobjclient-13.1` and UCX `-dev` headers (only runtime `libucx0` is present). Unblocking needs
Abseil-from-source + cuobjclient + UCX dev headers — a base-image change, not a benchmark change. The
adapter is writable the moment that build is solved (the API is the DeepEP clone, identical to
`ep_uccl.py`).

### FlashInfer EP / TensorRT-LLM NVLink one-sided AllToAll — BLOCKED on x86_64 (container capability)
FlashInfer is pre-installed and exposes `flashinfer.comm.MoeAlltoAll` and `trtllm_moe_alltoall` (the
TRT-LLM one-sided all-to-all). Both require a **symmetric multi-process MNNVL workspace**. The handle
type is hardcoded by arch:
- **x86_64 (H100/H200/B200):** `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` → needs `pidfd_getfd` →
  **CAP_SYS_PTRACE**, which the enroot/pyxis GHA container does not grant. Without it the cross-rank
  symmetric buffer can't be established, so the all-to-all can't run.
- **aarch64 (GB200/GB300):** `CU_MEM_HANDLE_TYPE_FABRIC` (CUDA fabric handles, no pidfd) — this path
  would work, but GB300 is capacity-limited and GB200 has no validated runner in the fleet.
So FlashInfer EP (and the TRT-LLM one-sided path through it) is a **GB300/GB200 (aarch64 FABRIC)**
candidate, blocked on x86_64 by the missing container capability. Documented rather than forcing a
`--cap-add SYS_PTRACE` launcher change (security-sensitive, and still wouldn't cover NVL72 multi-node).

## Precision matrix

### MXFP8 / MXFP4 / NVFP4 dispatch + combine — BLOCKED (kernel path)
DeepEP (V1 and V2) dispatch accepts **e4m3 fp8 only** (per-token block-128 scales). The micro-scaled /
NVFP4 formats need either FlashInfer's `MoeAlltoAll` (blocked above on x86_64) or a DeepEP fp4 dispatch
extension (does not exist). FlashInfer *has* fp4 quant kernels, but they're reachable only through the
MNNVL-gated EP path. So MX/NVFP4 EP dispatch is gated behind the same FlashInfer-EP blocker.
**Tractable subset (separate task):** direct-cast fp8 + per-token vs per-block scale-layout variants
on the existing DeepEP fp8 path.

### Quantized combine (MXFP8 / NVFP4 / direct-cast / FP32-accum combine) — BLOCKED (no kernel)
No backend wires a **quantized combine** kernel today; every backend's combine is bf16/none. The
capability axes exist (`combine_dtype`, `combine_quant_mode`, default bf16/none) and the schema carries
`shape.quant.*` + `combine_quant_in_timing` so a future run slots in with no schema break. Reserved
until ROCm/MoRI **PR311** (AMD) or a DeepEP quant-combine lands and is shown value-sensitive.

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
- **Framework all-reduce (SGLang quick / vLLM / AITER / FlashInfer one-shot/two-shot), all-gather
  DP-attention→TP-MoE shapes, RL mesh-to-mesh:** in progress as additional suites.
- **KV-cache backends NIXL / MoonCake / MoRI-IO:** declared but not wired (raw memcpy + CPU-pinned are
  wired); MoRI-IO is AMD-only (out of NVIDIA scope).

## Out of scope for "NVIDIA chips"
AMD SDMA copy path, MI355X cross-node EP, MoRI-IO KV backend — these are AMD/MI355X items.
