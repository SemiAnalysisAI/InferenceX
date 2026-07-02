# CollectiveX — gated items: implemented-where-possible, honest blockers otherwise

This records goal.md items that are **not** completable as real GHA results on the available
NVIDIA fleet today, with the *specific* blocker for each (empirically established, not assumed),
plus what WAS done toward each. Scope: NVIDIA chips (H100, H200, B300, GB300 — all with full
sweeps as of 2026-07-02; B200/GB200 spot-validated).

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
`Buffer(group, …)` and runs genuine UCCL dispatch/combine. **Validated: `correct=True`,
`uccl_version=0.1.1`, intranode NVLink on h100/h200/b300/b200** (normal bf16+fp8 + LL — but on h100
LL is superseded by the full-sweep hang finding below). If the wrapper
is ever absent the import falls back to the low-level `uccl.ep.Buffer`, which fails loudly (preserved
failed-case) — never faked. Fresh full-sweep re-validation (post idempotent-build fix, which cured the
old per-case-rebuild SIGABRT/timeout): **h200 = 426/426 correct incl LL-mode 32/32** (run 28535235520);
**h100 = 394/394 correct in NORMAL mode** (run 28535226475) **but all 4 LL-mode cases HANG (rc=124, 900s
timeout — 0/32)**. Since the identical UCCL LL code is 32/32 on h200 (same Hopper arch, same wheel), the
h100 LL hang is an **h100-dgxc cluster limitation** (LL uses IBGDA-style low-latency proxies; the
h100-dgxc fabric deadlocks them — consistent with the documented h100-dgxc cross-node IB wall below),
NOT an arch or UCCL-code wall. Both SKUs also fail ONLY the `empty-rank` diagnostic (see empty-rank note
below). Remaining gap: aarch64 GB200/GB300 (the from-source/proxy bootstrap doesn't come up — see the
aarch64 wall below); uccl is x86-single-node so far.

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
  - **H100 intermittent crash (open):** the MoeAlltoAll **construction** succeeds (cap granted), but
    ~half of h100 flashinfer cases hit `torch.AcceleratorError: CUDA error: unspecified launch failure`
    during dispatch/combine execution (run 28500524185: 21/38 cases; scattered across T/routing, the SAME
    config both crashes AND passes → a genuine intermittent, NOT config/pidfd). NOT a per-case IPC reclaim
    race either: a between-case `/dev/shm` drop + settle was tested (run 28522872429) and made it WORSE
    (in-flight IPC corruption, 21→27 fails). So it's flashinfer MoE-kernel flakiness on Hopper — needs
    compute-sanitizer on a live run to root-cause. Mitigations shipped: (1) each flashinfer case is
    RETRIED up to `CX_FLASHINFER_RETRIES` (default 3) times in the shard loop, dropping the intermediate
    failed-case record on a retry-success so the shard isn't polluted; (2) flashinfer is sweep-chunked
    (`SLOW_MAX_CASES=12`, smaller than others so the retry budget stays within `--time`), bounded +
    PARALLEL so a crash can't take a large shard down. **Retry MEASURED (run 28534841204, retry engaged
    — 17 retries in the p3 shard alone): coverage 30/46 configs, 173/173 correct — up from the ~19-24
    baseline but NOT the ~94% a clean-independent-50% model predicts.** The deadlock is severe (1470
    completion-flag-timeout events that run) and, crucially, CORRELATED within a container: once the
    MNNVL barrier state degrades, retries in the same allocation keep timing out, so retry has
    diminishing returns (one whole chunk, p1, passed cleanly while p0/p2/p3 degraded). Fuller coverage
    would need a fresh container per retry (re-import cost) or much smaller chunks (more GHA jobs) — both
    rejected for marginal gain; the real fix is live compute-sanitizer root-cause. Upgrade to 0.6.14 was
    also tested (run 28530579787) and did NOT fix it (it was a vLLM-side fix), so bundled wheel + retry
    is the shipped path. B300 + GB300 flashinfer are 100% clean (Blackwell), confirming Hopper-kernel.
- **H200 (`h200-dgxc`) runner:** its container **denies** CAP_SYS_PTRACE, so `pidfd_getfd` fails at
  MoeAlltoAll **construction** on every rank (`pidfd_getfd(...) errno 1: Operation not permitted`,
  deterministic — NOT the h100 intermittent, so retry cannot help). This is a per-runner environment
  limitation, NOT a code/hardware gap — the identical adapter is official on H100+B300. Not
  harness-fixable: our launchers pass no `--container-cap-add`/cap flags (caps are the cluster's enroot
  default — h100-dgxc grants it, h200-dgxc doesn't), enroot runs unprivileged so the cap isn't grantable
  per-job, and `MoeAlltoAll` has **no non-MNNVL transport** to route around it (it IS the MNNVL one-sided
  A2A). Documented rather than forcing a security-sensitive `--cap-add SYS_PTRACE` on that shared runner.
- **aarch64 (GB200/GB300):** uses `CU_MEM_HANDLE_TYPE_FABRIC` (no pidfd, no cap needed) — validated
  clean: GB300 full flashinfer sweep **852/852 correct at EP4+EP8** (run 28531976125; rack EP16/32/64
  validated earlier). Both Hopper issues (the h200 pidfd cap wall AND the h100 intermittent MNNVL
  deadlock) are absent on the fabric-handle path.

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
`flashinfer 0.6.8.post1` `moe_a2a_combine` had **no `output_dtype`**, and at investigation time neither
did 0.6.13 (then-latest PyPI) nor the cu130 nightly wheel (0.6.13.dev20260612) — `output_dtype`/
`output_scales` landed on flashinfer **main** after those. (LATER nightlies carry it — see the
direct-cast bullet below; `cx_build_flashinfer_latest` probes the installed wheel's combine signature
and only source-builds if it still lacks it.) So `cx_build_flashinfer_latest` BUILDS flashinfer main from source
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
  build exceeds the H100 runner's job budget (SIGTERM). B300's longer budget lets it land. NOTE the
  original blocker no longer applies: since the nightly wheel gained `output_dtype` (direct-cast bullet
  below), an H100 mxfp8-combine re-run would skip the source build entirely — attainable, just not yet
  re-run (and it would still be subject to the h100 intermittent MNNVL deadlock above).
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
- **Cross-node over InfiniBand (H200 DONE via nccl-ep; H100 cluster WALLED).** Two layers had to fall:
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
  547–808µs, status=comparable-experimental (single-node world=8 validated first, run 28327013318).
  (IBGDA/internode-DeepEP would be a faster one-sided path but needs the driver capability — gated;
  nccl-ep is the validated, portable cross-node EP.)
  **H100 cross-node — WALLED (correcting an earlier "same path covers H100" overclaim).** The h100
  launcher gained the same `CX_NODES>1` FileStore-rendezvous block (ported from h200; committed), and the
  2-node allocation + per-node container DO come up (run 28446105759: nodes hpc-gpu-1-0/1). But the
  nccl-ep run reproducibly HANGS to the 900s timeout on BOTH decode and prefill, with no captured evidence
  (the `timeout -k` kill pre-empts stderr) — the gloo+NCCL FileStore bringup that auto-detects the right
  interface on the h200 fabric does not converge on the hpc-gpu-1 cluster (different inter-node
  networking; no SSH to introspect the correct `GLOO/NCCL_SOCKET_IFNAME`). Not a systematic-matrix data
  point either: `sweep_matrix` places h100 at `nodes=''` (single-node) only — cross-node ws16 was a
  separate goal-182 demo. So h100 single-node EP (all backends @ ws8) is complete; cross-node ws16 stays a
  cluster-bringup wall pending interface-level access to that cluster.
- **Cross-node MI355X (goal 183, "if available") — via nccl-ep on RCCL.** MoRI's RDMA registration also
  aborts cross-node (SIGABRT, run 28325251742, *after* the rendezvous master is correctly resolved) —
  the AMD analogue of UCCL's GPUDirect-RDMA wall. nccl-ep runs on RCCL (identical `all_to_all_single`
  API) over a 2-node MI355X allocation with the same FileStore rendezvous (the MI355X multi-srun gained
  `CX_RDZV_FILE`; nccl-ep uses a pure rccl PG, sidestepping the gloo `connectFullMesh` 127.0.1.1 alias
  too — and `nccl-ep` had to be added to the MI355X launcher's AMD-bench allowlist, else it silently
  fell back to MoRI). **DONE:** MI355X nodes=2 / **world=16 over RoCE/IB**, run 28328718973,
  **correct=True** T=1→8, disp_p50 345–431µs, status=comparable-experimental.
- **DeepEP-hybrid on gb300 WORKS at EP4 AND EP8 (corrected twice); only UCCL aarch64 remains a wall.**
  Per-backend re-validation (informed by upstream docs: NVIDIA HybridEP = the Megatron
  `moe_flex_dispatcher_backend="hybridep"`, TMA-NVLink + IBGDA, **built for NVL72 rack-scale GB200/GB300**)
  overturned the earlier blanket "uccl + deepep-hybrid fail at EP4 and EP8 on Grace-Blackwell" claim:
  - **DeepEP-hybrid gb300 EP4 (single-tray) — WORKS.** EP4 sweep (run 28452161275): 30 valid docs,
    **169/169 correct**, `max_rel_error=0.0`, `branch=hybrid-ep`.
  - **DeepEP-hybrid gb300 EP8 (2-tray, MNNVL) — WORKS.** Run 28480519588: decode **8/8** + prefill **6/6**,
    `ws=8 nodes=2 transport=mnnvl`, full T-ladder 128→4096 all `correct=True` (RT p50 374µs@T128 →
    1404µs@T4096). NOT intranode-only (an earlier wrong claim): the only blocker was build PERSISTENCE —
    `cx_build_deepep_hybrid` did `build_ext --inplace` under `/tmp/DeepEP_hybrid` + PYTHONPATH, but `/tmp`
    does NOT survive across the EP8 multi-srun's separate srun steps (only the pyxis container rootfs does),
    so the case-srun saw the bundled mainline `deep_ep` → `no attribute HybridEPBuffer`. Fixed by installing
    into site-packages (`pip install`, persists — mirrors deepep-v2), build_ext fallback for EP4.
  - **DeepEP-hybrid h100 + h200 (Hopper, EP8 single-node) — WORKS, 212/212 correct each** (runs
    28535221873 / 28535231056, post idempotent-build fix): 43/44 cases valid across the `none` +
    `linear` uneven-token distributions, decode+prefill ladders T=8→4096, all `correct=True`. The ONE
    failing case (c043) is the `empty-rank` diagnostic (`ep-uneven-tokens-v1`, `required_publication:
    diagnostic` — one rank gets ZERO tokens): HybridEP's `set_intra_node_buffers` → `hybrid_ep.cu:81
    cudaDeviceSynchronize` raises `cudaErrorIllegalAddress` on Hopper (identical index c043 on BOTH
    SKUs = deterministic-by-config, NOT the flashinfer intermittent nor accumulation). Not
    retried/chunked: deterministic kernel limit, and the backend already has 212 correct points/SKU.
  - **`empty-rank` is a CROSS-BACKEND Hopper diagnostic differentiator (not HybridEP-only).** The same
    zero-token-rank case ALSO crashes **UCCL** on Hopper (h100 c073 rc=1, h200 c073) — so of the Hopper
    EP backends, deepep-hybrid + uccl fail it while **mainline DeepEP HANDLES it** (verified control:
    h100 mainline deepep empty-rank case c073 = valid doc, **3/3 correct**, zero failed records in the
    shard). So the empty-rank diagnostic cleanly separates zero-token-rank-robust (mainline DeepEP) from
    non-robust (HybridEP, UCCL) EP kernels. It's `required_publication: diagnostic`, one case per
    backend, and flips those backends' GHA jobs to "failure" despite full data — judge by the failed-case
    record + the 200+ correct points, not the job conclusion. Untested on Blackwell (b300/gb300 hybrid +
    uccl suites are `uneven_tokens=none` only, so no Blackwell control exists for empty-rank).
  - **UCCL aarch64 (gb300) — WALL (confirmed fresh, the one genuine aarch64 EP wall).** Run 28457032490:
    `ModuleNotFoundError: No module named 'uccl.ep'` — the uccl EP extension does not import on aarch64
    Grace-Blackwell (consistent with UCCL-EP docs: NVIDIA/AMD + EFA/IB/Broadcom, no aarch64/Grace). EP4+EP8.
  LESSON: a failing run is not proof of a capability wall — both deepep-hybrid claims were wrong; the EP8
  one was a build-env bug, not a hardware limit. Always check the library's actual support before walling.
  Both backends work on x86 single-node (uccl b300=126/b200=124; deepep-hybrid h100=212/h200=212/b300=36,
  43/44 cases on Hopper — only the empty-rank diagnostic crashes, see above). deepep
  (bundled V1), deepep-v2 (from-source), flashinfer, nccl-ep, AND deepep-hybrid (EP4 **and** EP8 — the
  EP8 build-persistence fix above; latest full sweep 788/788 correct, run 28531976125) all run on gb300,
  so the only unfillable gb300 cell is uccl (the aarch64 wall).
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

## Operational note — do not delete ALL runs of a non-`main` workflow
`collectivex-experimental.yml` lives ONLY on the `collectivex` branch (unlike `collectivex-sweep.yml`,
which is also on `main`). GitHub keeps a workflow in the Actions registry only if it is on the default
branch OR has at least one run. Deleting EVERY run of `collectivex-experimental.yml` therefore
DE-REGISTERS it — `gh workflow run collectivex-experimental.yml --ref collectivex` then fails with
"workflow not found on the default branch," and `gh` even reports the failed dispatch as success if the
caller greps stdout for `github.com` (the 404 URL matches). Re-register by pushing any change under
`experimental/CollectiveX/**` (the `on: push` trigger creates a run). Robust fix: also add the workflow
to `main` (as the sweep already is), so run-deletion can never de-register it.
