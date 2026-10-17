# CollectiveX EP Benchmark Methodology

CollectiveX runs expert-parallel (EP) communication benchmarks by hand on real accelerator
allocations and writes one result document per case into `collectivex/results/`. Committing those
files is the publish step. It does **not** rank, recommend, select, or decide what any consumer
displays. The frontend reads `results/cases.json` and the committed result documents and makes its
own coverage and display decisions. This document describes how a case is scheduled, measured,
checked, and recorded, not a publication or qualification contract.
[methodology-summary.md](methodology-summary.md) states the same rules in brief; this document keeps
the evidence behind each one. Where either disagrees with the code in `collectivex/bench/`, the code
is authoritative.

## Product Boundary

CollectiveX is a communication microbenchmark for:

- comparing EP libraries on one chip/topology.
- comparing EP latency and logical payload bandwidth across systems under the same workload.
- surfacing unsupported, failed, invalid, and unstable cases rather than hiding them.

It does not predict serving throughput without a separate correlation study.

## Matrix

The implemented workload is `deepseek-v4-pro` (DeepSeek-V4-Pro 1.6T, the `dsv4` model the rest of
InferenceX benchmarks): hidden 7168, top-k 6, 384 routed experts, packed placement, uniform routing,
and one pinned resource configuration per backend/topology, the one its reference engine runs. Its
token ladders live with the workload in `configs/sweep.json`: decode T=1..512 and prefill
T=1024..8192 tokens per rank, powers of two. `normal` mode runs both ladders with
`layout-and-dispatch-v1` semantics; `low-latency` runs each backend's decode-kernel semantics on the
decode ladder only. Its runnable set is narrower than and distinct from `normal`'s, so it is enabled
cell by cell from the registry's `ll_backends` map rather than assumed wherever `normal` runs.

`sweep_matrix.py` materializes the requested SKUs, backends, EP sizes, modes, precisions and ladders
into shards of one allocation each. A filter produces a smaller matrix, not a different contract.
`results/cases.json` is the committed snapshot of the full matrix, and a unit test fails when it no
longer matches the configs. A cell whose backend runs on that SKU at another EP size is classified
`unsupported` rather than silently skipped. Low-latency adds only runnable cells, so a low-latency
wall is recorded in prose under [Backends](#backends) rather than as a matrix row. Whether a cell is
attempted is a capability fact; whether it succeeded is decided only by the benchmark's return code.

### Ladder Clamps

A backend may clamp the ladder, and every clamped point is reported rather than dropped silently:
`workload.ladder_measured`, `ladder_dropped` and `ladder_cap` record what ran and what did not.
DeepEP V2, UCCL-EP and MoRI pre-allocate their low-latency receive at 256 tokens per rank (vLLM's
default batched-DP budget for the DeepEP-API backends; for MoRI, the harness's shared low-latency
cap), so their low-latency ladders drop the 512 rung; NCCL EP sizes its group
to the ladder's top and runs every rung. DeepEP V2's receive is sized from a constant rather than
from the ladder, so clamping the ladder around a kernel defect cannot change the footprint, which
sets both the transport's memory traffic and the FP8 dequant volume. That lever was used once:
DeepEP's low-latency combine corrupted the 256 rung on every Blackwell SKU (B200, GB200 and GB300,
EP8 and EP16, both precisions, MNNVL and RDMA alike) while Hopper stayed clean, stochastically at
roughly 1.5-3.3% per invocation, as one wrong token row whose norm still matched to 4 significant
figures. The ladder was clamped to 128 until the pin moved to upstream main, whose #642 retires the
combine consumer's shared-memory reads with a CTA-scope fence before its staging buffer is recycled.

### Precision

Combine is always BF16. Dispatch precision is a swept dimension: a BF16 control and, on the backends
whose FP8 dispatch is supported upstream (DeepEP V2, MoRI, UCCL-EP, FlashInfer EP), an FP8 dispatch.
`sweep_matrix.py`'s `BACKEND_PRECISIONS` holds each backend's set, and a backend never emits a case
for a precision it does not support. Every FP8 codec here is per-128-block e4m3 with FP32 scales:
7168 value bytes plus 224 scale bytes per copy at hidden 7168.

- MoRI's codec is what vLLM's MoRI `prepare()` runs for a block-quantized model (aiter `per_1x128`,
  `scale_dim = hidden/128`), and what SGLang runs, in e4m3fnuz on gfx942 and OCP e4m3fn on gfx950,
  as vLLM's `fp8_dtype()` selects.
- UCCL-EP quantizes per token and per 128 channels in e4m3fnuz (max 240) on gfx942 (MI300X, MI325X)
  and e4m3fn elsewhere, the arch-keyed dtype of its upstream bench (`ep_uccl.py`
  `_fp8_e4m3_dtype`).
- On Blackwell (CUDA capability 10.x or 12.x) the DeepEP and UCCL-EP low-latency kernels round each
  per-128 scale up to a power of two and hand the receiver packed UE8M0 exponents
  (`round_scale=True, use_ue8m0=True`), because that is what vLLM's DeepEP LL path asks for when its
  DeepGEMM expert kernel takes UE8M0 scales; on Hopper they keep plain FP32 scales, as vLLM does. The
  wire carries FP32 scales in both cases, so FP8 byte counts do not change, and those rows carry the
  `-ue8m0` generation suffix.
- The `normal`-mode codecs of DeepEP V2, UCCL-EP and FlashInfer EP keep unrounded FP32 scales, where
  vLLM's `per_token_group_quant_fp8` rounds them to powers of two whenever DeepGEMM's E8M0 path is on
  (vLLM main a9fe30e9f6 `fp8_utils.py:663-664`): byte-identical and timing-neutral, but not
  bit-identical to vLLM's quantize.
- NCCL EP's FP8 is its internal DS_FP8E3M4 recipe, for low-latency decode only, staged (never
  zero-copy) with a BF16 combine. It is the one realizable (backend, precision) pair off every
  deployed path: TensorRT-LLM's `NcclEP`, the only engine integration, sends DeepSeek-FP8 activations
  in BF16 (`uses_internal_fp8_dispatch = False`). So `sweep_matrix.py`'s `OFF_PATH_PRECISIONS` keeps
  it out of the default matrix, and naming the precision (`--precisions fp8`) opts it back in: the
  one place a precision filter adds rows rather than only removing them. NCCL EP's normal mode and
  prefill stay BF16-only here, although the library also offers HT a caller-quantized FP8 path with
  forwarded scales (`QUANT_FWD`).

The caller-side quantize is charged **inside the measured dispatch**, because a production forward
pass pays it on the critical path. It is one compiled kernel on DeepEP V2, UCCL-EP, FlashInfer EP and
MoRI, checked bitwise against its eager reference, and per row across batch sizes, before timing;
MoRI compiles it in `low-latency` too, because InterNodeV1LL takes a caller-quantized tensor. The DeepEP V2
and UCCL-EP `low-latency` kernels quantize internally from BF16. So an FP8 `normal` dispatch covers
quantize-plus-transport while its BF16 control covers transport alone.

Read that charge as a **fixed per-call cost, not a payload-proportional one**, and compare FP8 with
BF16 at the top of the ladder. On eager DeepEP V2 decode rows timed before per-sample alignment, FP8
dispatch p50 minus its BF16 control was 65 µs on h100, 59 µs on b200, 27 µs on b300 and 57 µs on
gb300 at T=1, flat within a microsecond or two through T=64, then decaying, and slightly negative by
T=512 on h100 and b300, where halved payload bytes more than repay it; FlashInfer EP carried ~107 µs
on gb300 under the same timing. At T=1 the FP8 path moves *fewer* bytes than BF16, and the charge
exceeded the fused quantize's own device time (1.5-3.6 µs per SKU) by more than an order of
magnitude: host launch cost landing in a window that opened with no host sync behind an idle stream.
The graph-replayed gb200 deepep-v2 EP8 rows do not show it (+2.1 µs at T=1 and -22.5 µs at T=512,
run 38523177277, 2026-10-15). FlashInfer's graph-replayed FP8 rows, and FP8 rows under the current
alignment and engine contracts, have not been compared yet.
Compiling the quantize reduces the charge rather than causing it: an eager quantize measured
33-39 µs worse per decode dispatch on h100 through the same window, while production issues one
fused quantize into an already-busy stream.

`implementation.fp8_consume` names which FP8 consumption path the chained roundtrip models.
`native`, the default, has the expert GEMM consume the dispatched FP8 and scales directly, as SGLang
and vLLM do for this workload's 128-block shape, so no conversion sits between the collectives.
`CX_FP8_CONSUME=dequant` stages every pair from its own dispatch, modelling vLLM's quant-format
mismatch fallback (vLLM dequantizes when `block_k` disagrees with DeepEP's block size). It is a
verification hatch rather than a second metric (never a sweep axis, never a default), because the
mismatched-config cost is derivable from what every run already emits:

    dequant roundtrip  ~=  roundtrip + stage        (+2.4% .. -0.1%, b200 LL fp8 ladder)

slightly high because chaining amortises launch overhead (median `rt/(d+s+c)` = 0.93 across the
corpus). The reverse does **not** hold: reconstructing native as `dequant - stage` errs by -11.6% at
T=1 and -5% at T=64 and converges only by T=256, worst in the decode regime the headline reports.
So measure native and derive dequant, never the other way round. Charging `stage` to the FP8
roundtrip inverted the fp8-vs-bf16 verdict in 39 of 51 comparisons.

## Backends

Each backend runs the configuration its reference engine selects (vLLM, except TensorRT-LLM for
NCCL EP low-latency), so a cross-backend comparison compares deployed contracts. `implementation.maturity` is `production` where an engine can select the configuration
today and `candidate` where the row describes the library (see [Result Document](#result-document)).

### DeepEP V2

DeepEP V2 means the `ElasticBuffer` introduced by
[DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605), not a newer legacy `Buffer` build.
The source pin in `runtime/common.sh` is upstream `main`, which carries #605 along with
[PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) (pure scale-up initialization when GIN is
unavailable), [PR #640](https://github.com/deepseek-ai/DeepEP/pull/640) (NCCL shared-memory mappings
are no longer misclassified as duplicate NCCL libraries) and
[PR #642](https://github.com/deepseek-ai/DeepEP/pull/642) (the low-latency combine fence for the
Blackwell top-rung corruption of [issue #700](https://github.com/deepseek-ai/DeepEP/issues/700)).
The adapter fails closed unless the loaded `deep_ep` exposes `ElasticBuffer`, which also catches an
image-bundled `deep_ep` shadowing the build.

`normal` runs vLLM's `deepep_v2` configuration (`library_config.engine_contract = "vllm"`) in both
phases. vLLM keys its prepare/finalize on the model-level flag (`use_cudagraph = not
enforce_eager`), so with CUDA graphs on, its default, every batch, prefill as well as decode,
dispatches with `do_cpu_sync=False` and `do_expand=False` into a receive bounded at the next power of
two of T, on vLLM's buffer flags: `prefer_overlap_with_compute=False`, under which DeepEP sizes its
kernels at max(its model, 64) = 64 SMs, and `allow_multiple_reduction=False`. No deepep-v2 dispatch
waits on the host. Decode replays a graph and prefill runs the same path eager, because vLLM runs
large prefill batches uncaptured.

Scale-up cases request NCCL Device API LSA and fail closed unless the realized LSA team covers the
full EP world (`library_config.lsa_team_ranks`, checked against the registry's scale-up domain),
which is also what makes the adapter's hybrid choice the one vLLM's
`VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1` autoselect would make: DeepEP goes hybrid only when the LSA team
is smaller than the world. x86 EP16 is that hybrid path, GIN across two logical scale-out domains
represented by two physical RDMA ranks, with eight scale-up ranks per domain. GB EP16 remains MNNVL
scale-up and uses LSA. Hybrid EP16 allocates 33 GIN QPs per peer rank (a fixed 512-QP budget)
instead of upstream's and vLLM's automatic 65/129, which exhaust the per-NIC QP budget on CX-7
(`ibv_create_qp` ENOMEM in `ncclDevCommCreate`; 49x16 initializes, 65x16 does not). The 64-SM
kernels ask for 64 x 16 + 1 QPs and are capped at that allocation, so the hybrid channels share 32
data QPs plus one notify QP (`library_config.num_allocated_qps`, `num_qps`). DeepEP's SM model reads
the scale-out rate from an `ibstat` probe that cannot see EFA, and in hybrid mode divides by it, so
on the EFA pools the registry's `network.rdma_gbs` supplies the per-GPU rate and the resulting count
is passed explicitly (`library_config.sm_model_rdma_gbs`). Elsewhere the count is left to DeepEP, as
vLLM leaves it. With overlap off the model's own 10-26 SMs are raised to 64 either way.

**A virtualized pool can make a scale-out row measure the hypervisor rather than the fabric.**
h200-dgxc EP16 on deepep-v2 paid three to eight times h100's cross-node cost while nccl-ep on the
same nodes ran at h100 rates. ElasticBuffer registers its GIN window `NCCL_WIN_STRICT_ORDERING`,
which strips PCIe relaxed ordering, and in h200's KVM guests strict-ordered GPU-NIC writes cap the
hop near 10 GB/s per GPU. The DeepEP build patches the registration behind a switch that a pool
opts into with `rdma_relaxed_ordering` in its `network` block. On h200-dgxc, bf16 decode at 512
tokens per rank drops from 3088 µs to 620 µs per round trip and the oracle passes in bf16 and fp8.
Upstream made the window strict on purpose (DeepEP #661/#674): with relaxed ordering a GIN signal
can in principle overtake the data it announces. That race has not been reproduced, and a passing
oracle does not rule it out. Those rows carry a `-relaxed-ordering` `kernel_generation` suffix and
never pool with the strict series; every other pool keeps upstream's registration.

`low-latency` drives the legacy `deep_ep.Buffer` IBGDA decode kernels
(`low_latency_dispatch`/`low_latency_combine`), which deliver a per-expert padded receive buffer and
apply the top-k gate weights inside a source-side combine (weighted-kernel-sum). It builds and calls
them as vLLM's `deepep_low_latency` path does: `num_qps_per_rank` equal to the local expert count,
NVLink allowed, MNNVL allowed only on the GB pools (vLLM's `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL`,
which its env doc says to turn on "for better latency on GB200 like system"), and each collective
split into a send kernel and a receive hook (`return_recv_hook=True`). vLLM's default non-DBO path
splits both dispatch and combine that way (vLLM main a9fe30e9f6 `modular_kernel.py:1214-1273`,
`1405-1446`; `deepep_ll.py:222-223`, `309`, `402`) and runs the hook straight after the send unless
a shared expert is scheduled in between; here nothing is, so the hook follows immediately.
Single-node EP8 runs over the intra-node NVLink path (`allow_nvlink_for_low_latency_mode`),
validated on H200 with `/dev/gdrdrv` absent. NVSHMEM/IBGDA carries payload on the wire only at
multi-node EP16, which among the x86 pools runs on B200 alone: the nscale bare-metal pool's
gdrdrv-backed IBGDA over native IB is what a low-latency scale-out needs, and no virtualized pool has
it. GB200/GB300 EP16 stays inside the MNNVL scale-up domain. The legacy Buffer still initializes
NVSHMEM with an RDMA transport even single-node, so both EFA pools (B300, H100) withhold
low-latency: on b300 NVSHMEM aborts with "nvshmem detect topo failed" (status 28) with and without
`NVSHMEM_REMOTE_TRANSPORT=none` (runs 34521825749, 34523594787), while the HT `ElasticBuffer` path,
which needs no NVSHMEM, passes.

### UCCL-EP

[UCCL](https://github.com/uccl-project/uccl) EP is a drop-in, API-identical DeepEP replacement that
keeps the legacy `Buffer` `dispatch`/`combine` (unweighted rank-sum) but routes scale-out over CPU
proxies that issue GPUDirect RDMA on plain `libibverbs`, with no NVSHMEM/IBGDA and with software
message ordering, atomics, and flow control. Its scale-up is single-node `cudaIpc` over NVLink/XGMI
(so the scale-up domain is one physical node, never MNNVL), and UCCL starts its proxies only across
nodes, so the b200-nscale EP16 `normal` rows are the only ones on the proxy path. No engine ships a
UCCL selector, so its rows stay `candidate`, but engines deploy it as a drop-in `deep_ep` (UCCL's
`deep_ep_wrapper`, the route UCCL documents for vLLM), and vLLM's DeepEP backends then drive this
Buffer. The adapter runs vLLM's configuration (main, 2026-10-15).

`normal` is the `deepep_ht` path. It asks `Buffer.get_dispatch_config` and `get_combine_config` for
the EP size on every call, and calls `set_num_sms` only under dual-batch overlap, which is off by
default, clamping it to 20 there, so the budget is UCCL's class default of 20 SMs at EP8 and EP16, on
HIP as on CUDA. FP8 scales go out row-major, as vLLM's block quantize emits them. The Buffer gets
1 GiB of NVLink buffer, plus 1 GiB of RDMA buffer and 10 QPs per rank when it spans nodes. Its
dispatch polls its receive counters on the host before launching the transfer, so `normal` stays
eager and its samples pass through the host gate ([Rank Alignment](#rank-alignment)).

`low-latency` issues the shared legacy-Buffer calls exactly as DeepEP V2 does (send kernel plus
receive hook), with the receive sized to vLLM's fixed 256-token batched-DP budget rather than to the
ladder, and the same 1 GiB NVLink buffer, which the low-latency kernels never touch. At EP8 the
adapter passes `is_intranode`, so those kernels run `cudaIpc` over NVLink rather than the proxy
path. vLLM's `enable_shrink=False` is the one argument the pinned Buffer rejects (uccl#1043 adds it).
Intranode low-latency replays graphs on every SKU and precision, as vLLM's captured decode step runs
it; an earlier b200 FP8 eager exception rested on a 0.98x-vs-1.04x pair-period gap measured before
per-sample alignment existed (b0d7acf17), which is drift-sized.

UCCL-EP is built from source at its pin in `runtime/common.sh` with UCCL's documented drop-in flags
`USE_DMABUF=1 PER_EXPERT_BATCHING=1` (the second changes the low-latency kernels).
`implementation.library_version` reads `uccl@<pin> deep_ep==<wrapper> <build flags>`, taken from the
record the prepare step writes beside the build, and the import probe rejects DeepSeek's `deep_ep`
(same API), any `deep_ep` loaded from outside the prepared cache, and any cache recorded for another
commit.

The AMD SKUs run UCCL-EP `normal` without low-latency: upstream raised `kNumMaxTopK` 9 -> 16
(uccl#1016, 2026-07-13) six days before our pin, and the low-latency host assert
`kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group` cannot hold on AMD, where that product is
16 for every CU count. uccl#1057 (341835e, 2026-09-27) checks the runtime `num_topk + 1` instead, so
the exclusion is the pin's, not the library's, and lasts until a pin bump and a correctness pass.
UCCL-EP EP16 is registered on b200-nscale only, where it runs at full health over bare-metal IB. On
mi355x its CPU-proxy RDMA is functional over the Pollara fabric but roughly 13x under its
upstream-documented bandwidth (~6 GB/s vs 82 GB/s), unchanged by GPU-memory registration mode
(DMA-BUF vs peer-memory) or traffic class: an ionic-driver-level suspect.

### MoRI

`normal` runs vLLM's MoRI configuration (`MoriAll2AllManager`) under MoRI's MANUAL launch mode,
which is MoRI's default and which neither vLLM nor SGLang overrides in code (this repo's mi355x FP4
SGLang recipes do set `MORI_EP_LAUNCH_CONFIG_MODE: AUTO` in their base environment,
`disagg-variants.yaml:41`). The launch tuple lives in the
`EpDispatchCombineConfig`, and dispatch and combine pass no per-call override, exactly as vLLM's
`prepare`/`finalize` call them. One node runs IntraNode at block_num 80, rdma_block_num 0 and
`warp_num_per_block` 16 on dispatch and combine alike, the tuple vLLM uses under both
`mori_high_throughput` and `mori_low_latency` and SGLang pins too; two nodes run InterNodeV1 at
64/32/8 on gfx950 (32/16/16 on gfx942). `num_qp_per_pe` is MoRI's default of 1, the input buffer is
external (MoRI's default, which SGLang sets explicitly), and combine `quant_type` is `none`.
`implementation.library_config` records the tuple the handle realized, as MoRI's `get_handle_info`
reports it, and the adapter fails closed if any field differs from the request;
`implementation.library_version` records the installed MoRI build.

The warp count and the buffer mode are pinned together deliberately: MoRI's tuning tables key combine
on `zero_copy`, selecting roughly 16 warps for external input against 4-8 for a registered buffer,
and the mismatch is measurable in both directions. On MI300X, MI325X and MI355X, **in
registered-buffer mode** 16 warps costs +13-18% combine at T=128 and +61-78% at T=512 against 8.
**In the external-input mode the engines actually run** the same 16 warps *wins* by 14-19% at T=128,
26-27% at T=256, and 9-14% at every prefill rung including T=8192. Below T=32 it gives up 0.2-2.5 µs,
the only range where 8 is ahead. All arms were correct at every rung. With an external input buffer
combine's push kernel writes each received row straight into its destination's staging slot. That
copy is the transfer itself, bounded by the receive count, so BF16 rows hand over the dispatch output
unchanged and report `stage` unavailable, while FP8 rows still stage for real, to dequantize the
received payload with its per-block scales.

These rows describe the engine-integrated configuration, not MoRI's peak: its shipped tuning tables
reach a faster combine with per-shape block and warp counts no engine selects, and AUTO would not
reproduce them uniformly. Gfx950 ships no IntraNodeLL combine table and no BF16 rule for normal-mode
IntraNode dispatch, so AUTO defaults exactly those two (coupling the result to whichever MoRI
revision is pinned) while tuning the other two, which is not one number about the hardware. How far
off peak is arch-dependent: across buffer modes on MI355X, with the registered mode's excluded BF16
stage added back, a registered buffer at 8 warps is still 15% faster at T=512 decode and 8% at
T=8192 prefill than the shipped pairing, but that did not reproduce on gfx942, so treat 0-15% as the
honest range for a configuration no engine runs.

MoRI EP16 runs InterNodeV1 on mi355x only, 2x8 over the Pollara RoCE rails with MoRI's default
STATIC_HEAP symmetric heap; MI300X and MI325X register no scale-out fabric, so their EP16 cells are
unsupported. Two caller-side defects had to be removed before the mi355x row could publish.
`combine()` takes the rank's own routing tensor, never dispatch's returned recv-slot indices:
InterNodeV1 keys its node predicate by the rank's own token id, so recv-layout indices silently drop
or pollute cross-node partials (the caller-side part of ROCm/mori#475, guarded upstream by
ROCm/mori#546). And forcing `MORI_SHMEM_MODE=VMM_HEAP` for scale-out corrupted combine
stochastically from T~64 up and near-certainly at prefill token counts: on ROCm 7.2 the HIP runtime
leaves VMM allocations requested uncached in the cached pool, so cross-node combine partials read
stale lines (ROCm/mori#610). A same-branch A/B on mi355x settled it: STATIC_HEAP is correct on every
bf16 and fp8 rung, decode 1..512 and prefill 1024..8192 (run 34939333022), while VMM_HEAP reproduces
the corruption (run 34941185534). A scale-out MoRI case now fails closed under any heap mode but
STATIC_HEAP.

`low-latency` follows vLLM's `mori_low_latency`, which differs from `mori_high_throughput` only
across nodes. One node builds the same IntraNode kernel under either selector, which the `normal`
decode rows already measure, so MoRI has no EP8 low-latency cell. Across nodes it runs
`InterNodeV1LL` with the arch's tuple (block_num 64, rdma_block_num 32 and 8 warps on gfx950; 32, 16
and 16 on gfx942) and one QP per PE (`MoriAll2AllManager._make_all2all_kwargs`; the selection
landed in vllm#41751 and ships from v0.27.0), called through the same single-call `dispatch()` and
`combine()` as every vLLM MoRI path (`MoriPrepareAndFinalize.supports_async()` is False). It keeps
InterNodeV1's rank-deduplicated layout, unweighted rank-sum combine and STATIC_HEAP requirement.

`MORI_ENABLE_SDMA` stays unset, MoRI's default (off), on every row, because vLLM never sets it.
MoRI reads it once, at shmem init, and calls it "currently only effective for AsyncLL"
(`dispatch_combine.cpp:138-140`), a kernel no vLLM path selects. A row run with it set on carries
the `-sdma` generation suffix and `library_config.sdma = true`. SGLang deploys MoRI low-latency
differently, as `AsyncLL` (64/32/8 with 2 QPs per PE, driven split-phase, selected by
`MORI_ENABLE_SDMA`), which this suite does not measure.

### NCCL EP

NCCL EP is NVIDIA's native MoE dispatch/combine on the NCCL Device API, shipped since v0.2 as the
`nccl-extensions` wheel (which owns `nccl.ep`; `nccl4py`, pinned alongside, provides `nccl.core`).
[nccl-ep.md](nccl-ep.md) documents its engine contract, row identity and wheel pin. It is NVIDIA-only
and CUDA 13 only, and runs EP8 and EP16 on all six NVIDIA SKUs: EP16 stays inside the MNNVL scale-up
domain on GB200/GB300 and is a 2x8 RDMA scale-out over the cross-node GIN path on H100/H200/B200/B300.
Under v0.1 that GIN path faulted inside `nccl_ep.cc` identically on RoCE and IB on every x86 pool;
v0.2 clears it, revalidated on-metal on B200, H100 and H200, and on B300 over EFA proxy GIN (full
decode and prefill ladders correct).

`normal` selects its `HIGH_THROUGHPUT` algorithm, whose FLAT `[N, hidden]` receive and unweighted
rank-sum combine match `layout-and-dispatch-v1` exactly, so the same oracle applies. No engine runs
it (TensorRT-LLM's `NcclEP`, the only engine integration, is low-latency rank-major only), so HT rows
are library rows (`candidate`) and reproduce no engine's cost. Every backend's timed HT dispatch
carries its routing work: uccl-ep calls `get_dispatch_layout` inside dispatch, deepep-v2, MoRI and
FlashInfer pass routing on every call, and nccl-ep binds it with `ncclEpUpdateHandle`, documented as a
"per-step collective: prepare the handle for the given top-k routing decisions", which the API
requires before every dispatch whose routing changed. The update's HT work is sized by the handle's
capacity, not by T: capacity-wide memsets, a routing-map conversion, an `ncclAllGather` of capacity
rows and a metadata scan over them (nccl-extensions `nccl_ep.cc:4057-4137`, `4281`). Every rung, the
smallest included, therefore carries a term set by the ladder's top (512 for decode, 8192 for
prefill). NVIDIA's own `ep_bench` excludes the update (CUDA events around dispatch and combine only)
and times it in a separate micro-benchmark. In low-latency mode `ncclEpUpdateHandle` returns
immediately and the kernel reads the cached routing inside the timed dispatch.

`low-latency` runs TensorRT-LLM's `NcclEP` configuration: rank-major receive with caller-side
weighted expert pre-reduction and an unweighted rank-sum combine, one group sized to the ladder's
top (TensorRT-LLM sizes it to `min(max_num_tokens, moe_max_num_tokens)`), and the engine's zero-copy
policy: graphed rows dispatch into a plain tensor that the library stages, as TensorRT-LLM does
under CUDA graph capture, eager rows write a registered window directly within one LSA/MNNVL domain,
and scale-out stages in both regimes. v0.1's combine recv pipeline, a port of DeepEP's pre-#642 code
missing the `fence.proxy.async.shared::cta`, raced on every rung (1-in-5 bimodal corruption at T=256
on gb300 EP8, error 0.47 vs 0.0039; a T<=128 ladder clamp reduced exposure but was never a safety
boundary). The v0.2 wheel ships that fence (`fence_view_async_shared` before the `emptyBarriers`
arrive in `ll_ep.cuh`), so the ladder runs unclamped, subject as always to the oracle. The adapter
binds one handle per group and rebinds it per ladder point, because two handles built from one group
share its double-buffer parity slots and corrupt each other's signalling
([NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303)).

### FlashInfer EP

FlashInfer EP is TensorRT-LLM's one-sided MNNVL `MoeAlltoAll`, in which each rank writes tokens
directly into its peers' workspace windows and combine reads them back, so there is no send/recv
pairing and no NVSHMEM. It is GB200/GB300-only for that reason, runs EP8 and EP16 inside the MNNVL
scale-up domain, and has one kernel family, so it runs `normal` only, across the full ladder: a
`low-latency` row would re-measure the same kernel under a mode that promises another. It runs
vLLM's integration (`--all2all-backend flashinfer_nvlink_one_sided`): dispatch sends
`[x, x_sf, topk_ids, topk_weights]` (no `x_sf` under BF16) with `invalid_token_expert_id=-1`, and
combine is handed a caller-owned payload with `payload_in_workspace` at its `False` default, so
FlashInfer's own staging copy of the filled slots into the workspace runs, and is timed, inside
`combine`, as in vLLM. Writing the expert output straight into the workspace instead is
TensorRT-LLM's path (and SGLang's with its `flashinfer_cutlass` runner only). The result lands in a
caller-owned tensor (`output=` from 0.6.16; before that the returned tensor is copied into it, as
vLLM's `combine_into` does). Its FP8 dispatch is DeepSeek blockwise FP8 (e4m3 with FP32 1x128 scales,
vLLM's `per_token_group_quant_fp8` at block 128) carried as a fourth dispatch payload, which vLLM
accepts on this transport from v0.29.0 (vllm#51924). The C++ `toNvDataType` accepts only
fp16/bf16/fp32 for combine, so an FP8 combine buffer would raise rather than corrupt.

Its combine is the one place a backend's accumulator precision changes the expectation rather than
the tolerance: through 0.6.15 the kernel holds its top-k accumulators in the payload dtype and
reduces them with a hand-unrolled pairwise tree, so every level rounds to BF16, and the oracle
reproduces that tree exactly (`combine_reduction = "topk-slot-tree"`) rather than loosening the gate
to absorb it. 0.6.16 rewrote the accumulator to FP32, and the adapter reads the installed version and
picks the matching model. Graph replay cuts the gb200 EP8 T=1 pair period from 151 µs to 43 µs and
leaves prefill flat, so it replays decode only.

## Workload Identity

One deterministic workload is generated over the global token batch from the workload's seed in
`configs/sweep.json` (part of the workload identity, baked into every scheduled case) and sliced by
source rank. A keyed BLAKE2b counter over the (token, slot, attempt, stream) coordinates produces
byte-identical expert indices and gate weights on every runtime, and the harness proves the
realized routing trace identical across ranks before a case can succeed.

Routing traffic distinguishes:

- token-expert assignments, which determine expert compute load.
- rank-deduplicated token payload copies, which determine EP activation traffic.

Adapters may not generate routing or reinterpret one quantity as the other.

## Measurement

`normal` mode uses `layout-and-dispatch-v1`: dispatch timing includes layout plus communication, and
combine returns activation payload through an unweighted rank-sum path. Expert-output staging is
outside isolated combine timing AND outside the measured paired roundtrip, so `roundtrip` means
dispatch then combine (the transport) in every row, and staging is reported as its own `stage`
component wherever it does device work. The `CX_FP8_CONSUME=dequant` verification hatch is the one
exception, putting the conversion back inside the chain on purpose.

Under FP8, treat `stage` as **harness scaffolding rather than a phase a serving stack has**: it
converts the received FP8 payload to the BF16 combine sends, something production never does
separately. In production the FP8 lands in the expert GEMM, which reads FP8 operands natively and
emits the BF16 combine receives. This suite measures the collective, not the layer, so `stage`
stands in for that GEMM, which is why it is excluded from `roundtrip` and why **`stage` must not be
summed into a total or compared between backends**: each adapter converts a different amount.
UCCL-EP converts only the received rows in `normal` mode, and MoRI in both modes. DeepEP V2 runs
`normal` without the host sync, so its receive is the worst-case `[next_pow2(T) * ranks, hidden]`
plane and combine must be handed that whole plane (it reads only the valid prefix on device); its
`stage` converts all of it, in decode and prefill alike. DeepEP V2 and UCCL-EP convert the whole
padded `[local experts, 256 * ranks, hidden]` receive in `low-latency`, regardless of token count,
and NCCL EP its whole low-latency receive plane. FlashInfer converts only the filled slots, and only
under FP8: its BF16 stage is a pointer hand-off, because its payload staging copy is part of its
`combine`. The one production path that *does* pay a separate materialised dequant is the
quant-format mismatch fallback, which `CX_FP8_CONSUME=dequant` models; it is not the default because
it is not the fast path.

Read `implementation.stage_excluded_from_roundtrip` as "there was device-work staging and it was
hoisted out of the chain", not as "this row's roundtrip is stage-free". It is gated on whether the
backend's `stage()` does device work at all, so `false` covers two unrelated situations: the backend
has nothing to stage (every BF16 row hands its receive buffer to combine by pointer, FlashInfer's
payload staging copy then running inside `combine`), or the `dequant` hatch put the conversion back
inside the chain (`implementation.fp8_consume = "dequant"`). Reading `false` alone as "roundtrip
includes staging" subtracts a cost the row never paid. Each component declares availability, origin,
and sample count; `stage` is measured only on eager rows whose `stage()` does device work, and is
unavailable elsewhere (graph replay never times it separately). `isolated_sum` is derived: the
isolated dispatch and combine percentiles added, never `stage`.

Headline latency is the **chained pair period** (`components.pair_period`, defined under
[Chained Pair Period](#chained-pair-period)) for every row, graph-replayed or eager; `summarize.py`
prints its p50 and p99. MAX is the fresh-entry family's reduction because a layer is not finished
until its slowest rank is, so MAX is the completion cost, and it charges inter-rank entry stagger to
whichever component the ranks entered unevenly. Before per-sample alignment (Rank Alignment) that
stagger depended on the code path AND the precision, not only on the fleet: on identical h200
low-latency decode cells the per-iteration spread was ~9.3 µs for deepep-v2 and uccl-ep at BF16 (they
share the legacy `Buffer` path) against ~2.6 µs for nccl-ep, and collapsed to ~2.8 µs for those same
two under FP8, where in-kernel quantisation makes the heavier dispatch self-align the ranks.
Alignment removes most entry stagger; ranks still leave a window unevenly (h200 uccl-ep prefill
spreads 11-67 µs at p50), and that term is not subtractable in any principled way, so MAX alone taxes
some rows more than others.

Every row therefore also carries `cross_rank_min_us` (the same iterations reduced with MIN, the
skew-excluded floor) and `cross_rank_spread_us` (per-iteration roundtrip MAX minus MIN). Read MAX and
MIN as a bracket: two cells whose MAX gap is smaller than the larger contender's spread are not
separated by the data. Call a fresh-entry winner only where MAX and MIN agree on the ordering. Do not
rank on p99 of MAX for multi-node decode cells, where it is dominated by worst-rank stalls rather
than transport, while p99 of MIN is the synchronized-cost tail beside it. The isolated components
inherit the preceding operation's per-rank exit stagger, so treat them as residual-wait diagnostics
rather than per-operation costs; the paired roundtrip is the comparable fresh-entry quantity.

### CUDA Graph Replay

Serving engines capture their decode step, so graph-compatible backend/mode pairs are measured
under `CUDAGraph.replay()` by default at decode, as the backend's engine captures it, without
changing its contract. Prefill stays eager on every backend, as the engines run it, and
`COLLX_CUDA_GRAPH=0` restores the eager pipeline everywhere.

- **deepep-v2** `normal` and `low-latency` decode (see [DeepEP V2](#deepep-v2)).
- **nccl-ep** low-latency and HT decode. Graphed low-latency dispatch writes a plain tensor that the
  library stages, as TensorRT-LLM's captured decode does, so it carries no `-zc`; eager rows within
  one domain take the window path (`-zc`). HT's per-step routing `ncclAllGather` is the one host
  NCCL collective in any capture, so nccl-ep's communicator runs NCCL graph usage mode 1 (one graph
  at a time): the default mixing mode wraps every captured collective in an external event wait and
  record, which left ~14 µs idle before each all-gather (h100 EP8 trace) and put the graphed HT
  decode pair period at 1.14x eager, against 1.03x at T=1 and 1.00x at T=256 under mode 1. Graphed
  HT rows carry the `-gum1` suffix; eager rows never do, since the mode only affects a capture.
  Across nodes the captured all-gather is still proxy-driven, and NCCL fronts it with a
  host-callback node on every replay (about +50 µs of dispatch at EP16).
- **mori** `normal` and `low-latency` decode, since vLLM captures at most 512 tokens by default on
  ROCm and runs its prefill batches uncaptured. ROCm torch before 2.13 rejects external events, so
  the timing events are captured with `hipEventRecordWithFlags(..., hipEventRecordExternal)`, the
  call newer torch makes.
- **flashinfer-ep** decode.
- **uccl-ep** low-latency, intranode only; `normal` host-syncs and stays eager.

Only the launch mechanism changes; every family keeps its eager meaning. The fresh-entry components
(`roundtrip`, `dispatch`, `combine`) each capture one pair with event nodes around the timed window,
behind the alignment captured ahead of the first timing event (Rank Alignment); their origin is
`cuda-graph-replay`, and `stage` is not separately timed. The chained family captures each sibling
chain as one graph of `chain_iters` unrolled pairs, the shape of a decode graph, replayed once untimed
to upload it and once timed. Capture uses torch's default (`global`) error mode, as vLLM's and
SGLang's decode captures do: an API call that is unsafe during capture, from any thread, fails the
capture instead of being recorded into a graph no engine could build. Every graphed row is
value-checked (Correctness, check 8). Graphed rows carry `-cudagraph-spin-ar` on `kernel_generation`
and eager rows `-eager-spin-ar`, so the two never pool; `implementation.cuda_graph_replay` and
`cuda_graph_supported` record the regime.

### Rank Alignment

Every fresh-entry sample opens the way DeepEP's `bench_kineto` does: a GPU spin, then a device
all-reduce, with no host sync before the start event. The spin lasts 2 ms of wall time
(`COLLX_ALIGN_SPIN_US`; DeepEP spins about 10 ms) and is calibrated after each series' warm-up,
because `torch.cuda._sleep` counts SM cycles. It outlasts the host's enqueue of the sample, so the
last rank to arrive already has its timed work queued when the all-reduce releases everyone, and
every rank's window opens at the barrier's exit instead of at its host's launch: the entry skew and
the first kernel's launch latency both leave the window (about 25 µs off `cross_rank_min_us` as well
as MAX on h200). A 500 µs spin covered the typical 50-70 µs enqueue but not its occasional stalls:
h200 uccl-ep normal decode roundtrip p99 roughly doubled and spread p99 rose from ~20 to 185-293 µs.

Eager rows queue the spin and the all-reduce ahead of each sample. Graphed rows capture them in each
graph ahead of its first timing event, with their stream dependencies, so no host-side handoff
follows the alignment. CUDA launches a replay whole: a rank whose host launches late reaches the
all-reduce late, and the other ranks wait for it there, before their windows open. Before the
all-reduce moved into the graph (#3727), a late launch stalled the other ranks inside the window
(gb200 flashinfer EP8 T=1 roundtrip p99 858 µs against a combine p99 of 30 µs); in-graph, that cell
measured 49 µs at p99 against 46 µs at p50 on 2026-10-15, and the graphed decode rows rechecked that
day ran their roundtrip p99 at 1.01-1.09x their p50 on h200 EP8 (deepep-v2, nccl-ep HT and
low-latency) and gb200 flashinfer EP8, and 1.03-1.19x on mi325x MoRI, whose own pair-period p99 ran
1.02-1.16x its p50, with the per-sample cross-rank spread at 5-26 µs at p99. That is why graphed rows
publish tails. HIP, by contrast, submits a replay node by node from the host, and its captured event
records become host-enqueued markers, so on ROCm a host that lags after submitting the all-reduce
lets its peers' windows open before its own dispatch is queued. The spin gives that host time to
submit the whole replay before the barrier releases anyone. Graphed mi300x MoRI decode behind the
all-reduce alone ran its roundtrip p99 at 1.30-1.55x its p50 at small T, with the cross-rank MIN
tight (one rank's tail), while the same node and image ran eager at 1.02-1.14x. With the spin
captured ahead of the all-reduce, graphed decode ran its roundtrip p99 at 1.02-1.13x its p50 on both
mi300x and mi325x at every rung, no replay call outlasted the spin, mi300x's small-T dispatch p99 fell
from 47.6-56.4 µs to 29.8-36.0 µs, and its per-sample cross-rank spread p99 from 15-28 µs to 9 µs:
the eager tails of the same SKU.

A sample counts as **missed** when some rank's host took longer than the spin to get from the spin's
enqueue to the end of its timed operation's enqueue (eager) or to the return of its replay call
(graphed), so that rank's window may have opened, or its timed work started, after the barrier
released the others. A dispatch that waits on the host returns only after its own device work, so
for those series the lead stops at the start event and the host gate below covers the rest. Garbage
collection is off for the length of every timed series, so a collection cannot stall a host inside a
window. Every row publishes `rank_alignment.missed_samples` per series and `missed_fraction`, the
largest share among roundtrip, dispatch and combine (staging is local work). The fresh-entry tails
(`roundtrip`, `dispatch`, `combine`, `isolated_sum`, `cross_rank_min_us` and the token rates) are
withheld once the samples that were missed or host-stalled (below), each counted once, could fill
everything above them: p99 at 1% of samples, p95 at 5%, p90 at 10%.
`implementation.rank_alignment` records the method (`gpu-spin-then-all-reduce` eager,
`in-graph-spin-then-all-reduce` graphed) and the spin.

`combine` (and eager `stage`) is timed behind its own untimed dispatch, as a pair runs it, in both
regimes: each combine sample is a fresh pair whose combine input is the stage hoisted once, untimed
(see `stage_excluded_from_roundtrip`), so the window opens at that dispatch's exit and inherits its
per-rank exit stagger. A dispatch that waits on the host would spend the spin on its wait if it
followed the barrier, so in those series (uccl-ep `normal`, which is always eager) it runs before the
alignment and the window opens at the barrier's exit instead.

**Host-synced series wait out a throttled host.** In modes whose dispatch waits on the host (uccl-ep
`normal` is the only one) the barrier aligns the dispatch launch, and the host's poll and relaunch
sit inside the window. That is their production cost, and it carries the host's speed into the
window. While uccl-ep EP16 runs, b200-nscale nodes intermittently run the cores of both sockets at
about 200 MHz effective (`scaling_cur_freq`, an APERF/MPERF average) for 100-200 ms, every 2-5 s,
while the GPUs keep their clocks and the thermal-throttle counters stay at zero. An idle pinned
process outside the benchmark runs ~16x slow over the same spans, then takes another 10-20 ms to
wind back up through 3-7x. The same nodes show none of it idle, with 40 cores busy, during uccl-ep
EP8, or during deepep-v2 EP16, whose RDMA is GPU-initiated where uccl-ep posts its own from 32
busy-polling CPU proxies per node; runs with and without the per-sample alignment see it at the same
rate. It is the platform's response to uccl-ep's multi-node host load, not to the alignment. The
50-100 µs of uccl-ep host work between the notify and the end event then takes 1-2 ms, every rank's
transfer waits on the late one, and `cross_rank_min_us` rises with MAX. On 2026-10-14 that hit
0.7-1.2% of EP16 decode samples, enough to put roundtrip p99 at 2-3.8 ms instead of ~0.6 ms at
whichever rungs collected more than 1%.

So each sample of these series first times a fixed loop of interpreter work (~20 µs at speed)
outside the window, and while it runs more than 2.5x slower than the lowest median any of the run's
series calibrated, the rank keeps probing instead of opening the sample; the other ranks wait at the
barrier. A second probe after the sample's final sync catches a throttle that began inside the
window. A sample waits at most 0.5 s and a run 60 s in total, so a host throttled for good cannot
stall the run; past the budget its throttled samples count as stalled. One node did that on
2026-10-14: under uccl-ep EP16 its second socket ran throttled a third of the time, 5-30% of the
samples counted as stalled, and every tail was withheld. Rows publish, per series,
`host_held_samples`: samples whose start some rank held for more than 1 ms (a shorter hold is an
interrupt on one probe), which then opened at speed and gate nothing; and `host_stalled_samples`:
samples where a probe on either side of some rank's window ran more than 5x slow, a throttle that
began inside the window or outlasted the hold, which gate the tails like missed ones. Each count
carries its largest series share (`host_held_fraction`, `host_stalled_fraction`); all four are null
in series whose dispatch does not wait on the host. The bars come from the same diagnostics: away
from throttles, a ~50 µs slice of Python on a rank's own core runs more than 2.5x its median in at
most 0.2% of samples and more than 5x in about 0.001%. Holding measures the library on a host
running at speed; a uccl-ep EP16 decode loop on this platform pays the throttles too, and
`host_held_fraction` shows how often they came.

Two limits remain. The probes see the host only at the two ends of a sample, a spin and a window
apart: a slowdown under 5x after the sample opened, or a throttle that starts and ends between them,
still lands in the window uncounted, and the chained family is held per sibling chain (below), not
per pair. And unlike `bench_kineto`, nothing flushes L2 between samples.

Each chain sibling is aligned once, before its first event. An eager sibling opens with the
all-reduce alone: a spin in front lets the host queue a backlog that the GPU then drains faster than
the host can refill it, so a host-bound chain's kept pairs mix two speeds. Eager mi300x MoRI decode,
which the host paces, published a 78 µs period with a +25 µs settle drift at T=64 behind a spin;
behind the all-reduce alone it runs 82-84 µs at every rung to T=128 with settle drift between -3.6
and -0.6 µs. Garbage collection is off inside both siblings. A graphed sibling keeps the captured
spin, since its replay is one host call. In modes whose dispatch waits on the host, each eager
sibling also passes through the host gate: it opens once this host runs at speed, and after its final
sync every rank's host probes again and the ranks agree, by a MAX all-reduce, whether any of them ran
throttled. If one did, every rank restarts the sibling together, up to three times.
`chain_health.host_retried_siblings` counts the restarts and `host_stalled_trials` the trials still
throttled after the last one; a point with any such trial publishes its `pair_period` p50 and
withholds its tails. Both counts are null where the dispatch does not wait on the host.

### Chained Pair Period

The components above measure **fresh entry**: drained around each timed window, so every sample
starts from an idle pipeline behind the rank barrier. A decode loop never stops, and what it pays per
MoE layer is the pipeline's steady-state **period**. Every row therefore also carries the chained
family, measured by `benchmark_chain`: dispatch→combine pairs issued back-to-back with CUDA events
enqueued on-stream and **no host synchronization inside the loop**, the first `chain_drop` (16) pairs
discarded as pipeline fill, pooled over `chain_trials` (4) per point on the same rotated ladder order
as the fresh-entry pass. The pairing is exactly `run_roundtrip`'s (dispatch, the staged combine input
(or an inline `stage` under the `CX_FP8_CONSUME=dequant` hatch), then combine), so paired-API
backends stay in contract and `pair_period` excludes expert-output staging on the same rule
`roundtrip` does.

Each trial runs **two sibling chains** of `chain_iters` (128) pairs, because the statistics must not
carry the instrumentation that collects them. A single chain with six `record()` calls per pair
charged its four inner records plus glue into the published period wherever the device drained
faster than the host enqueued (the bottom of the ladder), because every event then executes as
issued and the pair window degenerates to host elapsed time. The fleet exposed it before a profiler
did: period minus floor-sum sat at a roughly T-independent 10-30 µs on every vendor and fabric at
once (a host constant, not transport), inflating T=1 periods by 20-38%. So a **floors chain** runs
first carrying only the four op-window events, and a **period chain** runs second carrying only the
outer pair events, nothing between its two collectives: both records' host cost lands in the
inter-pair gap, outside the window, so the period carries what an uninstrumented caller pays. (An
eager launch floor remains, as for any eager caller, but a CUDA-graphs decode loop pays less host per
pair than any eager harness can.)

The two chains publish five statistics, and only five, beside the two host-gate counts above:

- `components.pair_period` (origin `chained-median`) is the per-pair period from the period chain,
  reduced across ranks by MEDIAN. MAX is right for a drained component (a layer finishes with its
  slowest rank), but the period is a **rate**: the collectives phase-lock every rank into one
  cadence, and a MAX would publish whichever rank hiccuped as the pipeline's speed.
- `chain_floor_us.dispatch` / `.combine` (origin `chained-cross-rank-min`) are each op's window from
  the floors chain, reduced across ranks by MIN: the last rank into a collective waited least, so its
  window is the op's floor, and it tracks profiler kernel time to ~10%, making it a free Kineto
  substitute. Read floor-vs-period as transport share, not an identity that must close to zero.
  `period - Σfloors` is a real quantity with a meaning in each sign, and neither sign is an error:
  - **Positive** is the per-pair inter-rank wait that the MIN deliberately strips. Where a backend is
    synchronization-dominated it rivals the floor sum and sits *flat in T*, as gb200 flashinfer-ep
    normal decode held ~70 µs bf16 / ~100 µs fp8 at every rung under eager timing (run 31089556516),
    vanishing by T=512 and in prefill as the floors grow into it. That is
    `period = max(sync budget, work)`, not instrumentation. It is invisible to `pair_spread_us`
    (2.7-8.5 µs against 62-107 µs gaps) because the period is conserved while the wait migrates
    between op windows, the same steady-state stagger that bans chained per-op medians below. The
    graph-replayed rows of that cell show +2.8 µs at T=1 and negative values from T=2 (2026-10-15).
  - **Negative** is the floors chain's own four-records-per-pair host cost (~10-12 µs) inflating
    *its* windows wherever the device outruns the host: the very effect the two-chain split evicted
    from the period, still present and harmless in the floors (gb200/h200 fp8 LL, run 31089556516).
    It is not overlap, and it is not cross-chain settling.

  Do not read a large positive residual as the six-events defect returning. That defect was a roughly
  T-independent host constant appearing on *every vendor and fabric at once*, and it lands in
  `interpair_gap_us`. A sync-dominated residual is backend-specific and leaves the gap small.
- `chain_health.pair_spread_us` measures per-iteration cross-rank max-minus-min of the pair. It is the
  *proof* the median means anything. Large next to `pair_period` means a paced or slow rank, and the
  point should not be read as a steady-state period at all.
- `chain_health.interpair_gap_us` is the start-to-start median minus the pair-window median, measured
  once per trial: the per-pair cost OUTSIDE the published window (the harness's own two `record()`
  calls plus any inter-pair stall), and the in-result regression guard against the six-events defect
  above.
- `chain_health.settle_drift_us` measures the late-half minus early-half period median, once per
  trial, signed, cross-rank max-magnitude. `chain_drop` *assumes* the chain settled before the kept
  pairs. This is the proof, and an unconverged chain (or a device clocking down mid-chain) publishes
  its drift instead of a clean-looking period.

**Chained per-op medians and p99s are never published.** Without a host sync the ranks arrive at
each collective at slightly different times, and the resulting wait parks in whichever op window a
given rank happens to block in. This placement is stable per rank (so one rank's per-op numbers look
clean and convincing), arbitrary across ranks, with rank 3's dispatch long exactly where rank 5's
combine is, and bistable across runs of an identical configuration, while the sum, the pair period,
is conserved. A chained per-op median measures where the wait sat on that run rather than what the
operation cost. The cross-rank MIN is the one reduction that removes it, which is why the floors are
published and the medians are not. A p99 of a chained window would mix the same noise back in through
the tail.

**The published period always means one thing: free-running.** The chain lets ranks drift by up to
about one iteration, which is only sound where the receive plane tolerates it: every backend
double-buffers per dispatch, enforces strict pairing by contract, or completes each op on a reusable
handle, and DeepEP V2's **normal** mode (the one genuinely unaudited cell) was hand-probed with 256
un-synchronized pairs at T=128, EP8+EP16, both precisions (2026-08-06, pin `01dc3aaa`, the
then-current dgxc pool's hybrid GIN over RoCE, under the earlier overlap-on, multiple-reduction
flags; not yet re-probed under vLLM's): all passed, outputs finite, timed inputs unchanged,
cross-rank period agreement within 1 µs, and the chain-vs-synced gap is the size of the effect this
section is about (EP8 105.4 vs 125.4 µs BF16, 216.6 vs 272.3 µs FP8, while EP16 838 vs 863 µs and 820
vs 897 µs). A backend that cannot run free belongs behind a fix, not a measurement variant:
re-aligning ranks between pairs adds its own ~10 µs and removes the cross-pair overlap the measurement
exists to capture. That is a differently-defined quantity that must never share a column with the
free-running period.

### Timing Profile

Every measured component uses one fixed timing profile, defined once in `configs/sweep.json` and
baked into every scheduled case:

- 256 trials x 8 timed iterations = 2048 observations for the fresh-entry family.
- 4 chain trials x (128 free-running pairs - 16 dropped for pipeline fill) = 448 observations for
  the pair period and another 448 for the op floors, each trial running the two sibling chains
  (floors first, then the lean period chain). The trial count is lower than the fresh-entry family's
  because one call already yields 128 pairs, and matching it would multiply the case's wall clock
  without buying convergence.
- 32 synchronized full dispatch-stage-combine warmups before each available measured component at
  every trial/point, and before each chain trial.
- component measurement order rotates each trial (`trial_order`) so every timed component occupies
  every position in the sequence, over a per-trial-rotated token ladder, which the chain trials
  rotate the same way.
- per-iteration maximum latency across ranks before nearest-rank p50/p90/p95/p99 (the chained family
  reduces by median and minimum, as described in Chained Pair Period).

`measurement.sampling` carries both halves of that profile, because `sample_count` alone cannot be
decomposed back into them and a 128x4 chain is not the same measurement as a 512x1 one. Decode and
prefill identify the serving regime represented by one MoE-layer collective; they do not change the
timed primitive at an otherwise identical shape. Ascending through the ladder, each shape is
conditioned with 8 untimed full roundtrips (settling clocks, fabric, and buffer state) before it is
correctness-checked. All timing happens after every shape is warmed and checked, and conditioning
rounds are never measured or emitted.

### Vendor Tables

Comparing these figures against a vendor table is not like-for-like, in a knowable direction. Every
fresh-entry sample here is one event window per call, opened behind a device barrier that removes
entry skew (as DeepEP's pre-iteration sleep and MoRI's barrier do) and reduced across ranks by MAX,
so it keeps any host work inside the window and the slowest rank's exit. Vendor microbenchmarks
published for these same kernels variously time the named kernel only via a profiler (DeepEP, UCCL
low-latency), replay CUDA graphs (MoRI), average across ranks instead of taking the max (all of
them), and report the best of a config sweep (DeepEP V1, MoRI). Expect the headline to read above
such a table by at least the max-versus-mean gap; `cross_rank_min_us` and `chain_floor_us` are the
per-row figures to place beside one. The like-for-like checks so far all ran earlier contracts and
need re-running before they are cited: skew-excluded MoRI dispatch at 0.96x MoRI's own shipped
tuning-config best at the same shape and byte count, DeepEP V2 on B300 within 3% of DeepEP's
published 8x2 figure (under the overlap-on, multiple-reduction flags), and FlashInfer EP within 4% of
NVIDIA's published one-sided kernel across eight byte-normalized points, on rows whose combine
skipped the payload staging copy.

### Bandwidth

Rows carry bytes and latency, not bandwidth: `bandwidth.py` and the dashboard derive logical payload
bandwidth as

`wire_payload_bytes / measured_latency_seconds`

Every row carries two byte accountings and each excludes expert metadata, padding, and backend buffer
capacity. `byte_provenance` is the canonical comparable basis: rank-deduplicated token-rank
activations, one copy per unique (token, dest-rank) pair. `wire_byte_provenance` is what the kernels
actually move: identical to the canonical basis for every layout that deduplicates per rank (every
`normal` mode, MoRI's InterNodeV1LL, whose combine is an unweighted rank-sum, and NCCL EP's default
rank-major low-latency layout, which deduplicates per destination rank), and one copy per (token,
expert) assignment for the low-latency kernels that apply top-k weights inside combine (DeepEP V2,
UCCL-EP, and NCCL EP's expert-major layout under `COLLX_NCCL_LL_LAYOUT=expert-major`). For a token
whose experts share a destination rank the deduplicated count is a lower bound on those kernels'
traffic (34% low on nccl-ep's expert-major low-latency EP8 at T=128, measured 2026-08-30), which is
why bandwidth divides the WIRE basis; a rate derived from `byte_provenance` on such a row is a lower
bound, not the wire rate, and is not comparable across backends. `logical_copies` states each row's
copy counts and wire basis (`routed`, `assignments`, `wire`), so the two are never silently mixed.
BF16 moves 2 bytes per value with no scale payload. An FP8 dispatch moves 1 byte per value, plus
per-128-block FP32 scales, for every FP8 codec here: DeepEP V2, UCCL-EP, MoRI, NCCL EP's DS-FP8, and
FlashInfer EP, which carries them as a fourth dispatch payload (the low-latency kernels send FP32
scales even where they hand the receiver packed UE8M0 ones). Combine stays BF16, so the dispatch and
combine directions can carry different byte counts and the roundtrip is their per-field sum. Latency
(the headline) is measured directly and is unaffected. Algorithm bandwidth, bus bandwidth, wire
utilization, and physical-link utilization are not derived without a defined primitive model or
transport counters, and logical bandwidth must never be labeled physical bandwidth. The row's
`token_rate_at_latency_percentile` divides global tokens by each fresh-entry roundtrip percentile:
a lower-tail service rate at p99 latency, not a p99 percentile of an inverted rate distribution.

## Correctness

An implementation-independent oracle uses an expert-specific deterministic transform so wrong expert
routing cannot pass an identity roundtrip. For every rank and point it verifies:

1. destination rank/expert, source token, multiplicity, gate weight, and receive counts.
2. dispatched payload and metadata before timing.
3. combined output before timing.
4. unchanged semantic inputs through all timed samples.
5. dispatched payload/metadata and combined output again after timing.
6. the **free-running chain's own final combined output**, once per chain trial, against a drained
   pair through the identical dispatch→combine path.
7. the same full check as 2-5 once more against the state the free-running chain leaves behind.
8. under graph replay, that every timed capture's poisoned output was rewritten by its replay, and
   that a separate capture with staging inside it, replayed once to upload, then replayed again after
   its inputs are scaled by exactly 2 in place and dispatch's output, the staged combine input and its
   result are overwritten with 0xFF bytes, matches an eager drained pair run on the same scaled inputs
   (`cuda_graph_output_rewritten`, `cuda_graph_last_output_passed`). A replay that re-sent its
   captured data instead of re-reading its inputs fails it. The inputs are scaled back afterwards.

Every check is folded into `correctness.passed`, and any failed rank or point makes the case
`invalid` in the result it writes. Checks 6 and 7 exist because checks 2-5 only ever see drained
calls: without them the headline `pair_period` would come from a regime no oracle had inspected, and a
backend that transports correctly when drained but corrupts under back-to-back pairs would present as
the fastest cell in the suite. They answer different questions. Check 6 is a regime A/B, not an
oracle: after each chain trial's closing synchronize, outside every timed window, the chain's last
combined output is compared elementwise (the oracle's tolerance, not bit equality, since combine
kernels are not required to be order-deterministic) against a freshly drained pair through the same
code path, so it catches corruption the chain wrote into its own results. Check 7 reruns the full
expert oracle after the final trial, on the settled communicator (`benchmark_chain` ends
synchronized), so it catches state the chain corrupted for whatever runs next. Interior chain pairs
stay unvalidated by design: each pair overwrites its predecessor's output, and holding or reducing
every output would put device work (or ~O(iters × T × hidden) memory) inside the timed loops the
chain exists to keep clean. A defect confined to interior pairs that heals by the final pair is
outside this suite's evidence. Check 6 is reported as `correctness.chain_last_output_passed` (ANDed
across trials) and check 7 as `correctness.post_chain_state_passed`, always a boolean.

Check 6 gates **only where the chain stages per pair**, and is reported as `null` elsewhere. That
boundary is measured, not assumed. Wherever `stage_excluded_from_roundtrip` holds (every FP8 adapter
by default, since `stage_device_work` *is* the FP8 flag), the chain hoists staging out of the timed
loop, capturing one warm-up dispatch's stand-in and reusing it for all 128 pairs. Neither the chain's
final combine nor the drained reference then consumes an input matching its own dispatch, so they are
two differently mismatched pairs and nothing requires them to agree.

The A/B, on h100/deepep-v2/EP8, identical in every respect but the hoist:

| staging | `chain_last_output_error` | vs `COMBINE_REL_TOL` |
|---|--:|--:|
| hoisted (`fp8_consume=native`) | 31 – 93 | **1000× – 2966×** |
| per pair (`fp8_consume=dequant`) | `0.0` at every rung | bit-identical |

in **both** `normal` and `low-latency` mode (runs 31180411148, 31185184372, 31185233991), against a
BF16 control that is `0.0` because BF16 never hoists. So the difference was an artifact of the hoist,
not a transport defect: gating on it under the hoist reddened every FP8 leg fleet-wide for something
the harness does deliberately. Passing the chain's staged input to the drained pair was tried first
and is *not* sufficient: it makes the two share an input, but one that matches neither dispatch. Only
per-pair staging makes the regimes comparable.

The check keeps its teeth exactly where it has meaning: every BF16 row, and any FP8 row run under the
`dequant` hatch. `null` there means the question was not asked, never a comparison that ran and
failed. If this boundary is ever moved, move it on a measured magnitude: the verdict alone justified
two wrong calls in one day, and the number settled it in one probe each time.

What the null leaves uncovered is the intersection of three conditions: a corruption that manifests
only under free-running pairs (the drained oracles are blind by regime), leaves no state behind (the
post-chain oracle is blind), and lives in a path the BF16 sibling row does not exercise (its
still-gating check is blind). Anything short of all three still reds a gate. The DeepEP low-latency
256-rung corruption hit both precisions, so its BF16 rows would red `chain_last_output_passed`, and a
wedge trips the per-case hang guard. Note the hoisted chain also never consumes its own FP8 receive,
so the exposure covers dispatch-side corruption as well as combine-side, and `chain_health` is a
consistency guard, not a work guard: a regime defect that uniformly shortens the combine would publish
a fast period no surviving check contradicts. The standing probe for the intersection is the `dequant`
hatch: `CX_FP8_CONSUME=dequant` stages every pair from its own dispatch, restoring this check on FP8's
own free-running pairs. Run one whenever an FP8 chained period moves in a way its BF16 sibling does
not, and before first publishing a new FP8 backend. flashinfer-ep's BF16 stage is a pointer hand-off
and its payload staging copy runs inside combine, so its BF16 rows carry this check like every BF16
row (a bit-exact match is expected), while its FP8 rows are null like every hoisted FP8 row and the
`dequant` hatch reaches them.

Check 6 also publishes `correctness.chain_last_output_error`, the worst chained-vs-drained relative
error, cross-rank MAX, **reported whether or not the verdict passed**. Read it before reading the
verdict. The check assumes a regime defect lands orders of magnitude past `COMBINE_REL_TOL` while
ordinary kernel non-determinism stays well inside it, and that assumption is not free: FlashInfer EP
before 0.6.16 (the image's 0.6.8.post1 included) accumulates its combine in the payload dtype (a BF16
slot-tree) rather than FP32, so its rounding over a large receive plane is far coarser than a backend
reducing in FP32. A verdict alone cannot separate "the transport corrupted this" from "this tolerance
is too tight for this accumulator", and the two call for opposite responses. The magnitude is the
discriminator. A failure reported without it should not be read as evidence of corruption.

Every `normal` mode, MoRI's InterNodeV1LL and nccl-ep's rank-major low-latency layout use an
activation-only, unweighted rank-sum combine. The oracle builds each rank's gate-weighted expert
aggregate before combine and derives the expected combine from the values
actually communicated: each destination rank casts its FP32 aggregate to the payload dtype (BF16)
exactly as the adapter does, and `implementation.combine_reduction` names how those rows are then
reduced. `domain-fp32` (uccl-ep, MoRI, nccl-ep HT, FlashInfer from 0.6.16) is the two-level
reduction: ranks sharing a scale-up domain (NVLink/MNNVL) reduce in FP32, and each domain casts its
aggregate to BF16 for the scale-out send before those partials are summed. A group that fits in one
scale-up domain (`ep_size <= scale_up_domain`, including every EP8 case and the MNNVL EP16 cases) has
a single domain and no scale-out rounding; a multi-node EP16 group carries one BF16 partial per node,
and omitting that cast left multi-node EP16 ~0.048 off, above the gate (observed while DeepEP V2 still
ran multiple reduction). deepep-v2 forms no such partials: it runs vLLM's
`allow_multiple_reduction=False`, under which hybrid forwarders relay each scale-up rank's BF16 row
unreduced into the source's top-k slot (DeepEP at the pin, `hybrid_combine.cuh:453-494`) and the source
sums the first slot of each unique destination rank in FP32 (`combine_reduce_epilogue.cuh:74-95`). Its
rows therefore use `rank-fp32`, the model nccl-ep low-latency rank-major also uses, at every scope;
inside one scale-up domain the two models agree up to the final BF16 cast. `topk-slot-tree` is
FlashInfer before 0.6.16. All keep the gate tight (max elementwise relative error, denominator
clamped at 0.02, below `8 * 2^-8`, the residual accumulation-order ambiguity). On x86 EP16 the single
reduction moves one scale-out row per unique remote destination rank instead of one per remote node,
about 2.6x the combine scale-out bytes for uniform top-6 routing over 384 experts on 2x8; the
canonical `token-rank` byte basis already counts it that way.

Low-latency adapters other than MoRI and nccl-ep rank-major instead use a source-side gate-weighted
combine: the kernel multiplies each expert's returned message by that assignment's top-k weight, so
the adapter stages the UNWEIGHTED per-expert transform and a dedicated per-(source, expert)-slot
oracle derives the expected combine as the gate-scaled sum of per-expert BF16 messages, with no
per-domain intermediate, since the low-latency kernels reduce at the source rank. Those rows publish
`combine_reduction` at its default, which this model ignores. The delivered (source, expert)
assignment multiset and per-expert counts are checked against the routing trace, and the same tight
combine gate applies. Under FP8 dispatch the oracle applies the backend's exact per-token cast
round-trip to its semantic payload before both the dispatched-payload compare and the combine
expectation, so the payload match stays bit-exact and the same tight gate holds. The quantization is
modeled, not absorbed into a wider tolerance. Pre/post dispatch behavior is checked against canonical
source-token metadata and expected output. Native receive slots may be assigned nondeterministically,
so physical receive order is not treated as a correctness property.

## Result Document

One raw case document, committed at `results/<sku>/<case_id>.json`, carries `record_type:
"case-attempt"`, the sweep `version`, and a `generated_at` timestamp, and contains:

- `identity`: `case_id`, `attempt_ordinal`, `case_factors` (the SKU and the scheduled case,
  including backend, EP size, mode, precision, phase, suite, workload, and the topology coordinate),
  and `allocation_factors` (source SHA; the run id and run attempt are null for hand runs).
- `workload`: `cross_rank_consistent`, whether the routing trace was proven identical across ranks,
  and `ladder_measured`, `ladder_dropped` and `ladder_cap`.
- `measurement`: dispatch and combine dtype (the realized wire formats, with combine always BF16 and
  dispatch BF16 or the SKU's FP8 format), `combine_semantics`, `payload_unit` (`token-rank`),
  `sampling`, and the per-point `rows`.
- `implementation`:
  - `name` and `kernel_generation` ([Identity](#identity)).
  - `maturity`: `production` where an inference engine can select this configuration today (vLLM's
    `--all2all-backend` or SGLang's `--moe-a2a-backend`, or, for a library vLLM does not expose, the
    engine that does: TensorRT-LLM's `NcclEP` for NCCL EP's low-latency rank-major BF16 rows), and
    `candidate` for a real transport or mode we benchmark that no engine ships a selector for, whose
    numbers describe the library rather than a deployable configuration. uccl-ep stays `candidate`
    although its rows run vLLM's DeepEP configuration on UCCL's drop-in `deep_ep`, which is how
    engines deploy it, because no engine ships a UCCL selector. nccl-ep sets maturity per row, while
    the registry's `backend_maturity` keeps one value per backend (`candidate` for nccl-ep).
  - `fp8_consume`, `rank_alignment` (method and spin), `stage_excluded_from_roundtrip`,
    `chained_period` (always true), `cuda_graph_supported` (whether the adapter declares this mode
    graph-safe) and `cuda_graph_replay` (whether this row replayed).
  - `combine_reduction` and `library_version`: which reduction the oracle held the kernel to, and the
    installed library it was chosen from. deepep-v2 reports the wheel string at the pinned commit,
    e.g. `2.1.0+local@01dc3aaac82068020353dce2c302e38153c0bfaa`, since the patched build cannot embed
    a clean git hash; uccl-ep, a from-source build, names its pinned commit and build flags beside the
    wrapper version; nccl-ep and flashinfer-ep report their wheel versions and mori its installed
    build.
  - `library_config`: the adapter's `EPBackend.implementation_details()`, what it chose or the
    library resolved that changes what a row measures, null where an adapter records none. deepep-v2
    records `engine_contract` (`vllm`) in both modes and, in `normal`, its SM and QP counts,
    allocated QPs, LSA team, hybrid choice, vLLM's buffer flags, host-sync, expand and receive sizing
    (`sm_model_rdma_gbs` on EFA only); the shared low-latency path, on deepep-v2 and uccl-ep alike,
    records `receive_capacity`, `num_qps_per_rank`, `return_recv_hook`, `zero_copy` and `fp8_scales`,
    plus deepep-v2's NVLink and MNNVL flags; uccl-ep adds its verified `deep_ep` identity, buffer
    bytes, QPs, `is_intranode` and running CPU-proxy threads, with `Buffer.num_sms` and the dispatch
    and combine `Config`s as the constructor rounded them in `normal` mode; mori the launch tuple its
    handle realized, QPs, quant type, SDMA, heap and launch-config modes and the contract (`vllm`) it
    follows; nccl-ep `capacity_tokens_per_rank`, `zero_copy`, `num_sms` (`dispatch`,
    `combine`) and `engine_contract` (`tensorrt-llm`, or null for a library row); flashinfer-ep its
    combine contract (`payload_in_workspace`, `combine_output`), dispatch scale format and workspace
    sizing.
- `topology`: `device_product` (the GPU name torch reports), `placement` (`packed`),
  `gpus_per_node`, `nodes`, `scale_up_domain`, `scope`, `topology_class`, `world_size`, and three
  distinct transport fields, `scale_up_transport` and `scale_out_transport` (the components), plus
  `transport`, the derived summary that folds them into one string (`nvlink` scale-up-only,
  `nvlink-rdma` once a case scales out).
- `runtime`: the realized software stack, `vendor`, `framework` (the torch version),
  `accelerator_runtime` (the CUDA or HIP version torch was built against), and `collective_library`
  (`nccl`/`rccl` and the version actually loaded into the process), plus `gpu_clocks`, what each
  rank's GPU ran at over the timed passes. A clamped or throttled GPU looks exactly like a slow
  library, and clocks are not locked, so every case records them, through NVML on NVIDIA and amdsmi
  on AMD (`gpu_clocks.source`):
  - per rank, the SM clock read right after each point's timed bursts and after each chain (min,
    p50, max, sample count), the clock limit, the power limit and the highest temperature. A read
    right after a burst still shows the burst's clock: an MI325X held ~1.5 GHz for the first
    millisecond after a power-bound burst and only then climbed back toward its 2.1 GHz limit;
  - `throttle_fraction`, the share of the timed passes the clock was held down, by cause: NVML's
    violation counters (power, thermal, sync boost, board limit, reliability) and amdsmi's residency
    accumulators (power, thermal, HBM and VR thermal, PROCHOT), null where the GPU does not report a
    cause, with the worst rank per cause in `throttle_fraction_max`;
  - on NVIDIA, the clock-event reasons seen other than idle;
  - `sm_clock_ratio_min`, the slowest rank's median clock over the fastest rank's, null unless every
    rank reported. Ranks run identical work, so their GPUs should agree;
  - `slow_gpu`, true when that ratio is below 0.95, when a rank reported a hardware, thermal or power
    brake slowdown, or when a rank spent at least 5% of the timed passes throttled by any cause but
    sync boost. The last two catch a tray clamped as a whole, which keeps the ratio at 1; only the
    last works on AMD, which reports no clock-event reasons. The busiest prefill cases measured spent
    0.01-0.02% power-capped. The run log and `summarize.py` call such cases out.

  Reads happen only between timed bursts, never inside a window. A probe that cannot start leaves the
  values null and records `unavailable_reason`: the writing rank's own failure, or the ranks whose
  probe did not start. It never fails the case.
- `provenance`: the mounted image tag, the source SHA, and `source_dirty` (whether tracked files under
  `collectivex/` had uncommitted edits; such a result is never published).
- `outcome`: `status` (`success` or `invalid`) and `reasons`.

Each `rows` entry carries point latency (`components`, graph-replayed by default where supported,
plus `components.pair_period`, `chain_floor_us` and `chain_health`), the skew bracket
(`cross_rank_min_us`, `cross_rank_spread_us`), the alignment's `rank_alignment` counts, `correctness`,
byte accounting (`byte_provenance`, `wire_byte_provenance`, `logical_copies`), the receive counts,
routing load and fanout, and the token rate; per-point statistics are summarized in place, not
emitted as separate documents. Each run of a case writes exactly this one raw result document,
replacing the case's previous file, while unsupported or never-run cells produce no synthetic record.

## Identity

Identifiers are readable factor strings:

- `case_id`: `{sku}-{backend}-{workload}-{mode}-{phase}-ep{ep}-{routing}-{precision}`, each factor
  slug-normalized. It names the result file, so a re-run replaces the committed result and git
  history keeps the one it replaced.
- `attempt_ordinal`: `COLLX_ATTEMPT_ID`, 1 unless set.

These IDs let a consumer group matched configurations and separate distinct ones. The backend does
not itself compute cohorts, controlled comparisons, sensitivity pairs, eligibility, or
recommendations. A reader decides which cases to surface and how to compare them.

`implementation.kernel_generation` names the kernel family a row ran, followed by the harness's
regime suffix (`-cudagraph-spin-ar` or `-eager-spin-ar`). Rows of one backend pool only within one
generation:

| Backend | Families |
|---|---|
| deepep-v2 | `v2-elastic-buffer-vllm-nosync`, plus `-relaxed-ordering` on h200 EP16; `legacy-buffer-ll-hook`, plus `-ue8m0` for Blackwell FP8 |
| uccl-ep | `uccl-legacy-buffer-vllm-ht`; `uccl-legacy-buffer-ll-hook`, plus `-ue8m0` for B200 FP8 |
| mori | `intranode-vllm`, `inter-node-v1-vllm` and `inter-node-v1ll-vllm`, each plus `-sdma` when `MORI_ENABLE_SDMA` was set on |
| nccl-ep | `nccl-ep-v02-ht-routed-zc-static` (plus `-gum1` when graphed), `nccl-ep-v02-ll-rm` (plus `-zc` on the eager window path) or `nccl-ep-v02-ll-em`, then `-fp8-ds-e3m4` on FP8 rows and `-cap<N>-<wheel version>` ([nccl-ep.md](nccl-ep.md)) |
| flashinfer-ep | `flashinfer-mnnvl-one-sided-staged-<wheel version>` |

Backend source pins live in `runtime/common.sh` and are enforced by exact fetched-commit comparison.

## Execution Isolation

Every non-MNNVL scale-out case needs the registry's `rdma_devices` for its operator-approved fabric,
and the launcher rejects a missing or malformed selector. `socket_ifname` pins the cross-node socket
interface; without it each node's default-route interface is used (h100-dgxc, h200-dgxc, b300).
After allocation every scale-out node must show its socket interface up and an ACTIVE port (with a
non-empty GID at the configured index, where one is set) on every listed device before backend
initialization; HCAs are never guessed. Every shard first clears the NCCL, NVSHMEM, MoRI and UCCL
selectors it inherited from the launching shell, and host-side utility steps receive only `HOME`,
`PATH`, `USER`, `XDG_CACHE_HOME` and `ENROOT_CACHE_PATH`. Scale-up and MNNVL cases stop there,
except that `single_node_rdma_devices` (an operator field no registry entry sets) exports
`NVSHMEM_HCA_LIST` on one-node runs, for a library that initializes RDMA even on one node, as
DeepEP's legacy low-latency Buffer does. On the IB and
RoCE pools, scale-out NVIDIA forces `NCCL_NET=IB`, so a socket fallback fails instead of being
mislabeled as RDMA, while AMD leaves net-plugin selection to RCCL; NCCL/RCCL and UCCL take the device
list with NCCL's `=` exact-match prefix, so `mlx5_1` cannot also select `mlx5_10`, and NVSHMEM and
MoRI get the same list. `ib_gid_index` applies only when every selected port reports an Ethernet link
layer, where it selects the operator-approved RoCE GID; native InfiniBand keeps explicit HCA and
service-level pinning but leaves the RoCE-only GID override unset so NVSHMEM and NCCL use the native
LID path, and mixed Ethernet and InfiniBand lists are rejected. `rdma_service_level` sets the
NVSHMEM, NCCL, MoRI, UCCL and DeepEP (`EP_OVERRIDE_RDMA_SL`) service-level variables, and `rdma_traffic_class` the MoRI and UCCL
traffic-class ones. Scale-out also pins `NCCL_IB_MERGE_NICS=0`: NCCL's dual-port fusion collapses each
card into one fused device, and any fused device disables NCCL GIN, which the DeepEP V2 EP16 hybrid
path requires. A rail-isolated fabric (`rail_isolated`, per-port rail subnets with no cross-rail
routing) adds `NCCL_CROSS_NIC=0`, because cross-NIC pairs black-hole at QP RTR there. Selectors come
from the tracked platform registry, whose `network` block wins over an operator document's network
keys, and appear only in mode-0600 private logs.

The EFA pools (`rdma_fabric: "efa"`: b300, a p6-b300.48xlarge pool with 16 EFA devices per node, and
h100-dgxc, a p5.48xlarge pool with 32) are not verbs fabrics to NCCL: their ports report an
unspecified link layer, there is no GID table, service level or traffic class, and NCCL reaches them
only through the aws-ofi-nccl libfabric plugin. The hosts' enroot hooks need not provide it (b300's
bind only `/dev/infiniband`), and the image's own EFA user space (rdma-core 50, libfabric 1.17)
predates the hosts' `efa.ko`, so multi-node shards bind the host's `/opt/amazon/{efa,ofi-nccl}` and
rdma-core read-only and put them first on the loader path. The probe accepts the unspecified link
layer for the listed devices (which must still be ACTIVE on every node), the IB selector family
(`NCCL_IB_*`, `NVSHMEM_HCA_LIST`/IBGDA, MoRI, UCCL) stays unset, `NCCL_NET_PLUGIN=ofi` and
`FI_PROVIDER=efa` select the plugin, NVSHMEM is pointed at its libfabric transport, and backend setup
fails unless the host libfabric reaches every listed device from inside the container.

NCCL GIN then rides the plugin's GIN through its CPU proxy (`NCCL_GIN_TYPE=2`), which requires
GDRCopy 2.5+ (`gdrdrv` loaded, `libgdrapi` present) on the node: without it the plugin's GIN init
fails, NCCL disables the Device API for the communicator, and `ncclEpCreateGroup` returns
`ncclInvalidUsage`. The type is pinned because NCCL otherwise picks the plugin's GPU-initiated EFA GDA
backend (type 5, aws-ofi-nccl 1.21), which has only weak signals, while nccl.ep and the pinned DeepEP
both require strong ones and fail device-communicator setup. Before 1.21 it picked its built-in GDAKI
on the nodes' two ConnectX-7 IB ports, which are not the GPU fabric, and most ranks failed window
registration while the rest hung. Where the host plugin predates 1.21 (h100's 1.20), multi-node
shards build aws-ofi-nccl 1.21.1 into the backend cache and load it instead: 1.20's proxy applies
every signal with an inline GPU read-modify-write (1.3-4.5x on b300 EP16). Proxy GIN still bounds
these rows well below the IB pools. b300's hosts carry the EFA GDA stack (EFA installer 1.50.0), but
besides weak-signal EP kernels, GPU-initiated GIN there needs NCCL 2.31 (2.32's GIN device ABI is
newer than aws-ofi-nccl 1.21.1 accepts), and even a weak-signal DeepEP fork could not register the
EFA doorbells with CUDA on the nodes' 580 driver; h100's p5 has first-generation EFA, which cannot
run it at all.

The single-slurm pools (H100, H200, B200, B300) also reject an allocation holding a GPU with an
active thermal slowdown or above 90 °C and retry on other nodes, up to three allocations: every
collective is a barrier, so one clamped device paces every rank (a B200 with one GPU at 120 MHz ran a
case 17x slower). b300 additionally checks that CUDA initializes and sees all eight GPUs.

Repository staging copies `collectivex/` into a mode-0700 execution child of the operator's
`stage_dir` (or, when none is configured, beside the image under `squash_dir`), which must be visible
at the same path on the login node and every allocated node. The parent process resolves the exact
execution child before copying. Backend preparation then runs from that tree on every allocated node.
Cleanup waits for confirmed allocation teardown and removes only that child. Pinned sources (DeepEP V2
with its `fmt` submodule, and UCCL-EP) are fetched on the submit host before allocation, at the exact
pinned revision, because compute nodes may have no network.

## Image Pinning And Build Isolation

Enroot imports configured container tags into one squash per (image platform, image reference),
staged once per cluster (once per node where the squash dir is node-local, as on mi355x) and reused
by every run of the same image. Freshness is decided against a
digest sidecar: the container registry's manifest digest is resolved from the submit host at launch,
and a resolved digest that differs from the sidecar stamp means the tag moved upstream and forces a
fresh import. An unresolved digest (no registry egress, a transient blip) reuses whatever is staged;
`COLLX_IMAGE_REFRESH=1` forces an update in that case, discarding only files staged before the launch
so concurrent shards still import once. Validity is still proven per use (`unsquashfs -l`) before any
reuse, and a squash another account owns is reused when readable and current but never replaced.

Backends the image does not ship live in a mode-0700 cluster-local backend cache mounted only as
`/cx-cache`: DeepEP V2 and UCCL-EP, built from source at their pins, the `nccl-extensions` wheels,
and aws-ofi-nccl 1.21.1 where an EFA host's plugin predates it. Each cache directory's name binds
the CPU architecture, the image and the pin, plus the GPU architecture for the GPU builds (for
DeepEP V2 also its torch and NVSHMEM specs; for DeepEP V2 and UCCL-EP a build-recipe generation),
and the cache is never
committed. The DeepEP V2 build applies one local patch, the opt-in relaxed-ordering switch on its
GIN window; UCCL-EP's AMD build swaps managed allocations for pinned host memory, because
`hipMallocManaged` fails on these CDNA nodes. MoRI and FlashInfer ship in the image and are only
import-checked. Per-execution stages remain isolated and disposable, and runtime probes fail closed
before reuse. The running account is inside the trusted cluster boundary: this cache guards against
stale or accidental mutation, not hostile same-UID jobs. Only an unpublished partial build may be
reset automatically. A cache that fails integrity or runtime checks is left intact and rejected so a
concurrent allocation cannot lose files it is using.

## Result Delivery

There is no results server or managed object store. Each shard runs one allocation and its per-case
result documents are copied to `results/<sku>/` in the checkout that ran it. A case counts as
successful on the benchmark's own return code, and failed or unsupported cells produce no synthetic
record. Results reach the dashboard only through a reviewed commit to `main`; the unit-test job checks
every committed result against `results/cases.json`, rejects ones measured from uncommitted code, and
rejects IP addresses and host-identifying fields.
