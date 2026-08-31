# CollectiveX

CollectiveX is an experimental inference-communication benchmark. Its EP suite measures MoE
dispatch, combine, and paired roundtrip latency across EP libraries and accelerator systems; its
KV-transfer suite measures disaggregated-serving KV-cache handoffs across transfer libraries and
fabrics. Both upload neutral result artifacts.

CollectiveX schedules benchmarks, executes them on real allocations, and uploads the neutral
artifacts each run emits. It does not validate those artifacts, promote, rank, recommend, select, or
decide what a consumer displays. Any downstream display or comparison is the consumer's
responsibility. The full measurement methodology is in [docs/methodology.md](docs/methodology.md).

## Execution Profile

The workload uses packed placement and one pinned `fixed-profile` resource configuration per
backend/topology; there is no tuning sweep. Combine is always BF16; dispatch precision is a swept
dimension — a BF16 control plus, on every backend whose FP8 dispatch is supported upstream
(DeepEP V2, MoRI, UCCL-EP, FlashInfer EP), an FP8 dispatch, caller-prequantized in `normal` mode (in
`low-latency` the DeepEP and UCCL-EP kernels quantize internally from BF16; MoRI stays
caller-prequantized; FlashInfer has no `low-latency` path). That caller-side quantize is charged
inside the measured dispatch, because a production forward pass pays it on the critical path. NCCL EP
is BF16-only this release, so it emits the control alone. Coverage is uniform routing only. Cases run
in one of two modes:

- `normal` uses `layout-and-dispatch-v1`, rank-deduplicated token payloads, and activation-only,
  unweighted rank-sum combine. It runs the full decode and prefill ladders.
- `low-latency` uses each backend's decode-optimized kernel family: on DeepEP the legacy
  `deep_ep.Buffer` IBGDA `low_latency_dispatch`/`low_latency_combine` (a per-expert padded receive
  and a source-side gate-weighted combine); on UCCL-EP the same legacy `Buffer` low-latency kernels,
  which at the scoped EP8 run `cudaIpc` over NVLink rather than its CPU-proxy transport; on MoRI the `IntraNodeLL` kernel (single-call,
  pure-intranode, same compact layout and unweighted rank-sum combine as `IntraNode`). It is a
  decode-phase-only, per-SKU-capability-gated addition whose runnable set differs from `normal`'s, so
  it is enabled from each SKU's `ll_backends` registry entry (currently DeepEP V2 at EP8 on
  H100/H200 and at EP8 *and EP16* on B200 — the nscale bare-metal pool, whose gdrdrv-backed IBGDA
  over native IB is what a low-latency scale-out needs and no virtualized pool has — and on
  GB200/GB300, whose EP16 stays inside the MNNVL scale-up domain;
  MoRI EP8 on MI300X/MI325X/MI355X; UCCL-EP EP8 on H100/H200/B200 only — UCCL's low-latency host
  assert `kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group` cannot hold on AMD, where
  `kNumMaxWarpGroups` is 16, since upstream raised `kNumMaxTopK` 9 -> 16 (uccl#1016, 2026-07-13) and
  our pin is six days later; the product is 16 for every CU count, so this is a dated regression
  rather than a hardware limit, and the AMD SKUs keep UCCL-EP normal mode without LL —
  and NCCL EP EP8 on all six NVIDIA SKUs, restored once the single-handle fix removed the
  [NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303) signal aliasing that had wedged them.
  B300 carries the `candidate` NCCL EP as its *only* low-latency row, so it has no production
  decode coverage; DeepEP V2 emits no LL row at all on B300 (the IBGDA address-handle wall in the
  backend table below), and `_ll_runnable` adds only runnable cells, so that wall is prose here
  rather than a classified matrix row).
  Scoped single-node EP8 runs over the intra-node NVLink/XGMI
  low-latency path (no `/dev/gdrdrv` needed — validated on H200 with it absent); NVSHMEM/IBGDA on the
  wire carries payload only on a multi-node scale-out (EP16) run; the legacy Buffer still
  self-enables IBGDA at EP8, which is why B300 fails there.

Cases use a fixed timing profile from `configs/sweep.json`: 256 trials x 8 timed iterations (2048
samples per component) with 32 synchronized full roundtrip warmups before each measured component at
every trial/point. Component measurement order rotates each trial so every timed component occupies
every position in the sequence; each iteration takes the cross-rank maximum before nearest-rank
p50/p90/p95/p99. A keyed BLAKE2b counter produces
byte-identical routing and gate weights on every runtime.

Those components all measure **fresh entry** — the GPU is drained around every timed window — the
latency of an idle pipeline, not what a decode loop pays. So every row also carries the **chained
pair period**: 4 trials x (128 dispatch→combine pairs issued back-to-back with no host sync, first
16 dropped as pipeline fill) = 448 observations, reduced across ranks by median. Each trial runs
**two sibling chains** — a floors chain carrying only per-op events, then a period chain carrying
only the outer pair events — because the first version's single six-events-per-pair chain charged
its four inner `record()` calls into the period wherever the device outran the host, publishing a
~flat 10–30µs host constant as transport (+20–38% at T=1, fleet-wide). `components.pair_period` is
the headline latency for every row that carries one (released 2026-08-06 after the b200/h200/gb200
hand references were confirmed against two-pass fleet artifacts), and `summarize.py` footnotes what
its starred columns hold either way. The floors chain publishes `chain_floor_us`, the
cross-rank minimum of each op's window; the period chain also yields `chain_health.pair_spread_us`
(cross-rank cadence proof), `interpair_gap_us` (the per-pair cost outside the published window — the
regression guard for instrumentation creeping back in, and the discriminator that keeps a
sync-dominated `period − Σfloors` gap from being misread as that defect returning; see the
methodology's `chain_floor_us` bullet for what each sign of that residual means) and
`settle_drift_us` (late-half minus
early-half period — the convergence proof `chain_drop` otherwise merely assumes). Chained per-op
*medians* are never published: inter-rank wait parks in whichever op window a rank blocks in, stable
per rank, arbitrary across ranks, and conserved only in the pair total. Nothing existing was renamed
or re-meant and the sweep `version` stays 1, so consumers key on the presence of
`components.pair_period`.

The chained regime is checked twice over: each chain trial's own final combined output is compared
against a drained pair through the identical code path (`correctness.chain_last_output_passed`,
with the size of any difference in `correctness.chain_last_output_error`), and the full oracle runs
once per ladder point against the state the chain leaves behind
(`correctness.post_chain_state_passed`). The second always gates; the first gates only where the
chain stages per pair, and is `null` where staging is hoisted out of the chain — under the hoist
neither regime combines an input matching its own dispatch, so the two are not comparable. That
boundary was measured rather than assumed: identical h100 cases differ by 1000×–2966× the combine
tolerance with the hoist and by exactly zero without it. See the methodology's Correctness section. The null is a deliberate, bounded gap — an FP8-only,
free-running-only, stateless corruption would red no gate — and the `CX_FP8_CONSUME=dequant` hatch
is the standing probe for it.

`roundtrip` means dispatch then combine — the transport — in every row. Expert-output staging sits
outside it and is reported separately as `stage`; under FP8 that component is harness scaffolding
standing in for the expert GEMM, which in production consumes FP8 operands natively rather than
materialising a BF16 copy, so `stage` must not be summed into a total or compared between backends.
Rows measured before this change carried the staging copy inside the chain for MoRI BF16 and
FlashInfer BF16, and the sweep `version` stays 1 across it, so
`implementation.stage_excluded_from_roundtrip` and whether a `stage` component is present are the
only way to tell the two generations apart. See [docs/methodology.md](docs/methodology.md) for the
full contract.

Correctness is checked against an implementation-independent oracle that reproduces the backend's
two-level reduction — intra-scale-up-domain FP32, then a BF16 cast of each domain's partial for the
scale-out send. The combine gate is a tight max elementwise relative error below `8 * 2^-8`
(denominator clamped at 0.02), which holds across scale-up and multi-node scale-out topologies
alike. Under FP8 dispatch the oracle applies the same per-token cast round-trip to its semantic
payload, so the dispatched-payload compare stays bit-exact and the combine gate is unchanged — the
quantization is modeled, not tolerated. Any failed rank or point makes the case ineligible in the
result it writes.

The matrix covers H100, H200, B200, B300, GB200, GB300, MI300X, MI325X, and MI355X. `sweep_matrix.py` materializes
the requested SKUs, backends, EP sizes, and token ladders, then extracts strict per-shard controls
and rejects missing, stale, malformed, or altered shard controls. `--only-sku`, `--exclude-skus`,
`--ep-sizes`, and `--precisions` select a subset; the matrix is generated per dispatch, with no
frozen digest or locked case count.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

Physical host count does not determine scope: both GB topologies stay inside one 72-GPU MNNVL
scale-up domain.

| Backend | Engine availability | Current scope |
|---|---|---|
| DeepEP V2 | `production` — vLLM `--all2all-backend deepep_v2`, SGLang `--moe-a2a-backend deepep` | `normal` mode is PR #605 `ElasticBuffer` plus exact upstream #630 and #640 fixes: LSA for scale-up and GIN for x86 EP16 scale-out. FP8 dispatch via `use_fp8_dispatch` (blockwise e4m3fn) alongside BF16. `low-latency` mode is the legacy `deep_ep.Buffer` IBGDA decode kernels (per-expert padded layout, weighted combine, `use_fp8` e4m3fn), decode only — EP8 wherever enabled, plus EP16 on GB200/GB300 (inside the MNNVL domain) and on B200's nscale bare-metal pool (IBGDA over native IB rails with `/dev/gdrdrv`; the prior virtualized b200 pool could never run it). B300 is an unsupported coverage row in `low-latency`: the legacy Buffer self-enables NVSHMEM IBGDA even for a single-node EP8 run, and on B300 address-handle creation fails (`ibgda.cpp:2234 Unable to create ah`), rc255 on all eight ranks. `NVSHMEM_DISABLE_IB=1` does not help — the Buffer re-enables IBGDA regardless, and the run fails identically with it set and unset (measured on b300-002 and b300-011) |
| MoRI | `production` — vLLM `--all2all-backend mori_*`, SGLang `--moe-a2a-backend mori` | `normal` mode uses the direct `IntraNode` kernel for scale-up EP8 on every CDNA SKU. EP16 is an unsupported coverage row on all three: the adapter pins `InterNodeV1` over 2x8 XGMI + RDMA, but its combine corrupts at the transport layer (ROCm/mori#475), so the registry ships `mori: [8]` and no EP16 case is dispatched. `low-latency` mode selects the `IntraNodeLL` decode kernel (single-call, pure-intranode, same compact layout and unweighted combine as `IntraNode`), decode/EP8 only. FP8 dispatch is caller-prequantized (per-SKU e4m3fnuz on gfx942, e4m3fn on gfx950); combine stays BF16 (`quant_type=none`) alongside BF16 dispatch |
| UCCL-EP | `candidate` — no engine exposes a UCCL-EP selector | [UCCL](https://github.com/uccl-project/uccl) EP: a drop-in, API-identical DeepEP replacement whose CPU proxies issue GPUDirect RDMA over plain `libibverbs` (no NVSHMEM/IBGDA), with software message ordering, atomics, and flow control; scale-up is single-node `cudaIpc` over NVLink/XGMI (never MNNVL). `normal` mode is the legacy `Buffer` `dispatch`/`combine` (unweighted rank-sum); `low-latency` reuses the legacy `low_latency_dispatch`/`low_latency_combine` decode kernels (weighted combine), decode/EP8 only. FP8 dispatch is caller-prequantized in `normal` mode (blockwise e4m3fn, per-SKU e4m3fnuz on gfx942); in `low-latency` mode the caller sends BF16 and the decode kernel quantizes to e4m3 internally (`use_fp8`). Combine is BF16. Runs on NVIDIA and AMD (H100/H200/B200 + MI300X/MI325X/MI355X), EP8 scale-up. Cross-node EP16 is functional (the internode RDMA path connects and the light case passes correctness) but its CPU-proxy throughput overruns the standardized per-case wall-clock budget on heavy token counts, so EP16 is an unsupported coverage row for now |
| NCCL EP | `candidate` — NVIDIA's own library, but no engine exposes an NCCL-EP selector | [NCCL EP](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep): NVIDIA's native MoE dispatch/combine on the NCCL Device API — LSA (NVLink load/store) intra-node, GIN (GPU-Initiated Networking) inter-node — driven through the `nccl4py` bindings. `normal` mode selects the `HIGH_THROUGHPUT` algorithm (FLAT `[N, hidden]` receive, unweighted rank-sum combine); the `LOW_LATENCY` algorithm carries an EP8 `ll_backends` row on all six NVIDIA SKUs, restored once the single-handle fix removed the NVIDIA/nccl#2303 signal aliasing. That LL decode ladder is clamped to T<=128, below its 256-slot receive: `nccl_ep`'s combine recv pipeline is a port of DeepEP's pre-#642 kernel and is missing the same shared-memory fence before `mbarrier_arrive`, which corrupted T=256 on GB300 in 1 of 5 executions — bimodal, healthy rows at 0.0039 relative error against 0.4704 on the failure. The fence is absent at NVIDIA/nccl master, so it is unfixed upstream. The clamp lowers exposure and is **not** a safety boundary: the fence is missing on every combine recv and T=256 is merely the rung with the most pipeline iterations, so lower rungs are less likely to hit the race rather than immune. Restore when a fixed wheel ships BF16 only: `contrib/nccl_ep/RELEASE.md` says "No FP8 support", so no FP8 case is emitted. That note is worth re-testing rather than trusting — the C library at our pinned commit does read `inputs->scales` and switch on e4m3/e5m2, the two documented FP8 exclusions are expert-major layouts we do not use, and `NVIDIA/nccl` has not moved since 2026-06-11 while `NVIDIA/nccl-extensions` has replaced that row outright. NVIDIA-only and CUDA 13 only. EP8 scale-up on H100/H200/B200/B300 plus EP8 and EP16 on GB200/GB300, where EP16 stays inside the MNNVL scale-up domain. x86 EP16 scale-out is an unsupported coverage row: the cross-node GIN path faults inside `nccl_ep.cc` identically on RoCE and IB across four SKUs, a GDAKI limit rather than a fabric-selection one |
| FlashInfer EP | `production` — vLLM `--all2all-backend flashinfer_nvlink_one_sided` | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `MoeAlltoAll`: TensorRT-LLM's one-sided MNNVL all-to-all, where each rank writes tokens straight into its peers' workspace windows and combine reads them back — no send/recv pairing and no NVSHMEM. `normal` mode only (there is one kernel family; no separate decode path), and GB200/GB300 only, since the transport is MNNVL. FP8 dispatch is caller-prequantized blockwise e4m3fn, carried as a fourth dispatch payload alongside its per-128-block FP32 scales, with the combine plane forced to BF16 — the C++ `toNvDataType` accepts only fp16/bf16/fp32 for combine, so an FP8 combine buffer would raise rather than corrupt. EP8 and EP16, both inside the scale-up domain. Unlike every other backend here, its combine accumulates in the PAYLOAD dtype rather than FP32: wheels before 0.6.16 reduce the top-k contributions with a pairwise BF16 tree that rounds at every level, so the oracle models that reduction directly (`combine_reduction = "topk-slot-tree"`) instead of widening the tolerance. 0.6.16 moved the accumulator to FP32, and the adapter switches models on the installed version |

DeepEP V2 means the `ElasticBuffer` implementation introduced by
[DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605), not a newer legacy `Buffer` build.
The pinned source is upstream `main`, which contains #605 along with
[PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) (fixes pure scale-up initialization when
GIN is unavailable), [PR #640](https://github.com/deepseek-ai/DeepEP/pull/640) (stops NCCL
shared-memory mappings being misclassified as duplicate NCCL libraries), and
[PR #642](https://github.com/deepseek-ai/DeepEP/pull/642) (the low-latency combine fence that fixes
the Blackwell top-rung corruption of
[issue #700](https://github.com/deepseek-ai/DeepEP/issues/700)), which the previous pin — the #630
head on the pre-merge #605 branch — predated. Scale-up cases request NCCL Device API LSA and fail closed
unless the realized LSA team covers the full EP world. x86 EP16 scale-out cases instead require the
hybrid path with GIN, two logical scale-out domains represented by two physical RDMA ranks, and eight
scale-up ranks per domain; GB EP16 remains MNNVL scale-up and therefore uses LSA. Whether a given
SKU/backend/EP cell is attempted is a capability fact; whether it succeeded is decided by the
benchmark's return code.

## KV-Cache Transfer Suite

`kv-transfer` legs run 2 nodes x 1 GPU — the per-worker prefill/decode pair a
disaggregated deployment actually forms — and move bursts of 1 to 32 concurrent
requests' paged KV as per-request layer-major descriptor lists over seed-keyed
random block tables (the post-fragmentation layout vLLM and SGLang post; a
burst posts every request's prepped transfer, then awaits them all), plus one
contiguous bulk row as the wire-speed ceiling. The workload is transcribed from
what vLLM allocates for the model it serves: `kv-dsv4` = DeepSeek-V4-Pro's mixed
cache (30 Compressed Sparse Attention layers at 4 tokens per 576 B entry plus
their 132 B indexer entries, 31 Heavily Compressed Attention layers at 128
tokens per entry, and the 128-token sliding-window cache on all 61 layers; fp8
by architecture); ISL
8k to 512k at page sizes 16 and 64 tokens; `pull` (READ, vLLM NixlConnector) and `push` (WRITE,
SGLang disagg) both timed from the initiator with offset-pattern verification on
the destination pool in both directions. Backends: `nixl` (what Dynamo, vLLM,
and SGLang ship), `mooncake` (NVIDIA-only; the wheel links libcuda at import),
and `mori-io` (AMD's native engine) where the registry's `kv_backends` map
enables them; no entry, no legs, mirroring `ll_backends`. A backend entry may
restrict ops, pin an image, or set a NIC filter (mooncake on mi355x is
push-only from AMD's atom-dev image over the GPU-paired Pollara NIC; upstream
ionic RDMA READ is broken; both b300 backends are pinned to one rail —
mooncake because the image's engine draws peer NICs blindly and cross-rail
draws stall ~1 s, nixl because UCX's own two-rail READ selection is unstable
across runs while one rail holds 49 GB/s at p95/p50 ≤ 1.01 — so b300 kv rows
are one-rail measurements). Fabrics: `rdma`
(torch pools) and, on GB racks, `mnnvl` (cuMem FABRIC pools; see the
methodology for the bulk-vs-paged lane inversion that row exists to publish).

## Workflow And Artifacts

`.github/workflows/collectivex-sweep.yml` has two jobs. `setup` generates a public-SKU matrix
(`backend`, `only_sku`, `exclude_skus`, `ep_sizes` inputs) and uploads the matrix.
`sweep` extracts a strict ignored `.shards/<id>.json` control per matrix entry, executes one
allocation per shard, fetches pinned DeepEP source before allocation when required, and uploads the
result artifacts with `always()` so a red or partial run still uploads.

Each shard emits per-case result JSON and a small mechanical summary. A case counts as successful on
the benchmark's own return code; there is no completeness or privacy validation step, and failed or
unsupported cells produce no synthetic record. No step promotes a run,
builds a dataset, or advances a channel; the neutral artifacts are the output. A consumer downloads
them and decides what to display.

No operator credentials are passed to the workflow or uploaded; runner-local overrides and any
selectors stay on the runner. Per-step runner logs are kept on the runner for postmortem, and
result artifacts carry only the fields listed in the methodology.

## Runner Configuration

Each SKU's Slurm and storage values come from its tracked baseline in the registry. An optional
runner-local JSON document at `$XDG_CONFIG_HOME/inferencex/collectivex.json` or
`COLLECTIVEX_OPERATOR_CONFIG` overlays that baseline per field; a runner with no registry entry, an
unknown field, and non-JSON input all fail closed, and configuration is never evaluated as shell.
Duplicate JSON keys are NOT rejected — `json.load` keeps the last silently — and runner keys other
than the one being resolved are not validated, so a typo'd SKU name is ignored rather than
reported. GHA passes no operator secret, so a SKU runs entirely from its tracked baseline unless a
runner-local document is present.

All public per-SKU platform data lives in the tracked `configs/platform_config.json` registry:
architecture/product, container image and platform, fixed placement, launcher, runnable backend/EP
pairs, the scale-out `fabric` identity (NIC and switch — so same-GPU clusters on different fabrics
are distinct entries, e.g. a second b200 cluster), tracked operator defaults, and scale-out RDMA
selectors. Operator documents can override the defaults. Launchers
declare and check the fields they actually require. `sweep_matrix.py` derives EP topology from the
placement fields; the sweep includes every registered SKU by default.

Every selected non-MNNVL EP16 placement additionally requires `socket_ifname` and `rdma_devices` for
its operator-approved fabric; optional `ib_gid_index`, `rdma_service_level`, `rdma_traffic_class`,
and `rail_isolated` are also allowlisted. Service level and traffic class are mapped into MoRI's
RDMA/IO QoS environment.
CollectiveX does not heuristically select a management route or HCA. After allocation, every
non-MNNVL scale-out node must prove that all configured interfaces and active HCA ports exist before
backend setup. Scale-up and MNNVL jobs clear these overrides. Scale-out NCCL/RCCL is pinned to `IB`
with exact-match HCA selectors so a socket fallback fails instead of being mislabeled as RDMA.
Scale-out also disables NCCL dual-port NIC fusion (`NCCL_IB_MERGE_NICS=0`): a fused device disables
NCCL GIN, which the DeepEP V2 EP16 hybrid path requires, and a rail-isolated fabric
(`rail_isolated=1`, e.g. B300's multi-plane RoCE) additionally sets `NCCL_CROSS_NIC=0`.

`ib_gid_index` is applied only when every selected HCA port reports an Ethernet link layer, where it
selects the operator-approved RoCE GID. Native InfiniBand profiles retain explicit HCA and service
level pinning but leave the RoCE-only GID override unset so NVSHMEM/NCCL can use the native LID path.
Mixed Ethernet and InfiniBand HCA lists are rejected.

`stage_dir` is a pre-existing, runner-owned, non-symlinked base outside the checkout and workflow
workspace. It is not group- or world-writable and is visible at the same path on the runner and every
allocated node. Jobs create only a marked mode-0700 execution child, prove cross-node read/write
visibility, and remove that exact child after allocation teardown; they never mount the runner
checkout or create a stage beneath image storage on AMD. When an AMD operator row omits `stage_dir`,
the runner derives a private base beside its standard `_work` directory on the shared runner
filesystem; the root-owned squash cache is never used as a repository stage.

H200, B200, and B300 runners may omit `stage_dir`; their isolated execution child is created under a
runner-owned mode-0700 base in the validated operating-system account home, independent of the
workflow's temporary `HOME`. H100 may also omit `stage_dir`; its private base is created beside, never
beneath, the configured shared container directory so it is compute-visible. Canonical B300 execution
ignores any legacy configured `stage_dir` and always uses the validated compute-visible account-home
base; an execution-ID suffix isolates parallel B300 workers. Canonical GB300 execution likewise
ignores its legacy group-writable `stage_dir` and derives an execution-specific private base beneath the
validated compute-visible account home. Backend preparation runs from that staged tree on every node.

Enroot imports the configured image tag into a per-run-scoped squash keyed by image tag and image
platform, so one run never reuses another run's imported filesystem. The image tag and platform are
per-SKU registry fields; the DeepEP V2 source pin lives in `runtime/common.sh` and its build is
fetched and verified at the pinned commit, checked for `ElasticBuffer`, and cached in a
cluster-local build cache keyed by architecture, image, and commit. Only the fixed `/cx-cache` mount
reaches the container.

## Local Checks

```bash
python3 -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
python3 experimental/CollectiveX/sweep_matrix.py --backend all --out /tmp/cx-matrix.json >/dev/null
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

Core paths are `configs/`, `sweep_matrix.py`, `summarize.py`, `bench/`, `runtime/`, `launchers/`,
and `tests/`.
