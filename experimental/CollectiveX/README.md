# CollectiveX

CollectiveX is an experimental MoE expert-parallel communication benchmark. It measures dispatch,
combine, and paired roundtrip latency across EP libraries and accelerator systems, then uploads
neutral result artifacts.

CollectiveX schedules benchmarks, executes them on real allocations, and uploads the neutral
artifacts each run emits. It does not validate those artifacts, promote, rank, recommend, select, or
decide what a consumer displays. Any downstream display or comparison is the consumer's
responsibility. The full measurement methodology is in [docs/methodology.md](docs/methodology.md).

## Execution Profile

The workload uses packed placement and one pinned `fixed-profile` resource configuration per
backend/topology. There is no tuning sweep. Combine is always BF16. Dispatch precision is a swept
dimension, with a BF16 control plus, on every backend whose FP8 dispatch is supported upstream
(DeepEP V2, MoRI, UCCL-EP, FlashInfer EP), an FP8 dispatch, caller-prequantized in `normal` mode (in
`low-latency` the DeepEP and UCCL-EP kernels quantize internally from BF16. MoRI stays
caller-prequantized, while FlashInfer has no `low-latency` path). That caller-side quantize is charged
inside the measured dispatch, because a production forward pass pays it on the critical path. NCCL
EP is BF16-only here, so it emits the control alone; wiring the FP8 dispatch support added in v0.2
is future work. Coverage is uniform routing only. Cases run in one of two modes:

- `normal` uses `layout-and-dispatch-v1`, rank-deduplicated token payloads, and activation-only,
  unweighted rank-sum combine. It runs the full decode and prefill ladders.
- `low-latency` uses each backend's decode-optimized kernel family: on DeepEP the legacy
  `deep_ep.Buffer` IBGDA `low_latency_dispatch`/`low_latency_combine` (a per-expert padded receive
  and a source-side gate-weighted combine). On UCCL-EP the same legacy `Buffer` low-latency kernels,
  which at the scoped EP8 run `cudaIpc` over NVLink rather than its CPU-proxy transport. On MoRI the `IntraNodeLL` kernel (single-call,
  pure-intranode, same compact layout and unweighted rank-sum combine as `IntraNode`). It is a
  decode-phase-only, per-SKU-capability-gated addition whose runnable set differs from `normal`'s, so
  it is enabled from each SKU's `ll_backends` registry entry (currently DeepEP V2 at EP8 on
  H100/H200 and at EP8 *and EP16* on B200 (the nscale bare-metal pool, whose gdrdrv-backed IBGDA
  over native IB is what a low-latency scale-out needs and no virtualized pool has) and on
  GB200/GB300, whose EP16 stays inside the MNNVL scale-up domain,
  plus MoRI EP8 on MI300X/MI325X/MI355X and UCCL-EP EP8 on H100/H200/B200 only (UCCL's low-latency host
  assert `kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group` cannot hold on AMD, where
  `kNumMaxWarpGroups` is 16, since upstream raised `kNumMaxTopK` 9 -> 16 (uccl#1016, 2026-07-13) and
  our pin is six days later. The product is 16 for every CU count, so this is a dated regression
  rather than a hardware limit, and the AMD SKUs keep UCCL-EP normal mode without LL),
  and NCCL EP at EP8 on all six NVIDIA SKUs and at EP16 on B200 (nscale), GB200 and GB300).
  Scoped single-node EP8 runs over the intra-node NVLink/XGMI
  low-latency path (no `/dev/gdrdrv` needed, as validated on H200 with it absent). NVSHMEM/IBGDA on the
  wire carries payload only on a multi-node scale-out (EP16) run. The legacy Buffer still
  self-enables IBGDA at EP8, which is why B300 fails there.

Cases use a fixed timing profile from `configs/sweep.json`: 256 trials x 8 timed iterations (2048
samples per component) with 32 synchronized full roundtrip warmups before each measured component at
every trial/point. Component measurement order rotates each trial so every timed component occupies
every position in the sequence, and each iteration takes the cross-rank maximum before nearest-rank
p50/p90/p95/p99. A keyed BLAKE2b counter produces
byte-identical routing and gate weights on every runtime.

Those components all measure **fresh entry** (the GPU is drained around every timed window), the
latency of an idle pipeline, not what a decode loop pays. So every row also carries the **chained
pair period**: 4 trials x (128 dispatch→combine pairs issued back-to-back with no host sync, first
16 dropped as pipeline fill) = 448 observations, reduced across ranks by median. Each trial runs
**two sibling chains** (a floors chain carrying only per-op events, then a period chain carrying
only the outer pair events), because the first version's single six-events-per-pair chain charged
its four inner `record()` calls into the period wherever the device outran the host, publishing a
~flat 10–30µs host constant as transport (+20–38% at T=1, fleet-wide). `components.pair_period` is
the headline latency for every row that carries one (released 2026-08-06 after the b200/h200/gb200
hand references were confirmed against two-pass fleet artifacts), and `summarize.py` footnotes what
its starred columns hold either way. The floors chain publishes `chain_floor_us`, the
cross-rank minimum of each op's window, while the period chain also yields `chain_health.pair_spread_us`
(cross-rank cadence proof), `interpair_gap_us` (the per-pair cost outside the published window, which is the
regression guard for instrumentation creeping back in, and the discriminator that keeps a
sync-dominated `period − Σfloors` gap from being misread as that defect returning. See the
methodology's `chain_floor_us` bullet for what each sign of that residual means) and
`settle_drift_us` (late-half minus
early-half period, the convergence proof `chain_drop` otherwise merely assumes). Chained per-op
*medians* are never published: inter-rank wait parks in whichever op window a rank blocks in, stable
per rank, arbitrary across ranks, and conserved only in the pair total. Nothing existing was renamed
or re-meant and the sweep `version` stays 1, so consumers key on the presence of
`components.pair_period`.

The chained regime is checked twice over: each chain trial's own final combined output is compared
against a drained pair through the identical code path (`correctness.chain_last_output_passed`,
with the size of any difference in `correctness.chain_last_output_error`), and the full oracle runs
once per ladder point against the state the chain leaves behind
(`correctness.post_chain_state_passed`). The second always gates. The first gates only where the
chain stages per pair, and is `null` where staging is hoisted out of the chain. Under the hoist
neither regime combines an input matching its own dispatch, so the two are not comparable. That
boundary was measured rather than assumed: identical h100 cases differ by 1000×–2966× the combine
tolerance with the hoist and by exactly zero without it. See the methodology's Correctness section. The null is a deliberate, bounded gap. An FP8-only,
free-running-only, stateless corruption would red no gate, and the `CX_FP8_CONSUME=dequant` hatch
is the standing probe for it.

`roundtrip` means dispatch then combine (the transport) in every row. Expert-output staging sits
outside it and is reported separately as `stage`. Under FP8 that component is harness scaffolding
standing in for the expert GEMM, which in production consumes FP8 operands natively rather than
materialising a BF16 copy, so `stage` must not be summed into a total or compared between backends.
Rows measured before this change carried the staging copy inside the chain for MoRI BF16 and
FlashInfer BF16, and the sweep `version` stays 1 across it, so
`implementation.stage_excluded_from_roundtrip` and whether a `stage` component is present are the
only way to tell the two generations apart. See [docs/methodology.md](docs/methodology.md) for the
full contract.

Correctness is checked against an implementation-independent oracle that reproduces the backend's
two-level reduction, with intra-scale-up-domain FP32, then a BF16 cast of each domain's partial for the
scale-out send. The combine gate is a tight max elementwise relative error below `8 * 2^-8`
(denominator clamped at 0.02), which holds across scale-up and multi-node scale-out topologies
alike. Under FP8 dispatch the oracle applies the same per-token cast round-trip to its semantic
payload, so the dispatched-payload compare stays bit-exact and the combine gate is unchanged. The
quantization is modeled, not tolerated. Any failed rank or point makes the case ineligible in the
result it writes.

The matrix covers H100, H200, B200, B300, GB200, GB300, MI300X, MI325X, and MI355X. `sweep_matrix.py` materializes
the requested SKUs, backends, EP sizes, and token ladders, then extracts strict per-shard controls
and rejects missing, stale, malformed, or altered shard controls. `--only-sku`, `--exclude-skus`,
`--ep-sizes`, and `--precisions` select a subset. The matrix is generated per dispatch, with no
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
| DeepEP V2 | `production`, with vLLM `--all2all-backend deepep_v2`, SGLang `--moe-a2a-backend deepep` | `normal` mode is PR #605 `ElasticBuffer` plus exact upstream #630 and #640 fixes: LSA for scale-up and GIN for x86 EP16 scale-out. FP8 dispatch via `use_fp8_dispatch` (blockwise e4m3fn) alongside BF16. `low-latency` mode is the legacy `deep_ep.Buffer` IBGDA decode kernels (per-expert padded layout, weighted combine, `use_fp8` e4m3fn), decode only, with EP8 wherever enabled, plus EP16 on GB200/GB300 (inside the MNNVL domain) and on B200's nscale bare-metal pool (IBGDA over native IB rails with `/dev/gdrdrv`, although the prior virtualized b200 pool could never run it). B300 is an unsupported coverage row in `low-latency`: the legacy Buffer self-enables NVSHMEM IBGDA even for a single-node EP8 run, and on B300 address-handle creation fails (`ibgda.cpp:2234 Unable to create ah`), rc255 on all eight ranks. `NVSHMEM_DISABLE_IB=1` does not help. The Buffer re-enables IBGDA regardless, and the run fails identically with it set and unset (measured on b300-002 and b300-011) |
| MoRI | `production`, with vLLM `--all2all-backend mori_*`, SGLang `--moe-a2a-backend mori` | `normal` mode uses the direct `IntraNode` kernel for scale-up EP8 on every CDNA SKU. EP16 remains an unsupported coverage row on all three CDNA SKUs. Part of the old ROCm/mori#475 corruption was this harness passing dispatch's returned recv-slot indices to `combine()` instead of the rank's own routing (root-caused upstream, guarded by ROCm/mori#546, kernels unchanged) — with the corrected call, single-shot InterNodeV1 is clean through T=512 on mi355x — but a residual stochastic corruption remains from T~128 up under repeated execution and is near-certain at prefill sizes (run 33045314017/33050026476; unaffected by per-pair drains, so not a buffer-reuse race; upstream cannot reproduce on ionic driver 26.03 vs our 25.11). The tw pairs additionally have no cross-node GPU fabric, and mi355x EP16 has no publishable transport today: uccl-ep's CPU-proxy RDMA is functional on Pollara but ~13x under its documented bandwidth (~6 GB/s vs 82; unchanged by registration mode or traffic class — ionic-driver suspect, same 25.11-vs-26.03 delta as the mori residual). `low-latency` mode selects the `IntraNodeLL` decode kernel (single-call, pure-intranode, same compact layout and unweighted combine as `IntraNode`), decode/EP8 only. FP8 dispatch is caller-prequantized (per-SKU e4m3fnuz on gfx942, e4m3fn on gfx950). Combine stays BF16 (`quant_type=none`) alongside BF16 dispatch |
| UCCL-EP | `candidate` (no engine exposes a UCCL-EP selector) | [UCCL](https://github.com/uccl-project/uccl) EP: a drop-in, API-identical DeepEP replacement whose CPU proxies issue GPUDirect RDMA over plain `libibverbs` (no NVSHMEM/IBGDA), with software message ordering, atomics, and flow control. Scale-up is single-node `cudaIpc` over NVLink/XGMI (never MNNVL). `normal` mode is the legacy `Buffer` `dispatch`/`combine` (unweighted rank-sum). `low-latency` reuses the legacy `low_latency_dispatch`/`low_latency_combine` decode kernels (weighted combine), decode/EP8 only. FP8 dispatch is caller-prequantized in `normal` mode (blockwise e4m3fn, per-SKU e4m3fnuz on gfx942). In `low-latency` mode the caller sends BF16 and the decode kernel quantizes to e4m3 internally (`use_fp8`). Combine is BF16. Runs on NVIDIA and AMD (H100/H200/B200 + MI300X/MI325X/MI355X), EP8 scale-up. Cross-node EP16 is functional (the internode RDMA path connects and the light case passes correctness) but its CPU-proxy throughput overruns the standardized per-case wall-clock budget on heavy token counts, so EP16 is an unsupported coverage row for now |
| NCCL EP | `candidate` (NVIDIA's own library, but no engine exposes an NCCL-EP selector) | [NCCL EP](https://github.com/NVIDIA/nccl-extensions/tree/main/nccl_ep): NVIDIA's native MoE dispatch/combine on the NCCL Device API, using LSA (NVLink load/store) intra-node and GIN (GPU-Initiated Networking) inter-node, driven through the `nccl-extensions` wheel's `nccl.ep` bindings (`nccl.core` comes from its `nccl4py` dependency). `normal` mode selects the `HIGH_THROUGHPUT` algorithm (FLAT `[N, hidden]` receive, unweighted rank-sum combine) and runs EP8 and EP16 on all six NVIDIA SKUs: on GB200/GB300 EP16 stays inside the MNNVL scale-up domain, and on H100/H200/B200/B300 it is a 2x8 RDMA scale-out over the cross-node GIN path. The `LOW_LATENCY` algorithm carries an `ll_backends` row at EP8 on all six NVIDIA SKUs and at EP16 on B200 (nscale), GB200 and GB300. The LL decode ladder runs to its full 256-slot receive, so only the 512 point is dropped. BF16 only; wiring the FP8 dispatch support added in v0.2 is future work. NVIDIA-only and CUDA 13 only. |
| FlashInfer EP | `production`, with vLLM `--all2all-backend flashinfer_nvlink_one_sided` | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `MoeAlltoAll`: TensorRT-LLM's one-sided MNNVL all-to-all, where each rank writes tokens straight into its peers' workspace windows and combine reads them back, with no send/recv pairing and no NVSHMEM. `normal` mode only (there is one kernel family and no separate decode path), and GB200/GB300 only, since the transport is MNNVL. FP8 dispatch is caller-prequantized blockwise e4m3fn, carried as a fourth dispatch payload alongside its per-128-block FP32 scales, with the combine plane forced to BF16. The C++ `toNvDataType` accepts only fp16/bf16/fp32 for combine, so an FP8 combine buffer would raise rather than corrupt. EP8 and EP16, both inside the scale-up domain. Unlike every other backend here, its combine accumulates in the PAYLOAD dtype rather than FP32: wheels before 0.6.16 reduce the top-k contributions with a pairwise BF16 tree that rounds at every level, so the oracle models that reduction directly (`combine_reduction = "topk-slot-tree"`) instead of widening the tolerance. 0.6.16 moved the accumulator to FP32, and the adapter switches models on the installed version |

DeepEP V2 means the `ElasticBuffer` implementation introduced by
[DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605), not a newer legacy `Buffer` build.
The pinned source is upstream `main`, which contains #605 along with
[PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) (fixes pure scale-up initialization when
GIN is unavailable), [PR #640](https://github.com/deepseek-ai/DeepEP/pull/640) (stops NCCL
shared-memory mappings being misclassified as duplicate NCCL libraries), and
[PR #642](https://github.com/deepseek-ai/DeepEP/pull/642) (the low-latency combine fence that fixes
the Blackwell top-rung corruption of
[issue #700](https://github.com/deepseek-ai/DeepEP/issues/700)), which the previous pin, the #630
head on the pre-merge #605 branch, predated. Scale-up cases request NCCL Device API LSA and fail closed
unless the realized LSA team covers the full EP world. x86 EP16 scale-out cases instead require the
hybrid path with GIN, two logical scale-out domains represented by two physical RDMA ranks, and eight
scale-up ranks per domain. GB EP16 remains MNNVL scale-up and therefore uses LSA. Whether a given
SKU/backend/EP cell is attempted is a capability fact. Whether it succeeded is decided by the
benchmark's return code.

## Workflow And Artifacts

`.github/workflows/collectivex-sweep.yml` has two jobs. `setup` generates a public-SKU matrix
(`backend`, `only_sku`, `exclude_skus`, `ep_sizes` inputs) and uploads the matrix.
`sweep` extracts a strict ignored `.shards/<id>.json` control per matrix entry, executes one
allocation per shard, fetches pinned DeepEP source before allocation when required, and uploads the
result artifacts with `always()` so a red or partial run still uploads.

Each shard emits per-case result JSON and a small mechanical summary. A case counts as successful on
the benchmark's own return code. There is no completeness or privacy validation step, and failed or
unsupported cells produce no synthetic record. No step promotes a run,
builds a dataset, or advances a channel. The neutral artifacts are the output. A consumer downloads
them and decides what to display.

No operator credentials are passed to the workflow or uploaded. Runner-local overrides and any
selectors stay on the runner. Per-step runner logs are kept on the runner for postmortem, and
result artifacts carry only the fields listed in the methodology.

## Runner Configuration

Each SKU's Slurm and storage values come from its tracked baseline in the registry. An optional
runner-local JSON document at `$XDG_CONFIG_HOME/inferencex/collectivex.json` or
`COLLECTIVEX_OPERATOR_CONFIG` overlays that baseline per field. A runner with no registry entry, an
unknown field, and non-JSON input all fail closed, and configuration is never evaluated as shell.
Duplicate JSON keys are NOT rejected. `json.load` keeps the last silently, and runner keys other
than the one being resolved are not validated, so a typo'd SKU name is ignored rather than
reported. GHA passes no operator secret, so a SKU runs entirely from its tracked baseline unless a
runner-local document is present.

All public per-SKU platform data lives in the tracked `configs/platform_config.json` registry:
architecture/product, container image and platform, fixed placement, launcher, runnable backend/EP
pairs, the scale-out `fabric` identity (NIC and switch, so same-GPU clusters on different fabrics
are distinct entries, e.g. a second b200 cluster), tracked operator defaults, and scale-out RDMA
selectors. Operator documents can override the defaults. Launchers
declare and check the fields they actually require. `sweep_matrix.py` derives EP topology from the
placement fields. The sweep includes every registered SKU by default.

Every selected non-MNNVL EP16 placement additionally requires `socket_ifname` and `rdma_devices` for
its operator-approved fabric. Optional `ib_gid_index`, `rdma_service_level`, `rdma_traffic_class`,
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
visibility, and remove that exact child after allocation teardown. They never mount the runner
checkout or create a stage beneath image storage on AMD. When an AMD operator row omits `stage_dir`,
the runner derives a private base beside its standard `_work` directory on the shared runner
filesystem. The root-owned squash cache is never used as a repository stage.

H200, B200, and B300 runners may omit `stage_dir`. Their isolated execution child is created under a
runner-owned mode-0700 base in the validated operating-system account home, independent of the
workflow's temporary `HOME`. H100 may also omit `stage_dir`. Its private base is created beside, never
beneath, the configured shared container directory so it is compute-visible. Canonical B300 execution
ignores any legacy configured `stage_dir` and always uses the validated compute-visible account-home
base. An execution-ID suffix isolates parallel B300 workers. Canonical GB300 execution likewise
ignores its legacy group-writable `stage_dir` and derives an execution-specific private base beneath the
validated compute-visible account home. Backend preparation runs from that staged tree on every node.

Enroot imports the configured image tag into a per-run-scoped squash keyed by image tag and image
platform, so one run never reuses another run's imported filesystem. The image tag and platform are
per-SKU registry fields. The DeepEP V2 source pin lives in `runtime/common.sh` and its build is
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
