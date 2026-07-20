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
backend/topology; there is no tuning sweep. Combine is always BF16; dispatch precision is a swept
dimension — a BF16 control plus a caller-prequantized FP8 dispatch on every backend. Coverage is
uniform routing only. Cases run in one of two modes:

- `normal` uses `layout-and-dispatch-v1`, rank-deduplicated token payloads, and activation-only,
  unweighted rank-sum combine. It runs the full decode and prefill ladders.
- `low-latency` uses each backend's decode-optimized kernel family: on DeepEP the legacy
  `deep_ep.Buffer` IBGDA `low_latency_dispatch`/`low_latency_combine` (a per-expert padded receive
  and a source-side gate-weighted combine); on UCCL-EP the same legacy `Buffer` low-latency kernels
  over its CPU-proxy transport; on MoRI the `IntraNodeLL` kernel (single-call,
  pure-intranode, same compact layout and unweighted rank-sum combine as `IntraNode`). It is a
  decode-phase-only, per-SKU-capability-gated addition whose runnable set differs from `normal`'s, so
  it is enabled from each SKU's `ll_backends` registry entry (currently DeepEP V2 EP8 on H100/H200/B200,
  MoRI EP8 on MI300X/MI325X/MI355X, and UCCL-EP EP8 on H100/H200/B200 only — UCCL's low-latency kernel
  trips a warp-group assertion on AMD's CU count, so the AMD SKUs keep UCCL-EP normal mode without LL).
  Scoped single-node EP8 runs over the intra-node NVLink/XGMI
  low-latency path (no `/dev/gdrdrv` needed — validated on H200 with it absent); NVSHMEM/IBGDA on the
  wire is only a multi-node scale-out (EP16) concern.

Cases use a fixed timing profile from `configs/sweep.json`: 256 trials x 8 timed iterations (2048
samples per component) with 32 synchronized full roundtrip warmups before each measured component at
every trial/point. Component measurement order rotates each trial so every timed component occupies
every position in the sequence; each iteration takes the cross-rank maximum before nearest-rank
p50/p90/p95/p99, and roundtrip p99 is the headline latency. A keyed BLAKE2b counter produces
byte-identical routing and gate weights on every runtime.

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

| Backend | Current scope |
|---|---|
| DeepEP V2 | `normal` mode is PR #605 `ElasticBuffer` plus exact upstream #630 and #640 fixes: LSA for scale-up and GIN for x86 EP16 scale-out. FP8 dispatch via `use_fp8_dispatch` (blockwise e4m3fn) alongside BF16. `low-latency` mode is the legacy `deep_ep.Buffer` IBGDA decode kernels (per-expert padded layout, weighted combine, `use_fp8` e4m3fn), decode/EP8 only |
| MoRI | `normal` mode uses the direct `IntraNode` kernel for scale-up EP8 on every CDNA SKU and pins `InterNodeV1` for EP16 over 2x8 XGMI + RDMA. `low-latency` mode selects the `IntraNodeLL` decode kernel (single-call, pure-intranode, same compact layout and unweighted combine as `IntraNode`), decode/EP8 only. FP8 dispatch is caller-prequantized (per-SKU e4m3fnuz on gfx942, e4m3fn on gfx950); combine stays BF16 (`quant_type=none`) alongside BF16 dispatch |
| UCCL-EP | [UCCL](https://github.com/uccl-project/uccl) EP: a drop-in, API-identical DeepEP replacement whose CPU proxies issue GPUDirect RDMA over plain `libibverbs` (no NVSHMEM/IBGDA), with software message ordering, atomics, and flow control; scale-up is single-node `cudaIpc` over NVLink/XGMI (never MNNVL). `normal` mode is the legacy `Buffer` `dispatch`/`combine` (unweighted rank-sum); `low-latency` reuses the legacy `low_latency_dispatch`/`low_latency_combine` decode kernels (weighted combine), decode/EP8 only. FP8 dispatch is caller-prequantized (blockwise e4m3fn, per-SKU e4m3fnuz on gfx942) with BF16 combine. Runs on NVIDIA and AMD (H100/H200/B200 + MI300X/MI325X/MI355X), EP8 scale-up. Cross-node EP16 is functional (the internode RDMA path connects and the light case passes correctness) but its CPU-proxy throughput overruns the standardized per-case wall-clock budget on heavy token counts, so EP16 is an unsupported coverage row for now |

DeepEP V2 means the `ElasticBuffer` implementation introduced by
[DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605), not a newer legacy `Buffer` build.
The pinned source is the [PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) head, whose parent
is the #605 merge tree, plus the exact one-line library matcher from upstream
[PR #640](https://github.com/deepseek-ai/DeepEP/pull/640). The first fixes pure scale-up
initialization when GIN is unavailable; the second prevents NCCL shared-memory mappings from being
misclassified as duplicate NCCL libraries. Scale-up cases request NCCL Device API LSA and fail closed
unless the realized LSA team covers the full EP world. x86 EP16 scale-out cases instead require the
hybrid path with GIN, two logical scale-out domains represented by two physical RDMA ranks, and eight
scale-up ranks per domain; GB EP16 remains MNNVL scale-up and therefore uses LSA. Whether a given
SKU/backend/EP cell is attempted is a capability fact; whether it succeeded is decided by the
benchmark's return code.

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
`COLLECTIVEX_OPERATOR_CONFIG` overlays that baseline per field; unknown runners, fields, duplicate
keys, and non-JSON input fail closed, and configuration is never evaluated as shell. GHA passes no
operator secret, so a SKU runs entirely from its tracked baseline unless a runner-local document is
present.

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
