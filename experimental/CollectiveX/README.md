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
backend/topology; there is no tuning sweep. Dispatch and combine are fixed BF16 on every backend;
precision is not a swept dimension. The explicit mode selects one of two contracts:

- Normal mode uses `layout-and-dispatch-v1`, rank-deduplicated token payloads, and activation-only
  combine. Uniform core coverage and one Zipf sensitivity remain; EPLB is measured only as the Zipf
  remedy.
- Low-latency mode uses `expert-packed-weighted-combine-v1`, token-expert payloads, and gate-weighted
  combine through genuine DeepEP V1 or UCCL low-latency APIs. It is decode-only. Other backends are
  recorded as unsupported for this suite.

Both modes use `fixed-512-v1`: 64 trials x 8 timed iterations with 32 synchronized full roundtrip
warmups before each measured component at every trial/point. Roundtrip is measured first; each
iteration takes the cross-rank maximum before nearest-rank p50/p90/p95/p99, and roundtrip p99 is the
headline latency. A stdlib integer counter produces byte-identical routing and gate weights.

Correctness is checked against the reference activation. The combine gate is `rtol=0.05, atol=0.02`
for the BF16 communication path. Any failed rank or point makes the case ineligible in the result
it writes.

The matrix covers H100, H200, B200, B300, GB200, GB300, and MI355X. `sweep_matrix.py` materializes
the requested SKUs, backends, EP sizes, and token ladders, then extracts strict per-shard controls
and rejects missing, stale, malformed, or altered shard controls. `--only-sku`, `--exclude-skus`, and
`--ep-sizes` select a subset; the matrix is generated per dispatch, with no frozen digest or locked
case count.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

Physical host count does not determine scope: both GB topologies stay inside one 72-GPU MNNVL
scale-up domain. The MI325X launcher/configuration path is retained for future versions but is not
referenced by any current suite or shard.

| Backend | Current scope |
|---|---|
| DeepEP V1 | Image-pinned `deep_ep.Buffer`: normal and native low-latency APIs; upstream v1.2.1 on x86 and the image's GB fork on arm64 |
| DeepEP V2 | PR #605 `ElasticBuffer` plus exact upstream #630 and #640 fixes: LSA for scale-up and GIN for x86 EP16 scale-out; source/SASS-bound reproducible JIT |
| DeepEP Hybrid | Pinned `HybridEPBuffer`: x86 EP16 multi-domain RDMA/DOCA; GB EP8/EP16 in one MNNVL communication domain |
| UCCL | Pinned 0.1.1 wheel and wrapper with normal and native low-latency APIs on Hopper; Blackwell is explicitly unsupported |
| NCCL/RCCL A2A | Portable rank-deduplicated payload plus expert/routing-metadata reference |
| MoRI | MI355X EP8 uses IntraNode; EP16 pins InterNodeV1 over 2x8 XGMI + RDMA |

DeepEP V2 means the `ElasticBuffer` implementation introduced by
[DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605), not a newer legacy `Buffer` build.
The pinned source is the [PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) head, whose parent
is the #605 merge tree, plus the exact one-line library matcher from upstream
[PR #640](https://github.com/deepseek-ai/DeepEP/pull/640). The first fixes pure scale-up
initialization when GIN is unavailable; the second prevents NCCL shared-memory mappings from being
misclassified as duplicate NCCL libraries. Scale-up cases request NCCL Device API LSA and fail closed
unless the realized LSA team covers the full EP world. x86 EP16 scale-out cases instead require the
hybrid path with GIN, two logical scale-out domains represented by two physical RDMA ranks, and eight
scale-up ranks per domain; GB EP16 remains MNNVL scale-up and therefore uses LSA. The isolated build
records the API, source, loaded libraries, generated JIT source, and executable SASS; raw CUBIN bytes
stay private diagnostics. Whether a given SKU/backend/EP cell is attempted is a capability fact;
whether it succeeded is decided by the benchmark's return code.

## Workflow And Artifacts

`.github/workflows/collectivex-sweep.yml` has two jobs. `setup` generates a public-SKU matrix
(`backend`, `suites`, `only_sku`, `exclude_skus`, `ep_sizes` inputs), fetches the pinned backend
source archive, and uploads the matrix.
`sweep` extracts a strict ignored `.shards/<id>.json` control per matrix entry, executes one
allocation per shard, and uploads the result artifacts with `always()` so a red or partial run still
uploads.

Each shard emits per-case result JSON, detached sample JSON, and a small mechanical summary. A case
counts as successful on the benchmark's own return code; there is no schema, completeness, or privacy
validation step, and failed or unsupported cells produce no synthetic record. No step promotes a run,
builds a dataset, or advances a channel; the neutral artifacts are the output. A consumer downloads
them and decides what to display.

Private host, address, device, NIC, credential, workspace, and path data stays in encrypted config,
ignored operator notes, or bounded mode-0600 runner logs; it is never uploaded.

## Runner Configuration

Runner-local Slurm and storage values use a strict per-SKU JSON document at
`$XDG_CONFIG_HOME/inferencex/collectivex.json` or `COLLECTIVEX_OPERATOR_CONFIG`. The mode-0600,
same-owner, non-symlink file is outside the checkout and never uploaded. Unknown runners, fields,
duplicate keys, endpoint literals, unsafe paths, and non-JSON input fail closed; configuration is
never evaluated as shell. GHA passes encrypted `COLLECTIVEX_OPERATOR_CONFIG_V1` content only to the
launcher, which validates it, exports the selected SKU's allowlisted values, and deletes the temporary
copy before allocation. Required JSON fields are:

| SKU | Variables |
|---|---|
| `h100-dgxc` | `partition`, `account`, `squash_dir` |
| `h200-dgxc` | `partition`, `squash_dir` |
| `b200-dgxc` | `partition`, `account`, `squash_dir` |
| `b300` | `partition`, `account`, `squash_dir` |
| `gb200` | `partition`, `account`, ordered `storage_roots` |
| `gb300` | `partition`, `account`, `squash_dir`, `enroot_cache_path` |
| `mi325x`, `mi355x` | `partition`, `squash_dir`, `stage_dir` |

Every selected non-MNNVL EP16 placement additionally requires `socket_ifname` and `rdma_devices` for
its operator-approved fabric; optional `ib_gid_index`, `rdma_service_level`, and `rdma_traffic_class`
are also allowlisted. Service level and traffic class are mapped into MoRI's RDMA/IO QoS environment.
CollectiveX does not heuristically select a management route or HCA. After allocation, every
non-MNNVL scale-out node must prove that all configured interfaces and active HCA ports exist before
backend setup. Scale-up and MNNVL jobs clear these overrides. Scale-out NCCL/RCCL is pinned to `IB`
with exact-match HCA selectors so a socket fallback fails instead of being mislabeled as RDMA.

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
base; a hashed execution-ID suffix isolates parallel B300 workers. Canonical GB300 execution likewise
ignores its legacy group-writable `stage_dir` and derives an execution-hashed private base beneath the
validated compute-visible account home. The workflow proves every derived base is visible from all
allocated nodes before launch.

Before import, each Docker Hub tag is resolved with bounded registry requests and must match its
pinned digest; digest-qualified overrides are rejected. Enroot imports use a fixed filesystem epoch
and a versioned, registry-digest-bound cache key. Every mounted squash is freshly hashed. Image-provided
DeepEP is checked against exact wheel and installed-file fingerprints; source-built backends use pinned
commits and runtime-verified GPU targets. DeepEP V2's mode-0700 cluster-local build cache is keyed by a
versioned build recipe, verified image, architecture, upstream trees, and dependency pins; only its
fixed `/cx-cache` mount reaches the container, and it never enters result artifacts. Pinned V2 and
Hybrid sources are fetched once per workflow, validated whole, and extracted to their exact backend
root before staging.

## Local Checks

```bash
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python experimental/CollectiveX/sweep_matrix.py --backends all --out /tmp/cx-matrix.json >/dev/null
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

Core paths are `capability.py`, `configs/`, `identity.py`, `sweep_matrix.py`, `summarize.py`,
`bench/`, `runtime/`, `launchers/`, and `tests/`.
