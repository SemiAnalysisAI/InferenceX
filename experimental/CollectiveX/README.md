# CollectiveX

<div align="center">

**English** | [中文](./README_zh.md)

</div>

CollectiveX is an experimental MoE expert-parallel communication benchmark. It measures dispatch,
combine, and paired roundtrip latency across EP libraries and accelerator systems.

> Publication hold: historical schema 3-5 data is diagnostic. No current dataset is approved for
> rankings, recommendations, or regression baselines.

> Development status: the V1 precision, point-level publication, branch-only delivery, and frontend
> contracts are implemented. Validated B200 EP8, H200 EP8, and GB200 EP8/EP16 probe outcomes are
> recorded, together with seven completed H100 EP8 outcomes; 50 native precision cells remain
> provisional, so V1 counts/digests are not frozen.

## Implemented Pre-V1 Execution Profile

The BF16/BF16 control and endpoint precision suites use packed placement and one pinned
`fixed-profile` resource configuration per backend/topology/precision. V1 performs no tuning sweep.
The explicit mode selects one of two contracts:

- Normal mode uses `layout-and-dispatch-v1`, rank-deduplicated token payloads, and activation-only
  combine. Uniform core coverage and one Zipf sensitivity remain; EPLB is measured only as the Zipf
  remedy.
- Low-latency mode uses `expert-packed-weighted-combine-v1`, token-expert payloads, and gate-weighted
  combine through genuine DeepEP V1 or UCCL low-latency APIs. It is decode-only and never shares a
  ranking cohort with normal mode. Other backends are explicitly unsupported for this suite.

Both modes use `fixed-512-v1`: 64 trials x 8 timed iterations with 32 synchronized full roundtrip
warmups before each measured component at every trial/point. Roundtrip is measured first; each
iteration takes the cross-rank maximum before nearest-rank p50/p90/p95/p99, and roundtrip p99 is the
headline latency. A stdlib integer counter produces byte-identical routing and gate weights.

Correctness is checked against the semantic value after the declared communication codec. The
frozen combine gates are `rtol=0.05, atol=0.02` for BF16, `rtol=0.06, atol=0.03` for native
logfmt10, and `rtol=0.08, atol=0.04` for native FP8 direct-cast combine. These are fail-closed
publication thresholds, not estimates of codec error. FP8 direct-cast evidence also counts
saturation on the exact transformed native combine input; any saturated value fails the point.

The BF16 planning baseline covers H100, H200, B200, B300, GB200, GB300, MI325X, and MI355X. It
requests
608 cases / 1,600 token points: 338 runnable cases / 868 points, emitted as 54 executable workflow
shards/allocation cells, plus 270 explicit unsupported cases / 732 points. `sweep_matrix.py`
materializes every token ladder and rejects missing, stale, malformed, or altered shard controls.
Shards are emitted round-robin by SKU so the bounded GHA matrix uses every runner pool early.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI325X/MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

Physical host count does not determine scope: both GB topologies stay inside one 72-GPU MNNVL
scale-up domain.

| Backend | Current scope |
|---|---|
| DeepEP V1 | Image-pinned `deep_ep.Buffer`: normal and native low-latency APIs; upstream v1.2.1 on x86 and the image's GB fork on arm64 |
| DeepEP V2 | PR #605 `ElasticBuffer` plus exact upstream #630 and #640 fixes: LSA for scale-up and GIN for x86 EP16 scale-out; source/SASS-bound reproducible JIT |
| DeepEP Hybrid | Pinned `HybridEPBuffer`: x86 EP16 multi-domain RDMA/DOCA; GB EP8/EP16 in one MNNVL communication domain |
| UCCL | Pinned 0.1.1 wheel and wrapper with normal and native low-latency APIs on Hopper; Blackwell is explicitly unsupported |
| NCCL/RCCL A2A | Portable rank-deduplicated payload plus expert/routing-metadata reference |
| MoRI | EP8 uses MI325X AsyncLL or MI355X IntraNode; EP16 pins InterNodeV1 over 2x8 XGMI + RDMA |

FlashInfer is outside v1 because its exercised EP path failed intermittently at runtime. It is not
misreported as a platform capability limitation and can return after a stable pinned path is proven.

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
records the API, source, loaded libraries, generated JIT source, executable SASS, and raw CUBIN
diagnostics. The current H100 runner pool is explicitly unsupported for V2 because NCCL 2.30.4
reports that its EP8 communicator lacks Device API symmetric-memory support; re-enabling that pool
requires an all-rank CUDA P2P/LSA-capable runtime. Other NVIDIA SKUs remain unvalidated until their
GPU outcomes pass the native correctness and publication gates.

H100 EP16 is planned unsupported on the current runner pool because allocated compute nodes expose
no active RDMA device. EP8 remains in scope and derives its private stage beside the existing shared
container directory because the runner account home is not mounted on compute nodes.

Axes not implemented in this baseline include cached-layout `[cl]`, runtime-visible `[rv]`, FP8,
quantized combine,
extra routing distributions, activation profiles, uneven allocation, placement permutations, model
envelopes, and scaling studies.

## Workflow And Artifacts

`.github/workflows/collectivex-sweep.yml` generates a public-SKU matrix, extracts a strict ignored
`.shards/<id>.json` control, executes one allocation per shard, privacy-checks result JSON, and uploads
raw GitHub artifacts. Runs default to `release_tag=unversioned` and are diagnostic-only. A V1 run
must explicitly select `release_tag=v1`; setup then requires the locked full-matrix digest and emits
a run/attempt/source-bound `cxrelease-v1-*` marker. Partial and filtered runs cannot receive it.

The main-registered `.github/workflows/collectivex-sweep.yml` provides `probe-precision`, `sweep`,
`publish-v1`, and `refresh-v1` operations, so its branch revision can be dispatched with
`--ref collectivex` without a standalone branch-only workflow. Probes are unversioned, bounded,
native-runtime checks and cannot publish. Publication accepts exactly three successful first-attempt tagged
sweep run IDs from one source SHA, revalidates their metadata and release markers, and runs
`publisher.py` in a disposable runner-local workspace. Refresh revalidates and reuploads only the
exact content-addressed sanitized dataset. Raw artifacts and the private publisher workspace are
never exposed to the frontend.

There is no results server, attached store, Vercel storage, GCP, Neon, managed database, or managed
object store. With the existing server-side `GITHUB_TOKEN`, the frontend discovers the latest
successful version-scoped publication workflow, downloads its NDJSON artifact just in time, verifies
the ZIP layout, UTF-8/NDJSON shape, schema, promotion state, and SHA-256, then serves versioned channel
and immutable dataset URLs. The UI keeps an explicit benchmark-version selector; V2 and later
versions must use separate release tags and publication identities. The full validation contract is
in [docs/methodology.md](docs/methodology.md).

## Runner Configuration

Runner-local Slurm and storage values use a strict per-SKU JSON document at
`$XDG_CONFIG_HOME/inferencex/collectivex.json` or `COLLECTIVEX_OPERATOR_CONFIG`. The mode-0600,
same-owner, non-symlink file is outside the checkout and never uploaded. Unknown runners, fields,
duplicate keys, endpoint literals, unsafe paths, and non-JSON input fail closed; configuration is
never evaluated as shell. GHA passes encrypted `COLLECTIVEX_OPERATOR_CONFIG_V1` content only to the
launcher, which validates it, exports the selected SKU's allowlisted values, and deletes the
temporary copy before allocation. Required JSON fields are:

| SKU | Variables |
|---|---|
| `h100-dgxc` | `partition`, `account`, `squash_dir` |
| `h200-dgxc` | `partition`, `squash_dir` |
| `b200-dgxc` | `partition`, `account`, `squash_dir` |
| `b300` | `partition`, `account`, `squash_dir` |
| `gb200` | `partition`, `account`, ordered `storage_roots` |
| `gb300` | `partition`, `account`, `squash_dir`, `enroot_cache_path` |
| `mi325x`, `mi355x` | `partition`, `squash_dir`, `stage_dir` |

Every selected non-MNNVL EP16 placement additionally requires `socket_ifname` and `rdma_devices`
for its operator-approved fabric; optional
`ib_gid_index` and `rdma_service_level` are also allowlisted. CollectiveX does not heuristically
select a management route or HCA. After allocation, every non-MNNVL scale-out node must prove that
all configured interfaces and active HCA ports exist before backend setup. Scale-up and MNNVL jobs
clear these overrides. Scale-out NCCL/RCCL is pinned to `IB` with exact-match HCA selectors so a
socket fallback fails instead of being mislabeled as RDMA.

`ib_gid_index` is applied only when every selected HCA port reports an Ethernet link layer, where it
selects the operator-approved RoCE GID. Native InfiniBand profiles retain explicit HCA and service
level pinning but leave the RoCE-only GID override unset so NVSHMEM/NCCL can use the native LID path.
Mixed Ethernet and InfiniBand HCA lists are rejected.

`stage_dir` is a pre-existing, runner-owned, non-symlinked base outside the checkout and workflow
workspace. It is not group- or world-writable and is visible at the same path on the runner and every
allocated node. Jobs create only a marked mode-0700 execution child, prove cross-node read/write
visibility, and remove that exact child after allocation teardown; they never mount the runner
checkout or create a stage beneath image storage on AMD.

When an AMD operator row omits `stage_dir`, the self-hosted runner derives a private base beside its
standard `_work` directory on the shared runner filesystem. The root-owned squash cache is never
used as a repository stage. The derived parent must be runner-owned, non-symlinked, and not writable
by group or world before any benchmark source is copied.

H200, B200, and B300 runners may omit `stage_dir`. Their isolated execution child is then created
under a runner-owned mode-0700 base in the validated operating-system account home, independent of
the workflow's temporary `HOME`. A symlinked account-home entry is resolved once to its canonical,
runner-owned target; the single hidden staging base directly beneath it must itself be non-symlinked
and not writable by other users. The workflow still proves that base is visible from every allocated
node before launch. The execution child is validated, marked, cross-node checked, and removed using
the same contract. H100 may also omit `stage_dir`; its private base is created beside, never beneath,
the configured shared container directory so it is compute-visible. All other runners require the
dedicated `stage_dir` above; no canonical run stages source beneath shared container storage.
Canonical B300 execution ignores any legacy configured `stage_dir` and always uses the validated
compute-visible account-home base. Its UID-mapped Actions shell may accept that exact base when its
owner matches the private parent owner; this exception is not available to configured stages or
other runners. A hashed execution-ID suffix isolates parallel B300 workers without exposing private
runner identity. On this NFS export, a newly created hashed base may realize as UID 0; only that
creation path is accepted, while a pre-existing root-owned base is rejected.
Canonical GB300 execution likewise ignores its legacy group-writable `stage_dir` and derives an
execution-hashed private base beneath the validated compute-visible account home.

Before import, each Docker Hub tag is resolved with bounded registry requests and must match its
pinned digest; digest-qualified overrides are rejected. Enroot imports use a fixed filesystem epoch
and a versioned, registry-digest-bound cache key. Every mounted squash is freshly hashed. The
verified registry digest and local squash hash are both recorded. Image-provided DeepEP is checked
against exact wheel and installed-file fingerprints; source-built backends use pinned commits and
runtime-verified GPU targets. DeepEP V2's mode-0700 cluster-local build cache is keyed by a versioned
build recipe, verified image, architecture, upstream trees, and dependency pins; only its fixed
`/cx-cache` mount reaches the container, and it never enters result artifacts.
Pinned V2 and Hybrid sources are fetched once per workflow. Each job validates the complete archive,
extracts only its exact backend root, permits only contained relative leaf symlinks to archived
regular files, and revalidates the Git tree and submodule pins before staging.
Compute containers receive an explicit environment allowlist. Private host, address, device, NIC,
credential, workspace, and path data stays in encrypted config, ignored operator notes, or bounded
mode-0600 runner logs; it is never uploaded.

## Local Checks

```bash
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python experimental/CollectiveX/sweep_matrix.py --backends all --out /tmp/cx-matrix.json >/dev/null
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python experimental/CollectiveX/publisher.py --store-root "$COLLECTIVEX_STORE_ROOT" verify
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

Core paths are `capability.py`, `configs/`, `contracts.py`, `schemas/`, `sweep_matrix.py`,
`publisher.py`, `runtime/`, `launchers/`, and `tests/`.
