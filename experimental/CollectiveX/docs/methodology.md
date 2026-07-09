# CollectiveX EP Benchmark Methodology

CollectiveX schedules expert-parallel (EP) communication benchmarks, executes them on real
accelerator allocations, and uploads the neutral artifacts each run emits. It does **not** validate
those artifacts, promote, rank, recommend, select, hide, or decide what any consumer displays. The
frontend reads the neutral matrix, result, summary, and catalog artifacts and makes its own coverage
and display decisions. This document describes how a case is scheduled, measured, checked, and
recorded — not a publication or qualification contract.

## Product Boundary

CollectiveX is a communication microbenchmark for:

- comparing EP libraries on one chip/topology;
- comparing EP latency and logical payload bandwidth across systems under the same workload; and
- surfacing unsupported, failed, invalid, and unstable cases rather than hiding them.

It does not predict serving throughput without a separate correlation study.

## Matrix

The implemented workload is `deepseek-v3`: hidden 7168, top-k 8, 256 routed experts, packed
placement, and one pinned fixed resource profile per backend/topology. Dispatch and combine are
fixed BF16 on every backend; precision is not a swept dimension. Each case explicitly selects normal
`layout-and-dispatch-v1` or low-latency `expert-packed-weighted-combine-v1` semantics.

- `ep-core`: uniform routing; decode T=1..128 powers of two; prefill T=256/512.
- `ep-low-latency`: DeepEP V1 native low-latency APIs; uniform decode T=1..128 powers of
  two; other backends are recorded as unsupported rather than fabricating a low-latency path.

`sweep_matrix.py` materializes the requested SKUs, backends, EP sizes, and token ladders into a
matrix document, then extracts strict per-shard controls. `--only-sku`, `--exclude-skus`, and
`--ep-sizes` select a subset; a subset produces a smaller matrix, not a different contract. The
matrix is generated per dispatch; there is no frozen matrix digest or locked case count.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

Physical host count does not define scope. Both GB cells remain inside one 72-GPU MNNVL scale-up
domain. The MI325X launcher/configuration path is retained for future versions but is not referenced
by any current suite or shard.

Unsupported combinations are explicitly classified in the matrix, not silently skipped coverage. DeepEP V2 is the
`ElasticBuffer` introduced by PR #605, pinned with upstream PR #630's minimal pure-scale-up fix and
the exact upstream PR #640 library matcher that excludes NCCL shared-memory mappings. Scale-up cases
request NCCL Device API LSA and fail closed unless the realized LSA team covers the full EP world.
x86 EP16 scale-out uses the hybrid path with GIN and requires two logical scale-out domains
represented by two physical RDMA ranks, with eight scale-up ranks per domain. GB EP16 remains MNNVL
scale-up and uses LSA. MoRI EP8 uses MI355X IntraNode in normal mode; EP16 uses pinned InterNodeV1
over 2x8 XGMI + RDMA with 96 blocks, 64 RDMA blocks, 8 warps, one QP per PE, and external input.
MoRI's AsyncLL transport is not the low-latency suite contract and is never labeled as such. Whether
a given SKU/backend/EP cell is attempted is a capability fact; whether it succeeded is decided only
by the emitted artifact.

## Workload Identity

One canonical workload is generated over the global token batch and sliced by source rank. Expert
indices and gate weights are serialized. Activations use a versioned integer counter formula whose
BF16 values are exact across runtimes. The manifest records shape, EP, generator, and oracle
coordinates, and loading regenerates the expected routing arrays for direct equality validation.

Routing traffic distinguishes:

- token-expert assignments, which determine expert compute load; and
- rank-deduplicated token payload copies, which determine EP activation traffic.

Adapters may not generate routing or reinterpret one quantity as the other.

## Measurement

Normal mode uses `layout-and-dispatch-v1`: dispatch timing includes layout plus communication, and
combine returns activation payload through an unweighted rank-sum path. Low-latency mode uses
`expert-packed-weighted-combine-v1`: native DeepEP V1 APIs dispatch token-expert assignments
and perform gate-weighted combine. Expert-output staging is outside isolated combine timing and
inside the measured paired roundtrip. Each component declares availability, origin, start/end states,
stage scope, and sample count. A paired-only API reports null isolated components. `isolated_sum` is
derived. Normal and low-latency evidence describe different measurement contracts and are not
directly comparable; the artifact records the mode so a reader can keep them separate.

Every measured component uses `fixed-512-v1`:

- 64 trials x 8 timed iterations = 512 observations;
- 32 synchronized full dispatch-stage-combine warmups before each available measured component at
  every trial/point;
- roundtrip first, then isolated dispatch and combine, with a fixed per-phase conditioning ladder; and
- per-iteration maximum latency across ranks before nearest-rank p50/p90/p95/p99.

Measured roundtrip p99 is the headline latency. Retries remain separate attempts; a later success
does not erase earlier failures. Decode and prefill identify the serving regime represented by one
MoE-layer collective; they do not change the timed primitive at an otherwise identical shape.

The versioned conditioning contract is part of scheduled and evidence identity.

Logical payload bandwidth is:

`logical_payload_bytes / measured_latency_seconds`

Normal-mode payload bytes use rank-deduplicated token-rank activations; low-latency bytes use
token-expert assignments. Both add required scale bytes at the named boundary and exclude expert
metadata, padding, and backend buffer capacity. Algorithm bandwidth, bus bandwidth, wire
utilization, and physical-link utilization are not emitted without a defined primitive model or
transport counters. Logical bandwidth must never be labeled physical bandwidth. Payload and token
rates are named `rate_at_latency_percentile`: bytes or tokens divided by the matching latency
percentile. They are lower-tail service rates at p99 latency, not p99 percentiles of an inverted
rate distribution.

## Correctness

An implementation-independent oracle uses an expert-specific deterministic transform so wrong expert
routing cannot pass an identity roundtrip. For every rank and point it verifies:

1. destination rank/expert, source token, multiplicity, gate weight, and receive counts;
2. dispatched payload and metadata before timing;
3. combined output before timing;
4. unchanged semantic inputs through all timed samples; and
5. dispatched payload/metadata and combined output again after timing.

Normal-mode adapters use activation-only, unweighted rank-sum combine. The oracle builds each rank's
gate-weighted expert aggregate before combine, independently derives `sum(gate * expert(token))`, and
checks the dispatch metadata and transformed output. Low-latency adapters separately verify the
expert-packed source/expert assignment, native gate weights, and gate-weighted combined output. Both
contracts compare against the reference activation. The combine gate is `rtol=0.05, atol=0.02` for
the BF16 communication path. This threshold is a correctness gate, not an estimate of transport
error. Any failed rank or point makes the case ineligible in the result it writes.
Pre/post dispatch behavior is checked against canonical source-token metadata and expected output.
Native receive slots may be assigned nondeterministically, so physical receive order is not treated
as a correctness property.

## Result Artifact

One raw case document uses `format: "collectivex.ep.v1"`, carries `schema_version: 1`, rejects
unknown fields, and contains:

- `case`: stable case ID, suite, and coordinate;
- `workload`: canonical identity and logical MoE shape;
- `measurement`: sampling, component states, timing, and byte accounting;
- `implementation`: instantiated class/API, pinned source, loaded libraries, and resources;
- `topology`: requested and realized SKU, devices, placement, scale-up domain, and transport;
- `provenance`: source SHA, image/squash hashes, allocation, run, and attempt;
- `rows`: point latency, byte accounting, token rate, correctness, load, fanout, and anomaly
  evidence; and
- `outcome`: `success`, `failed`, `invalid`, `unsupported`, with `diagnostic` and reasons.

Exact per-point samples are emitted as detached `collectivex.samples.v1` documents referenced by path
and byte count, so the raw document stays compact. Each dispatched case writes its raw result document;
unsupported or never-run cells produce no synthetic record. Private
environment details (hosts, addresses, device selectors, credentials, workspace paths) remain in
local mode-0600 logs and ignored operator notes and never enter an emitted artifact.

## Identity

Identifiers are readable factor strings:

- `case_id`: `{sku}-{backend}-{workload}-{mode}-{phase}-ep{ep}-{routing}` with the remaining
  case factors appended in stable key order;
- `attempt_id`: `case_id` plus `-a{ordinal:02d}`; and
- `point_id`: `case_id` plus `-t{tokens_per_rank}`.

Canonical workload files use readable routing and shape coordinates and are validated against the
deterministic generator. Detached sample documents are referenced by path and byte count. Content
SHA-256 is retained only for the mounted squash; source and library revisions use Git commits and
trees. DeepEP V2 uses a fixed NVCC random seed and validates the complete generated-kernel set.
Hybrid binds its realized auto-tuned config and complete kernel-key set. Pinned source trees, build
recipes, runtime versions, and dependencies remain bound to the case factors.

These IDs let a consumer group matched configurations and separate distinct ones. The backend does
not itself compute cohorts, controlled comparisons, sensitivity pairs, eligibility, or
recommendations — a reader decides which cases to surface and how to compare them.

## Execution Isolation

Every non-MNNVL scale-out case uses operator-pinned socket and RDMA selectors. The launcher rejects
missing or partial profiles, then probes every allocated node for the configured interface, active
HCA port, and configured GID before backend initialization. It never substitutes a default route,
inherited runner environment, or transport fallback. Scale-up and MNNVL cases clear the profile;
scale-out NCCL/RCCL forces `NCCL_NET=IB` and exact HCA matching. Selector values remain in encrypted
config and mode-0600 private logs.

Repository staging uses a pre-existing, runner-owned, group/world non-writable shared base outside
the checkout and workflow workspace. The parent process resolves the exact execution child before
copying, claims it with a runner-owned marker, and verifies that all allocated nodes can read and
write the same bytes. Cleanup waits for confirmed allocation teardown and removes only that child,
including a safely identified partial claim. V2 and Hybrid source is fetched before allocation at an
exact pinned revision, followed by exact Git tree, submodule, and local-patch validation.

H200, B200, and B300 may derive that private base beneath the validated operating-system account home
when it is compute-visible. H100 instead derives a sibling of its shared container directory, never a
child of image storage. The launcher still proves cross-node visibility before any benchmark starts.
Canonical B300 execution ignores the legacy operator `stage_dir` field and always derives the base
from the validated shared account home. Its UID-mapped Actions shell may accept that exact base when
its owner matches the private parent owner; explicit stages and all other runners retain the strict
effective-UID ownership rule. An execution-ID suffix isolates parallel B300 workers. The current
NFS export may realize a newly created base as
UID 0; only that creation path is accepted, while a pre-existing root-owned base is rejected.
Canonical GB300 execution likewise ignores its legacy group-writable `stage_dir` and derives an
execution-specific private base beneath the validated compute-visible account home.

## Image Pinning And Build Isolation

Enroot imports configured container tags with a fixed `SOURCE_DATE_EPOCH` and versioned cache
generation; every mounted squash is freshly hashed. Image-provided DeepEP is also checked against
exact package versions and its expected API. Source-built DeepEP V2 uses
a separate mode-0700 cluster-local cache mounted only as `/cx-cache`. Its content key binds a
versioned build recipe, CPU/GPU architecture, upstream source trees, and pinned
build dependencies. The cache is never an artifact; per-execution source/results stages remain
isolated and disposable, and marker plus runtime probes fail closed before reuse. The runner UID is
inside the trusted cluster boundary: this cache guards against stale or accidental mutation, not
hostile same-UID jobs. Only an unpublished partial build may be reset automatically; a cache that
fails integrity or runtime checks is left intact and rejected so a concurrent allocation cannot lose
files it is using.

## Neutral Artifact Delivery

There is no results server, attached store, or managed object store. Each shard runs one allocation,
emits per-case result JSON plus detached sample JSON and a small mechanical summary, and uploads them
as GitHub artifacts with `always()` so a red or partial run still uploads. A case counts as successful
on the benchmark's own return code; there is no schema, completeness, detached-sample, or privacy
validation step before upload, and failed or unsupported cells produce no synthetic record.

No step promotes a run, builds a dataset, or advances a channel; the artifacts are the output. Any
downstream display or comparison is the consumer's responsibility.

## Legacy Data

Historical numeric schemas 3-5 are outside this benchmark's artifacts. They remain historical
diagnostic evidence and are not produced or consumed by the current sweep.
