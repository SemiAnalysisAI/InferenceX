# CollectiveX EP Benchmark Methodology

CollectiveX schedules expert-parallel (EP) communication benchmarks, executes them on real
accelerator allocations, and uploads the neutral artifacts each run emits. It does **not** validate
those artifacts, promote, rank, recommend, select, hide, or decide what any consumer displays. The
frontend reads the neutral matrix, result, and summary artifacts and makes its own coverage
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
fixed BF16 on every backend; precision is not a swept dimension. Every case uses the normal
`layout-and-dispatch-v1` semantics.

- `ep-core`: uniform routing over the workload's token ladders — for `deepseek-v3`, decode
  T=1..512 powers of two and prefill T=1024..8192 powers of two. Ladders are model-specific and
  live with the workload in `configs/sweep.json`.

`sweep_matrix.py` materializes the requested SKUs, backends, EP sizes, and token ladders into a
matrix document, then extracts strict per-shard controls. `--only-sku`, `--exclude-skus`, and
`--ep-sizes` select a subset; a subset produces a smaller matrix, not a different contract. The
matrix is generated per dispatch; there is no frozen matrix digest or locked case count.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

Physical host count does not define scope. Both GB cells remain inside one 72-GPU MNNVL scale-up
domain.

Unsupported combinations are explicitly classified in the matrix, not silently skipped coverage. DeepEP V2 is the
`ElasticBuffer` introduced by PR #605, pinned with upstream PR #630's minimal pure-scale-up fix and
the exact upstream PR #640 library matcher that excludes NCCL shared-memory mappings. Scale-up cases
request NCCL Device API LSA and fail closed unless the realized LSA team covers the full EP world.
x86 EP16 scale-out uses the hybrid path with GIN and requires two logical scale-out domains
represented by two physical RDMA ranks, with eight scale-up ranks per domain. GB EP16 remains MNNVL
scale-up and uses LSA. MoRI EP8 uses the direct IntraNode kernel on every CDNA SKU; EP16 uses pinned
InterNodeV1 over 2x8 XGMI + RDMA with 96 blocks, 64 RDMA blocks, 8 warps, one QP per PE, and external
input. No cell runs a low-latency-family kernel: throughput-oriented kernels are measured across the
full token ladder on both vendors.
Whether a given SKU/backend/EP cell is attempted is a capability fact; whether it succeeded is
decided only by the emitted artifact.

## Workload Identity

One deterministic workload is generated over the global token batch from the workload's seed in
`configs/sweep.json` (part of the workload identity, baked into every scheduled case) and sliced by
source rank; a keyed BLAKE2b counter over the (token, slot, attempt, stream) coordinates produces
byte-identical expert indices and gate weights on every runtime, and the harness proves the
realized routing trace identical across ranks before a case can succeed.

Routing traffic distinguishes:

- token-expert assignments, which determine expert compute load; and
- rank-deduplicated token payload copies, which determine EP activation traffic.

Adapters may not generate routing or reinterpret one quantity as the other.

## Measurement

Normal mode uses `layout-and-dispatch-v1`: dispatch timing includes layout plus communication, and
combine returns activation payload through an unweighted rank-sum path. Expert-output staging is
outside isolated combine timing and inside the measured paired roundtrip. Each component declares
availability, origin, and sample count. A paired-only API reports null isolated components.
`isolated_sum` is derived. The artifact records the mode so a reader can keep distinct measurement
contracts separate.

Every measured component uses one fixed timing profile, defined once in `configs/sweep.json`
and baked into every scheduled case:

- 256 trials x 8 timed iterations = 2048 observations;
- 32 synchronized full dispatch-stage-combine warmups before each available measured component at
  every trial/point;
- component measurement order rotates each trial (`trial_order`) so every timed component occupies
  every position in the sequence, over a per-trial-rotated token ladder; and
- per-iteration maximum latency across ranks before nearest-rank p50/p90/p95/p99.

Measured roundtrip p99 is the headline latency. Decode and prefill identify the serving regime
represented by one MoE-layer collective; they do not change the timed primitive at an otherwise
identical shape. Ascending through the ladder, each measured shape is conditioned with 8 untimed
full roundtrips — settling clocks, fabric, and buffer state — before it is correctness-checked;
all timing happens after every shape is warmed and checked. Conditioning rounds are never
measured or emitted.

Logical payload bandwidth is:

`logical_payload_bytes / measured_latency_seconds`

Normal-mode payload bytes use rank-deduplicated token-rank BF16 activations (2 bytes per value, no
scale payload) and exclude expert metadata, padding, and backend buffer capacity. Algorithm bandwidth, bus bandwidth, wire
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
gate-weighted expert aggregate before combine and derives the expected combine from the values
actually communicated, reproducing the two-level reduction: each destination rank casts its FP32
aggregate to the payload dtype (BF16) exactly as the adapter does; ranks sharing a scale-up domain
(NVLink/MNNVL) reduce in FP32, and each domain casts its aggregate to BF16 for the scale-out send
before those partials are summed. A group that fits in one scale-up domain (`ep_size <=
scale_up_domain` — every EP8 case and the MNNVL EP16 cases) has a single domain and no scale-out
rounding; a multi-node RoCE EP16 group carries one BF16 partial per node. Modelling that per-domain
cast is what lets the gate stay tight — max elementwise relative error (denominator clamped at 0.02)
below `8 * 2^-8`, the residual accumulation-order ambiguity — across scale-up and scale-out topologies
alike (omitting it left multi-node EP16 ~0.048 off, above the gate). It is a correctness gate, not an
estimate of transport error. Any failed rank or point makes the case ineligible in the result it writes.
Pre/post dispatch behavior is checked against canonical source-token metadata and expected output.
Native receive slots may be assigned nondeterministically, so physical receive order is not treated
as a correctness property.

## Result Artifact

One raw case document carries `record_type: "case-attempt"` and the single `version`, and contains:

- `identity`: `case_id`, `attempt_ordinal`, `case_factors` (SKU and the scheduled case — backend,
  EP size, mode, phase, suite, workload, and the topology coordinate), and `allocation_factors`
  (run id, run attempt, source SHA);
- `workload`: `cross_rank_consistent`, whether the routing trace was proven identical across ranks;
- `measurement`: dispatch/combine dtype and semantics, `sampling`, and the per-point `rows`;
- `implementation`: backend name and kernel generation;
- `topology`: requested SKU/product, placement, nodes, scale-up domain, transport, and world size;
- `provenance`: the mounted image tag and source SHA; and
- `outcome`: `status` (`success` or `invalid`) and `reasons`.

Each `rows` entry carries point latency, byte accounting, token rate, correctness, load, and fanout;
per-point statistics are summarized in place, not emitted as separate documents. Each dispatched
case writes exactly this one raw result document; unsupported or never-run cells produce no
synthetic record.

## Identity

Identifiers are readable factor strings:

- `case_id`: `{sku}-{backend}-{workload}-{mode}-{phase}-ep{ep}-{routing}`, each factor
  slug-normalized; and
- `attempt_ordinal`: a positive integer distinguishing repeat executions of one `case_id`.

Backend source pins live in `runtime/common.sh` and are enforced by exact fetched-commit comparison;
the loaded DeepEP V2 build is checked for the required `ElasticBuffer` API.

These IDs let a consumer group matched configurations and separate distinct ones. The backend does
not itself compute cohorts, controlled comparisons, sensitivity pairs, eligibility, or
recommendations — a reader decides which cases to surface and how to compare them.

## Execution Isolation

Every non-MNNVL scale-out case uses operator-pinned socket and RDMA selectors. The launcher rejects
missing or partial profiles, then probes every allocated node for the configured interface, active
HCA port, and configured GID before backend initialization. It never substitutes a default route,
inherited runner environment, or transport fallback. Scale-up and MNNVL cases clear the profile;
scale-out NVIDIA forces `NCCL_NET=IB`, while AMD leaves plugin selection to RCCL. Both use exact HCA
matching. Scale-out also pins `NCCL_IB_MERGE_NICS=0` so dual-port NIC fusion cannot disable NCCL GIN
— which the DeepEP V2 EP16 hybrid path requires — and a rail-isolated fabric (`rail_isolated`) adds
`NCCL_CROSS_NIC=0`. Selectors come from the tracked platform registry, optionally overlaid by an
operator config, and appear only in mode-0600 private logs.

Repository staging uses a pre-existing, runner-owned, group/world non-writable shared base outside
the checkout and workflow workspace. The parent process resolves the exact execution child before
copying; backend preparation then runs from that tree on every allocated node. Cleanup waits for
confirmed allocation teardown and removes only that child. DeepEP V2 source is fetched before allocation at an
exact pinned revision, initializes its pinned `fmt` submodule, and applies the required local patch.

H200, B200, and B300 may derive that private base beneath the validated operating-system account home
when it is compute-visible. H100 instead derives a sibling of its shared container directory, never a
child of image storage.
Canonical B300 execution ignores the legacy operator `stage_dir` field and always derives the base
from the validated shared account home. Its UID-mapped Actions shell may accept that exact base when
its owner matches the private parent owner; explicit stages and all other runners retain the strict
effective-UID ownership rule. An execution-ID suffix isolates parallel B300 workers. The current
NFS export may realize a newly created base as
UID 0; only that creation path is accepted, while a pre-existing root-owned base is rejected.
Canonical GB300 execution likewise ignores its legacy group-writable `stage_dir` and derives an
execution-specific private base beneath the validated compute-visible account home.

## Image Pinning And Build Isolation

Enroot imports configured container tags into a per-run-scoped squash keyed by the image tag and
image platform, so one run never reuses another run's imported filesystem. Image-provided DeepEP is
also checked against exact package versions and its expected API. Source-built DeepEP V2 uses
a separate mode-0700 cluster-local cache mounted only as `/cx-cache`. Its path binds CPU/GPU
architecture, image, and upstream commit. The cache is never an artifact; per-execution
source/results stages remain isolated and disposable, and runtime probes fail closed before reuse. The runner UID is
inside the trusted cluster boundary: this cache guards against stale or accidental mutation, not
hostile same-UID jobs. Only an unpublished partial build may be reset automatically; a cache that
fails integrity or runtime checks is left intact and rejected so a concurrent allocation cannot lose
files it is using.

## Neutral Artifact Delivery

There is no results server, attached store, or managed object store. Each shard runs one allocation,
emits per-case result JSON and a small mechanical summary, and uploads them as GitHub artifacts with
`always()` so a red or partial run still uploads. A case counts as successful on the benchmark's own
return code; there is no completeness or privacy validation step before upload, and failed or
unsupported cells produce no synthetic record.

No step promotes a run, builds a dataset, or advances a channel; the artifacts are the output. Any
downstream display or comparison is the consumer's responsibility.

## Legacy Data

Historical numeric schemas 3-5 are outside this benchmark's artifacts. They remain historical
diagnostic evidence and are not produced or consumed by the current sweep.
