# CollectiveX EP Pre-V1 Baseline

<div align="center">

**English** | [中文](./methodology_zh.md)

</div>

This document describes the implemented BF16 baseline. It is not yet the V1 qualification contract.
Before any V1-tagged run, this English document must match the implemented precision, measurement,
publication, and frontend contracts; counts and digests remain unfrozen. Chinese documentation
synchronization is explicitly deferred for the V1 implementation phase.
H100, H200, B200, B300, GB200, and GB300 precision cells are terminal. Five MI325X/MI355X
precision cells remain provisional.

## Product Boundary

CollectiveX is a communication microbenchmark for:

- comparing EP libraries on one chip/topology;
- comparing EP latency and logical payload bandwidth across systems under the same workload; and
- exposing unsupported, failed, invalid, and unstable evidence without contaminating decisions.

It does not predict serving throughput without a separate correlation study.

## Implemented Matrix

The implemented workload is `deepseek-v3-v1`: hidden 7168, top-k 8, 256 routed experts, packed
placement, and one pinned fixed resource profile per backend/topology/precision. The BF16/BF16
control is portable; endpoint precision suites schedule only allowlisted native communication-format
profiles after real-hardware probes resolve their capability cells. Each case explicitly selects
normal `layout-and-dispatch-v1` or low-latency `expert-packed-weighted-combine-v1` semantics.

- `ep-core-v1`: uniform routing; decode T=1..128 powers of two; prefill T=256/512.
- `ep-routing-v1`: Zipf with EPLB off/on; decode T=128; prefill T=512.
- `ep-low-latency-v1`: DeepEP V1/UCCL native low-latency APIs; uniform decode T=1..128 powers of
  two; the capability contract rejects every other backend instead of fabricating a low-latency path.
- `ep-precision-normal-v1`: nonbaseline native dispatch/combine profiles at decode T=128 and prefill
  T=512; BF16/BF16 endpoint controls are referenced rather than duplicated.
- `ep-precision-low-latency-v1`: nonbaseline native low-latency profiles at decode T=128.
- Current planning matrix: 656 requested cases / 1,648 token points; 379 runnable cases / 916
  points in 54 executable workflow shards/allocation cells; 277 unsupported cases / 732 points.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI325X/MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

Physical host count does not define scope. Both GB cells remain inside one 72-GPU MNNVL scale-up
domain.

Unsupported combinations are terminal outcomes, not silently skipped coverage. DeepEP V2 is the
`ElasticBuffer` introduced by PR #605, pinned with upstream PR #630's minimal pure-scale-up fix and
the exact upstream PR #640 library matcher that excludes NCCL shared-memory mappings.
Scale-up cases request NCCL Device API LSA and fail closed unless the realized LSA team covers the
full EP world. x86 EP16 scale-out uses the hybrid path with GIN and requires two logical scale-out
domains represented by two physical RDMA ranks, with eight scale-up ranks per domain. GB EP16
remains MNNVL scale-up and uses LSA. NVIDIA capabilities declared in source remain unvalidated until
GPU outcomes pass the native oracle and publisher gates. H100 V2 on the current runner pool is a
declared unsupported combination in v1 because NCCL 2.30.4 reports no Device API symmetric-memory
support for its EP8 communicator; that pool can return only after all-rank CUDA P2P/LSA support is
restored. This baseline omits `[cl]`, `[rv]`, quantization, alternate activation/routing profiles,
uneven allocation, placement permutations, model envelopes, and scaling.
H100 EP16 is supported on the healthy runner subset. The private runner overlay excludes pods that
do not expose the host RDMA devices in their network namespace, and allocation validation requires
the complete operator-pinned RoCE profile on every selected node before image import. H100 EP8
remains in scope with a private stage beside its configured shared container directory; unlike the
B300 runner account home, the H100 account home is not compute-visible. B300 EP16 is terminal
unsupported for V1 publication because its currently functional fallback HCAs are not the
GPU-adjacent product fabric; this is an operational topology decision, not a library limitation.
FlashInfer is excluded from v1 after repeatable intermittent execution failures; those failures are
not converted into planned-unsupported coverage.
MoRI EP8 uses MI325X AsyncLL or MI355X IntraNode in normal mode. EP16 uses pinned InterNodeV1 over
2x8 XGMI + RDMA with 96 blocks, 64 RDMA blocks, 8 warps, one QP per PE, and external input. MoRI's
AsyncLL transport is not the genuine low-latency suite contract and is never labeled as such.

## Workload Identity

One canonical workload is generated over the global token batch and sliced by source rank. Expert
indices and gate weights are serialized. Activations use a versioned integer counter formula whose
BF16 values are exact across runtimes; its full identity is bound into the manifest. The manifest
also binds shape/EP coordinates and oracle version. SHA-256 covers canonical bytes and parameters;
library RNG regeneration is not proof of identity.

Routing traffic distinguishes:

- token-expert assignments, which determine expert compute load; and
- rank-deduplicated token payload copies, which determine EP activation traffic.

Adapters may not generate routing or reinterpret one quantity as the other.

## Measurement

Normal mode uses `layout-and-dispatch-v1`: dispatch timing includes layout plus communication, and
combine returns activation payload through an unweighted rank-sum path. Low-latency mode uses
`expert-packed-weighted-combine-v1`: native DeepEP V1/UCCL APIs dispatch token-expert assignments and
perform gate-weighted combine. Expert-output staging is outside isolated combine timing and inside
measured paired roundtrip. Each component declares availability, origin, start/end states, stage
scope, and sample count. A paired-only API reports null isolated components. `isolated_sum` is
derived and never used for throughput or recommendations. Mode is series identity, and normal and
low-latency evidence cannot share a ranking cohort.

Every measured component uses `fixed-512-v1`:

- 64 trials x 8 timed iterations = 512 observations;
- 32 synchronized full dispatch-stage-combine warmups before each available measured component at
  every trial/point;
- roundtrip first, then isolated dispatch and combine, with a fixed per-phase conditioning ladder; and
- per-iteration maximum latency across ranks before nearest-rank p50/p90/p95/p99.

Measured roundtrip p99 is the headline latency. Retries remain separate attempts; a later success
does not erase earlier failures. Decode and prefill identify the serving regime represented by one
MoE-layer collective; they do not change the timed primitive at an otherwise identical shape.

The NCCL/RCCL reference is an end-to-end Python adapter, not a bare fabric primitive. Its dispatch
boundary includes layout, count exchange, a device-to-host split synchronization, fresh receive
allocation, and four payload/metadata all-to-all calls; activation-only combine adds one all-to-all plus
scatter/reduction. Its p99 therefore measures the complete reference-adapter boundary and can be
host/scheduler-sensitive. It is useful for portable system controls but must not be labeled fabric,
link, bus, or single-collective latency.

The versioned conditioning and EPLB planner contracts (reference trace, redundant count, and
placement/remap version) are part of scheduled and evidence identity.

Logical payload bandwidth is:

`logical_payload_bytes / measured_latency_seconds`

Normal-mode payload bytes use rank-deduplicated token-rank activations; low-latency bytes use
token-expert assignments. Both add required scale bytes at the named boundary and exclude expert
metadata, padding, and backend buffer capacity. Algorithm bandwidth, bus bandwidth, wire utilization,
and physical-link utilization are not published without a defined primitive model or transport
counters. Logical bandwidth must never be labeled physical bandwidth. Published payload and token
rates are named `rate_at_latency_percentile`: bytes or tokens divided by the matching latency
percentile. They are lower-tail service rates at p99 latency, not p99 percentiles of an inverted rate
distribution.

## Correctness

An implementation-independent oracle uses an expert-specific deterministic transform so wrong
expert routing cannot pass an identity roundtrip. For every rank and point it verifies:

1. destination rank/expert, source token, multiplicity, gate weight, and receive counts;
2. dispatched payload and metadata before timing;
3. combined output before timing;
4. unchanged semantic inputs through all timed samples; and
5. dispatched payload/metadata and combined output again after timing.

Normal-mode adapters use activation-only, unweighted rank-sum combine. The oracle builds each rank's
gate-weighted expert aggregate before combine, independently derives `sum(gate * expert(token))`,
and checks the dispatch metadata and transformed output. Low-latency adapters separately verify the
expert-packed source/expert assignment, native gate weights, and gate-weighted combined output. Both
contracts compare against the semantic value after the declared communication codec. The frozen
combine gates are `rtol=0.05, atol=0.02` for BF16, `rtol=0.06, atol=0.03` for native logfmt10, and
`rtol=0.08, atol=0.04` for native FP8 direct-cast combine. These thresholds are correctness gates,
not estimates of codec error. Direct-cast saturation is measured on the exact transformed native
combine input, and the required saturation count is zero. Any failed rank or point makes the case
ineligible.
Pre/post dispatch evidence is hashed in canonical source-token order. Native receive slots may be
assigned nondeterministically, so physical receive order is not treated as a correctness property.

## Native Result

One raw case document uses `format: "collectivex.ep.v1"`, rejects unknown fields, and contains:

- `case`: stable case ID, suite, required tier, and coordinate;
- `workload`: canonical identity and logical MoE shape;
- `measurement`: sampling, component states, timing, and byte accounting;
- `implementation`: instantiated class/API, pinned source, loaded libraries, and resources;
- `topology`: requested and realized SKU, devices, placement, scale-up domain, and transport;
- `provenance`: source SHA, image/squash hashes, allocation, run, and attempt;
- `rows`: point latency, byte accounting, token rate, correctness, load, fanout, and anomaly evidence; and
- `outcome`: `success`, `failed`, `invalid`, `diagnostic`, or `unsupported`, with reasons.

Raw result documents and exact samples pass through transient GitHub delivery artifacts before the
publisher archives them in the private bundle; they never enter the public tree. Private environment
details remain in local mode-0600 logs and ignored operator notes; they are never archived or
published. Every expected case has one terminal selected outcome while every attempt remains retained.

## Identity And Comparisons

Canonical JSON produces three full SHA-256 IDs:

- `series_id`: all locked factors except token coordinate and repeat allocation;
- `point_id`: `series_id` plus token coordinate; and
- `evidence_id`: `point_id` plus allocation/run/attempt/sample checksum.

Locked factors include workload bytes, measurement and sampling contract, resources, realized
topology, implementation/build, loaded libraries, image/squash, runtime, and source SHA.
Deferred code generation is captured before measurement and recaptured afterward. DeepEP V2 uses a
fixed NVCC random seed and binds final cache keys plus generated-source and executable-SASS hashes;
raw CUBIN bytes remain private diagnostics. Hybrid binds its realized auto-tuned config and complete
kernel-key set while retaining rank-local shared-object hashes as private diagnostics. Locally built
extension hashes are diagnostic; their pinned source trees, build recipe, runtime, and dependencies
remain series-bound.
The series identity includes the case ID, which binds the complete scheduled token ladder and the
frozen percentile, rank-reduction, conditioning, warmup, and correctness semantics.

A controlled comparison declares one contrast:

- `library`: backend implementation and its pinned fixed resource profile may differ; the realized system,
  workload, EP, resource policy, source, and measurement remain matched;
- `chip`: a controlled platform contrast. The full realized system/topology and pinned resource
  profile may differ while workload, EP, placement class, resource policy, backend lineage, source,
  and measurement remain matched. It is not a silicon-only comparison;
- `system`: all hardware/backend differences stay visible while workload, EP, and measurement match;
- `routing`: routing distribution/EPLB differs while the static implementation build/generator,
  system, model shape, resource profile, and measurement remain matched. Uniform and Zipf without
  EPLB reuse the same generated implementation; EPLB's physical-expert/JIT configuration remains an
  explicit treatment difference.

Any undeclared mismatch rejects the overlay. Chip/system results describe measured systems, not
silicon alone.

## Evidence Policy

Capability declarations say what may be attempted; artifacts determine evidence status. Promotion
requires exact expected coverage with no missing, extra, duplicate, malformed, or heterogeneous
case. Public coverage preserves each matrix disposition; promotion requires every runnable case to
succeed and every planned-unsupported case to remain unsupported in every selected run. Only the
pinned canonical full-v1 matrix, with a decision-grade library, chip, system, and routing cohort,
may advance `dev-latest`; partial matrices remain diagnostic. The full-matrix digest intentionally
pins the exact workflow shard grouping as well as the requested cases, so changing `--max-cases`
or the SKU round-robin scheduling order produces diagnostic-only runs even when case coverage is
unchanged. Superseded retries,
planned-unsupported outcomes, and unstable comparison cohorts may render diagnostically but cannot
rank or recommend; every successful required series in a promoted dataset remains decision-grade.
Any failed, invalid, or diagnostic retry of a runnable case blocks promotion even if a later retry
succeeds. Routing cohorts are comparable-experimental sensitivities and never produce configuration
recommendations; official library/platform/system cohorts own actionable recommendations.

A point becomes decision-grade only after three independent workflow runs and allocation IDs pass
correctness, identity, provenance, tail gates, p50/p99 repeat-stability thresholds, and stable ordering. The
publisher, not the frontend, computes eligibility, controlled cohorts, sensitivity pairs, and
recommendations.

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
including a safely identified partial claim. The same-run V2/Hybrid source archive is fully validated
under fixed member and expanded-size bounds, and only the selected pinned root is extracted; a
symlink is accepted only when it is a relative leaf pointing to a regular member inside the same
backend root, followed by exact Git tree/submodule validation.
H200, B200, and B300 may derive that private base beneath the validated operating-system account
home when it is compute-visible. H100 instead derives a sibling of its shared container directory,
never a child of image storage. The launcher still proves cross-node visibility before any benchmark
starts.
Canonical B300 execution ignores the legacy operator `stage_dir` field and always derives the base
from the validated shared account home. Its UID-mapped Actions shell may accept that exact base when
its owner matches the private parent owner; explicit stages and all other runners retain the strict
effective-UID ownership rule. A hashed execution-ID suffix isolates parallel B300 workers without
exposing private runner identity. The current NFS export may realize a newly created hashed base as
UID 0; only that creation path is accepted, while a pre-existing root-owned base is rejected.
Canonical GB300 execution likewise ignores its legacy group-writable `stage_dir` and derives an
execution-hashed private base beneath the validated compute-visible account home.

## Artifact Validation And JIT Delivery

There is no self-hosted service, Vercel storage, GCP, Neon, managed database, or managed object
store. The publication workflow uses runner-local temporary storage only as a disposable validation
and promotion workspace:

```text
$COLLECTIVEX_STORE_ROOT/
  private/incoming/          # write-once downloaded GHA attempts
  private/bundles/<sha256>/  # immutable source archives, native results/samples, matrix, checksums
  private/quarantine/        # rejected attempts plus machine-readable reasons
  public/datasets/<sha256>/  # immutable sanitized frontend datasets
  public/channels/           # promoted dev-latest pointer; never served from persistent storage
  locks/
```

Private and public trees use separate permissions. JSON manifests and checksums are authoritative;
a rebuildable catalog is only an index. Raw sweep artifacts are transient publisher input; only the
sanitized promoted NDJSON is retained as a frontend publication artifact.

Container tags are checked against pinned registry digests. Enroot imports use a fixed
`SOURCE_DATE_EPOCH` and versioned cache generation; every mounted squash is freshly hashed into
series identity. Image-provided DeepEP is also checked against exact per-architecture wheel and
installed-file fingerprints, so a stale cache cannot inherit the pinned source identity.
Source-built DeepEP V2 uses a separate mode-0700 cluster-local cache mounted only as `/cx-cache`.
Its content key binds a versioned build recipe, verified image digest, CPU/GPU architecture,
upstream source trees, and pinned build dependencies. The cache is never an artifact or publisher
input; per-execution source/results stages remain isolated and disposable, and marker plus runtime
probes fail closed before reuse. The runner UID is inside the trusted cluster boundary: this cache
guards against stale or accidental mutation, not hostile same-UID jobs. Only an unpublished partial
build may be reset automatically; a published cache that fails integrity or runtime checks is left
intact and rejected so a concurrent allocation cannot lose files it is using.

Publication is fail-closed:

1. acquire an exclusive filesystem lock and stage on the destination filesystem;
2. archive source bytes before parsing;
3. require the exact matrix-declared artifact set and reject every unconsumed archive member;
4. validate strict schemas, privacy, checksums, identities, timing, and exact matrix outcomes;
5. write checksums and `COMPLETE`, fsync, then atomically rename the private bundle;
6. build and validate the sanitized content-addressed dataset, fsync, then atomically rename it;
7. atomically replace `dev-latest.json` only when every promotion gate passes.

Rejected attempts remain only in the disposable private workspace and short-lived source artifacts;
they never advance `dev-latest` or enter a production channel. The workspace is destroyed with the
publication runner and is never attached to the frontend. No publication artifact is emitted unless
all three selected bundles advance `dev-latest`.

`publisher.py ingest` accepts the exact matrix plus one `--artifact` directory or ZIP per GitHub
artifact. `promote` accepts explicit immutable bundle IDs. Default `verify` requires
the private workspace; it also verifies `dev-latest` when present, while an explicit
`--channel dev-latest` requires it. The workflow copies only the verified sanitized dataset to a
one-record `collectivex_public_v1_<sha256>.ndjson` artifact. Raw artifacts and private workspace
content are never bundled into the application.

Sweeps default to `release_tag=unversioned`. The main-registered `collectivex-sweep.yml` owns
`sweep`, `publish-v1`, and `refresh-v1`, so its branch revision remains dispatchable. V1 emits a
marker bound to the run ID, first attempt, qualification index, source SHA, and locked matrix digest.
Publication accepts exactly three unique successful run IDs from one source SHA with qualification
indices 1, 2, and 3, downloads their immutable artifacts, and passes the same provenance assertions
to `publisher.py ingest`. Refresh requires an exact source run and dataset digest and reuploads the
same validated sanitized bytes. Partial, filtered, untagged, cross-source, rerun, failed, expired,
or digest-mismatched inputs fail closed.

Using a server-side GitHub read token, the frontend discovers the latest successful version-scoped
publication run and downloads the publication artifact just in time. It requires exactly one root
NDJSON entry, validates UTF-8, schema, promotion status, and filename/body SHA-256, then exposes a
short-lived versioned channel pointer and immutable versioned dataset URL. The benchmark-version
selector currently exposes V1; later versions require separate release and publication identities.
The frontend never invents missing values, selects retries, or recomputes decision eligibility.

## Legacy Data

Numeric schemas 3-5 are outside the v1 publisher and frontend reader. They remain historical
diagnostic evidence and cannot seed `dev-latest` or drive v1 decisions.
