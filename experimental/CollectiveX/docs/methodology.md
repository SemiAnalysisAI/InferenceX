# CollectiveX EP v1 Technical Design

This is the tracked technical design for new CollectiveX expert-parallel results. Active work and
exit criteria live in `../goal.md`; historical run narratives are evidence, not contract.

The result namespace is `collectivex.ep.v1`. New producers must use it end to end: matrix,
benchmark, bundle, projection, and frontend. Numeric schemas 3 through 5 are import-only legacy.

## Product boundary

CollectiveX measures MoE dispatch, combine, and their paired roundtrip so users can:
- compare EP libraries on one chip and topology;
- compare EP latency and logical payload bandwidth across chips at the same logical workload; and
- inspect failures, unsupported cells, topology effects, and tail stability without contaminating rankings.

This is a communication microbenchmark. It does not claim to predict serving throughput unless a
separate end-to-end correlation study demonstrates that relationship.

## Record model

Each JSON result document has `format: "collectivex.ep.v1"` and exactly one terminal outcome per
expected case. Unknown fields, invalid enums, missing nested identity, or zero parsed documents fail.

Required top-level groups are:
- `case`: stable case ID, suite membership, required evidence tier, and swept coordinate;
- `workload`: logical MoE shape and canonical routing identity;
- `measurement`: timing boundary, sampling schedule, component availability, and byte accounting;
- `implementation`: library, instantiated API, build, runtime, and resource identity;
- `topology`: requested and realized placement and transport;
- `provenance`: source, image, loaded libraries, allocation, attempt, and timestamps;
- `rows`: per-point latency, bandwidth, correctness, and tail evidence; and
- `outcome`: `success`, `failed`, `invalid`, `diagnostic`, or `unsupported`, with reasons.

Raw samples and private environment data live in the immutable run bundle, not the public row; every
result and failure retains its case ID and attempt ID.

## Workload contract

A workload is generated once over the global token batch. Every rank materializes only its assigned
slice; adapters may not generate their own routing. The serialized canonical workload includes:

- phase, tokens per rank, hidden size, top-k, expert count, EP size, and source-token allocation;
- dispatch and combine dtypes, quantization/scaling layout, alignment, and capacity policy;
- routing distribution, seed, routing step, expert placement, EPLB mapping, and trace checksum; and
- exact input values, gate weights, expected receive counts, and oracle version.

The headline shape is DeepSeek-V3-like (`hidden=7168`, `top_k=8`, `experts=256`), but every shape is
named and checksummed. Decode and prefill are distinct cases; dropped points are terminal outcomes.

## Promoted v1 matrix

The promoted matrix is intentionally finite:

- `ep-core-v1`: uniform routing, the full decode ladder, and prefill 256/512 (T=128 is measured once
  in the decode ladder because phase does not change the kernel);
- `ep-routing-v1`: one Zipf trace with EPLB off/on at decode 128 and prefill 512/2048; and
- 39 runnable stack/topology cells, producing 232 cases and 618 token points before repeat allocations.

Every promoted case is normal mode, BF16 dispatch/combine, backend-tuned resources, canonical
`deepseek-v3-v1`, and `layout-and-dispatch-v1`. Balanced, rank-local, hotspot, heavier Zipf, temporal,
uneven-token, model-envelope, placement, scaling, and quantized-combine sweeps are manual diagnostics
or follow-on studies, not missing v1 coverage.

DeepEP PR #605 V2 is not a runnable v1 cell yet. Historical V2-labelled runs used legacy `Buffer`;
the real `ElasticBuffer` adapter must land before V2 re-enters the matrix. It will add eight cells,
48 cases, and 128 points, yielding the final 47-cell/280-case/746-point v1 target.

## Measurement contracts

The timing boundary is named and immutable. Implementations advertise supported contracts; an
unsupported pairing must fail before allocation or emit `unsupported` without timing.

### `layout-and-dispatch-v1`

Dispatch includes routing-layout generation and communication. Input quantization and receive-side
dequantization are outside the timed region. This is the common library-comparison boundary only
when every selected adapter can implement the same start and stop states.

### `cached-layout-comm-only-v1`

The exact routing layout or handle is prepared and validated before timing, then reused. The timer
covers dispatch from that cached state, which may still include packing, local movement, handle work,
and communication. Handle reuse is bound to the routing checksum. This contract is never overlaid
with a layout-inclusive result.

### `runtime-visible-v1`

Timing starts at the runtime-visible input state and ends when the expert input or combined token
output is consumable. Any cast, scale generation, layout, dequantization, event wait, or staging
inside that boundary is recorded in `stage_scope` and timed consistently for isolated components
and paired roundtrip.

Only `layout-and-dispatch-v1` enters the promoted v1 matrix. Cached-layout (`[cl]`) is a decomposition
diagnostic, not a communication-only portable contract. Runtime-visible (`[rv]`) duplicates the BF16
path and is retained only for a future targeted quantization-cost study. Native low-latency (LL)
remains manual until it has matched normal-mode semantics, correct byte accounting, one honest timing
contract, and evidence-gated platform support. Legacy `[cl]`, `[rv]`, and LL rows remain importable and
displayable but cannot rank or recommend.

### Component semantics

`dispatch`, `combine`, and `roundtrip` each have `availability`, `origin`, `start_state`, and
`end_state`. Unmeasured components are null. A paired-only implementation, such as a stateful
roundtrip protocol, must not copy roundtrip samples into dispatch or combine. `isolated_sum` is a
derived diagnostic and is never a measured latency, throughput denominator, or recommendation.

## Sampling and timing

Every scored point uses `fixed-512-v1`:

- 64 trials;
- 8 timed iterations per trial, for 512 observations per measured component; and
- 32 synchronized, untimed, full dispatch-stage-combine warmups immediately before each
  trial and point.

The realized point order, warmup schedule, retry policy, attempt count, and all failed attempts are
recorded. Backend-specific warmup or sampling changes create a different contract and cannot enter
the same contrast.

Device work is timed with events on the stream that performs the work, with explicit dependencies
for multi-stream operations. Host monotonic time is retained as a diagnostic. Each iteration is
reduced by maximum latency across ranks before percentiles are computed. Report p50, p90, p95, and
p99; measured roundtrip p99 is the headline configuration latency.

Retries never replace earlier attempts. Selection rules operate on the full attempt history so a
successful retry cannot hide instability or bias a curve. Tail gates use suite-versioned thresholds
for p99/p50, exceedance rate, adjacent-point discontinuity, and cross-allocation variation; a failed
tail gate makes the point diagnostic.

## Correctness

Correctness uses an implementation-independent oracle. For each routed token copy it verifies the
destination rank, expert, source token, multiplicity, gate weight, and source-order reconstruction.
A deterministic expert-specific transform ensures that routing to the wrong expert cannot pass as
an identity roundtrip.

For every rank and point, the benchmark must:

1. verify expected and realized receive counts;
2. validate dispatch metadata and payload against the oracle;
3. validate combine output against the oracle before timing;
4. run all timed samples without mutating the semantic input; and
5. validate payload and metadata again after timing.

Quantized paths declare the exact format, scale layout, accumulation behavior, absolute and relative
tolerances, and the reason for each tolerance. A whole document cannot be marked correct from one
implementation or one pre-timing smoke check. Any failed rank or point prevents that case from being
comparison eligible.

## Latency and bandwidth

All latency fields use microseconds. The document records the formula and byte-accounting version
for each bandwidth field.

- `logical_payload_bytes` counts actual routed activation and required scale bytes at the named
  operation boundary. Metadata and padding are reported separately.
- `logical_bandwidth_Bps = logical_payload_bytes / measured_latency_seconds` for that operation.
- paired roundtrip accounting records dispatch and combine payload separately before summing them;
- `roundtrip_tokens_per_second` uses measured paired roundtrip, never `isolated_sum`;
- primitive `algbw` and operation-adjusted `busbw` remain primitive-specific metrics; and
- physical wire utilization is null unless measured transport counters support it.

Logical payload bandwidth is useful for comparing the same EP semantics. It is not physical link
bandwidth and must not be labeled as such. Charts expose byte definitions, units, and denominators.

## Identity and controlled comparisons

Identity is canonical JSON hashed with SHA-256. Three related IDs avoid hiding differences:
- `series_id`: all locked factors except the swept token coordinate and repeat allocation;
- `point_id`: `series_id` plus the swept coordinate; and
- `evidence_id`: `point_id` plus allocation, run, attempt, and sample-set checksum.

Locked factors include workload bytes and routing; measurement contract and component states;
sampling, order, warmups, and retries; requested and achieved resources; physical placement and
transport; instantiated backend API/class/build; loaded libraries; image; runtime; and source SHA.

A comparison declares exactly one contrast axis:
- `library`: backend implementation may differ; workload, chip, topology, resource policy, and
  measurement remain matched;
- `chip`: hardware and realized topology may differ; workload, EP size, placement class, resource
  policy, implementation contract, and measurement remain matched;
- `system`: chip, topology, and backend may differ; workload, EP size, measurement, and declared
  resource policy remain matched, and every varied field remains visible; or
- `resource`: requested resource profile may differ; all other locked factors remain matched.

The validator excludes only the declared axis; any additional difference rejects the overlay. Chip
and system contrasts are measured systems, not silicon-only claims. `standardized`, `normalized`,
and backend-tuned resource policies are distinct classes and are never silently mixed.

## Topology and provenance

Requested and realized topology are both mandatory: chip SKU and architecture, nodes, GPUs per
node, world size, rank-to-node/device/tray map, scale-up domain, locality, transport, fabric, and a
topology fingerprint. Validate `world_size == placement ranks`, allocation capacity, packed-case
occupancy, and platform-registry compatibility before timing.

Placement labels are valid only if execution applies and records that placement. Contradictory SKU,
node, tray, or transport metadata makes the case invalid.

Implementation identity names the instantiated class and probed API, not an inferred package major
version. Legacy DeepEP `Buffer`, PR #605 `ElasticBuffer`, native NVIDIA `contrib/nccl_ep`, and a
PyTorch `all_to_all_single` reference are separate implementations. Record source commit, patches,
native GPU targets, build inputs, image digest, and actually loaded libraries after dynamic builds.

Private hostnames, addresses, device IDs, NIC IDs, and paths are retained only in the private bundle
and removed from the public projection.

## Capability and evidence policy

Capability declarations describe combinations the resolver may attempt; they do not prove that a
cell works or that its measurements are comparable. Evidence status is derived from artifacts:

- `unsupported`: the library or platform cannot represent the requested contract;
- `failed`: setup or execution did not produce a complete result;
- `invalid`: correctness, timing, identity, topology, or schema failed;
- `diagnostic`: valid evidence that does not satisfy comparison or repeat requirements; and
- `eligible`: complete, conforming evidence that may enter a controlled contrast.

Every requested matrix case has one terminal outcome. Missing, extra, duplicate, malformed,
heterogeneous, or wrong-status cases block channel promotion but remain visible as evidence.
Machine-readable quarantine is applied before plotting or decision generation.

A p99 point becomes decision-grade only after three complete independent allocation IDs agree under
the same point identity and pass correctness, coverage, provenance, and tail-stability gates. The
public UI may show diagnostic evidence, but only decision-grade measured roundtrip p99 can drive a
ranking or recommendation.

## Isolated artifact store

Development storage uses one self-hosted machine and one persistent filesystem. It must not depend
on Vercel storage, GCP, Neon, another managed database, or a third-party object store.

`$COLLECTIVEX_STORE_ROOT/private` contains incoming attempts, content-addressed immutable run
bundles, quarantined attempts, raw samples, environments, matrix definitions, outcomes, schemas,
and checksums. `$COLLECTIVEX_STORE_ROOT/public` contains only sanitized content-addressed datasets
and mutable channel pointers such as `dev-latest.json`. The two trees have separate permissions.

`bundle_id` hashes the canonical manifest and file checksums. `dataset_id` hashes projection format,
selection policy, source bundle IDs, and projected checksums; publication time is excluded. JSON
manifests are authoritative. A rebuildable catalog is an index, not a database.

Publication is fail-closed and atomic:

1. take an exclusive filesystem lock;
2. stage on the same filesystem as the destination;
3. verify checksums and strict schemas;
4. compare the full expected matrix with terminal outcomes;
5. verify homogeneous identities and realized timing schedules;
6. write checksums and `COMPLETE`, then fsync files and directories;
7. atomically rename the private run bundle;
8. build, sanitize, validate, fsync, and atomically rename the public dataset; and
9. atomically replace the channel pointer only after all prior steps succeed.

Invalid or incomplete attempts may update a sanitized `latest-attempt` diagnostic pointer but never
`dev-latest`. Channel responses use `no-cache`; immutable dataset responses may use long-lived
caching. GitHub Actions artifacts are transient delivery inputs, not durable authority.

## Legacy imports

Numeric schema versions 3, 4, and 5 are immutable historical inputs. Importers preserve original
bytes, source availability, schema, sampling, timing, and quarantine reasons. They must not rewrite
legacy records as `collectivex.ep.v1`, synthesize missing components, seed `dev-latest`, or drive
rankings, budgets, crossovers, and recommendations.

Legacy data may appear in an explicitly historical evidence view. New comparable results begin only
with native `collectivex.ep.v1` producers and a publisher-created dataset.
