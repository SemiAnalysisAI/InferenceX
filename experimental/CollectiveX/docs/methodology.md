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
placement, and one pinned fixed resource profile per backend/topology. Combine is always BF16;
dispatch precision is a swept dimension — a BF16 control and, on the backends whose FP8 dispatch is
supported upstream (DeepEP V2, MoRI, UCCL-EP, FlashInfer EP), an FP8 dispatch (`bf16`, `fp8`),
caller-prequantized in `normal` mode (the `low-latency` kernels quantize FP8 internally from BF16 on
DeepEP and UCCL-EP, and stay caller-prequantized on MoRI). Because `normal`-mode FP8 is
caller-prequantized, that quantize is a cost a production forward pass pays on the critical path, so
it is charged **inside the measured dispatch** rather than prepared ahead of the timing window; it is
issued as one fused kernel and guarded bitwise against its eager reference. This means an FP8
`normal` dispatch number covers quantize-plus-transport while its BF16 control covers transport
alone, and it is why FP8 `normal` rows are not comparable to runs published before sweep version 2.
`low-latency` rows are unaffected: those kernels either quantize internally (nothing for the caller
to charge) or take pre-quantized input by API contract. NCCL EP is BF16-only this release, so its
cells carry the control alone; the per-backend precision set lives in `sweep_matrix.py`'s
`BACKEND_PRECISIONS` and a backend never emits a case for a precision it does not support.
`normal`-mode cases use the
`layout-and-dispatch-v1` semantics; `low-latency` cases use each backend's decode-kernel semantics
(detailed below).

- `ep-core`: uniform routing over the workload's token ladders — for `deepseek-v3`, decode
  T=1..512 powers of two and prefill T=1024..8192 powers of two. Ladders are model-specific and
  live with the workload in `configs/sweep.json`.

`sweep_matrix.py` materializes the requested SKUs, backends, EP sizes, and token ladders into a
matrix document, then extracts strict per-shard controls. `--only-sku`, `--exclude-skus`,
`--ep-sizes`, and `--precisions` select a subset; a subset produces a smaller matrix, not a
different contract. The matrix is generated per dispatch; there is no frozen matrix digest or locked
case count.

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
scale-up and uses LSA. MoRI EP8 uses the direct IntraNode kernel on every CDNA SKU; its EP16 InterNodeV1 path is
configured but unsupported (transport-layer combine corruption, ROCm/mori#475) and never dispatched.
MoRI runs under its MANUAL launch mode with a pinned launch config, because that is what the engines
run: neither vLLM nor SGLang sets `MORI_EP_LAUNCH_CONFIG_MODE`, and both pin block_num 80 with
rdma_block_num 0. These numbers therefore describe the engine-integrated configuration, not MoRI's
peak -- its own shipped tuning tables reach roughly 1.3-1.4x faster combine at T=256 on gfx950 with a
config no engine selects, and AUTO would not reproduce those tables anyway (there is no BF16 gfx950
dispatch rule and no gfx950 IntraNodeLL combine table, so AUTO falls back to hard-coded defaults and
couples the result to whichever MoRI revision is pinned). UCCL-EP is a drop-in, API-identical DeepEP replacement that keeps the legacy `Buffer`
`dispatch`/`combine` (unweighted rank-sum) but routes it over CPU-proxy GPUDirect RDMA on plain
`libibverbs` — no NVSHMEM/IBGDA — with software message ordering, atomics, and flow control; its
scale-up is single-node `cudaIpc` over NVLink/XGMI (so the scale-up domain is one physical node,
never MNNVL) and its EP16 scale-out uses the same per-SKU RDMA rails as the other backends. NCCL EP
is NVIDIA's native MoE dispatch/combine on the NCCL Device API, driven through the `nccl4py`
bindings; `normal` mode selects its `HIGH_THROUGHPUT` algorithm, whose FLAT `[N, hidden]` receive and
unweighted rank-sum combine match `layout-and-dispatch-v1` exactly, so the same oracle applies. It is
NVIDIA-only and CUDA 13 only, and runs EP8 scale-up on H100/H200/B200/B300 plus EP8 and EP16 on
GB200/GB300, where EP16 stays inside the MNNVL scale-up domain; x86 EP16 scale-out is an unsupported
coverage row, its cross-node GIN path faulting inside `nccl_ep.cc` identically on RoCE and IB across
four SKUs — a GDAKI limit, not a fabric-selection one. FlashInfer EP is TensorRT-LLM's one-sided MNNVL `MoeAlltoAll`, in which each rank writes tokens directly into its peers' workspace windows and combine reads them back, so there is no send/recv pairing and no NVSHMEM; it is GB200/GB300-only for that reason, and runs EP8 and EP16 inside the MNNVL scale-up domain. Its combine is the one place a backend's accumulator precision changes the expectation rather than the tolerance: through 0.6.15 the kernel holds its top-k accumulators in the payload dtype and reduces them with a hand-unrolled pairwise tree, so every level rounds to BF16, and the oracle reproduces that tree exactly rather than loosening the gate to absorb it (0.6.16 rewrote the accumulator to FP32; the adapter reads the installed version and picks the matching model). Those throughput kernels run across the full token ladder in the `normal` mode.

A second `low-latency` mode adds each backend's decode-optimized kernel family. On DeepEP it drives
the legacy `deep_ep.Buffer` low-latency decode kernels (`low_latency_dispatch`/`low_latency_combine`),
which deliver a per-expert padded receive buffer and apply the top-k gate weights inside a source-side
combine (weighted-kernel-sum). For the scoped single-node EP8 cells these run over the intra-node
NVLink low-latency path (`allow_nvlink_for_low_latency_mode`); NVSHMEM/IBGDA (and thus `/dev/gdrdrv`)
is only exercised on the wire by a multi-node scale-out (EP16) run, and single-node EP8 was validated
on H200 with `/dev/gdrdrv` absent. On MoRI it selects the `IntraNodeLL` kernel — a single-call,
pure-intranode decode kernel that keeps the same rank-deduplicated compact layout and plain unweighted
rank-sum combine as the throughput `IntraNode` kernel, so it differs only by kernel type and timing
(the split-phase RDMA-staged `AsyncLL` kernel is deliberately not used — its separate receive phase
does not fit the single-call dispatch/combine contract). Low latency is a decode-phase-only addition
whose runnable set is narrower than and distinct from the throughput kernels', so it is enabled
cell-by-cell from the registry's `ll_backends` map rather than assumed wherever `normal` runs; it is
currently enabled for DeepEP V2 EP8 on H100/H200/B200, MoRI
EP8 on MI300X/MI325X/MI355X, and UCCL-EP EP8 on H100/H200/B200 only (the legacy `Buffer` low-latency kernels; at EP8 these
run `cudaIpc` over NVLink, not the CPU-proxy RDMA path, because the adapter passes `is_intranode`
and UCCL then never starts its proxies. The AMD SKUs drop LL: upstream raised `kNumMaxTopK` 9 -> 16
six days before our pin, and the resulting host assert cannot hold on AMD's 16 warp groups), and NCCL EP EP8 on all six NVIDIA SKUs — its
`LOW_LATENCY` algorithm is the DeepEP-derived decode path, EXPERT_MAJOR receive with a source-side
weighted-kernel-sum combine. Those rows were dropped while every LL leg wedged on stale peer signals
([NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303)) and restored once the single-handle
adapter removed the aliasing that caused it. B300, GB200 and GB300 carry NCCL EP as their only
low-latency row, and it is a `candidate` transport, so those three SKUs publish no production decode
coverage. Whether a given SKU/backend/EP/mode cell is attempted is a capability
fact; whether it succeeded is decided only by the emitted artifact.

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
outside isolated combine timing AND outside the measured paired roundtrip, so `roundtrip` means
dispatch then combine — the transport — in every row. It is reported as its own `stage` component
wherever it does device work. The one exception is the `CX_FP8_CONSUME=dequant` verification hatch,
which puts the conversion back inside the chain on purpose.

Read `implementation.stage_excluded_from_roundtrip` as "there was device-work staging and it was
hoisted out of the chain", not as "this row's roundtrip is stage-free". It is gated on whether the
backend's `stage()` does device work at all, so it is `false` in two unrelated situations, and the
`stage` component is what separates them: **absent** means the backend has nothing to stage (the
staging is a bare pointer assignment, as for NCCL EP and for every BF16 row that hands the receive
buffer straight to combine), while **present alongside `false`** means the `dequant` hatch put the
conversion back inside the chain. A reader that treats `false` alone as "roundtrip includes staging"
will wrongly subtract a cost the row never paid. Each component declares
availability, origin, and sample count. A paired-only API reports null isolated components.
`isolated_sum` is derived.

Headline latency is the p99 of the per-iteration cross-rank MAX (`p50` is emitted alongside it, and
`summarize.py` prints both; the p99 is the figure the published cohorts rank on). MAX is the
reduction because a layer is not finished until
its slowest rank is, so MAX is the completion cost, and it charges inter-rank entry stagger to
whichever component the ranks entered unevenly. How much stagger there is depends on the code
path AND the precision, not only on the fleet: on identical h200 low-latency decode cells the
per-iteration spread is ~9.3 us for deepep-v2 and uccl-ep at BF16 (they share the legacy
`Buffer` path) against ~2.6 us for nccl-ep, and it collapses to ~2.8 us for those same two under
FP8, where the kernel quantises in-kernel and the heavier dispatch self-aligns the ranks. So the
term is not subtractable in any principled way, and MAX alone taxes some rows more than others.

Every row therefore also carries `cross_rank_min_us` (the same iterations reduced with MIN — the
skew-excluded floor) and `cross_rank_spread_us` (per-iteration MAX minus MIN). Read MAX and MIN
as a bracket. Two cells whose MAX gap is smaller than the larger contender's spread are not
separated by the data: rank on roundtrip p50 and call a winner only where MAX and MIN agree on
the ordering. Do not rank on p99 of MAX for multi-node decode cells, where it is dominated by
worst-rank stalls rather than transport — p99 of MIN is the synchronized-cost tail beside it. The
isolated components inherit the preceding operation's per-rank exit stagger, so treat them as
residual-wait diagnostics rather than per-operation costs; the paired roundtrip is the
comparable quantity.

One backend's timed window omits a cost the others pay, deliberately. nccl-ep binds routing with
`ncclEpUpdateHandle`, a collective whose cost scales with the group's token capacity rather than with
the token count, so charging it per iteration would import a ladder-max-proportional term into
dispatch -- the same shape of artifact that sizing HT's combine input to the ladder maximum used to
put under combine. It is therefore bound during the untimed warm-up, which is also what NVIDIA's own
`ep_bench` does (CUDA events around dispatch and combine only, handle update outside the loop). In
low-latency mode there is nothing to exclude: `ncclEpUpdateHandle` returns immediately and the kernel
reads the cached routing inside the timed dispatch. Every other backend pays its layout per timed
call -- uccl-ep calls `get_dispatch_layout` inside dispatch; deepep-v2, MoRI and FlashInfer pass
routing on every call -- and those costs scale with tokens, so they belong in the window.

The artifact records the mode so a reader can keep distinct measurement
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

Comparing these figures against a vendor table is not like-for-like, in a knowable direction.
Every sample here is an eager per-call measurement that includes kernel-launch cost and
inter-rank entry skew, reduced across ranks by MAX. Vendor microbenchmarks published for these
same kernels variously time the named kernel only via a profiler (DeepEP, UCCL low-latency),
replay CUDA graphs (MoRI), average across ranks instead of taking the max (all of them), delete
entry skew with a pre-iteration sleep or an amortized barrier (DeepEP, MoRI), and report the
best of a config sweep (DeepEP V1, MoRI). Expect our headline to read roughly 5-10% above such
a table on a healthy fabric; `cross_rank_min_us` is the per-row figure to place beside one.
Where a like-for-like comparison exists we match or beat: our skew-excluded MoRI dispatch is
0.96x MoRI's own shipped tuning-config best at the same shape and byte count, DeepEP V2 on B300
reproduces DeepEP's published 8x2 figure within 3%, and FlashInfer EP matches NVIDIA's published
one-sided kernel within 4% across eight byte-normalized points.

Logical payload bandwidth is:

`logical_payload_bytes / measured_latency_seconds`

Payload bytes use rank-deduplicated token-rank activations and exclude expert metadata,
padding, and backend buffer capacity. BF16 moves 2 bytes per value with no scale payload; an FP8
dispatch moves 1 byte per value, plus per-128-block FP32 scales for DeepEP's and UCCL-EP's blockwise
codec (none for MoRI's plain e4m3 cast), while combine stays BF16 — so the dispatch and combine directions can carry
different byte counts and the roundtrip is their per-field sum. The rank-deduplicated count is exact
for the normal-mode layout. It is also exact for a low-latency kernel that deduplicates per rank
(MoRI's `IntraNodeLL`, whose combine is an unweighted rank-sum). The low-latency kernels that apply
top-k weights inside combine instead send one copy per (token, expert) assignment rather than per
(token, rank), so for a token whose experts share a destination rank this logical count is a lower
bound on the bytes those kernels actually move. Each row states which basis it used in
`logical_copies`, so the two are never silently mixed. Latency (the headline) is
measured directly and is unaffected. Algorithm bandwidth, bus bandwidth,
wire utilization, and physical-link utilization are not emitted without a defined primitive model or
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
alike (omitting it left multi-node EP16 ~0.048 off, above the gate).

Low-latency adapters instead use a source-side gate-weighted combine: the kernel multiplies each
expert's returned message by that assignment's top-k weight, so the adapter stages the UNWEIGHTED
per-expert transform and a dedicated per-(source, expert)-slot oracle derives the expected combine as
the gate-scaled sum of per-expert BF16 messages — no per-domain intermediate, since the low-latency
kernels reduce at the source rank. The delivered (source, expert) assignment multiset and per-expert
counts are checked against the routing trace, and the same tight combine gate applies. Under FP8
dispatch the oracle applies the backend's exact per-token cast round-trip to its semantic payload before both the
dispatched-payload compare and this combine expectation, so the payload match stays bit-exact and the
same tight gate holds — the quantization is modeled, not absorbed into a wider tolerance. It is a
correctness gate, not an estimate of transport error. Any failed rank or point makes the case ineligible in the result it writes.
Pre/post dispatch behavior is checked against canonical source-token metadata and expected output.
Native receive slots may be assigned nondeterministically, so physical receive order is not treated
as a correctness property.

## Result Artifact

One raw case document carries `record_type: "case-attempt"` and the single `version`, and contains:

- `identity`: `case_id`, `attempt_ordinal`, `case_factors` (SKU and the scheduled case — backend,
  EP size, mode, precision, phase, suite, workload, and the topology coordinate), and
  `allocation_factors` (run id, run attempt, source SHA);
- `workload`: `cross_rank_consistent`, whether the routing trace was proven identical across ranks;
- `measurement`: dispatch/combine dtype (the realized wire formats — combine always BF16, dispatch
  BF16 or the SKU's FP8 format) and semantics, `sampling`, and the per-point `rows`;
- `implementation`: backend name, kernel generation, and `maturity` — whether a production
  inference engine can select this transport today (`production` = exposed by vLLM's
  `--all2all-backend` or SGLang's `--moe-a2a-backend`; `candidate` = a real transport we
  benchmark that no engine ships a selector for, so its numbers describe the library rather
  than a deployable configuration). The same map is in the registry's `backend_maturity`;
- `topology`: requested SKU/product, placement, nodes, scale-up domain, transport, and world size;
- `provenance`: the mounted image tag and source SHA; and
- `outcome`: `status` (`success` or `invalid`) and `reasons`.

Each `rows` entry carries point latency, byte accounting, token rate, correctness, load, and fanout;
per-point statistics are summarized in place, not emitted as separate documents. Each dispatched
case writes exactly this one raw result document; unsupported or never-run cells produce no
synthetic record.

## Identity

Identifiers are readable factor strings:

- `case_id`: `{sku}-{backend}-{workload}-{mode}-{phase}-ep{ep}-{routing}-{precision}`, each factor
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
