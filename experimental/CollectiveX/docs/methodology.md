# CollectiveX EP Benchmark Methodology

CollectiveX schedules expert-parallel (EP) communication benchmarks, executes them on real
accelerator allocations, and uploads the neutral artifacts each run emits. It does **not** validate
those artifacts, promote, rank, recommend, select, hide, or decide what any consumer displays. The
frontend reads the neutral matrix, result, and summary artifacts and makes its own coverage
and display decisions. This document describes how a case is scheduled, measured, checked, and
recorded, not a publication or qualification contract.

## Product Boundary

CollectiveX is a communication microbenchmark for:

- comparing EP libraries on one chip/topology.
- comparing EP latency and logical payload bandwidth across systems under the same workload.
- surfacing unsupported, failed, invalid, and unstable cases rather than hiding them.

It does not predict serving throughput without a separate correlation study.

## Matrix

The implemented workload is `deepseek-v4-pro` (DeepSeek-V4-Pro 1.6T, the `dsv4` model the rest of
InferenceX benchmarks): hidden 7168, top-k 6, 384 routed experts, packed
placement, and one pinned fixed resource profile per backend/topology. Combine is always BF16.
Dispatch precision is a swept dimension, with a BF16 control and, on the backends whose FP8 dispatch is
supported upstream (DeepEP V2, MoRI, UCCL-EP, FlashInfer EP), an FP8 dispatch (`bf16`, `fp8`),
caller-prequantized in `normal` mode (the `low-latency` kernels quantize FP8 internally from BF16 on
DeepEP and UCCL-EP, and stay caller-prequantized on MoRI). That caller-side quantize is charged
**inside the measured dispatch**, because a production forward pass pays it on the critical path. It
is one fused kernel on DeepEP V2, UCCL-EP and FlashInfer EP, guarded bitwise against its eager
reference. MoRI's is a plain dtype cast and needs neither. So an FP8 `normal` dispatch covers
quantize-plus-transport while its BF16 control covers transport alone, and is not comparable to a row
measured before that change. The sweep `version` deliberately stays 1 across it, so
`implementation.stage_excluded_from_roundtrip` and the presence of a `stage` component are the
discriminators for THAT change.

Those two fields do not separate every generation, and it is worth being exact about which ones
they miss, because the version tag will not help. A same-cell comparison across the durable store
puts numbers on it: the FP8 quantize charge moved dispatch by +64% to +110% (median, small T). The
staging hoist widening moved FlashInfer roundtrip −46% and MoRI BF16 −23%. MoRI's move to the
external input buffer with 16 warps moved FP8 combine +33% to +65% on gfx950. Sizing the NCCL EP
HT combine to its receive count moved combine −10% to −41%. The two-pass chain moved
`pair_period` −5% to −42% everywhere. Of those, only the first two are keyed by the fields above.

One more key, for the rest. **`chain_health.interpair_gap_us` present** marks the two-pass chain:
rows from the single six-events chain carry `chain_health` with `pair_spread_us` alone, and their
`pair_period` is inflated by a host constant rather than being a different-but-valid quantity, so
treat those rows as defective, not merely older. For the MoRI buffer-mode change and the NCCL EP
combine sizing there is **no per-row discriminator at all**. `kernel_generation` reads the same on
both sides, so pre-change rows for those two backends cannot be separated from post-change rows by
any field, and must be excluded from the store rather than keyed around.

If the sweep version is ever bumped, **skip 2 and go to 3.** A short-lived bump to 2 during this
branch's development left stored documents tagged `version: 2` that are currently invisible only
because the frontend reader accepts `[1]`. Publishing support for 2 would silently resurface that
mid-development churn as valid data.

The `fp8_consume` derivation, in full, since the code now points here for it. `dequant` is a
verification hatch rather than a second metric (never a sweep axis, never a default) because the
mismatched-config cost is derivable from what every run already emits:

    dequant roundtrip  ~=  roundtrip + stage        (+2.4% .. -0.1%, b200 LL fp8 ladder)

slightly high because chaining amortises launch overhead (median `rt/(d+s+c)` = 0.93 across the
corpus). The reverse does **not** hold: reconstructing native as `dequant - stage` errs by -11.6% at
T=1, -5% at T=64, and converges only by T=256. This is worst precisely in the decode regime the headline
reports. So measure native and derive dequant, never the other way round. The hatch reproduces
historical deepep-v2/uccl-ep numbers for regression checks (302.0µs against 302.5µs in run
30177021271 at T=1). It does not reproduce MoRI fp8, whose stage now casts only the rows dispatch
filled, nor any pre-hoist BF16 roundtrip.

Read that charge as a **fixed per-call cost, not a payload-proportional one**. On DeepEP V2 decode,
FP8 dispatch p50 minus its BF16 control is 65us on h100, 59us on b200, 27us on b300 and 57us on
gb300 at T=1, flat within a microsecond or two through T=64, then decaying. By T=512 it is slightly
negative on h100 and b300, where halved payload bytes more than repay it. FlashInfer EP carries
~107us on gb300 (its own codec, and a fourth dispatch payload under FP8). At T=1 the FP8 path moves
*fewer* bytes than BF16, so this is per-call work, not transport, and it exceeds the fused quantize's
own device time (1.5-3.6us per SKU) by more than an order of magnitude: the timing window has no host
sync before its start event (see below), so a near-idle stream at T=1 lets host-side launch cost land
inside it. Compiling the quantize reduces this rather than causing it. An eager quantize measures
33-39us worse per decode dispatch on h100 through the same window, but production issues one custom
quantize op into an already-busy stream, so the small-T end of an FP8 `normal` row is the least
production-representative number the suite emits. Compare FP8 and BF16 at the top of the ladder.
`low-latency` rows are unaffected: those kernels quantize internally or take pre-quantized input by
API contract. NCCL EP is BF16-only this release, so its cells carry the control alone. The
per-backend precision set lives in `sweep_matrix.py`'s `BACKEND_PRECISIONS` and a backend never
emits a case for a precision it does not support. `normal`-mode cases use the
`layout-and-dispatch-v1` semantics. `low-latency` cases use each backend's decode-kernel semantics
(detailed below).

- `ep-core`: uniform routing over the workload's token ladders, which for `deepseek-v4-pro` include decode
  T=1..512 powers of two and prefill T=1024..8192 powers of two. Ladders are model-specific and
  live with the workload in `configs/sweep.json`.

A backend may clamp the ladder below that, and every clamped point is reported in the artifact
rather than dropped silently. `workload.ladder_measured`, `ladder_dropped` and `ladder_cap`
record what ran and what did not. DeepEP V2 in `low-latency` mode pre-allocates a fixed receive,
so its ladder cannot exceed that buffer. Both are 256, so the decode ladder runs to its full
extent and only the 512 point is dropped.

That clamp was briefly load-bearing. DeepEP's low-latency combine corrupted the 256 rung on every
Blackwell SKU (B200, GB200 and GB300, EP8 and EP16, both precisions, MNNVL and RDMA alike) while
Hopper stayed clean, stochastically at roughly 1.5-3.3% per invocation, surfacing as one wrong token
row whose norm still matched to 4 significant figures. Upstream PR #642 fixed it with a CTA-scope
fence retiring the combine consumer's shared-memory reads before its staging buffer is recycled. The
commit we pinned was a pre-merge branch head that predated it, so we clamped the measured ladder to
128 until the pin moved to upstream main. The receive is sized from a constant rather than from
`max(ladder)` so clamping cannot change the footprint, which drives both the transport's memory
traffic and the FP8 dequant volume.

`sweep_matrix.py` materializes the requested SKUs, backends, EP sizes, and token ladders into a
matrix document, then extracts strict per-shard controls. `--only-sku`, `--exclude-skus`,
`--ep-sizes`, and `--precisions` select a subset. A subset produces a smaller matrix, not a
different contract. The matrix is generated per dispatch. There is no frozen matrix digest or locked
case count.

| Systems | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink, scale-up | 2x8 NVLink + RDMA, scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI, scale-up | 2x8 XGMI + RDMA, scale-out |
| GB200/GB300 | 2x4 MNNVL, scale-up | 4x4 MNNVL, scale-up |

**A virtualized pool can make a scale-out row measure the hypervisor rather than the fabric.**
h200-dgxc EP16 pays roughly three times the cross-node cost of b300 or h100 on identical topology
and identical traffic, while its EP8 rows are correct. The deficit is confined to the hop. It
sustains ~34 GB/s per node against a nominal 8x400G (~4.2 GB/s per GPU-NIC pair) where bare-metal
h100 reaches wire rate. Reordering the NIC-PE mapping to pair each rank with its socket-local NIC
changed nothing (478µs against a 480µs baseline), which rules the selector out and points at the
GDR path being degraded wholesale inside the guest. The retired b200-nscale pool showed the same
shape. Treat EP16 rows from a virtualized pool as a lower bound on the hardware until the host's
ACS/IOMMU configuration is confirmed.

Physical host count does not define scope. Both GB cells remain inside one 72-GPU MNNVL scale-up
domain.

Unsupported combinations are explicitly classified in the matrix, not silently skipped coverage. DeepEP V2 is the
`ElasticBuffer` introduced by PR #605, pinned at upstream main, which carries that PR plus #630's
minimal pure-scale-up fix, the #640 library matcher that excludes NCCL shared-memory mappings, and
the #642 low-latency combine fence. Scale-up cases
request NCCL Device API LSA and fail closed unless the realized LSA team covers the full EP world.
x86 EP16 scale-out uses the hybrid path with GIN and requires two logical scale-out domains
represented by two physical RDMA ranks, with eight scale-up ranks per domain. GB EP16 remains MNNVL
scale-up and uses LSA. MoRI EP8 uses the direct IntraNode kernel on every CDNA SKU. Its EP16 InterNodeV1 path is
configured but unsupported and never dispatched: `combine()` takes the rank's own routing tensor,
not dispatch's returned recv-slot indices (part of the ROCm/mori#475 corruption was that
caller-side mix-up, guarded upstream by ROCm/mori#546), and with the corrected call single-shot
combine is clean through T=512 — but a residual stochastic corruption remains from T~128 under
repeated execution and at prefill token counts, independent of per-pair drains (not a
buffer-reuse race; suspected driver-level, ionic 25.11 vs upstream's non-reproducing 26.03).
The tw pairs additionally have no cross-node GPU fabric. mi355x EP16 currently has no
publishable transport: UCCL-EP's CPU-proxy RDMA is functional on the Pollara fabric but roughly
13x under its upstream-documented bandwidth (~6 GB/s vs 82 GB/s), unchanged by GPU-memory
registration mode (DMA-BUF vs peer-memory) or traffic class — an ionic-driver-level suspect.
UCCL-EP EP16 is registered on b200-nscale, where it runs at full health on bare-metal IB.
MoRI runs under its MANUAL launch mode with a pinned launch config, because that is what the engines
run: neither vLLM nor SGLang sets `MORI_EP_LAUNCH_CONFIG_MODE`, and both pin block_num 80,
rdma_block_num 0, and `warp_num_per_block` 16 for the intra-node kernel, on dispatch and combine
alike (neither passes a per-call override), with an external input buffer, MoRI's default, which
SGLang sets explicitly. The two are pinned together deliberately: MoRI's tuning tables key combine on
`zero_copy`, selecting roughly 16 warps for external input against 4-8 for a registered buffer, and
the mismatch is measurable in both directions. On MI300X, MI325X and MI355X,
**in registered-buffer mode** 16 warps costs +13-18% combine at T=128 and +61-78% at T=512 against 8.
**In the external-input mode the engines actually run** the same 16 warps *wins* by 14-19% at T=128,
26-27% at T=256, and 9-14% at every prefill rung including T=8192. Below T=32 it gives up 0.2-2.5us,
the only range where 8 is ahead. All arms were correct at every rung. With an external input buffer
the kernel does its own staging copy, bounded by the receive count, so BF16 rows hand over the
dispatch output unchanged and declare `stage` as an explicit unavailable marker (null percentiles,
zero samples). FP8 rows still stage for real, to dequantize the received payload. These numbers
describe the engine-integrated configuration, not MoRI's peak: its shipped tuning tables reach a
faster combine with per-shape block and warp counts no engine selects, and AUTO would not reproduce
them uniformly. Gfx950 ships no IntraNodeLL combine table and no BF16 rule for normal-mode
IntraNode dispatch, so AUTO defaults exactly those two (coupling the result to whichever MoRI
revision is pinned) while tuning the other two, which is not one number about the hardware. How far
off peak is arch-dependent: across buffer modes on MI355X, with
the registered mode's excluded BF16 stage added back, a registered buffer at 8 warps is still 15%
faster at T=512 decode and 8% at T=8192 prefill than the shipped pairing, but that did not reproduce
on gfx942, so treat 0-15% as the honest range for a configuration no engine runs. The low-latency
arm has no engine-integrated configuration to match at all: SGLang's low-latency path pins `AsyncLL`
at 8 warps while this suite uses `IntraNodeLL` (`AsyncLL` is split-phase and fails silently under a
single-call harness), so its launch config is inherited from the normal-mode tuple by choice rather
than by precedent. UCCL-EP is a drop-in, API-identical DeepEP replacement that keeps the legacy `Buffer`
`dispatch`/`combine` (unweighted rank-sum) but routes it over CPU-proxy GPUDirect RDMA on plain
`libibverbs`, with no NVSHMEM/IBGDA and with software message ordering, atomics, and flow control. Its
scale-up is single-node `cudaIpc` over NVLink/XGMI (so the scale-up domain is one physical node,
never MNNVL) and its EP16 scale-out uses the same per-SKU RDMA rails as the other backends. NCCL EP
is NVIDIA's native MoE dispatch/combine on the NCCL Device API, driven through the `nccl4py`
bindings. `normal` mode selects its `HIGH_THROUGHPUT` algorithm, whose FLAT `[N, hidden]` receive and
unweighted rank-sum combine match `layout-and-dispatch-v1` exactly, so the same oracle applies. It is
NVIDIA-only and CUDA 13 only, and runs EP8 scale-up on H100/H200/B200/B300 plus EP8 and EP16 on
GB200/GB300, where EP16 stays inside the MNNVL scale-up domain. X86 EP16 scale-out is an unsupported
coverage row, its cross-node GIN path faulting inside `nccl_ep.cc` identically on RoCE and IB across
four SKUs. This is a GDAKI limit, not a fabric-selection one. FlashInfer EP is TensorRT-LLM's one-sided MNNVL `MoeAlltoAll`, in which each rank writes tokens directly into its peers' workspace windows and combine reads them back, so there is no send/recv pairing and no NVSHMEM. It is GB200/GB300-only for that reason, and runs EP8 and EP16 inside the MNNVL scale-up domain. Its combine is the one place a backend's accumulator precision changes the expectation rather than the tolerance: through 0.6.15 the kernel holds its top-k accumulators in the payload dtype and reduces them with a hand-unrolled pairwise tree, so every level rounds to BF16, and the oracle reproduces that tree exactly rather than loosening the gate to absorb it (0.6.16 rewrote the accumulator to FP32. The adapter reads the installed version and picks the matching model). Those throughput kernels run across the full token ladder in the `normal` mode. Its FP8 dispatch is the one (backend, precision) pair here that is realizable but off every deployed path. vLLM accepts only nvfp4/mxfp8/bf16 on this transport, so `sweep_matrix.py`'s `OFF_PATH_PRECISIONS` keeps it out of the default matrix and a production sweep measures only configurations an engine can select. Naming the precision explicitly (`--precisions fp8`) opts it back in for transport comparison against DeepEP V2/UCCL-EP at matching bytes and block size: the one place a precision filter ADDS rows rather than only removing them.

A second `low-latency` mode adds each backend's decode-optimized kernel family. On DeepEP it drives
the legacy `deep_ep.Buffer` low-latency decode kernels (`low_latency_dispatch`/`low_latency_combine`),
which deliver a per-expert padded receive buffer and apply the top-k gate weights inside a source-side
combine (weighted-kernel-sum). For the scoped single-node EP8 cells these run over the intra-node
NVLink low-latency path (`allow_nvlink_for_low_latency_mode`). NVSHMEM/IBGDA (and thus `/dev/gdrdrv`)
is only exercised on the wire by a multi-node scale-out (EP16) run, and single-node EP8 was validated
on H200 with `/dev/gdrdrv` absent. On MoRI it selects the `IntraNodeLL` kernel, a single-call,
pure-intranode decode kernel that keeps the same rank-deduplicated compact layout and plain unweighted
rank-sum combine as the throughput `IntraNode` kernel, so it differs only by kernel type and timing
(the split-phase RDMA-staged `AsyncLL` kernel is deliberately not used because its separate receive phase
does not fit the single-call dispatch/combine contract). Low latency is a decode-phase-only addition
whose runnable set is narrower than and distinct from the throughput kernels', so it is enabled
cell-by-cell from the registry's `ll_backends` map rather than assumed wherever `normal` runs. It is
currently enabled for DeepEP V2 at EP8 on H100/H200 and at EP8 and EP16 on B200 (the nscale
bare-metal pool: IBGDA over native IB rails with `/dev/gdrdrv`, which an x86 low-latency scale-out
needs and no virtualized pool has) and on GB200/GB300 (EP16 inside the MNNVL scale-up domain), MoRI
EP8 on MI300X/MI325X/MI355X, and UCCL-EP EP8 on H100/H200/B200 only (the legacy
`Buffer` low-latency kernels, which at EP8 run `cudaIpc` over NVLink, not the CPU-proxy RDMA path,
because the adapter passes `is_intranode` and UCCL then never starts its proxies. The AMD SKUs drop
LL: upstream raised `kNumMaxTopK` 9 -> 16 six days before our pin, and the resulting host assert
cannot hold on AMD's 16 warp groups). NCCL EP low-latency is HELD on every SKU. Its
`LOW_LATENCY` algorithm is the DeepEP-derived decode path, EXPERT_MAJOR receive with a source-side
weighted-kernel-sum combine — but its combine recv pipeline is a port of DeepEP's PRE-FIX code,
missing the `fence.proxy.async.shared::cta` DeepEP added in #642, and the shared-memory race that
fence closes is present on every rung, not only the T=256 rung where it was observed (1-in-5,
bimodal error 0.47 vs 0.0039 on gb300 EP8). The earlier T<=128 ladder clamp reduced exposure; it
was never a safety boundary, so a green clamped row is not publication-valid and the rows are held
in the registry until a fenced wheel ships. (Historically those rows also wedged on stale peer
signals, [NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303), fixed by the single-handle
adapter.) B300 therefore publishes no low-latency coverage beyond DeepEP V2 EP8.
Whether a given SKU/backend/EP/mode cell is attempted is a capability
fact. Whether it succeeded is decided only by the emitted artifact.

## Workload Identity

One deterministic workload is generated over the global token batch from the workload's seed in
`configs/sweep.json` (part of the workload identity, baked into every scheduled case) and sliced by
source rank. A keyed BLAKE2b counter over the (token, slot, attempt, stream) coordinates produces
byte-identical expert indices and gate weights on every runtime, and the harness proves the
realized routing trace identical across ranks before a case can succeed.

Routing traffic distinguishes:

- token-expert assignments, which determine expert compute load.
- rank-deduplicated token payload copies, which determine EP activation traffic.

Adapters may not generate routing or reinterpret one quantity as the other.

## Measurement

Normal mode uses `layout-and-dispatch-v1`: dispatch timing includes layout plus communication, and
combine returns activation payload through an unweighted rank-sum path. Expert-output staging is
outside isolated combine timing AND outside the measured paired roundtrip, so `roundtrip` means
dispatch then combine (the transport) in every row, and staging is reported as its own `stage`
component wherever it does device work. The `CX_FP8_CONSUME=dequant` verification hatch is the one
exception, putting the conversion back inside the chain on purpose.

Under FP8, treat `stage` as **harness scaffolding rather than a phase a serving stack has**: it
converts the received FP8 payload to the BF16 combine sends, something production never does separately. In production,
the FP8 lands in the expert GEMM, which reads FP8 operands natively and emits the BF16 combine
receives. This suite measures the collective, not the layer, so `stage` stands in for that GEMM,
which is why it is excluded from `roundtrip` and why **`stage` must not be summed into a total or
compared between backends**: each adapter converts a different amount. DeepEP V2 and UCCL-EP convert
only the received rows in `normal` mode but the whole padded plane in `low-latency`, where the
receive buffer is `[experts, cap * ranks, hidden]` regardless of token count. MoRI converts only the
received rows. FlashInfer only converts the filled slots. The one production path that *does* pay a separate
materialised dequant is a quant-format mismatch fallback (vLLM dequantises when `block_k` disagrees
with DeepEP's block size), which `CX_FP8_CONSUME=dequant` models. It is not the default because it
is not the fast path.

Read `implementation.stage_excluded_from_roundtrip` as "there was device-work staging and it was
hoisted out of the chain", not as "this row's roundtrip is stage-free". It is gated on whether the
backend's `stage()` does device work at all, so `false` covers two unrelated situations that the
`stage` component separates: **absent** means the backend has nothing to stage (a bare pointer
assignment, as for NCCL EP and every BF16 row that hands the receive buffer straight to combine),
**present alongside `false`** means the `dequant` hatch put the conversion back inside the chain.
Reading `false` alone as "roundtrip includes staging" subtracts a cost the row never paid. Each component declares
availability, origin, and sample count. A paired-only API reports null isolated components.
`isolated_sum` is derived.

Headline latency is the **chained pair period** (`components.pair_period`, defined under Chained
Pair Period below) for every row that carries one, and the p99 of the per-iteration cross-rank MAX
of `roundtrip` for rows measured before that field existed. The flip shipped **held** while the
six-events-per-pair chain described below, whose inner records inflated small-T periods
fleet-wide, was replaced by the two-pass chain, and was released on 2026-08-06 once the b200, h200
and gb200 hand references were confirmed against two-pass fleet artifacts (runs 31092783122 and
31089556516). Both `p50` and `p99` are
emitted either way and `summarize.py` prints both. MAX is the fresh-entry family's reduction because
a layer is not finished until its slowest rank is, so MAX is the completion cost, and it charges
inter-rank entry stagger to whichever component the ranks entered unevenly. That stagger depends on
the code path AND the precision, not only on the fleet: on identical h200 low-latency decode cells
the per-iteration spread is ~9.3 us for deepep-v2 and uccl-ep at BF16 (they share the legacy
`Buffer` path) against ~2.6 us for nccl-ep, and collapses to ~2.8 us for those same two under FP8,
where in-kernel quantisation makes the heavier dispatch self-align the ranks. The term is not
subtractable in any principled way, so MAX alone taxes some rows more than others.

Every row therefore also carries `cross_rank_min_us` (the same iterations reduced with MIN, the
skew-excluded floor) and `cross_rank_spread_us` (per-iteration MAX minus MIN). Read MAX and MIN as a
bracket: two cells whose MAX gap is smaller than the larger contender's spread are not separated by
the data. Rank on roundtrip p50 and call a winner only where MAX and MIN agree on the ordering. Do
not rank on p99 of MAX for multi-node decode cells, where it is dominated by worst-rank stalls
rather than transport, while p99 of MIN is the synchronized-cost tail beside it. The isolated components
inherit the preceding operation's per-rank exit stagger, so treat them as residual-wait diagnostics
rather than per-operation costs. The paired roundtrip is the comparable quantity.

### Chained Pair Period

Everything above measures **fresh entry**: drained around each timed window, so every
sample starts from an idle pipeline and the ranks re-stagger before each one. A decode loop never
stops, and what it pays per MoE layer is the pipeline's steady-state **period**. Every row therefore
carries the chained family, measured by `benchmark_chain`: dispatch→combine pairs issued
back-to-back with CUDA events enqueued on-stream and **no host synchronization inside the loop**,
the first `chain_drop` (16) pairs discarded as pipeline fill, pooled over `chain_trials` (4) per
point on the same rotated ladder order as the rest of Pass 2. The pairing is exactly
`run_roundtrip`'s (dispatch, the staged combine input (or an inline `stage` under the
`CX_FP8_CONSUME=dequant` hatch), then combine), so paired-API backends stay in contract and
`pair_period` excludes expert-output staging on the same rule `roundtrip` does.

Each trial runs **two sibling chains** of `chain_iters` (128) pairs, because the statistics must
not carry the instrumentation that collects them. The first version ran ONE chain with six
`record()` calls per pair. Wherever the device drains faster than the host enqueues (the bottom
of the ladder), every event executes as issued, the pair window degenerates to host elapsed time,
and the four inner records plus glue were charged into the published period. The fleet exposed it
before a profiler did: period minus floor-sum sat at a roughly T-independent 10–30µs on every
vendor and fabric at once (a host constant, not transport), inflating T=1 periods by 20–38%. So a
**floors chain** runs first carrying only the four op-window events, and a **period chain** runs
second carrying only the outer pair events, nothing between its two collectives: both records' host
cost lands in the inter-pair gap, outside the window, so the period carries what an uninstrumented
caller pays. (An eager-mode launch floor remains, as for any eager caller, but a CUDA-graphs decode loop
pays less host per pair than any eager harness can.)

The two chains publish five statistics, and only five:

- `components.pair_period` (origin `chained-median`) is the per-pair period from the period chain,
  reduced across ranks by MEDIAN. MAX is right for a drained component (a layer finishes with its
  slowest rank), but the period is a **rate**: the collectives phase-lock every rank into one
  cadence, and a MAX would publish whichever rank hiccuped as the pipeline's speed.
- `chain_floor_us.dispatch` / `.combine` (origin `chained-cross-rank-min`) are each op's window from
  the floors chain, reduced across ranks by MIN: the last rank into a collective waited least, so
  its window is the op's floor, and it tracks profiler kernel time to ~10%, making it a free Kineto
  substitute. Read floor-vs-period as transport share, not an identity that must close to zero.
  `period − Σfloors` is a real quantity with a meaning in each sign, and neither sign is an error:
  - **Positive** is the per-pair inter-rank wait that the MIN deliberately strips. Where a backend
    is synchronization-dominated it rivals the floor sum and sits *flat in T*, as gb200
    flashinfer-ep normal decode (run 31089556516) holds ~70µs bf16 / ~100µs fp8 at every rung,
    vanishing by T=512 and in prefill as the floors grow into it. That is `period = max(sync
    budget, work)`, not instrumentation. It is invisible to `pair_spread_us` (2.7–8.5µs against
    62–107µs gaps) because the period is conserved while the wait migrates between op windows. This is
    the same steady-state stagger that bans chained per-op medians below.
  - **Negative** is the floors chain's own four-records-per-pair host cost (~10–12µs) inflating
    *its* windows wherever the device outruns the host: the very effect the two-chain split
    evicted from the period, still present and harmless in the floors (gb200/h200 fp8 LL, run
    31089556516). It is not overlap, and it is not cross-chain settling.

  Do not read a large positive residual as the six-events defect returning. That defect was a
  roughly T-independent host constant appearing on *every vendor and fabric at once*, and it lands
  in `interpair_gap_us`. A sync-dominated residual is backend-specific and leaves the gap small.
- `chain_health.pair_spread_us` measures per-iteration cross-rank max-minus-min of the pair. It is the *proof*
  the median means anything. Large next to `pair_period` means a paced or slow rank, and the
  point should not be read as a steady-state period at all.
- `chain_health.interpair_gap_us` is the start-to-start median minus pair-window median, measured once per
  trial: the per-pair cost OUTSIDE the published window (the harness's own two `record()` calls
  plus any inter-pair stall), and the in-artifact regression guard against the six-events defect
  above.
- `chain_health.settle_drift_us` measures the late-half minus early-half period median, once per trial,
  signed, cross-rank max-magnitude. `chain_drop` *assumes* the chain settled before the kept
  pairs. This is the proof, and an unconverged chain (or a device clocking down mid-chain)
  publishes its drift instead of a clean-looking period.

**Chained per-op medians and p99s are never published.** Without a host sync the ranks arrive at
each collective at slightly different times, and the resulting wait parks in whichever op window a
given rank happens to block in. This placement is stable per rank (so one rank's per-op numbers look clean and
convincing), arbitrary across ranks, with rank 3's dispatch long exactly where rank 5's combine is,
and bistable across runs of an identical configuration, while the sum, the pair period, is
conserved. A chained per-op median measures where the wait sat on that run rather than what the
operation cost. The cross-rank MIN is the one reduction that removes it, which is why the floors are
published and the medians are not. A p99 of a chained window would mix the same noise back in
through the tail.

The fresh-entry family keeps its meaning exactly: `components.roundtrip`, `dispatch`, `combine`,
`stage`, `isolated_sum`, `cross_rank_min_us` and `cross_rank_spread_us` are measured and reduced
as they always were, at the same 256×8 sampling, and no stored row was re-meant or re-measured. A
consumer keys on the presence of `components.pair_period`. A row without it predates the chain.
Never rank a chained cell against a pre-chain one on the headline column. `summarize.py` footnotes
its table whenever both appear in it.

**The published period always means one thing: free-running.** The chain lets ranks drift by up to
about one iteration, which is only sound where the receive plane tolerates it: every backend
double-buffers per dispatch, enforces strict pairing by contract, or completes each op on a reusable
handle, and DeepEP V2's **normal** mode (the one genuinely unaudited cell) was hand-probed with
256 un-synchronized pairs at T=128, EP8+EP16, both precisions (2026-08-06, pin `01dc3aaa`, the
then-current dgxc pool's hybrid GIN over RoCE): all passed, outputs finite, timed inputs unchanged,
cross-rank period agreement within 1µs, and the chain-vs-synced gap is the size of the effect this
section is about (EP8 105.4 vs 125.4µs BF16, 216.6 vs 272.3µs FP8, while EP16 838 vs 863µs and 820 vs
897µs). A backend that cannot run free belongs behind a fix, not a measurement variant: re-aligning
ranks between pairs adds its own ~10µs and removes the cross-pair overlap the measurement exists to
capture. That is a differently-defined quantity that must never share a column with the free-running
period.

Every backend's timed HT dispatch now carries its routing work. nccl-ep binds routing with
`ncclEpUpdateHandle`, documented as a "per-step collective: prepare the handle for the given top-k
routing decisions" — and production routing changes every MoE layer, so a serving step pays that
collective before every dispatch, at the handle's full token capacity. Earlier generations excluded
it (as NVIDIA's own `ep_bench` does: CUDA events around dispatch and combine only, handle update
outside the loop) on the argument that its capacity-proportional cost would import a ladder-max
term into dispatch; that argument describes exactly what production pays, since engines size the
handle to their max token capacity and update it per step. The timed window now includes the
update; rows carry `kernel_generation` `nccl-ep-ht-routed`, and pre-change `nccl-ep-ht` rows are a
different measurement contract — the discriminator the earlier NCCL changes lacked. Low-latency
mode has nothing to include: `ncclEpUpdateHandle` returns immediately there and the kernel reads
the cached routing inside the timed dispatch. The other backends already carried this cost --
uccl-ep calls `get_dispatch_layout` inside dispatch, while deepep-v2, MoRI and FlashInfer pass
routing on every call.

Every measured component uses one fixed timing profile, defined once in `configs/sweep.json`
and baked into every scheduled case:

- 256 trials x 8 timed iterations = 2048 observations for the fresh-entry family.
- 4 chain trials x (128 free-running pairs - 16 dropped for pipeline fill) = 448 observations for
  the pair period and another 448 for the op floors, each trial running the two sibling chains
  (floors first, then the lean period chain). The trial count is lower than the fresh-entry family's
  because one call already yields 128 pairs, and matching it would multiply the leg's wall clock
  without buying convergence.
- 32 synchronized full dispatch-stage-combine warmups before each available measured component at
  every trial/point, and before each chain trial.
- component measurement order rotates each trial (`trial_order`) so every timed component occupies
  every position in the sequence, over a per-trial-rotated token ladder, which the chain trials
  rotate the same way.
- per-iteration maximum latency across ranks before nearest-rank p50/p90/p95/p99 (the chained
  family reduces by median and minimum, as described in Chained Pair Period).

`measurement.sampling` carries both halves of that profile, because `sample_count` alone cannot be
decomposed back into them and a 128x4 chain is not the same measurement as a 512x1 one.

The chained pair period is the headline latency where a row carries one, and measured roundtrip
p99 otherwise. Decode and prefill identify the serving regime
represented by one MoE-layer collective. They do not change the timed primitive at an otherwise
identical shape. Ascending through the ladder, each measured shape is conditioned with 8 untimed
full roundtrips (settling clocks, fabric, and buffer state) before it is correctness-checked.
All timing happens after every shape is warmed and checked. Conditioning rounds are never
measured or emitted.

Comparing these figures against a vendor table is not like-for-like, in a knowable direction.
Every sample here is an eager per-call measurement that includes kernel-launch cost and
inter-rank entry skew, reduced across ranks by MAX. Vendor microbenchmarks published for these
same kernels variously time the named kernel only via a profiler (DeepEP, UCCL low-latency),
replay CUDA graphs (MoRI), average across ranks instead of taking the max (all of them), delete
entry skew with a pre-iteration sleep or an amortized barrier (DeepEP, MoRI), and report the
best of a config sweep (DeepEP V1, MoRI). Expect our headline to read roughly 5-10% above such
a table on a healthy fabric. `cross_rank_min_us` is the per-row figure to place beside one.
Where a like-for-like comparison exists we match or beat: our skew-excluded MoRI dispatch is
0.96x MoRI's own shipped tuning-config best at the same shape and byte count, DeepEP V2 on B300
reproduces DeepEP's published 8x2 figure within 3%, and FlashInfer EP matches NVIDIA's published
one-sided kernel within 4% across eight byte-normalized points.

Logical payload bandwidth is:

`logical_payload_bytes / measured_latency_seconds`

Payload bytes use rank-deduplicated token-rank activations and exclude expert metadata,
padding, and backend buffer capacity. BF16 moves 2 bytes per value with no scale payload. An FP8
dispatch moves 1 byte per value, plus per-128-block FP32 scales for every blockwise codec here (
DeepEP V2, UCCL-EP and FlashInfer EP, which carries them as a fourth dispatch payload), and none for
MoRI's plain e4m3 cast, while combine stays BF16, so the dispatch and combine directions can carry
different byte counts and the roundtrip is their per-field sum. The rank-deduplicated count is exact
for the normal-mode layout, and for a low-latency kernel that deduplicates per rank (MoRI's
`IntraNodeLL`, whose combine is an unweighted rank-sum). The low-latency kernels that apply top-k
weights inside combine instead send one copy per (token, expert) assignment rather than per
(token, rank), so for a token whose experts share a destination rank this logical count is a lower
bound on the bytes those kernels move. Each row states which basis it used in `logical_copies`, so
the two are never silently mixed. Latency (the headline) is
measured directly and is unaffected. Algorithm bandwidth, bus bandwidth,
wire utilization, and physical-link utilization are not emitted without a defined primitive model or
transport counters. Logical bandwidth must never be labeled physical bandwidth. Payload and token
rates are named `rate_at_latency_percentile`: bytes or tokens divided by the matching latency
percentile. They are lower-tail service rates at p99 latency, not p99 percentiles of an inverted
rate distribution.

## Correctness

An implementation-independent oracle uses an expert-specific deterministic transform so wrong expert
routing cannot pass an identity roundtrip. For every rank and point it verifies:

1. destination rank/expert, source token, multiplicity, gate weight, and receive counts.
2. dispatched payload and metadata before timing.
3. combined output before timing.
4. unchanged semantic inputs through all timed samples.
5. dispatched payload/metadata and combined output again after timing.
6. the **free-running chain's own final combined output**, once per chain trial, against a
   drained pair through the identical dispatch→combine path.
7. the same full check as 2-5 once more against the state the free-running chain leaves behind.

Checks 6 and 7 exist because checks 2-5 only ever see drained calls: without them the headline
`pair_period` would come from a regime no oracle had inspected, and a backend that transports
correctly when drained but corrupts under back-to-back pairs would present as the fastest cell in
the suite. They answer different questions. Check 6 is a regime A/B, not an oracle: after each
chain trial's closing synchronize, outside every timed window, the chain's last combined output
is compared elementwise (the oracle's tolerance, not bit equality, since combine kernels are not
required to be order-deterministic) against a freshly drained pair through the same code path, so
it catches corruption the chain wrote into its own results. Check 7 reruns the full expert oracle
after the final trial, on the settled communicator (`benchmark_chain` ends synchronized), so it
catches state the chain corrupted for whatever runs next. Interior chain pairs stay unvalidated by
design: each pair overwrites its predecessor's output, and holding or reducing every output would
put device work (or ~O(iters × T × hidden) memory) inside the timed loops the chain exists to keep
clean. A defect confined to interior pairs that heals by the final pair is outside this suite's
evidence. Check 6 is reported per row as `correctness.chain_last_output_passed` (ANDed across
trials) and check 7 as `correctness.post_chain_state_passed`. Check 7 is **folded into
`correctness.passed`**, so its failure fails the leg exactly as any other oracle failure does.

Check 6 gates **only where the chain stages per pair**, and is reported as `null` elsewhere. That
boundary is measured, not assumed. Wherever `stage_excluded_from_roundtrip` holds (every FP8
adapter by default, since `stage_device_work` *is* the FP8 flag), the chain hoists staging out of
the timed loop, capturing one warm-up dispatch's stand-in and reusing it for all 128 pairs. Neither
the chain's final combine nor the drained reference then consumes an input matching its own
dispatch, so they are two differently mismatched pairs and nothing requires them to agree.

The A/B, on h100/deepep-v2/EP8, identical in every respect but the hoist:

| staging | `chain_last_output_error` | vs `COMBINE_REL_TOL` |
|---|--:|--:|
| hoisted (`fp8_consume=native`) | 31 – 93 | **1000× – 2966×** |
| per pair (`fp8_consume=dequant`) | `0.0` at every rung | bit-identical |

in **both** `normal` and `low-latency` mode (runs 31180411148, 31185184372, 31185233991), against a
BF16 control that is `0.0` because BF16 never hoists. So the difference was an artifact of the
hoist, not a transport defect: gating on it under the hoist reddened every FP8 leg fleet-wide for
something the harness does deliberately.

Passing the chain's staged input to the drained pair was tried first and is *not* sufficient. It
makes the two share an input, but a shared input that matches neither dispatch. Only per-pair
staging makes the regimes comparable.

The check keeps its teeth exactly where it has meaning: every BF16 row, and any FP8 row run under
the `dequant` hatch. `null` there means the question was not asked, never a comparison that ran and
failed. If this boundary is ever moved, move it on a measured magnitude: the verdict alone
justified two wrong calls in one day, and the number settled it in one probe each time.

What the null leaves uncovered is the intersection of three conditions: a corruption that manifests
only under free-running pairs (the drained oracles are blind by regime), leaves no state behind
(the post-chain oracle is blind), and lives in a path the BF16 sibling row does not exercise (its
still-gating check is blind). Anything short of all three still reds a gate. The DeepEP
low-latency 256-rung corruption hit both precisions, so its BF16 rows would red
`chain_last_output_passed` today, and a wedge trips the per-case hang guard. Note the hoisted chain
also never consumes its own FP8 receive, so the exposure covers dispatch-side corruption as well as
combine-side, and `chain_health` is a consistency guard, not a work guard: a regime defect that
uniformly shortens the combine would publish a fast period no surviving check contradicts. The
standing probe for the intersection is the `dequant` hatch. `CX_FP8_CONSUME=dequant` stages every
pair from its own dispatch, restoring this check on FP8's own free-running pairs. Run one whenever
an FP8 chained period moves in a way its BF16 sibling does not, and before first publishing a new
FP8 backend. flashinfer-ep hoists at BF16 as well (its stage is the workspace staging copy), so its
rows are null at every precision and the hatch does not reach it, and its coverage is an open
follow-up.

Check 6 also publishes `correctness.chain_last_output_error`, the worst chained-vs-drained
relative error, cross-rank MAX, **reported whether or not the verdict passed**. Read it before
reading the verdict. The check assumes a regime defect lands orders of magnitude past
`COMBINE_REL_TOL` while ordinary kernel non-determinism stays well inside it, and that assumption
is not free: FlashInfer EP accumulates its combine in the payload dtype (a BF16 slot-tree) rather
than FP32, so its rounding over a large receive plane is far coarser than a backend reducing in
FP32. A verdict alone cannot separate "the transport corrupted this" from "this tolerance is too
tight for this accumulator", and the two call for opposite responses. The magnitude is the
discriminator. A failure reported without it should not be read as evidence of corruption.
(`post_chain_state_passed` was previously published as `chain_regime_passed`, a name that
overclaimed. A fresh post-chain oracle proves the state, not the chain's outputs. Artifacts from
older harnesses carry the old name, where `null` meant the chain never ran. A validated chain
budget is now required up front, so `post_chain_state_passed` is always a boolean in new
artifacts. `chain_last_output_passed` is the separate case described above and is legitimately `null`
wherever the chain hoists staging, which is every FP8 row outside the `dequant` hatch.)

Normal-mode adapters use activation-only, unweighted rank-sum combine. The oracle builds each rank's
gate-weighted expert aggregate before combine and derives the expected combine from the values
actually communicated, reproducing the two-level reduction: each destination rank casts its FP32
aggregate to the payload dtype (BF16) exactly as the adapter does. Ranks sharing a scale-up domain
(NVLink/MNNVL) reduce in FP32, and each domain casts its aggregate to BF16 for the scale-out send
before those partials are summed. A group that fits in one scale-up domain (`ep_size <=
scale_up_domain`, including every EP8 case and the MNNVL EP16 cases) has a single domain and no scale-out
rounding. A multi-node RoCE EP16 group carries one BF16 partial per node. Modelling that per-domain
cast is what lets the gate stay tight (max elementwise relative error (denominator clamped at 0.02)
below `8 * 2^-8`, the residual accumulation-order ambiguity) across scale-up and scale-out topologies
alike (omitting it left multi-node EP16 ~0.048 off, above the gate).

Low-latency adapters instead use a source-side gate-weighted combine: the kernel multiplies each
expert's returned message by that assignment's top-k weight, so the adapter stages the UNWEIGHTED
per-expert transform and a dedicated per-(source, expert)-slot oracle derives the expected combine as
the gate-scaled sum of per-expert BF16 messages, with no per-domain intermediate, since the low-latency
kernels reduce at the source rank. The delivered (source, expert) assignment multiset and per-expert
counts are checked against the routing trace, and the same tight combine gate applies. Under FP8
dispatch the oracle applies the backend's exact per-token cast round-trip to its semantic payload before both the
dispatched-payload compare and this combine expectation, so the payload match stays bit-exact and the
same tight gate holds. The quantization is modeled, not absorbed into a wider tolerance. It is a
correctness gate, not an estimate of transport error. Any failed rank or point makes the case ineligible in the result it writes.
Pre/post dispatch behavior is checked against canonical source-token metadata and expected output.
Native receive slots may be assigned nondeterministically, so physical receive order is not treated
as a correctness property.

## Result Artifact

One raw case document carries `record_type: "case-attempt"`, the single `version`, and a
`generated_at` timestamp, and contains:

- `identity`: `case_id`, `attempt_ordinal`, `case_factors` (SKU and the scheduled case, including backend,
  EP size, mode, precision, phase, suite, workload, and the topology coordinate), and
  `allocation_factors` (run id, run attempt, source SHA).
- `workload`: `cross_rank_consistent`, whether the routing trace was proven identical across ranks.
- `measurement`: dispatch/combine dtype (the realized wire formats, with combine always BF16 and dispatch
  BF16 or the SKU's FP8 format) and semantics, `payload_unit` (`token-rank`), `sampling`, and the
  per-point `rows`.
- `implementation`: backend name, kernel generation, and `maturity`, which indicates whether a production
  inference engine can select this transport today (`production` = exposed by vLLM's
  `--all2all-backend` or SGLang's `--moe-a2a-backend`, while `candidate` = a real transport we
  benchmark that no engine ships a selector for, so its numbers describe the library rather
  than a deployable configuration). The same map is in the registry's `backend_maturity`. It also
  carries `fp8_consume` (which FP8 consumption path the chained roundtrip modelled (see above)),
  `combine_reduction` and `library_version` (which reduction the oracle held the kernel to, and
  the installed library that selected it), and two generation discriminators:
  `stage_excluded_from_roundtrip` (whether `roundtrip` excludes expert-output staging, discussed
  above) and `chained_period` (whether this document's rows carry the chained family at all).
- `topology`: requested SKU/product, placement, `gpus_per_node`, nodes, scale-up domain, `scope`,
  `topology_class`, world size, and three distinct transport fields, `scale_up_transport` and
  `scale_out_transport` (the components), plus `transport`, the derived summary that folds them
  into one string (`nvlink` scale-up-only, `nvlink-rdma` once a case scales out).
- `runtime`: the realized software stack, `vendor`, `framework` (the torch version),
  `accelerator_runtime` (the CUDA or HIP version torch was built against), and
  `collective_library` (`nccl`/`rccl` and the version actually loaded into the process).
- `provenance`: the mounted image tag and source SHA, and
- `outcome`: `status` (`success` or `invalid`) and `reasons`.

Each `rows` entry carries point latency (the fresh-entry `components` plus the chained
`components.pair_period`, `chain_floor_us` and `chain_health` (see Chained Pair Period)), byte
accounting, token rate, correctness, load, and fanout, while
per-point statistics are summarized in place, not emitted as separate documents. Each dispatched
case writes exactly this one raw result document, while unsupported or never-run cells produce no
synthetic record.

## Identity

Identifiers are readable factor strings:

- `case_id`: `{sku}-{backend}-{workload}-{mode}-{phase}-ep{ep}-{routing}-{precision}`, each factor
  slug-normalized, and
- `attempt_ordinal`: a positive integer distinguishing repeat executions of one `case_id`.

Backend source pins live in `runtime/common.sh` and are enforced by exact fetched-commit comparison, and
the loaded DeepEP V2 build is checked for the required `ElasticBuffer` API.

These IDs let a consumer group matched configurations and separate distinct ones. The backend does
not itself compute cohorts, controlled comparisons, sensitivity pairs, eligibility, or
recommendations. A reader decides which cases to surface and how to compare them.

## Execution Isolation

Every non-MNNVL scale-out case uses operator-pinned socket and RDMA selectors. The launcher rejects
missing or partial profiles, then probes every allocated node for the configured interface, active
HCA port, and configured GID before backend initialization. It never substitutes a default route,
inherited runner environment, or transport fallback. Scale-up and MNNVL cases clear the profile.
Scale-out NVIDIA forces `NCCL_NET=IB`, while AMD leaves plugin selection to RCCL. Both use exact HCA
matching. Scale-out also pins `NCCL_IB_MERGE_NICS=0` so dual-port NIC fusion cannot disable NCCL GIN
(which the DeepEP V2 EP16 hybrid path requires), and a rail-isolated fabric (`rail_isolated`) adds
`NCCL_CROSS_NIC=0`. Selectors come from the tracked platform registry, optionally overlaid by an
operator config, and appear only in mode-0600 private logs.

Repository staging uses a pre-existing, runner-owned, group/world non-writable shared base outside
the checkout and workflow workspace. The parent process resolves the exact execution child before
copying. Backend preparation then runs from that tree on every allocated node. Cleanup waits for
confirmed allocation teardown and removes only that child. DeepEP V2 source is fetched before allocation at an
exact pinned revision, initializes its pinned `fmt` submodule, and applies the required local patch.

H200, B200, and B300 may derive that private base beneath the validated operating-system account home
when it is compute-visible. H100 instead derives a sibling of its shared container directory, never a
child of image storage.
Canonical B300 execution ignores the legacy operator `stage_dir` field and always derives the base
from the validated shared account home. Its UID-mapped Actions shell may accept that exact base when
its owner matches the private parent owner. Explicit stages and all other runners retain the strict
effective-UID ownership rule. An execution-ID suffix isolates parallel B300 workers. The current
NFS export may realize a newly created base as
UID 0. Only that creation path is accepted, while a pre-existing root-owned base is rejected.
Canonical GB300 execution likewise ignores its legacy group-writable `stage_dir` and derives an
execution-specific private base beneath the validated compute-visible account home.

## Image Pinning And Build Isolation

Enroot imports configured container tags into a per-run-scoped squash keyed by the image tag and
image platform, so one run never reuses another run's imported filesystem. Image-provided DeepEP is
also checked against exact package versions and its expected API. Source-built DeepEP V2 uses
a separate mode-0700 cluster-local cache mounted only as `/cx-cache`. Its path binds CPU/GPU
architecture, image, and upstream commit. The cache is never an artifact. Per-execution
source/results stages remain isolated and disposable, and runtime probes fail closed before reuse. The runner UID is
inside the trusted cluster boundary: this cache guards against stale or accidental mutation, not
hostile same-UID jobs. Only an unpublished partial build may be reset automatically. A cache that
fails integrity or runtime checks is left intact and rejected so a concurrent allocation cannot lose
files it is using.

## Neutral Artifact Delivery

There is no results server, attached store, or managed object store. Each shard runs one allocation,
emits per-case result JSON and a small mechanical summary, and uploads them as GitHub artifacts with
`always()` so a red or partial run still uploads. A case counts as successful on the benchmark's own
return code. There is no completeness or privacy validation step before upload, and failed or
unsupported cells produce no synthetic record.

No step promotes a run, builds a dataset, or advances a channel. The artifacts are the output. Any
downstream display or comparison is the consumer's responsibility.

## Legacy Data

Historical numeric schemas 3-5 are outside this benchmark's artifacts. They remain historical
diagnostic evidence and are not produced or consumed by the current sweep.
