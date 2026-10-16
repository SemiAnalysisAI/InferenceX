# CollectiveX Methodology in Brief

What CollectiveX measures, how it measures it, and how to read the results.
[methodology.md](methodology.md) holds the evidence behind each rule, and the
[README](../README.md) holds usage and the coverage tables. Where the docs and the code in
`collectivex/bench/` disagree, the code is authoritative.

## Scope

CollectiveX measures one MoE layer's expert-parallel communication on real allocations. **Dispatch**
sends each token's activations to the ranks that own its top-k experts. **Combine** returns the
expert outputs to the token's rank. No expert compute runs, so the suite measures neither
compute/communication overlap nor serving throughput. It records what each backend does and does
not rank or recommend.

## Workload

- **Shape** `deepseek-v4-pro`: hidden 7168, top-k 6, 384 routed experts, packed expert placement.
- **Routing** is uniform and deterministic (a keyed BLAKE2b stream from the workload seed). Every
  case proves the routing trace identical on all ranks before it can succeed.
- **`T` is tokens per rank**: decode 1-512 and prefill 1024-8192, in powers of two. A rung a backend
  cannot reach goes in `ladder_dropped` rather than vanishing.
- **Precision**: dispatch is BF16, or FP8 where the backend supports it upstream. Combine is always
  BF16. Every FP8 codec is per-128-block e4m3 with FP32 scales on the wire. nccl-ep's FP8 is off
  every engine's path, so only `--precisions fp8` runs it.
- **Scale**: EP8 is one 8-GPU node (2x4 MNNVL on GB200/GB300). EP16 is 2x8 over RDMA on the x86 and
  AMD pools and 4x4 MNNVL on GB200/GB300, so GB EP16 stays scale-up.
- **Coverage**: `configs/platform_config.json` says which backend runs where (the README's coverage
  table), and `results/cases.json` lists every requested case, runnable or not.

## Backends run as engines run them

Each backend runs the configuration its reference engine selects (vLLM, and TensorRT-LLM for
nccl-ep low-latency), not a common denominator, so a cross-backend comparison
compares deployed contracts.

| Backend | Maturity | Launch | Contract |
|---|---|---|---|
| deepep-v2 | production | graph; normal prefill eager | vLLM's `deepep_v2` in both phases: overlap off (64 SMs), a single reduction, no CPU sync, receive sized to the next power of two of `T`. LL as vLLM's `deepep_low_latency`: send kernel plus receive hook, MNNVL on GB only, UE8M0 FP8 scales on Blackwell. The LL receive is fixed at 256, so LL stops at T=256. |
| nccl-ep | production for LL rank-major BF16 (TensorRT-LLM's `NcclEP`); candidate otherwise | graph; HT prefill eager | LL runs TensorRT-LLM's configuration: rank-major, pre-reduced, one group sized to the ladder's top. Graphed dispatch stages, because TensorRT-LLM dispatches into a plain tensor under capture; eager dispatch is zero-copy inside one NVLink/MNNVL domain. HT is a library row: dispatch includes the per-step routing update, whose memsets, all-gather and scan are sized by the capacity (`-cap512` decode, `-cap8192` prefill), not by T. The communicator uses graph usage mode 1. FP8 (DS_FP8E3M4, LL decode) is off by default because no engine selects it. |
| uccl-ep | candidate | normal eager (host-synced); LL graph | DeepEP's legacy API over CPU-proxy RDMA (proxies run only at b200 EP16), in vLLM's DeepEP configuration on UCCL's drop-in `deep_ep`: `deepep_ht` launch configs at 20 SMs and 1 GiB buffers; LL called as deepep-v2's, with a 256-token receive. |
| mori | production | graph decode; eager prefill | vLLM's MoRI configuration: IntraNode at 80 blocks, 16 warps (EP8), InterNodeV1 at 64/32/8 on STATIC_HEAP (EP16), external input buffer, one QP per PE. LL is vLLM's `mori_low_latency`: InterNodeV1LL at 64/32/8 on STATIC_HEAP, EP16 only (one node runs the normal rows' IntraNode). `MORI_ENABLE_SDMA` unset, as vLLM leaves it. |
| flashinfer-ep | production | graph decode; eager prefill | GB200/GB300 only (one-sided MNNVL). vLLM's integration: combine stages a caller-owned payload (`payload_in_workspace=False`) into a caller-owned output. FP8 is DeepSeek blockwise, which vLLM selects from v0.29.0. `kernel_generation` carries the wheel version. |

`production` means an engine can select the configuration today: vLLM or SGLang, or, for a library
vLLM does not expose, the engine that does (TensorRT-LLM's `NcclEP` for nccl-ep's low-latency
rank-major BF16 rows). `candidate` rows describe the library, not a deployable configuration.
nccl-ep sets maturity per row. uccl-ep runs vLLM's DeepEP configuration but stays `candidate`,
because no engine ships its selector.

## Timing

Every window is a pair of CUDA events on the stream. Each point produces two families of samples.

**Fresh entry** times one drained call per sample for `roundtrip`, `dispatch` and `combine`: 256
trials x 8 = 2048 samples. Each sample is reduced across ranks by MAX, because a layer finishes
with its slowest rank. `cross_rank_min_us` (MIN) and `cross_rank_spread_us` (roundtrip MAX - MIN)
come from the same samples.

**Chained** times dispatch-combine pairs issued back to back with no host sync, as a decode loop
issues them: 4 trials x (128 pairs - 16 pipeline-fill pairs) = 448 pairs. Each trial runs two
sibling chains so the instrumentation stays out of the result:

- The **period chain** records events only around each pair. `pair_period` is each pair's window,
  reduced across ranks by MEDIAN, with percentiles over the 448 pairs. The collectives lock every
  rank to one cadence, so a MAX would only publish the rank that hiccuped. **`pair_period` is the
  headline.**
- The **floors chain** records events around each op. `chain_floor_us.dispatch` and `.combine` take
  the cross-rank MIN: the last rank into a collective waits least, so its window tracks kernel time
  to within about 10%.
- **Health** fields:
  - `pair_spread_us`: the cross-rank spread of each pair. A large value means a paced rank.
  - `interpair_gap_us`: start-to-start minus the window, the cost outside the window.
  - `settle_drift_us`: the late-half minus the early-half period.
- **Chained per-op medians are never published.** Without a sync, each rank's wait lands in
  whichever op it happened to block in. Only the pair window (the period) and the MIN (the floor)
  are stable.

Free-running pairs let ranks drift apart by up to one pair. Every current backend tolerates that,
through a double-buffered receive, strict pairing, or a reusable handle.

**Launch.** Graph-safe backend/mode pairs replay CUDA graphs, as engines capture decode, and the
rest run eager. Under graphs the chain is one graph of 128 unrolled pairs. Capture uses torch's
default (`global`) error mode, as vLLM and SGLang capture. Graphed and eager rows carry different
`kernel_generation` suffixes (`-cudagraph-spin-ar`, `-eager-spin-ar`) and never pool.

**Rank alignment** keeps launch skew out of the cross-rank MAX. Every fresh-entry sample opens
with a 2 ms GPU spin, then a device all-reduce, with no host sync (DeepEP's `bench_kineto` method):

- **Eager**: queued ahead of each sample. A sample whose host took longer than the spin to finish
  enqueuing its timed op is a **miss**. Garbage collection is off inside every timed series.
- **Graphed**: captured in each graph before its first timing event. HIP submits a replay node by
  node from the host, so on ROCm the spin is what lets a lagging host queue the whole replay before
  the barrier releases its peers; a replay call that outlasts it is a **miss**.
- **Chains** align once per sibling: graphed chains keep the captured spin, eager siblings open
  with the all-reduce alone, so a host-bound chain cannot queue a backlog it then drains at a
  second speed.
- **`combine`** is timed behind its own untimed dispatch, a fresh pair with the stage hoisted, in
  both regimes.
- **Host-synced eager series** (uccl-ep normal, the only dispatch that waits on the host) also
  wait out a throttled host. A CPU probe holds the sample while the probe runs more than 2.5x slow,
  capped at 0.5 s per sample and 60 s per run. A probe more than 5x slow on either side of the
  window marks the sample **stalled**. Their untimed dispatch ahead of a timed `combine` runs
  before the alignment, so its wait cannot spend the spin. Their chain siblings are held the same
  way and restart, on every rank together, when any host ran throttled (at most 3 times); a point
  with a trial still throttled after the last restart publishes only its period p50.

**Order.** Each shape is warmed in ascending order (8 untimed round trips) and checked for
correctness before any timing. Every measured component and chain trial follows 32 warmup pairs, and
component and ladder order rotate per trial.

## What is published

- **Headline**: `pair_period` p50, with its p99 beside it. Fresh-entry `roundtrip` p50 is the
  isolated-call latency.
- **Fresh-entry tails**: every row, graphed or eager, withholds p99, p95 and p90 once misses plus
  stalls reach 1%, 5% and 10% of samples, and reports the counts in `rank_alignment`.
- **Bandwidth** is wire payload bytes divided by latency, derived by `bandwidth.py` and the
  dashboard; rows carry the bytes, the latencies and a token rate.
  - `wire_byte_provenance` counts what the kernels move: one copy per (token, rank) for
    rank-deduplicated layouts, and one per (token, expert) for the LL kernels that weight inside
    combine.
  - Every FP8 codec adds per-128-value FP32 scales.
  - This is a logical payload rate, never physical link utilization.
- **FP8 `stage`** stands in for the expert GEMM consuming FP8. It is excluded from `roundtrip` and
  `pair_period`, and timed separately on eager rows only. Do not sum it into a total or compare it
  across backends.

## Correctness

Every row is gated. Any failed rank or point makes the case `invalid`.

- Each expert applies a distinct deterministic transform, so misrouting cannot pass as an identity
  round trip.
- **Before and after timing**, the oracle checks destination and source metadata, receive counts,
  the dispatched payload and the combined output. Timing must leave the inputs unchanged.
- **After the chain**, its last output must match a drained pair, and the full oracle reruns on the
  state the chain leaves. The last-output check is null where FP8 staging is hoisted out of the
  chain.
- **The expected combine models each backend's reduction**:
  - `domain-fp32`: FP32 inside a scale-up domain, with one BF16 cast per domain for scale-out;
  - `rank-fp32`: one BF16 row per destination rank, summed in FP32 at the source (deepep-v2,
    nccl-ep LL rank-major);
  - `topk-slot-tree`: FlashInfer's BF16 tree before 0.6.16;
  - source-side gate weighting for the weighted low-latency kernels (deepep-v2, uccl-ep, nccl-ep
    expert-major), whose `combine_reduction` field the model ignores.

  FP8 is modelled by applying the backend's exact cast. The tolerance is 8 x 2^-8, relative.
- **Graphed rows**: every timed capture's output is poisoned and must be rewritten by its replay.
  A separate check scales the inputs by 2 and poisons dispatch's output and the staged combine input
  before a replay whose output must then match an eager pair on the same inputs, so a replay that
  re-sent stale data fails.

## Reading results

- **Rank on `pair_period` p50.** Each case is a single run and run-to-run drift is not
  characterized (before the host gate, host-synced uccl-ep EP16 periods moved 27-40% across nine
  b200 runs), so treat a small gap as no result. For fresh-entry rows, call a winner only where MAX
  and MIN agree.
- **Do not rank multi-node decode on fresh-entry p99 of MAX.** Worst-rank stalls own that tail.
- **Generations do not pool.** `kernel_generation` names the kernel family, its configuration and the
  graph or eager regime (plus capacity and wheel version where those change what a row measures);
  rows of one backend pool only within one generation.
- **Cross-backend rows compare contracts.**
  - nccl-ep's rank-major LL moves fewer bytes than the expert-major layout DeepEP and UCCL-EP use
    (27% fewer at b200 EP8, T=8).
  - Graphed and eager rows differ in host cost.
  - Capacity policies differ (see the table).
- **FP8 normal dispatch includes the caller-side quantize.** Compare FP8 with BF16 at the top of the
  ladder.
- **EFA pools** (b300, h100) run EP16 GIN through aws-ofi-nccl's CPU proxy, so those rows measure
  the proxy, not EFA line rate.
- **h200 deepep-v2 EP16** relaxes PCIe ordering on its GIN window (`-relaxed-ordering`).
- **Vendor tables are not like-for-like.** They time kernels in a profiler, average across ranks,
  replay graphs, or report a best-of-sweep. Put `cross_rank_min_us` or `chain_floor_us` beside them.

## Limits

- No expert compute runs between ops, so SM contention and overlap are not modelled.
- Routing is uniform. Real routing has hot experts.
- L2 is not flushed between samples.
- GPU clocks are not locked. Each case records every rank's clock and throttle time in
  `runtime.gpu_clocks`, and `slow_gpu` flags a GPU below 95% of its peers' clock, one reporting a
  hardware slowdown, or one throttled for 5% of the timed passes. Do not compare a flagged case.
- Held samples measure the library on a host running at speed. A platform that throttles under that
  host load costs production more; b200-nscale under uccl-ep EP16 does.
- Interior chain pairs are not validated; only the last pair and the post-chain state are.

## Provenance and publishing

A result records its source SHA, image tag, library versions, `kernel_generation` and topology. Only
results measured from committed code publish, by a reviewed commit of `results/<sku>/<case_id>.json`
to `main` ([README](../README.md#running-by-hand)).
