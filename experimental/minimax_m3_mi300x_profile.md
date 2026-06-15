# MiniMax M3 MI300X Profile and Optimization

Date: 2026-06-15

This report covers only the `minimaxm3-fp8-mi300x-vllm` recipe. The requested
profile points are the lowest, median, and highest useful concurrency for both
sequence lengths:

| Sequence | Concurrency | Parallelism |
| --- | ---: | --- |
| 1k1k | 1 | TP8 |
| 1k1k | 16 | TP8 |
| 1k1k | 256 | TP8 + EP8 |
| 8k1k | 1 | TP8 |
| 8k1k | 16 | TP8 |
| 8k1k | 256 | TP8 + EP8 |

The profiling branch was first brought up to current `main`
(`92a2b544`). Profiles use one request batch, eager execution, and one
steady-state decode step so ROCTracer exposes all kernels. End-to-end results
use the normal HIP graph path and ten request batches.

## Executive Findings

- All original off/control/fused profiles execute on one GPU stream. There is
  no compute-compute or compute-communication overlap in those traces.
- Enabling vLLM's existing shared-expert auxiliary stream on ROCm creates a
  second HIP stream and moves exactly 57 shared-expert triplets to it, but the
  measured cross-stream overlap is `0 us` at all six points. The GPU finishes
  the routed-expert branch before starting the shared-expert branch.
- The current shared-expert stream patch should not be enabled for M3 on
  MI300X. Its theoretical bound is only about 1-2 ms per decode step, it
  realizes none of that bound, and the tested spans move in the wrong direction.
- Same-batch TP communication cannot be hidden behind the following M3 compute:
  the attention reduction feeds the following Gemma RMSNorm and MoE, while the
  FFN reduction feeds the next layer. Cross-microbatch scheduling such as DBO
  is the relevant compute-communication overlap mechanism.
- The highest-value remaining non-collective c256 kernels are expert GEMMs
  (28.19-30.17 ms), the long-context index score (10.86 ms at 8k/c256), and
  sparse attention plus merge (5.19 ms at 8k/c256). Extending the existing
  gfx94x MXFP8 backend to EP8 and an FP8 index-key cache are the first two
  experiments; corrected FP8 main-KV storage and sparse-attention split-K
  tuning follow.
- c256 is a saturation-throughput point, not a latency operating point. The
  final 8k/c256 run improves TPOT by 9.23% and throughput by 4.22%, while mean
  TTFT rises 14.75% because requests queue behind a saturated decode batch.

## Evidence

- Starting sweep: [run 27510667862](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27510667862)
- Pre-profile MXFP8 baseline: [run 27519117381](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27519117381)
- AITER control profiles: [run 27534984155](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27534984155)
- AITER fused profiles: [run 27534992530](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27534992530)
- AITER-off profiles: [run 27537450736](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27537450736)
- Graph-capture smoke: [run 27537660155](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27537660155)
- Six-point graph benchmark: [run 27538604485](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27538604485)
- c1 graph policy validation: [run 27540379158](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27540379158)
- Shared-expert stream off profiles: [run 27556171480](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27556171480)
- Shared-expert stream on profiles: [run 27556179709](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27556179709)

The profile publish job also succeeded. The six original off-mode traces are
available through the Perfetto relay backed by `InferenceX-trace-storage`. All
18 original off/control/fused traces were reprocessed with the final phase
analyzer; they predate the later index-score launch change and are the discovery
baseline. The 12 shared-stream off/on traces use the current tuned launch and
were also reprocessed locally with the final analyzer.

The stream-on c1 job exposed an analyzer bug rather than a model failure:
Perfetto emitted each stream in order without globally sorting the two streams.
The analyzer now sorts by kernel timestamp before segmenting layers and has a
regression test for that case. The uploaded raw trace reprocesses successfully.
Launch-gap accounting also now uses the union of all stream intervals, so
multistream idle time is not double-counted.

## Decode Inventory

MiniMax M3 has 60 transformer layers:

- 3 dense full-attention layers
- 57 sparse-attention MoE layers
- 60 fused QK RMSNorm, RoPE, and KV-cache insertion calls
- 60 attention residual/collective-norm boundaries
- 60 FFN residual/collective-norm boundaries
- 1 model-input collective-norm boundary

The analyzer recognizes the following complete decode path:

| Group | Named phases |
| --- | --- |
| Setup | decode metadata, decode setup, model-input collective norm |
| Attention projection | QKV projection, fused QK norm/RoPE/cache, output projection |
| Sparse indexing | index score, partial top-k, top-k merge |
| Attention core | sparse attention, sparse merge, dense attention, dense KV write |
| MoE routing | router projection, BF16-to-FP32 cast, router top-k |
| MoE preparation | expert input preparation, token alignment, token sort |
| MoE experts | expert GEMM 1, activation, expert GEMM 2, weighted sum |
| Shared expert | gate/up projection, activation, down projection, gated combine |
| Dense FFN | gate/up projection, activation, down projection |
| Residuals | attention collective norm, FFN collective norm |
| Output | output preparation, logits projection, logits all-gather, sampling |

All 18 off/control/fused traces were recognized. In every trace, the sum of
phase kernel counts equals the trace kernel count, so no GPU kernel in the
selected decode window is unclassified.

## Dependency and Overlap Audit

The sparse-layer critical path is:

```text
QKV + index projections
  -> fused QK/index norm + RoPE + cache writes
  -> index score
  -> index top-k
  -> sparse attention
  -> attention output projection
  -> TP all-reduce + Gemma RMSNorm
  -> FP32 router + routed/shared experts
  -> shared/routed combine
  -> TP all-reduce + Gemma RMSNorm
  -> next layer
```

The resulting overlap opportunities are:

| Candidate | Independent work exists? | Measured/remaining budget | Decision |
| --- | --- | ---: | --- |
| Shared expert vs routed experts | Yes | 1.05-1.90 ms in current traces | Tested; two streams, zero overlap |
| Index branch vs sparse attention | No | Index top-k is an attention dependency | Do not split |
| Main QK/cache vs index QK/cache | Partly, before score | 0.26-0.30 ms total over 60 layers | Too small; already fused |
| Attention TP reduction vs MoE | No | MoE consumes normalized reduction output | Fusion, not overlap |
| FFN TP reduction vs next layer | No | Next layer consumes normalized reduction output | Fusion, not overlap |
| TP reduction vs another microbatch | Yes | Up to about 4.9 ms at fused 8k/c256 | DBO follow-up, not ready for M3 |
| EP dispatch/combine vs shared expert | Theoretically | Included in shared branch test | Current ROCm scheduling serialized it |
| Sparse-attention split-K chunks | Internal parallelism | 5.19 ms current; 5.47 ms discovery | Tune chunk count; not branch overlap |

The main M3 and index Q/K paths are already horizontally fused into one
projection and one norm/RoPE/cache kernel. The index score and top-k then
produce the sparse-attention block list, so there is no expensive independent
attention branch to move to a second stream.

## Profile Results

Baseline eager decode timing, in milliseconds:

| Point | Decode span | Collective + norm | Sparse index | Sparse attention | Expert GEMMs | Layer p50 | Layer p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k/c1 | 102.42 | 91.52 | 1.08 | 0.64 | 3.10 | 1.62 | 1.71 |
| 1k/c16 | 112.50 | 95.63 | 1.14 | 0.76 | 7.32 | 1.76 | 2.14 |
| 1k/c256 | 105.76 | 57.88 | 3.35 | 3.37 | 30.31 | 1.67 | 1.86 |
| 8k/c1 | 102.09 | 90.73 | 1.15 | 0.69 | 3.15 | 1.62 | 1.69 |
| 8k/c16 | 109.85 | 90.78 | 2.36 | 0.90 | 8.06 | 1.73 | 1.86 |
| 8k/c256 | 112.11 | 51.20 | 16.00 | 5.18 | 28.10 | 1.71 | 2.42 |

Fused AITER reduced the eager decode span at all six points:

| Point | Fused span (ms) | Change |
| --- | ---: | ---: |
| 1k/c1 | 97.01 | -5.28% |
| 1k/c16 | 96.19 | -14.50% |
| 1k/c256 | 100.84 | -4.65% |
| 8k/c1 | 96.90 | -5.08% |
| 8k/c16 | 103.32 | -5.95% |
| 8k/c256 | 100.44 | -10.41% |

Custom all-reduce kernels spin while waiting for peers, while AITER can move
part of that wait into inter-kernel gaps. Kernel busy time therefore cannot be
treated as end-to-end speedup. HIP graph benchmark results are the deciding
measurement.

At fused 8k/c256, the remaining 65.8 ms of classified kernel time is led by:

| Phase | Time (ms) | Calls |
| --- | ---: | ---: |
| Expert GEMM 1 + 2 | 28.79 | 114 |
| Sparse index score | 15.19 | 57 |
| Sparse attention + merge | 5.47 | 114 |
| Attention + FFN + input collective norm | 4.88 | 242 |
| QKV projection | 1.49 | 60 |
| Dense attention | 1.36 | 3 |
| Shared expert gate/up + down | 1.69 | 114 |
| Attention output projection | 1.01 | 60 |

The largest remaining launch gaps occur around MoE token alignment and shared
expert projections. After the changes below, expert GEMMs and sparse indexing
are the primary kernel targets at high concurrency.

The current stream-off validation run gives a full-model trace check of the
index-score launch tuning. Against the original AITER-off trace at the same
active batch, score time falls from 15.66 to 10.86 ms at 8k/c256 (`-30.6%`) and
from 3.09 to 2.70 ms at 1k/c256 (`-12.5%`). This is consistent with the
standalone microbenchmark and leaves the expert GEMMs as the dominant c256
compute target.

The two long-context cache readers are also large enough to treat as bandwidth
projects. The current 8k/c256 trace contains 216 active requests. At that
selected batch, the BF16 index-key path reads about 25.8 GB per decode step and
sustains about 2.38 TB/s over its 10.86 ms kernel time. Main sparse attention
reads about 12.9 GB of BF16 K/V data for the selected 16 blocks and sustains
about 2.62 TB/s over its 4.93 ms attend kernel. The separate attention merge
costs another 0.26 ms. At a full nominal batch of 256, the corresponding
traffic scales to 30.6 GB and 15.3 GB. These are minimum traffic estimates and
exclude metadata, query, output, and cache effects.

### Eager Launch Starvation

Every profiled kernel has a correlated HIP launch. The custom all-reduce kernel
spins while waiting for peers, so a long collective can keep the traced rank
"busy" even when useful work is waiting on another rank. When AITER shortens
that synchronization kernel, the eager Python launch rate becomes visible as
global GPU idle:

| Point | Span (ms) | Union busy (ms) | Global idle (ms) | Definite host starvation (ms) | Launch call in gap (ms) | Post-launch device gap (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fused 1k/c16 | 96.19 | 14.59 | 81.60 | 67.32 | 7.37 | 6.91 |
| Fused 8k/c256 | 100.44 | 65.88 | 34.56 | 26.53 | 3.83 | 4.20 |
| Shared-stream 1k/c1 | 106.81 | 14.40 | 92.41 | 75.02 | 9.64 | 7.75 |
| Current off 8k/c256 | 103.78 | 60.99 | 42.79 | 33.83 | 4.16 | 4.80 |

Several original off/control traces show only 0.8-0.9 ms of global idle because
their spinning collectives keep the queue occupied. The current matched
stream-off traces show that this is not stable across ranks and shapes: idle
ranges from 0.81 to 92.09 ms, with 42.79 ms at 8k/c256. A low idle value does
not imply more useful GPU work. Eager profile span is useful for dependency and
kernel attribution, while normal HIP-graph serving is the performance decision.

The per-stream gap total is retained for debugging, but it is not a wall-time
metric when multiple streams exist. The report uses global union-of-streams
idle and cross-stream interval overlap.

### Shared-Expert Stream Validation

vLLM's existing `SharedExperts` path records a synchronization point, enqueues
the routed expert work on the main stream, launches the shared MLP on an
auxiliary stream, and makes the main stream wait before combining the outputs.
The ROCm experiment applies the one-line platform eligibility change from
[vLLM PR 38665](https://github.com/vllm-project/vllm/pull/38665).

The enabled traces contain two streams:

- main stream: attention, routing, routed experts, collectives, output
- auxiliary stream: 57 repetitions of shared gate/up GEMM, SwiGLU, and shared
  down GEMM, exactly 171 kernels

The complete matched result is:

| Point | Active off/on | Off span (ms) | On span (ms) | Delta (ms) | On shared MLP (ms) | Streams | Overlap (us) | Max concurrent streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k/c1 | 1/1 | 103.207 | 106.805 | +3.598 | 1.250 | 2 | 0 | 1 |
| 1k/c16 | 16/16 | 108.778 | 111.602 | +2.824 | 1.055 | 2 | 0 | 1 |
| 1k/c256 | 256/256 | 112.480 | 116.916 | +4.436 | 1.885 | 2 | 0 | 1 |
| 8k/c1 | 1/1 | 103.270 | 112.123 | +8.853 | 1.050 | 2 | 0 | 1 |
| 8k/c16 | 16/16 | 112.868 | 114.714 | +1.846 | 1.059 | 2 | 0 | 1 |
| 8k/c256 | 216/217 | 103.778 | 111.530 | +7.752 | 1.903 | 2 | 0 | 1 |

The span deltas are directional rather than a precise regression measurement:
the jobs used different physical runners, custom all-reduce spin time varies
between ranks, and the final c256 annotations differ by one active request. The
stream evidence is exact. Every enabled trace has 57 gate/up, 57 activation,
and 57 down-projection kernels on the auxiliary stream;
`cross_stream_overlap_us=0` and `max_concurrent_kernel_streams=1` in all six.

Launch correlation distinguishes host lateness from device serialization:

| Point | Shared gate launch lead over routed end, p50 (ms) | Layers launched before routed end | Gate start after routed end, p50 (us) |
| --- | ---: | ---: | ---: |
| 1k/c1 | -0.117 | 0/57 | 122.342 |
| 1k/c16 | 17.818 | 57/57 | 4.528 |
| 1k/c256 | 23.492 | 57/57 | 5.090 |
| 8k/c1 | 12.013 | 57/57 | 4.527 |
| 8k/c16 | 5.732 | 57/57 | 4.408 |
| 8k/c256 | 0.300 | 57/57 | 4.688 |

At five points, every auxiliary gate launch completes before the routed
weighted sum finishes, yet the gate kernel waits until the routed branch is
done. The c16, 1k/c256, and 8k/c1 points provide 5.7-23.5 ms median ready
windows, much longer than the 1-2 ms shared branch, so their zero overlap is a
device scheduling/resource result rather than a late host launch. The 1k/c1
point is instead host-starved.

The existing implementation launches the auxiliary branch only after all
routed kernels have been submitted. Refactoring it into separate "launch
shared", "launch routed", and "join" operations could change queue order, but
the maximum removable work is still only about 1-2 ms. It would enlarge the
ready window at 1k/c1 and 8k/c256, but four other points already expose ample
ready time without overlap, so this is a low-confidence experiment. ROCm graph
capture also adds material risk. The development history in
[vLLM PR 43718](https://github.com/vllm-project/vllm/pull/43718) records:

- side-stream output allocation can hang on first graph replay
- preallocated outputs can avoid that specific hang
- torch.compile/full-graph variants can collapse replay to stream 0
- decode-sized AITER/BF16 kernels can still show zero useful overlap because
  of CU/resource contention

Because eager execution already realizes zero overlap even when both branches
are queued, a graph-mode M3 benchmark was not dispatched. Graph replay is
unlikely to recover work that the hardware scheduler serializes, and it adds a
known hang/collapse failure surface. The current shared-expert stream path is
rejected for M3 on MI300X.

## AITER Decision

vLLM already has the required AITER primitive:
`CustomAllreduce.fused_ar_rms`. M3's accuracy-tuned custom Gemma norm is opaque
to the current compilation pattern, so the profiling recipe directly invokes
the primitive from M3's existing `fused_allreduce_gemma_rms_norm` helper.

The vLLM starting points were
[PR 43953](https://github.com/vllm-project/vllm/pull/43953) and
[PR 44437](https://github.com/vllm-project/vllm/pull/44437).
The M3-specific compile path now also exists in
[vLLM PR 45639](https://github.com/vllm-project/vllm/pull/45639).
[InferenceX PR 1770](https://github.com/SemiAnalysisAI/InferenceX/pull/1770)
is the corresponding MI355X hardware-validation PR. A new InferenceX kernel PR
for the same AR+Gemma-RMS operation would be duplicate work.

The implementation:

- enables only the AITER custom all-reduce dependency
- disables all independently selectable AITER attention, linear, MoE, RMSNorm,
  FP8/FP4 BMM, RoPE, and shared-expert paths
- initializes the communicator before HIP graph capture
- defers both attention and FFN/MoE output reductions into the following Gemma
  RMSNorm helper
- checks source and patched SHA256 fingerprints before serving
- installs checksummed `amd-aiter 0.1.15.post1+rocm7.2` and `flydsl` wheels
- uses the graph-safe two-stage primitive because that AITER tag predates the
  one-stage exit-barrier fix

The two-stage graph path improves c16/c256 but regresses c1. A same-node 1k/c1
comparison measured 23.169 tok/s/GPU with AITER off versus 22.653 with it
enabled, a 2.28% regression. When `fused` is requested, the recipe therefore
forces graph c1 back to `off` while retaining fused eager c1 profiling. The
AITER path remains an explicit experiment for c16/c256 rather than the default
sweep path because it replaces the image wheel at runtime and has mixed
shape-dependent behavior.

The newest released AITER wheel is
[`v0.1.15.post1`](https://github.com/ROCm/aiter/releases/tag/v0.1.15.post1).
It contains the two-stage memory-ordering fix from
[AITER PR 2890](https://github.com/ROCm/aiter/pull/2890), but its tag does not
contain the one-stage exit-barrier fix from
[AITER PR 3514](https://github.com/ROCm/aiter/pull/3514). Once a released wheel
contains PR 3514, retest one-stage c1. At TP8, one M3 token is 12 KiB of BF16
hidden state, so c1 fits the current 128 KiB one-stage gate while c16 does not.

## Sparse Index Optimization

No compatible AITER MiniMax index-score/top-k kernel exists. The generic DSV4
sparse indexer has different layout and scoring contracts, so substituting it
would not be correct.

The checked-in Triton score kernel was microbenchmarked over the exact six
benchmark shapes. The selected launch uses a target grid of 2048 programs,
2 warps, and 3 stages only when `batch >= 128` or when
`batch >= 16 and max_block >= 64`.

| Shape | Current (us) | Tuned (us) | Change |
| --- | ---: | ---: | ---: |
| 1k/c1 | 22.36 | 23.19 | -3.7% |
| 1k/c16 | 21.07 | 21.73 | -3.2% |
| 1k/c256 | 51.96 | 43.73 | +15.8% |
| 8k/c1 | 21.24 | 21.90 | -3.1% |
| 8k/c16 | 27.14 | 23.89 | +12.0% |
| 8k/c256 | 320.92 | 227.15 | +29.2% |

The shape policy enables the tuned launch only for the three winning shapes.
All comparisons had zero maximum absolute output error.

## End-to-End Results

Throughput is total tokens per second per GPU. c1 rows are from the final
policy commit `df9598f3`; c16/c256 rows are inherited unchanged from
`c4d057aa`.

| Point | Starting sweep | MXFP8 baseline | Final | vs MXFP8 | vs start |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k/c1 | 23.34 | 23.15 | 23.16 | +0.06% | -0.76% |
| 1k/c16 | 203.08 | 214.95 | 218.75 | +1.77% | +7.71% |
| 1k/c256 | 782.65 | 852.48 | 857.83 | +0.63% | +9.61% |
| 8k/c1 | 99.88 | 100.12 | 100.47 | +0.35% | +0.59% |
| 8k/c16 | 669.16 | 697.18 | 712.55 | +2.20% | +6.48% |
| 8k/c256 | 1199.22 | 1216.81 | 1268.13 | +4.22% | +5.75% |

Latency from the same final policy:

| Point | Mean TTFT (s) | p99 TTFT (s) | Mean TPOT (ms) | p99 TPOT (ms) | Mean E2EL (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k/c1 | 0.157 | 0.549 | 10.60 | 10.60 | 9.91 |
| 1k/c16 | 0.203 | 0.814 | 17.79 | 18.35 | 16.48 |
| 1k/c256 | 1.091 | 13.208 | 71.74 | 77.20 | 67.23 |
| 8k/c1 | 0.450 | 0.481 | 10.66 | 10.67 | 10.36 |
| 8k/c16 | 0.918 | 6.626 | 23.69 | 27.67 | 22.61 |
| 8k/c256 | 52.821 | 175.323 | 165.69 | 204.11 | 205.92 |

Relative to the MXFP8 baseline, mean TPOT changes by -0.08%, -1.77%, -0.66%,
-0.34%, -2.33%, and -9.23% in table order. Mean TTFT is stable or better
through c16, except for noise below 2%. At c256 it rises 2.91% for 1k and
14.75% for 8k even though TPOT improves. The 8k/c256 batch is saturated: faster
decode reduces service time, but request admission and queue position dominate
mean TTFT. Its p99 TTFT still improves 3.96%.

The 52.82 s mean TTFT at 8k/c256 is therefore not a single-kernel latency
problem. A latency-oriented deployment needs a lower active-decode/admission
cap or a separate SLO-tuned configuration; c256 should remain the saturation
throughput measurement. Kernel work at that point should be judged primarily
by TPOT and throughput.

The final c1 validation logs explicitly show the requested `fused` experiment
being changed to effective `off` by the graph policy.

GSM8K strict exact match at 8k/c256 was `0.95830` (1264/1319), versus
`0.95679` (1262/1319) before these changes. The difference is within sampling
error and shows no accuracy regression.

## Ranked Next Experiments

### 1. Extend the Existing gfx94x MXFP8 Backend to EP8

This is the largest remaining compute target. The two expert GEMMs consume
30.06 ms at fused 1k/c256 and 28.79 ms at fused 8k/c256; the current
post-tuning control traces independently measure 30.17 and 28.19 ms. The
current hybrid gfx94x backend intentionally falls back to BF16 for all
expert-parallel runs because the first native implementation was slower at the
measured large local batches.

[vLLM PR 45567](https://github.com/vllm-project/vllm/pull/45567) is already the
native gfx94x MXFP8 implementation used by this branch. Do not create a
duplicate kernel PR. Extend and validate that backend's dispatch for the exact
M3 EP8 route:

- 16 local experts per GPU
- hidden size 6144 and expert intermediate size 3072
- top-k 4 with the observed c256 route distribution
- native E4M3FNUZ weights and per-32 E8M0 scales
- separate GEMM1 and short-K GEMM2 tuning for gfx942

Do not simply remove the EP fallback. Capture the routed-token histogram,
microbenchmark the exact per-layer shapes, and select native versus BF16 by
shape. Long-context memory is a constraint: the current policy already stores
BF16-only weights every fifth layer to fit the model, so retaining both native
and BF16 copies for all EP layers is not viable without a new memory policy.

### 2. FP8 Index-Key Cache

Launch tuning reduced the full-trace score kernel from 15.66 to 10.86 ms at
8k/c256 and from 3.09 to 2.70 ms at 1k/c256, but it remains the second-largest
high-concurrency kernel. M3's index cache is currently hard-coded to BF16 even
though the attention config has an `indexer_kv_dtype` field and the NVIDIA
model already plumbs it.

At the 216-request annotation selected by the current 8k/c256 trace, 57 sparse
layers read approximately 25.8 GB of BF16 index-key data per decode step before
accounting for cache effects; a full 256-request step is 30.6 GB. FP8 storage
halves that traffic and the side-cache footprint. This has materially more
upside than only fusing top-k because the score kernel dominates the index path.

Required work:

- plumb `indexer_kv_dtype` through the AMD M3 model and cache specification
- add FP8 insertion for the normalized index keys
- load and convert FP8 keys in `_decode_index_score_kernel`
- compare top-k block agreement, logits, and full GSM8K against BF16
- profile both 1k/c256 and 8k/c256 before enabling any lower-concurrency shape

Index keys are normalized, which makes unit-scale FP8 plausible, but accuracy
must be measured rather than assumed.

### 3. Corrected FP8 Main KV Cache

The current MI300X recipe deliberately keeps the main attention cache in BF16.
That decision was made while the ROCm path reinterpreted FNUZ cache bytes as
FN, which severely corrupted attention. Open
[vLLM PR 45563](https://github.com/vllm-project/vllm/pull/45563) fixes the
dtype view and reports full GSM8K strict exact match recovering from `0.0099`
to `0.9575`; its separate InferenceX eval reached `0.9606`.

This needs a matched performance and accuracy A/B on the final profiling
branch, not an assumption that FP8 is now free. The upside is material:

- the attend kernel is 5.21 ms in the discovery trace and 4.93 ms in the
  current 216-request trace, plus a 0.26 ms merge in both
- selected BF16 K/V traffic is about 12.9 GB at 216 requests and 15.3 GB at 256
- correct FNUZ FP8 storage halves that K/V traffic and cache footprint

Apply the dtype fix, verify physical FNUZ cache bytes, and compare BF16 versus
FP8 at 1k/c256 and 8k/c256 plus full GSM8K. Retune split-K only after choosing
the cache dtype because FP8 changes the kernel's bandwidth/compute balance.
At an 8k active context with c256, main-KV FP8 saves roughly 32 GB per GPU;
combining it with an FP8 index cache saves about 47.5 GB. That may also change
whether retaining both native and BF16 EP weights is feasible.

### 4. Sparse-Attention Split-K and Single-Chunk Fast Path

At c256, the current `TARGET_GRID=256` policy chooses one top-k chunk. Each
program then walks all eight valid blocks at 1k or all 16 at 8k. The attend
kernel costs 3.12 ms at 1k/c256 and 4.93 ms at 8k/c256; the unconditional merge
adds 0.29 ms and 0.26 ms respectively even though there is only one chunk.

Microbenchmark `num_topk_chunks` 1, 2, 4, and 8 on the exact c256 shapes.
More chunks may improve occupancy and shorten each program despite the partial
buffer and merge cost. For the one-chunk winner, write directly to the final
output and skip the merge and unnecessary LSE output. Keep lower-concurrency
shape policies separate: c1/c16 already choose more chunks to reach the target
grid.

### 5. Score + Partial Top-K Fusion

The decode index path currently writes a full FP32 score tensor and launches a
separate partial top-k kernel. Fuse score production with per-chunk top-k and
write only the final candidates. This removes one launch and the score
store/read round trip. The c256 path already has one chunk and therefore no
separate top-k merge kernel.

The bound is smaller than FP8 index storage: partial top-k costs 0.29-0.32 ms
at current c256, while index-key reads dominate the 10.86 ms long-context
score. Prioritize this after the cache-dtype experiment unless a fused kernel
also improves score tiling or cache reuse.

### 6. Fuse the Shared Expert into Native MXFP8 MoE

The separate shared MLP costs about 1.0-1.9 ms, plus 0.31-0.82 ms for the serial
combine. Since side-stream overlap failed, the more promising design is to
append the shared expert as an always-routed expert slot in the native MXFP8
grouped MoE:

- remap shared gate/up/down weights into the expert tensors
- widen top-k from 4 routed slots to 5 total slots
- give the shared slot unit weight
- preserve routed scaling without applying it to the shared slot
- validate TP/EP weight mapping and GSM8K

This follows the AITER fused-shared-expert model used by
[vLLM PR 44313](https://github.com/vllm-project/vllm/pull/44313) and
[vLLM PR 44434](https://github.com/vllm-project/vllm/pull/44434), but M3's
MXFP8 backend is not compatible with that AITER path. The M3 Amdahl bound is
also much smaller than the GLM/Qwen gains reported in those PRs.

### 7. DBO TP All-Reduce/Compute Overlap

[vLLM PR 44677](https://github.com/vllm-project/vllm/pull/44677) demonstrates
95.7% TP all-reduce hiding and up to 8.1% prefill throughput gain on H200 by
running two microbatches on compute and communication streams. This is the
right dependency model for M3 communication overlap because it overlaps one
microbatch's collective with another microbatch's compute.

It is not directly ready for the current M3 recipe:

- c256 uses expert parallelism, while the PR still requires a DeepEP backend
  whenever EP is enabled
- splitting c16 into two 8-token microbatches crosses the current gfx94x hybrid
  dispatch boundary from native MXFP8 (`9..831`) to BF16 (`<=8`)
- the direct AITER fused AR+RMS helper bypasses the generic
  `tensor_model_parallel_all_reduce` wrapper where DBO yields are inserted
- ROCm multistream graph behavior requires dedicated replay validation

At fused 8k/c256, all remaining collective-norm work is 4.88 ms, so the
same-step upper bound is about 5% before microbatch overhead. Revisit DBO after
the EP backend and fused-collective integration are explicit.

### Lower-Priority or Rejected Paths

- Splitting M3's main/index QK preparation across streams can hide at most
  0.26-0.30 ms over the full 60-layer step and adds graph complexity.
- Quantizing the router is not recommended. M3 deliberately computes router
  logits in FP32 for accuracy.
- Router/top-k/alignment/input preparation totals about 2.76 ms at fused
  8k/c256. Kernel fusion there is useful, but below expert GEMM and index-cache
  work.
- [vLLM AsyncTP](https://github.com/vllm-project/vllm/pull/17882) is not a
  current MI300X path. The pass manager imports `AsyncTPPass` only for the CUDA
  platform, not ROCm, and its supported GEMM+reduce-scatter/all-gather patterns
  do not match M3's direct AITER all-reduce+Gemma-RMS helper.
- The current shared-expert stream path is rejected: two streams are visible,
  actual overlap is zero, and graph-mode failure risk is documented upstream.

## Implementation

The main implementation surfaces are:

- `.github/workflows/profile.yml`
- `benchmarks/single_node/fixed_seq_len/minimaxm3_fp8_mi300x.sh`
- `benchmarks/single_node/fixed_seq_len/minimaxm3_mi300x_deferred_ffn_ar.patch`
- `benchmarks/single_node/fixed_seq_len/minimaxm3_mi300x_index_topk.patch`
- `utils/install_minimaxm3_aiter.py`
- `utils/patch_minimaxm3_aiter_ar_rms.py`
- `utils/patch_vllm_mi300x_rank0_profiler.py`
- `utils/patch_vllm_rocm_shared_experts_stream.py`
- `utils/analyze_profile_trace.py`
- `experimental/minimax_m3_index_score_bench.py`

Profiler scratch data is written under `/tmp/inferencex-profile`; only the
final compressed trace file is staged into `/workspace`.

The branch leaves `perf-changelog.yaml` untouched.
