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

## Evidence

- Starting sweep: [run 27510667862](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27510667862)
- Pre-profile MXFP8 baseline: [run 27519117381](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27519117381)
- AITER control profiles: [run 27534984155](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27534984155)
- AITER fused profiles: [run 27534992530](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27534992530)
- AITER-off profiles: [run 27537450736](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27537450736)
- Graph-capture smoke: [run 27537660155](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27537660155)
- Six-point graph benchmark: [run 27538604485](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27538604485)
- c1 graph policy validation: [run 27540379158](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27540379158)

The profile publish job also succeeded. The six off-mode traces are available
through the Perfetto relay backed by `InferenceX-trace-storage`. All 18 raw
traces were reprocessed with the final phase analyzer; the profile runs predate
the later index-score launch change and are the discovery baseline for it.

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

## AITER Decision

vLLM already has the required AITER primitive:
`CustomAllreduce.fused_ar_rms`. The M3 ROCm model cannot use the normal
`torch.compile` pattern matcher, so the recipe directly invokes the primitive
from M3's existing `fused_allreduce_gemma_rms_norm` helper.

The vLLM starting points were
[PR 43953](https://github.com/vllm-project/vllm/pull/43953) and
[PR 44437](https://github.com/vllm-project/vllm/pull/44437).

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

The final c1 validation logs explicitly show the requested `fused` experiment
being changed to effective `off` by the graph policy.

GSM8K strict exact match at 8k/c256 was `0.95830` (1264/1319), versus
`0.95679` (1262/1319) before these changes. The difference is within sampling
error and shows no accuracy regression.

## Implementation

The main implementation surfaces are:

- `.github/workflows/profile.yml`
- `benchmarks/single_node/fixed_seq_len/minimaxm3_fp8_mi300x.sh`
- `benchmarks/single_node/fixed_seq_len/minimaxm3_mi300x_deferred_ffn_ar.patch`
- `benchmarks/single_node/fixed_seq_len/minimaxm3_mi300x_index_topk.patch`
- `utils/install_minimaxm3_aiter.py`
- `utils/patch_minimaxm3_aiter_ar_rms.py`
- `utils/patch_vllm_mi300x_rank0_profiler.py`
- `utils/analyze_profile_trace.py`
- `experimental/minimax_m3_index_score_bench.py`

Profiler scratch data is written under `/tmp/inferencex-profile`; only the
final compressed trace file is staged into `/workspace`.

The branch leaves `perf-changelog.yaml` untouched.
