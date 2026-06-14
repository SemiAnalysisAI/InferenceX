# DeepSeek-V4 B300 TRT Huawei-Style Offline Benchmark

This is a branch-only benchmark. It does not use the normal InferenceX serving
pipeline and is not intended to merge into `main`.

## What It Measures

The benchmark reproduces the measurement contract implemented by Huawei's
DeepSeek-V4 offline recipe:

- fixed global batch
- full-batch prefill before decode
- two warmup decode rounds
- 256 measured decode rounds
- MTP3
- first measured round skipped
- upper-IQR outlier removal
- raw decode-round TPOT
- throughput calculated as `global batch / TPOT / devices`
- MTP accepted-token yield reported separately

The reference implementation is:

```text
/Users/bshan/Documents/cann-recipes-infer-master/
  docs/models/deepseek-v4/deepseek_v4_inference_guide.md
  models/deepseek-v4/models/model_infer.py
  executor/utils/common_utils.py
```

The hardware is not identical. Huawei's published DeepSeek-V4 Pro table uses
16 950DT chips and hybrid MXFP8/MXFP4. This branch uses one eight-GPU B300 node
and the staged FP4 TensorRT-LLM checkpoint.

## Fixed Workload

| Setting | Value |
|---|---|
| Model | `/scratch/models/DeepSeek-V4-Pro` |
| TRT image | `ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066` |
| TRT source | `c185066` |
| Topology | DEP8: attention DP 8, expert parallel 8, MoE TP 1 |
| Global batch sizes | `16`, `64`, `128` |
| Input length | exactly 8192 real token IDs |
| MTP depth | 3 draft tokens |
| Warmup | at least two full-batch decode rounds |
| Measurement | first 256 consecutive full-batch decode rounds |
| Sampling | temperature 1, top-p 1, top-k 0, EOS ignored |
| Seed | TRT engine-global seed 42 |
| Routing | perfect router |
| Overlap scheduler | disabled |

The prompt is built from pinned InfiniteBench `longbook_qa_eng.jsonl` data and
DeepSeek-V4 chat formatting. Prompt construction fails instead of inserting
pad or synthetic token IDs.

## Global And Local Batch

There is one authoritative `global_batch_size`. Every per-rank capacity is
derived from it:

```text
local_batch_size = global_batch_size / 8
max_batch_size = local_batch_size
cuda_graph_batch_size = local_batch_size
max_num_tokens = local_batch_size * 8192
kv_cache.free_gpu_memory_fraction = 0.60
moe.max_num_tokens = 65536
```

| GBS | Local batch/rank | TRT max_num_tokens/rank |
|---:|---:|---:|
| 16 | 2 | 16384 |
| 64 | 8 | 65536 |
| 128 | 16 | 131072 |

KV capacity remains memory-derived at a fixed 60% fraction. Do not set
`kv_cache.max_tokens`: this pinned one-model MTP implementation accounts for
target and draft KV separately, and an explicit exact-sequence cap
underprovisions the target schedule and staggers prefill. The fixed
65536-token MoE cap applies inside a fused-MoE invocation. It lets TRT
internally chunk the very large prefill/autotune tensor while the executor
still schedules the complete local batch in one prefill iteration. Decode has
at most 128 tokens node-wide, so the measured decode rounds never hit this
cap.

The old harness used approximately one prompt's prefill token budget even for
large global batches. TRT therefore queued and staggered requests. Dividing
the global request count by mean per-request TPOT assumed requests that were
not actually decoding together, which is why the derived number could be
roughly twice wall throughput.

`LLM.generate()` also enqueues the batch request by request while the executor
is live. The branch-local MPI entry shim therefore holds the idle executor
until exactly one complete global batch is present, then releases all requests
to attention-DP routing together. Each pass must show one prefill iteration:

```text
numContextRequests = local_batch_size
numGenRequests = 0
numScheduledRequests = local_batch_size
numActiveRequests = local_batch_size
numQueuedRequests = 0
numPausedRequests = 0
```

Decode must then begin with:

```text
numContextRequests = 0
numGenRequests = local_batch_size
numScheduledRequests = local_batch_size
numActiveRequests = local_batch_size
numQueuedRequests = 0
numPausedRequests = 0
```

The decode conditions must hold for 256 consecutive TRT iterations.

## Timing

TRT is created with:

```text
enable_iter_perf_stats = true
max_stats_len = 2048
disable_overlap_scheduler = true
```

The explicit `2048`-entry history bounds memory while covering prefill plus the
worst-case 1024 decode iterations needed when no MTP drafts are accepted. The
pinned executor also supports `-1` as unbounded, but this benchmark does not
need an unbounded history.

After generation, the worker reads `LLM.get_stats()` and selects the first 256
consecutive full-local-batch decode iterations. It then applies the same
filter as Huawei's `process_infer_time`:

1. Skip the first selected decode round.
2. Calculate Q1 and Q3 from the remaining 255 values.
3. Calculate `upper_fence = Q3 + 1.5 * (Q3 - Q1)`.
4. Drop only latencies above the upper fence.
5. Average retained round latencies.

The timer scope is the closest TRT runtime equivalent, not the same
instrumentation point. TRT's `iterLatencyMS` covers one complete executor
iteration with overlap disabled. Huawei records and sums the main-model and
MTP model timing regions inside its CANN decode loop. Treat the result as a
matched workload and aggregation contract, with a runtime-specific timer
scope.

The headline fields are:

```text
decode_round_tpot_ms
decode_step_tput_per_gpu =
    global_batch_size / decode_round_tpot_seconds / 8
```

The generation output cap is 1025 tokens:

```text
1 first token + 256 rounds * (1 target + 3 drafts)
```

That guarantees no request can finish before 256 MTP3 decode rounds. Generation
may run longer when acceptance is lower, but only the first 256 validated
full-batch rounds enter the headline metric.

## MTP Yield

MTP yield is not folded into Huawei's raw table throughput. The benchmark
reports it separately:

```text
observed_tokens_per_step = 1 + accepted_drafts_per_step
output_tput_per_gpu =
    decode_step_tput_per_gpu * observed_tokens_per_step
equivalent_output_tpot_ms =
    decode_round_tpot_ms / observed_tokens_per_step
```

MTP yield must come from TRT's iteration-level speculative counters for the
same validated 256-round window. A row fails validation if those counters are
unavailable; whole-request telemetry is not substituted into the headline
window.

Huawei publishes 1.44 accepted drafts per MTP3 round, or 2.44 output tokens
per round.

## Reference Rows

| GBS | Huawei chips | TPOT ms | Decode steps/s/chip |
|---:|---:|---:|---:|
| 16 | 16 | 17.64 | 56.70 |
| 64 | 16 | 19.03 | 210.16 |
| 128 | 16 | 20.61 | 388.23 |

The direct comparison is B300 `decode_step_tput_per_gpu` divided by Huawei
`decode_step_tput_per_chip`. The result also reports an output-token ratio
using each stack's observed or published MTP yield.

## Execution Chain

1. Dispatch `.github/workflows/e2e-tests.yml`.
2. Fan out one job per requested global batch.
3. Allocate one exclusive eight-GPU B300 Slurm node per job.
4. Build the exact 8192-token corpus.
5. Start one fresh TensorRT-LLM engine.
6. Run the short full-batch warmup.
7. Run one measured generation.
8. Validate and filter 256 iteration stats.
9. Upload per-job result/debug artifacts.
10. Collect `offline_aggregate.json`, `offline_summary.md`, and `agg_bmk.json`.

There is no HTTP server, request-rate generator, generic benchmark client,
serial scheduler tuning, or normal InferenceX matrix processing.

## Result Fields

`offline_aggregate.json` is authoritative:

- `decode_round_tpot_ms`: Huawei-style filtered full-batch round latency
- `decode_step_tput_per_gpu`: raw Huawei-comparable step rate
- `observed_tokens_per_step`: measured TRT MTP output yield
- `output_tput_per_gpu`: step rate multiplied by output yield
- `wall_output_tput_per_gpu`: whole `LLM.generate()` diagnostic
- `filter.retained_rounds`: retained values after first-round skip/outliers
- `filter.outlier_rounds`: upper-IQR values removed
- `schedule_validation`: iteration range and exact local-batch proof
- `b300_to_huawei_decode_step_ratio`: direct raw-step comparison
- `b300_to_huawei_output_ratio`: output rate using each stack's yield

`results_bmk/agg_bmk.json` remains renderer-compatible. Standard throughput
and TPOT fields use acceptance-adjusted output-token metrics. Custom flat
fields retain raw decode-round TPOT, decode-step throughput, GBS, local batch,
token yield, and measured round count.

## Dispatch

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT Huawei fixed GBS' \
  -f 'inputs[global_batch_sizes]=16,64,128' \
  -f 'inputs[salloc-time]=150' \
  -f 'inputs[worker-timeout]=7200'
```

Monitor:

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml --event workflow_dispatch --limit 1 \
  --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
```

Download:

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  -n offline-trt-summary -D ./offline-summary
jq '.rows' ./offline-summary/offline_aggregate.json
jq '.' ./offline-summary/agg_bmk.json
```

## Debugging

Each job uploads:

- `offline_result_gbsN.json`
- `offline_controller_gbsN.log`
- `offline_gpu_metrics_gbsN.csv`
- `offline_debug_gbsN.tar.gz`

The debug archive contains the corpus manifest, worker log/result,
`warmup_iteration_stats.json`, `measured_iteration_stats.json`, and every-rank
perfect-router marker.

Debug in this order:

1. Read the result's `failure_kind`, `phase`, and `error`.
2. Read `offline_controller_gbsN.log`.
3. Extract the debug archive and inspect `worker.log`.
4. Use `jq` on iteration stats; do not dump the full JSON into the chat.
5. Confirm context-only iterations precede a consecutive full-batch decode
   range.
6. Confirm all rank markers are `0..7` with identical fixed environment.

Do not fix failures by reducing the global batch, splitting prefill across
executor iterations, enabling overlap scheduling, weakening schedule
validation, reducing MTP depth, or reverting to per-request TPOT. Internal
fused-op chunking is allowed only when the schedule proof still shows one
complete full-batch prefill iteration.
