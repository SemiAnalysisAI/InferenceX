# TRT Bench Working Notes

These are operational notes for running and debugging `trt-bench`.

## Current Contract

- Branch-only; never merge to `main`.
- One B300 node, eight GPUs, DEP8.
- DeepSeek-V4 Pro FP4, TRT commit `c185066`.
- Exact GBS `16`, `64`, `128`.
- Exact 8192-token prompts.
- MTP3, temperature 1, engine-global seed 42, EOS ignored.
- Perfect router, LM-head TP, heuristic sparse top-k, ConfigurableMoE.
- Overlap scheduler disabled.
- Two warmup decode rounds.
- 256 measured full-batch decode rounds.
- First measured round skipped and upper-IQR outliers removed.
- Headline throughput is `GBS / decode-round TPOT / 8`.
- MTP yield is separate.

## Why The Old Result Was Too High

Run `27483465692` used a request pool and mean per-request decode windows. At
GBS 1024 it reported:

```text
derived output: 1582.72 tok/s/GPU
wall output:     731.58 tok/s/GPU
```

Those are not two measurements of the same schedule. The derived formula
multiplied each request's active-window speed by all 1024 requests even though
TRT queued and staggered them. The implied average active decode population
was about 473 requests, not 1024.

That run and all deleted experiment matrices are historical optimization data,
not Huawei-comparable benchmark rows.

## Source Audit

Huawei:

```text
/Users/bshan/Documents/cann-recipes-infer-master/
```

Relevant implementation:

- `models/deepseek-v4/infer.py`: one warmup generation, then one measured
  generation.
- `models/deepseek-v4/models/model_infer.py`: warmup stops after two decode
  rounds; measured decode runs `max_new_tokens=256` rounds; each recorded time
  is main-model time plus all MTP model times.
- `executor/utils/common_utils.py`: skips the first recorded decode round,
  removes values above `Q3 + 1.5 * IQR`, and averages retained values.
- The guide's throughput is exactly `GBS / TPOT / chips`.

Pinned TRT:

- `LLM.get_stats()` returns per-iteration `iterLatencyMS` and inflight batch
  counts when `enable_iter_perf_stats=true`.
- Set `max_stats_len=2048`. It keeps the history bounded while retaining
  prefill plus the worst-case 1024 decode iterations. The pinned executor
  treats `-1` as unbounded.
- `max_batch_size` is per attention-DP rank.
- `TLLM_METRICS_ALL_RANKS=1` adds a per-iteration collective and must remain
  disabled for timing.
- `iterLatencyMS` measures a complete TRT executor iteration. Huawei sums
  main-model and MTP timing regions inside CANN. The workload, decode window,
  and filtering match; the runtime instrumentation point does not.

## Capacity Table

| GBS | Local/rank | `max_batch_size` | CUDA graph | `max_num_tokens` |
|---:|---:|---:|---:|---:|
| 16 | 2 | 2 | 2 | 16384 |
| 64 | 8 | 8 | 8 | 65536 |
| 128 | 16 | 16 | 16 | 131072 |

`max_num_tokens` is intentionally much larger than the old recipe. It permits
all local 8192-token prompts to prefill in the same iteration. KV capacity is
memory-derived with a fixed 0.60 free-memory fraction. Do not set an exact
`kv_cache.max_tokens`: the pinned DeepSeek-V4 multi-pool cache manager did not
turn the exact-sequence quota into sufficient physical capacity, and the cap
used in run `27486168511` admitted only half of each local batch into prefill.
TRT fused MoE is capped at 65536 tokens per internal invocation so its
synthetic autotune and prefill tensors are chunked without splitting the
executor-level prefill iteration. Measured decode is far below the cap.

The pinned engine also uses `max_num_tokens` to build synthetic shapes during
`PyTorchModelEngine.warmup()`. GBS128 run `27487131935` completed GBS16 and
GBS64,
but spent 62 minutes in GBS128 initialization before cancellation because it
was tuning a synthetic 131072-token prefill shape. The MPI shim now
temporarily limits only this internal warmup to 65536 tokens. It restores
`max_num_tokens=131072` in a `finally` block before the benchmark's real
warmup and measured generations. The run remains invalid unless schedule
validation proves one local-batch-16 prefill and 256 full-batch decode rounds.

Run `27488380128` exposed an implementation mistake in the first cap: it
patched the abstract `ModelEngine.warmup`, while the runtime calls the
overridden `PyTorchModelEngine.warmup`. GPU activity stopped after initial
setup and the job remained in engine initialization for 52 minutes. The shim
and its unit test now target the concrete override explicitly.

Run `27486396235` proved memory-derived KV restores a full GBS16 prefill, but
GBS64 then exposed a separate pinned-kernel limit: the packed-FP8 CUDA
quantizer rejected the 65536-row MTP `h_proj` launch during engine warmup.
Each MPI rank now restricts that fused tactic to at most 32768 rows and uses
TRT's existing Triton quantizer above the limit. Decode matrices are only
local batch 2/8/16, so headline timing remains on the original fused path.

## Schedule Gate

The branch-local MPI entry shim waits at each idle pass boundary until exactly
one complete GBS has been enqueued. This prevents the live TRT executor from
starting prefill after only the first few `generate_async()` submissions.

There must be exactly one prefill iteration with:

```text
context == local_batch
generation == 0
scheduled == local_batch
active == local_batch
queued == 0
paused == 0
```

The first decode iteration and the next 255 must all show:

```text
context == 0
generation == local_batch
scheduled == local_batch
active == local_batch
queued == 0
paused == 0
```

Iteration IDs must be consecutive. Any mixed context/decode iteration or
partial first decode batch is `fixed_batch_validation`.

## Output Cap

TRT stops by emitted tokens, while Huawei stops by decode rounds. The measured
cap is:

```text
1 + 256 * (3 + 1) = 1025 tokens
```

Therefore no request can finish before 256 MTP3 rounds. The first 256 valid
rounds are measured; later rounds only let lower-acceptance requests reach the
common output cap.

The warmup cap is six tokens, which guarantees at least two MTP3 rounds. Only
the first two valid warmup rounds are required.

## Dispatch

Full sweep:

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

Canary:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT fixed GBS16 canary' \
  -f 'inputs[global_batch_sizes]=16' \
  -f 'inputs[salloc-time]=90' \
  -f 'inputs[worker-timeout]=5400'
```

Find and watch:

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml --event workflow_dispatch --limit 1 \
  --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
```

## Result Interpretation

Authoritative offline row:

- `decode_round_tpot_ms`: filtered full-batch TRT iteration latency
- `decode_step_tput_per_gpu`: direct Huawei-comparable rate
- `observed_tokens_per_step`: MTP output multiplier
- `output_tput_per_gpu`: step rate times output multiplier
- `wall_output_tput_per_gpu`: whole generation diagnostic
- `filter`: first-round skip and retained/outlier counts
- `schedule_validation`: exact selected iteration and local-batch proof

Flat renderer row:

- `tput_per_gpu` and `output_tput_per_gpu`: acceptance-adjusted output rate
- `mean_tpot`: equivalent output-token TPOT
- `decode_round_tpot_ms`: custom raw-round field
- `decode_step_tput_per_gpu`: custom raw-step field
- `global_batch_size`, `local_batch_size`, `measured_decode_rounds`: custom
  workload proof

Do not put the custom aggregate wrapper in `results_bmk`; the unofficial-run
API expects every JSON object there to be a flat benchmark row.

## Debug Checklist

1. Inspect `Show result headline`.
2. Download `offline-trt-job-gbsN`.
3. Read `offline_controller_gbsN.log`.
4. Extract `offline_debug_gbsN.tar.gz`.
5. Inspect `worker_result.json` and `worker.log`.
6. Query iteration stats:

   ```bash
   jq -r '
     .[] |
     [
       .iter,
       .iterLatencyMS,
       .numActiveRequests,
       .numQueuedRequests,
       .inflightBatchingStats.numContextRequests,
       .inflightBatchingStats.numGenRequests,
       .inflightBatchingStats.numScheduledRequests
     ] | @tsv
   ' measured_iteration_stats.json | head -40
   ```

7. Confirm rank markers are exactly `0..7`.
8. Check GPU telemetry for OOM, idle prefill gaps, or a hung rank.

Do not weaken the fixed-batch gate. Infrastructure errors may be retried on a
fresh node; capacity or schedule failures require an implementation fix.
