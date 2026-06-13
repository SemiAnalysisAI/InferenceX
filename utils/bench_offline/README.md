# DeepSeek-V4 B300 TensorRT-LLM Offline Benchmark

This is a disposable `trt-bench` branch experiment. It is not designed to
merge into `main`, populate inferencex.com, or share the normal serving sweep.

## Execution Chain

The benchmark is dispatched through the existing
`.github/workflows/e2e-tests.yml` path because GitHub only permits
`workflow_dispatch` for workflow files that also exist on the default branch.
On `trt-bench`, that workflow is replaced with this offline-only chain:

1. Validate either a concurrency list or up to ten JSON experiment entries.
2. Fan out at most ten `b300` runner jobs in parallel.
3. Allocate one exclusive B300 Slurm node with eight GPUs.
4. Run the pinned TRT image with `/scratch/models/DeepSeek-V4-Pro`.
5. Reuse an image-keyed CuTe DSL compile cache from shared `/data`.
6. Build an exact 8192-token InfiniteBench corpus.
7. For an experiment entry, start one fresh engine, warm it up once, and
   measure one pass. The legacy mode still tunes at most six fresh engines
   and repeats the winner on another fresh engine.
8. Upload one result and debug bundle per experiment.
9. Collect the rows with `utils/bench_offline/summarize.py`.

There is no server, HTTP client, request-rate generator, matrix config,
`process_result.py`, `summarize.py`, or generic `collect-results` stage in this
path. `LLM.generate()` receives one fixed batch of token-ID prompts directly.

## Pinned Inputs

- Hardware: one B300 node, eight GPUs allocated; four or eight are active
  depending on the explicit recipe shape.
- Image:
  `ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066`
- TRT source identity: `c185066`.
- Model path: `/scratch/models/DeepSeek-V4-Pro`.
- Dataset: `xinrongzhang2022/InfiniteBench`,
  `longbook_qa_eng.jsonl`.
- Dataset revision:
  `90f0394333616266d9fe85824ceaf505093cbaa5`.
- Concurrencies: `4,8,16,32,64,128,256,512,1024`.
- Prompt length: exactly 8192 real token IDs, with no padded IDs.
- Generated output: exactly 625 tokens per request with EOS ignored.
- Speculation: fixed MTP3.
- Huawei-primary parallelism: DEP8; LM-head TP is an explicit candidate
  toggle.
- Secondary production-recipe shapes: TP4 and DEP4. They remain offline but
  do not receive Huawei comparison ratios.
- Sampling: temperature `1.0`, top-p `1.0`, top-k `0`.
- Pinned TRT PyTorch sampler seed: one engine-global generator seeded to
  `42`.

The prompt retains the CANN recipe's literal instruction to summarize the
story in `256 words`. That prompt text is independent of the 625-token
generation budget.

The pinned TRT PyTorch sampler does not apply request-level
`SamplingParams.seed`; it advances one global generator for the lifetime of
an engine. CANN also globally seeds its temperature-1 sampler to `42`, but
the two implementations use different sampling algorithms. Output sequences
can therefore change between passes and scheduler candidates. Digests expose
that variation; they are not an assertion that every pass must emit the same
tokens.

## Parallelism

The candidate records one of three checked-in B300 recipe shapes:

- `DEP8`: eight active GPUs, attention DP on, expert parallel 8, MoE TP 1.
  This is the Huawei-primary shape.
- `TP4`: four active GPUs, attention DP off, expert parallel 1, MoE TP 4.
  This is the low/mid-concurrency production recipe.
- `DEP4`: four active GPUs, attention DP on, expert parallel 4, MoE TP 1.
  This is the production attention-DP recipe.

LM-head TP is recorded per attention-DP candidate. MTP max draft length stays
at three unless a separately labeled non-comparable experiment changes it.

The worker validates these resolved TRT arguments after every engine starts,
including engine `max_batch_size`, `max_num_tokens`, and CUDA graph batch
sizes. It also requires exactly ranks `0..N-1` for the selected active-GPU
count; an eight-rank launch cannot be mislabeled as TP4 or DEP4.

Perfect routing is enabled with `ENABLE_PERFECT_ROUTER=1`. TRT's MPI pool only
explicitly forwards `TRTLLM_*` and `TLLM_*` variables, so
`TRTLLM_ENABLE_PERFECT_ROUTER=1` is also set. Before creating `LLM`, the worker
replaces TRT's pinned `GenerationExecutorProxy.worker_main` reference with
`trt_mpi_entry.worker_main`. That shim recreates the unprefixed variable and
writes a marker before each rank imports TRT's real model worker. The benchmark
requires one marked entrypoint per active rank. `sitecustomize.py` provides an
additional early-process alias but is not the primary proof.

Fresh TRT engines spend most startup time in model initialization and
Blackwell CuTe DSL compilation, not in the measured pass. The workflow mounts
the image-keyed shared path
`/data/trtllm-cache/dsv4-c185066-sm100a/cute-dsl` and sets
`CUTE_DSL_CACHE_DIR`. Because TRT's MPI pool filters unprefixed variables,
`TRTLLM_BENCH_CUTE_DSL_CACHE_DIR` carries the path into
`trt_mpi_entry.worker_main`, which restores it before importing TRT on every
rank. Result markers prove propagation only. Cache-prime run `27475868179`
left zero files in that directory, so this variable is not currently proven
to cache CuTe compilation and must not be credited with reducing startup.

Perfect routing intentionally changes expert selection, so generated text is
not an accuracy result. This harness is performance-only, matching the
load-balance-idealized comparison shape.

## Prompt Construction

`prompts.py` uses TRT's pinned
`tensorrt_llm.tokenizer.deepseek_v4.DeepseekV4Tokenizer`.

For each context it:

1. Applies the CANN InfiniteBench prefix and suffix.
2. Renders DeepSeek-V4 chat mode, ending in
   `<｜Assistant｜></think>`.
3. Adjusts the number of context tokens until the rendered prompt contains
   exactly 8192 token IDs.
4. If BPE boundary merges skip directly from 8191 to 8193, tries a
   deterministic whitespace separator between the context and suffix. As a
   final fallback it trims a small number of decoded context-tail characters.
5. Records any boundary or tail adjustment in the corpus manifest.
6. Fails instead of padding if no exact length can be constructed.
7. Writes prompts to a little-endian `uint32` binary corpus.

`corpus_manifest.json` records the dataset checksum, corpus checksum, prompt
checksums, token counts, and context-trimming adjustments. The large
`corpus.bin` file is excluded from the uploaded debug archive.

## Timing And Throughput

For every request:

```text
ttft = first_token_time - arrival_time
decode_window = last_token_time - first_token_time
decode_tokens = output_tokens - 1 = 624
decode_iterations = last_iter - first_iter
token_tpot = decode_window / decode_tokens
```

The headline `mean_token_tpot_ms` is the arithmetic mean of per-request
`token_tpot` values in the measured pass.

Each request also records a SHA-256 digest of its 625 generated token IDs.
Per-pass sequence digests make fresh-engine tuning variation visible without
uploading generated text or full token arrays.

`derived_output_tput_per_gpu` is:

```text
concurrency / mean_token_tpot_seconds / active_gpu_count
```

`mean_step_tpot_ms` is the arithmetic mean of:

```text
decode_window / decode_iterations
```

`derived_step_tput_per_gpu` is:

```text
concurrency / mean_step_tpot_seconds / active_gpu_count
```

`wall_output_tput_per_gpu` is:

```text
all generated output tokens / total measured batch wall time / active_gpu_count
```

When TRT populates its speculative counters, raw `acceptance_rate` is
weighted:

```text
total accepted draft tokens / total proposed draft tokens
```

The pinned TRT PyTorch MTP path currently returns zero for both counters even
while iteration telemetry proves MTP is active. In that case
`raw_speculative_metrics_available=false`, `acceptance_rate=null`, and the
benchmark reports:

```text
effective_accepted_drafts_per_step = observed_tokens_per_step - 1
effective_acceptance_rate = effective_accepted_drafts_per_step / 3
```

`observed_tokens_per_step` is:

```text
total decode tokens / total decode iterations
```

The Huawei table's throughput values are decode-step throughput: each value is
exactly `global batch size / step TPOT / chips`. Results therefore report
three distinct comparisons instead of treating acceptance as a hardware
multiplier:

```text
step-rate ratio =
    B300 derived step throughput/GPU / Huawei step throughput/chip

actual output ratio =
    B300 derived output throughput/GPU
    / (Huawei step throughput/chip * Huawei published 2.44 tokens/step)

same-yield diagnostic =
    B300 derived output throughput/GPU
    / (Huawei step throughput/chip * TRT observed tokens/step)
```

Do not multiply Huawei throughput by raw acceptance rate. Huawei publishes
1.44 accepted drafts per MTP3 step, so its output yield is `1 + 1.44 = 2.44`
tokens/step. Raw acceptance omits the mandatory target token.

Huawei references are attached only to matching rows:

| TRT conc | Huawei GBS | Step TPOT ms | Step tput/chip |
|---:|---:|---:|---:|
| 8 | 16 | 17.64 | 56.70 |
| 32 | 64 | 19.03 | 210.16 |
| 64 | 128 | 20.61 | 388.23 |

## Tuning

The recommended optimization path is
`utils/bench_offline/b300_huawei_experiments.json`. Each entry runs one fresh
engine with one full-shape warmup and one measured pass. This avoids the
old seven-engine serial runtime and lets independent B300 nodes evaluate up to
ten configurations concurrently.

The checked-in first matrix remains DEP8/MTP3 and covers only c8, c32, and c64,
the batch-per-chip matches for Huawei. It tests:

- controls reproducing the prior selected scheduler settings
- LM-head TP in attention DP
- wait 0 versus wait 30
- attention-DP balance on versus off
- the production attention-DP knobs projected onto DEP8, including omitted
  `timeout_iters`; this is not the normal recipe's TP4 parallelism

After validating a DEP8 candidate, use separate experiment IDs for TP4/DEP4
recipe tests. Their result rows record active GPU count and parallelism, and
their Huawei ratios remain null by construction.

The checked-in second-stage matrix is
`utils/bench_offline/b300_stage2_experiments.json`. It repeats the c32 DEP8
control and LM-head-TP candidate on fresh engines, then tests TP4 at c8/c32
and DEP4 scheduler variants at c64.

The checked-in third-stage matrix is
`utils/bench_offline/b300_stage3_local_batch_experiments.json`. TRT documents
ADP `max_batch_size` as a per-local-rank capacity. Legacy controls retain the
old global sizing, while `attention_dp_batch_mode=local-rank` uses
`ceil(global_concurrency / attention_dp_ranks)` for engine and CUDA graph
capacity. At c8/c32/c64 on DEP8, that captures graph batches 1/4/8 instead of
8/32/64 and avoids padding each local decode batch to global concurrency.

The checked-in fourth-stage matrix is
`utils/bench_offline/b300_stage4_kernel_experiments.json`. It keeps matched
one-pass controls while testing:

- TRT's CuTE DSL FP4 paged-MQA-logits kernel at c8/c32/c64
- a block-aligned `max_seq_len=8832` control at c64
- native TRT iteration logging disabled at c64 while harness progress remains
- DEP4 local-rank batch/graph sizing at c64

The CuTE DSL override supplies only
`algorithm=deepseek_v4` and `use_cute_dsl_paged_mqa_logits=true`. Pinned TRT
then reconstructs checkpoint-derived compression ratios, window size,
indexer dimensions, and top-k. The worker validates that the flag,
`max_seq_len`, and `print_iter_log` resolve exactly as requested. A DSL row
also fails before engine initialization unless the image reports
`IS_CUTLASS_DSL_AVAILABLE=true`; TRT's silent fallback is not accepted.

The legacy concurrency-only mode remains available for reproducing the older
serial tuner:

Every tuning attempt creates and destroys a fresh TRT engine, performs one
full-shape warmup, and records one measured pass. A later candidate replaces
the current winner only when it is at least 3% faster in derived output-token
throughput per GPU. The wider threshold prevents one sampled MTP trajectory
from selecting a scheduler setting on a marginal difference.

The intended six-attempt order is:

1. wait 30, attention-DP balance on, overlap on
2. wait 0
3. wait 10
4. wait 60
5. best scheduler candidate with balance off
6. best prior candidate with overlap off

CUDA graphs capture either the legacy global concurrency or the configured
local-rank ADP batch size, with padding. If graph initialization or execution
fails, the same candidate is immediately retried with graphs disabled. That
retry consumes an attempt, all later candidates remain graph-off, and the
lowest-priority candidate may be omitted.

An engine-init or full-shape-warmup OOM/capacity failure stops pointless
scheduler retries. Concurrencies 512 and 1024 emit `capacity_failure` result
rows instead of disappearing from the collection.

A worker timeout is terminal for that concurrency because orphaned MPI or CUDA
state cannot be assumed reusable for another fresh-engine attempt.

## Runtime Logging

The live Actions log and `offline_controller_EXPERIMENT_ID.log` show:

- launcher and benchmark start details
- corpus preparation start/completion
- each tuning and final worker start
- a 60-second heartbeat while a worker remains active
- the latest explicit worker phase seen at each heartbeat
- candidate completion/failure and headline metrics
- final benchmark status

Detailed TRT output remains in each `*_worker.log` inside the debug archive.
Those logs include engine initialization, warmup, every measured pass,
perfect-router validation, aggregation, shutdown, and failure tracebacks.
Progress logging does not alter measured intervals or benchmark settings.

## Dispatch

Push `trt-bench`, then dispatch the ten single-candidate experiments:

```bash
BENCH_REF="$(git rev-parse HEAD)"
EXPERIMENTS="$(jq -c . utils/bench_offline/b300_huawei_experiments.json)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT offline optimization' \
  -f "inputs[experiments]=$EXPERIMENTS" \
  -f 'inputs[salloc-time]=90' \
  -f 'inputs[worker-timeout]=3600'
```

The top-level `ref=trt-bench` selects this branch's workflow implementation.
`inputs[ref]=$BENCH_REF` pins every matrix checkout to the same commit. The
launcher records the actual checked-out `git rev-parse HEAD` in result
provenance.

Find and monitor the run:

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml --event workflow_dispatch --limit 1 \
  --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
```

The workflow enforces a maximum of ten parallel jobs. Every experiment still
uses direct in-process `LLM.generate()`; no serving process or HTTP client is
introduced.

Dispatch the fourth-stage matrix after the third-stage jobs release the ten
B300 runner slots:

```bash
BENCH_REF="$(git rev-parse HEAD)"
EXPERIMENTS="$(
  jq -c . utils/bench_offline/b300_stage4_kernel_experiments.json
)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT offline kernel optimization' \
  -f "inputs[experiments]=$EXPERIMENTS" \
  -f 'inputs[salloc-time]=90' \
  -f 'inputs[worker-timeout]=3600'
```

## Artifacts And Collected Values

Each matrix job uploads `offline-trt-job-EXPERIMENT_ID`:

- `offline_result_EXPERIMENT_ID.json`: authoritative result row
- `offline_controller_EXPERIMENT_ID.log`: controller and tokenizer output
- `offline_gpu_metrics_EXPERIMENT_ID.csv`: one-second GPU telemetry
- `offline_debug_EXPERIMENT_ID.tar.gz`: candidate configs, worker JSON,
  worker logs, corpus manifest, and perfect-router markers

The collector uploads `offline-trt-summary`:

- `offline_aggregate.json`: all discovered full results plus concise `rows`
- `offline_summary.md`: the table shown in the Actions step summary

The collected row fields mean:

- `mean_token_tpot_ms`: mean per-request output-token TPOT
- `mean_step_tpot_ms`: mean per-request TRT decode-step TPOT
- `derived_output_tput_per_gpu`: concurrency/TPOT calculation
- `derived_step_tput_per_gpu`: concurrency/step-TPOT calculation
- `wall_output_tput_per_gpu`: measured whole-batch output throughput
- `observed_tokens_per_step`: TRT token output per decode iteration
- `effective_acceptance_rate`: `(observed_tokens_per_step - 1) / 3`
- `acceptance_rate`: accepted/proposed TRT counters, or null when unavailable
- `mean_ttft_ms`, `p99_ttft_ms`: first-token latency
- `huawei_published_dataset_token_tput_per_chip`: Huawei reference at its
  published 2.44 tokens/step
- `b300_to_huawei_published_output_ratio`: emitted-output comparison using
  Huawei's own published 2.44-token yield
- `b300_to_huawei_step_rate_ratio`: direct decode-step-rate comparison
- `b300_to_huawei_trt_yield_normalized_ratio`: same-yield diagnostic

This is not `agg_bmk.json`; generic InferenceX result fields such as
`tput_per_gpu` are intentionally not synthesized.

Download a finished summary:

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  -n offline-trt-summary -D ./offline-summary
jq '.rows' ./offline-summary/offline_aggregate.json
```

## Debugging Order

1. Inspect the failed matrix job's `Show result headline` output.
2. Read `offline_controller_EXPERIMENT_ID.log`.
3. Extract `offline_debug_EXPERIMENT_ID.tar.gz`.
4. Find the last `*_worker.json`; its `phase`, `failure_kind`, `error`, and
   traceback identify whether failure occurred in engine init, warmup,
   generation, metrics, or shutdown.
5. Match it to the adjacent worker log and candidate JSON.
6. Confirm the perfect-router marker contains one
   `source=trt_mpi_entry` PID per active GPU.
7. Check `gpu_metrics` and Slurm logs for node-level OOM or GPU faults.

Known interpretations:

- `cuda_graph`: controller retries the same candidate graph-off.
- `oom` or `capacity` during init/warmup at 512/1024: expected row-level
  capacity result.
- fewer `trt_mpi_entry` perfect-router PIDs than active GPUs: MPI entry patch
  or environment propagation failed.
- resolved parallelism mismatch: TRT changed or ignored an argument; do not
  accept the numbers.
- output token count other than 625: `ignore_eos` or sampling behavior changed;
  do not accept the numbers.
- prompt count/length/checksum mismatch: rebuild or tokenizer behavior changed;
  do not accept the numbers.

See `/TRT_BENCH_NOTES.md` for pinned-source facts and first-run risks.
