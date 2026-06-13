# DeepSeek-V4 B300 TensorRT-LLM Offline Benchmark

This is a disposable `trt-bench` branch experiment. It is not designed to
merge into `main`, populate inferencex.com, or share the normal serving sweep.

## Execution Chain

The benchmark is dispatched through the existing
`.github/workflows/e2e-tests.yml` path because GitHub only permits
`workflow_dispatch` for workflow files that also exist on the default branch.
On `trt-bench`, that workflow is replaced with this offline-only chain:

1. Validate the requested concurrency list.
2. Fan out one `b300` runner job for each concurrency.
3. Allocate one exclusive B300 Slurm node with eight GPUs.
4. Run the pinned TRT image with `/scratch/models/DeepSeek-V4-Pro`.
5. Build an exact 8192-token InfiniteBench corpus.
6. Tune at most six fresh TRT engines.
7. Start another fresh engine with the winner, warm it up, and measure three
   passes.
8. Upload one result and debug bundle per concurrency.
9. Collect the rows with `utils/bench_offline/summarize.py`.

There is no server, HTTP client, request-rate generator, matrix config,
`process_result.py`, `summarize.py`, or generic `collect-results` stage in this
path. `LLM.generate()` receives one fixed batch of token-ID prompts directly.

## Pinned Inputs

- Hardware: one B300 node, eight GPUs.
- Image:
  `ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066`
- TRT source identity: `c185066`.
- Model path: `/scratch/models/DeepSeek-V4-Pro`.
- Dataset: `xinrongzhang2022/InfiniteBench`,
  `longbook_qa_eng.jsonl`.
- Dataset revision:
  `90f0394333616266d9fe85824ceaf505093cbaa5`.
- Concurrencies: `8,32,64,128,256,512,1024`.
- Prompt length: exactly 8192 real token IDs, with no padded IDs.
- Generated output: exactly 625 tokens per request with EOS ignored.
- Speculation: fixed MTP3.
- Sampling seed: `20260613 + request_index`.

The prompt retains the CANN recipe's literal instruction to summarize the
story in `256 words`. That prompt text is independent of the 625-token
generation budget.

## Parallelism

TRT uses `tensor_parallel_size=8` to create its eight-rank world, but the
effective model execution is labeled `DEP8`:

- attention data parallel: enabled
- expert parallel size: 8
- MoE tensor parallel size: 1
- LM-head TP in attention DP: disabled
- MTP max draft length: 3

The worker validates these resolved TRT arguments after every engine starts.

Perfect routing is enabled with `ENABLE_PERFECT_ROUTER=1`. TRT's MPI pool only
explicitly forwards `TRTLLM_*` and `TLLM_*` variables, so
`TRTLLM_ENABLE_PERFECT_ROUTER=1` is also set. Before creating `LLM`, the worker
replaces TRT's pinned `GenerationExecutorProxy.worker_main` reference with
`trt_mpi_entry.worker_main`. That shim recreates the unprefixed variable and
writes a marker before each rank imports TRT's real model worker. The benchmark
requires eight marked rank entrypoints. `sitecustomize.py` provides an
additional early-process alias but is not the primary proof.

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
4. Fails instead of padding if no exact length can be constructed.
5. Writes prompts to a little-endian `uint32` binary corpus.

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
`token_tpot` values pooled across the three final measured passes.

Each request also records a SHA-256 digest of its 625 generated token IDs.
Per-pass sequence digests make fresh-engine tuning nondeterminism visible
without uploading generated text or full token arrays.

`derived_output_tput_per_gpu` is:

```text
concurrency / mean_token_tpot_seconds / 8
```

`wall_output_tput_per_gpu` is:

```text
all generated output tokens / total measured batch wall time / 8
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

This benchmark times emitted decode tokens, not CANN decode steps. Therefore,
the Huawei step-throughput references are converted with:

```text
Huawei published step throughput/chip * TRT observed tokens/step
```

Do not multiply Huawei throughput by raw acceptance rate. Acceptance omits the
mandatory target token; `observed_tokens_per_step` includes it.

The Huawei guide also publishes 1.44 accepted drafts per MTP3 step, or 2.44
tokens/step. Results retain both the corresponding published-dataset token
throughput and the Huawei value normalized to TRT's observed token yield.

Huawei references are attached only to matching rows:

| TRT conc | Huawei GBS | Step TPOT ms | Step tput/chip |
|---:|---:|---:|---:|
| 8 | 16 | 17.64 | 56.70 |
| 32 | 64 | 19.03 | 210.16 |
| 64 | 128 | 20.61 | 388.23 |

## Tuning

Every tuning attempt creates and destroys a fresh TRT engine. A later
candidate replaces the current winner only when it is at least 1% faster in
derived output-token throughput per GPU.

The intended six-attempt order is:

1. wait 30, attention-DP balance on, overlap on
2. wait 0
3. wait 10
4. wait 60
5. best scheduler candidate with balance off
6. best prior candidate with overlap off

CUDA graphs capture the exact requested concurrency with padding. If graph
initialization or execution fails, the same candidate is immediately retried
with graphs disabled. That retry consumes an attempt, all later candidates
remain graph-off, and the lowest-priority candidate may be omitted.

An engine-init or full-shape-warmup OOM/capacity failure stops pointless
scheduler retries. Concurrencies 512 and 1024 emit `capacity_failure` result
rows instead of disappearing from the collection.

A worker timeout is terminal for that concurrency because orphaned MPI or CUDA
state cannot be assumed reusable for another fresh-engine attempt.

## Dispatch

Push `trt-bench`, then dispatch the branch version of the existing workflow:

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f 'inputs[ref]=trt-bench' \
  -f 'inputs[test-name]=DSV4 B300 TRT offline' \
  -f 'inputs[concurrencies]=8,32,64,128,256,512,1024' \
  -f 'inputs[salloc-time]=500' \
  -f 'inputs[worker-timeout]=3600'
```

The top-level `ref=trt-bench` selects this branch's workflow implementation.
`inputs[ref]=trt-bench` selects the code checkout used by every job.

Find and monitor the run:

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml --event workflow_dispatch --limit 1 \
  --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
```

For a first bring-up, dispatch only `8`. After it succeeds, dispatch the full
matrix.

## Artifacts And Collected Values

Each matrix job uploads `offline-trt-conc-N`:

- `offline_result_concN.json`: authoritative result row
- `offline_controller_concN.log`: controller and tokenizer output
- `offline_gpu_metrics_concN.csv`: one-second GPU telemetry
- `offline_debug_concN.tar.gz`: candidate configs, worker JSON, worker logs,
  corpus manifest, and perfect-router markers

The collector uploads `offline-trt-summary`:

- `offline_aggregate.json`: all discovered full results plus concise `rows`
- `offline_summary.md`: the table shown in the Actions step summary

The collected row fields mean:

- `mean_token_tpot_ms`: mean per-request output-token TPOT
- `derived_output_tput_per_gpu`: concurrency/TPOT calculation
- `wall_output_tput_per_gpu`: measured whole-batch output throughput
- `observed_tokens_per_step`: TRT token output per decode iteration
- `effective_acceptance_rate`: `(observed_tokens_per_step - 1) / 3`
- `acceptance_rate`: accepted/proposed TRT counters, or null when unavailable
- `mean_ttft_ms`, `p99_ttft_ms`: first-token latency
- `huawei_published_dataset_token_tput_per_chip`: Huawei reference at its
  published 2.44 tokens/step
- `huawei_estimated_token_tput_per_chip`: Huawei reference normalized to TRT's
  observed tokens/step
- `b300_to_huawei_ratio`: B300 derived throughput divided by converted Huawei

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
2. Read `offline_controller_concN.log`.
3. Extract `offline_debug_concN.tar.gz`.
4. Find the last `*_worker.json`; its `phase`, `failure_kind`, `error`, and
   traceback identify whether failure occurred in engine init, warmup,
   generation, metrics, or shutdown.
5. Match it to the adjacent worker log and candidate JSON.
6. Confirm the perfect-router marker contains eight
   `source=trt_mpi_entry` PIDs.
7. Check `gpu_metrics` and Slurm logs for node-level OOM or GPU faults.

Known interpretations:

- `cuda_graph`: controller retries the same candidate graph-off.
- `oom` or `capacity` during init/warmup at 512/1024: expected row-level
  capacity result.
- fewer than eight `trt_mpi_entry` perfect-router PIDs: MPI entry patch or
  environment propagation failed.
- resolved parallelism mismatch: TRT changed or ignored an argument; do not
  accept the numbers.
- output token count other than 625: `ignore_eos` or sampling behavior changed;
  do not accept the numbers.
- prompt count/length/checksum mismatch: rebuild or tokenizer behavior changed;
  do not accept the numbers.

See `/TRT_BENCH_NOTES.md` for pinned-source facts and first-run risks.
