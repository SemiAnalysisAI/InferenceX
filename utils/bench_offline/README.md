# DeepSeek-V4 TensorRT-LLM Offline Benchmark

This branch contains a self-contained DeepSeek-V4 TensorRT-LLM benchmark.
It replaces `.github/workflows/e2e-tests.yml` on `trt-bench`, does not use
the normal InferenceX serving matrix, and is not intended to merge into
`main`.

Read these files in this order:

1. This README: runnable specification and operating commands.
2. `AGENTS.md`: invariants that must not change while debugging.
3. [`TRT_BENCH_NOTES.md`](../../TRT_BENCH_NOTES.md): commit history,
   completed-run ledger, exact flat renderer rows, and failure history.

Never edit `.github/configs/nvidia-master.yaml` or `perf-changelog.yaml` for
this benchmark.

## Benchmark Contracts

The branch supports three intentionally separate contracts:

| Contract | Profiles | Hardware | Purpose |
|---|---|---|---|
| Huawei fixed GBS | `huawei` | B300 x8 or GB300 x16 | Reproduce Huawei's fixed-batch offline method |
| PR maximum | `pr-tp32-mtp3`, `pr-tp16-mtp3`, `pr-tp8-mtp1` | GB300 x32/x16/x8 | Copy PR #1689 attempt 14 decode recipes and find saturated offline throughput |
| Rack | `rack-tp8x9-mtp1` | GB300 x72 | Run nine synchronized copies of the fastest TP8 recipe |

Do not compare timing fields without checking both
`benchmark.benchmark_profile` and `aggregate.timing_source`.

All contracts use:

- real DeepSeek-V4 chat-formatted prompts of exactly 8192 token IDs
- one fixed request population for the selected measurement
- 256 validated full-batch decode rounds
- temperature 1, top-p 1, top-k 0, and EOS ignored
- engine-global seed 42
- MTP accepted-token yield measured from the same selected rounds
- only upper-IQR latency outliers removed

## What Fixed GBS Means

`global_batch_size` is the number of requests that must be decoding together.
It is not HTTP concurrency and it is not a client-side request limit.

For a single attention-DP engine:

```text
local_batch_size = global_batch_size / active_gpu_count
```

For the 72-GPU rack:

```text
engine_global_batch_size = rack_global_batch_size / 9
local_batch_size = rack_global_batch_size / 72
```

The fixed population is enforced as follows:

1. The workflow accepts only profile-supported GBS values.
2. The MPI shim installs a request-queue gate while TRT initializes.
3. The gate stays disarmed during TRT's internal calibration requests.
4. After `LLM(...)` returns, rank 0 atomically creates the arm file.
5. Each real `LLM.generate()` submission is held until the exact complete
   batch is present.
6. The result selector requires 256 consecutive iterations where context is
   zero and active, scheduled, and generation counts all equal the exact
   local batch, with no queued or paused requests.

Huawei mode additionally requires one context-only full-batch prefill
iteration immediately before decode. PR and rack modes permit staged setup
because a direct offline engine must ingest its own 8K prompts, but the
selected decode window remains exact.

This is why the old serving-concurrency-derived result is not an equivalent
comparison. A configured concurrency can include requests in prefill, queues,
or staggered decode. Multiplying per-request TPOT by that configured value
overstates simultaneous decode throughput.

## End-To-End Execution

The benchmark runs through the branch-local
`.github/workflows/e2e-tests.yml`:

1. `prepare` validates `hardware-profile`, `benchmark-profile`, and GBS, then
   emits a sequential matrix.
2. `benchmark` checks out the exact `inputs.ref` into
   `${GITHUB_WORKSPACE}/offline-bench`.
3. The job invokes one of:
   - `benchmarks/single_node/offline/dsv4_fp4_b300_trt.sh`
   - `benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh`
   - `benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_trt.sh`
4. The launcher allocates Slurm resources, proves the rank map and NVLink
   fabric, starts a fresh container, and launches TensorRT-LLM through
   external MPI.
5. `utils/bench_offline/run.py` builds the exact prompt corpus and controls
   the worker.
6. `utils/bench_offline/trt_mpi_entry.py` installs the fixed-batch gate and
   narrowly scoped runtime workarounds.
7. `utils/bench_offline/trt_worker.py` constructs the engine, arms the gate,
   runs warmup when required, executes one measured pass, validates the
   schedule, and writes the result.
8. Rack runs repeat the engine path on nine disjoint two-node pairs, release
   all measured passes through one barrier, and combine them with
   `utils/bench_offline/aggregate_rack.py`.
9. Each matrix job uploads its result and debug evidence.
10. `collect-results` downloads every `offline-trt-job-*` artifact and runs
    `utils/bench_offline/summarize.py`.

There is no HTTP server, load generator, Dynamo frontend, NATS, etcd,
srt-slurm controller, or normal InferenceX matrix generator in this path.

## Huawei Method

The reference implementation is:

```text
/Users/bshan/Documents/cann-recipes-infer-master/
  docs/models/deepseek-v4/deepseek_v4_inference_guide.md
  models/deepseek-v4/models/model_infer.py
  executor/utils/common_utils.py
```

The copied measurement contract is:

- GBS 16, 64, and 128
- full-batch prefill before decode
- two request warmup decode rounds
- MTP3
- overlap disabled
- 256 measured decode rounds
- first measured round skipped
- values above `Q3 + 1.5 * IQR` removed
- raw round throughput calculated as `GBS / TPOT / devices`
- accepted-token yield reported separately

Huawei publishes:

| GBS | Chips | Decode-round TPOT ms | Decode steps/s/chip |
|---:|---:|---:|---:|
| 16 | 16 | 17.64 | 56.70 |
| 64 | 16 | 19.03 | 210.16 |
| 128 | 16 | 20.61 | 388.23 |

Huawei also reports 1.44 accepted drafts per MTP3 round, or 2.44 output
tokens per round. Hardware and precision are not identical: Huawei uses 16
950DT chips with hybrid MXFP8/MXFP4; this branch uses B300 or GB300 FP4.

## PR #1689 Recipe

The GB300 image and topology come from:

- [InferenceX PR #1689](https://github.com/SemiAnalysisAI/InferenceX/pull/1689)
- [Actions run 27164980476 attempt 14](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27164980476/attempts/14)
- image
  `nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1`
- TensorRT-LLM source
  `34a563ac6d8cc0ca7068c7f619e869fb8a625333`

PR maximum profiles:

| Profile | Decode topology | Fixed GBS | Local/rank | MTP | Max batch | KV |
|---|---|---:|---:|---:|---:|---:|
| `pr-tp32-mtp3` | TP32/EP32 | 192, 256 | 6, 8 | 3 | 8 | 0.70 |
| `pr-tp16-mtp3` | TP16/EP16 | 400, 512 | 25, 32 | 3 | 32 | 0.70 |
| `pr-tp8-mtp1` | TP8/EP8 | 3440, 4096 | 430, 512 | 1 | 512 | 0.80 |

The first point approximates attempt 14's active decode population; the
second fills the copied engine. These are fixed offline populations, not the
PR's HTTP concurrency.

The copied engines keep their complete CUDA graph lists, overlap scheduler,
learned router, EPLB maps, PDL, LM-head TP, MegaMoE backend, and
low-precision combine settings. The resolved recipes leave
`attention_dp_config` unset and do not use perfect routing.

Serving decode workers receive transferred KV and use small decode-only
`max_num_tokens` values. The direct offline engines must prefill locally, so
the PR and rack profiles use `max_num_tokens=32768`. Every result records
this adaptation.

## Rack Topology

`rack-tp8x9-mtp1` uses the complete 18-node, 72-GPU GB300 allocation:

```text
18 nodes x 4 GPUs = 72 GPUs
2 adjacent nodes x 4 GPUs = 8 ranks per engine
9 disjoint node pairs = 9 TP8/EP8 MTP1 engines
```

The result is `9xDEP8`, not TP72. Every GPU is active, but each TensorRT/NCCL
world remains inside its assigned two-node pair.

Before any engine starts, the parent requires:

- ranks exactly `0..71`
- four local ranks on each of exactly 18 hosts
- Fabric `Completed` and `Success` on every GPU
- one shared non-empty `ClusterUUID` and `CliqueId`

Rack batches:

| Rack GBS | Engine GBS | Local/GPU | Purpose |
|---:|---:|---:|---|
| 72 | 8 | 1 | Match Huawei GBS16 local batch |
| 288 | 32 | 4 | Match Huawei GBS64 local batch |
| 576 | 64 | 8 | Match Huawei GBS128 local batch |
| 30960 | 3440 | 430 | 9x attempt-14 active population |
| 36864 | 4096 | 512 | 9x copied engine capacity |

Model loading is admitted one engine at a time. A child has 600 seconds to
finish loading, up to three attempts, and a 15-second retry delay. Already
initialized engines stay alive. Once all nine children are ready, replica 0
publishes a common start timestamp 90 seconds in the future. This absorbs
shared-filesystem visibility delay; aggregation rejects measured start skew
above 10 seconds.

For logical rack round `i`:

```text
rack_round_latency[i] =
    max(replica_0_host_step[i], ..., replica_8_host_step[i])
```

The rack result skips eight logical startup rounds, applies the upper-IQR
filter, and computes throughput from that slowest-replica round series.

## Timing And Throughput

Huawei mode uses non-overlap TRT `iterLatencyMS`. PR and rack modes use
rank-0 `print_iter_log` `host_step_time`; overlap-mode `iterLatencyMS` spans
the wrong scope and remains diagnostic only.

Headline formulas:

```text
decode_step_tput_per_gpu =
    global_batch_size / decode_round_tpot_seconds / active_gpu_count

observed_tokens_per_step =
    1 + accepted_drafts_per_step

output_tput_per_gpu =
    decode_step_tput_per_gpu * observed_tokens_per_step

equivalent_output_tpot_ms =
    decode_round_tpot_ms / observed_tokens_per_step
```

`wall_output_tput_per_gpu` covers the complete `LLM.generate()` call,
including setup and the deliberately long output cap. It is diagnostic and
must not replace the validated decode-window result.

### Huawei Comparison

Use the raw step ratio for the hardware/scheduler comparison:

```text
hardware_to_huawei_decode_step_ratio =
    GB300 decode steps/s/GPU / Huawei decode steps/s/chip
```

To express the same comparison in output tokens without giving either stack
credit for a different MTP depth or acceptance rate:

```text
huawei_output_at_measured_yield =
    Huawei decode steps/s/chip * GB300 observed_tokens_per_step

hardware_to_huawei_same_yield_output_ratio =
    GB300 output tok/s/GPU / huawei_output_at_measured_yield
```

The same-yield output ratio is mathematically equal to the raw step ratio.
The separate `hardware_to_huawei_output_ratio` uses each stack's own measured
or published MTP yield and answers a different question.

The aggregate row stores the two equal-yield values as
`huawei_output_tput_per_chip_at_measured_tokens_per_step` and
`hardware_to_huawei_same_yield_output_ratio`.

## Result Files

Each benchmark job uploads `offline-trt-job-EXPERIMENT_ID` with:

- `offline_result_EXPERIMENT_ID.json`
- `offline_controller_EXPERIMENT_ID.log`
- GPU telemetry
- GB300 allocation, rank-map, topology, world, timing, and completion files
- `offline_debug_EXPERIMENT_ID.tar.gz`

For rack runs, the debug archive contains all nine child trees under
`.offline_rack_*/replicas/r00` through `r08`. Child result JSON stays hidden
there so collection sees exactly one 72-GPU row.

`collect-results` produces two artifacts:

| Artifact | File | Meaning |
|---|---|---|
| `offline-trt-summary` | `offline_aggregate.json` | Authoritative result rows plus missing/failure status |
| `offline-trt-summary` | `offline_summary.md` | Human-readable table and metric definitions |
| `offline-trt-summary` | `agg_bmk.json` | Flat renderer-compatible rows |
| `results_bmk` | `agg_bmk.json` | Same flat rows under InferenceMAX's expected artifact name |

Important renderer fields:

| Field | Offline meaning |
|---|---|
| `conc` | Compatibility alias for fixed `global_batch_size`; not serving concurrency |
| `mean_tpot` | Equivalent output-token TPOT, not raw decode-round latency |
| `tput_per_gpu` | Acceptance-adjusted output-token throughput |
| `output_tput_per_gpu` | Same acceptance-adjusted output-token throughput |
| `decode_round_tpot_ms` | Raw validated full-batch round latency |
| `decode_step_tput_per_gpu` | Raw round/request-step throughput |
| `observed_tokens_per_step` | Measured MTP output yield |
| `measured_decode_rounds` | Required validated round count, normally 256 |
| `timing_source` | Instrumentation used for the raw round |

`agg_bmk.json` must be a top-level JSON array of flat objects. A wrapper such
as `{"rows": [...]}` is not renderable by the unofficial-run endpoint.

## Known-Good Results

The complete result ledger and exact flat rows are in
[`TRT_BENCH_NOTES.md`](../../TRT_BENCH_NOTES.md).

| Run | Hardware/profile | Result |
|---:|---|---|
| [27493336994](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27493336994) | B300 x8, Huawei GBS16/64/128 | Valid fixed-GBS baseline |
| [27517035480](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27517035480) | GB300 x16, Huawei GBS16/64/128 | Valid NVL16 baseline |
| [27545752641](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27545752641) | GB300 x72, rack GBS30960/36864 | Valid rack maximum |
| [27555308252](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27555308252) | GB300 x72, rack GBS72/288/576 | Valid Huawei-local-batch comparison |

The best validated branch result is rack GBS36864:

```text
decode_round_tpot_ms: 90.548343
decode_step_tput_per_gpu: 5654.438070
observed_tokens_per_step: 1.814993
output_tput_per_gpu: 10262.766175
```

It is 5.946593% above attempt 14's TP8
`9686.735465` output tok/s/decode-GPU under the same decode-GPU denominator.

The final Huawei-local-batch rack comparison is:

| Rack GBS | Local/GPU | Steps/s/GPU | Tok/step | Output tok/s/GPU | Huawei at GB300 yield | Same-yield ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 72 | 1 | 53.504297 | 1.780816 | 95.281306 | 100.972266 | 0.943638 |
| 288 | 4 | 172.497882 | 1.795247 | 309.676373 | 377.289193 | 0.820793 |
| 576 | 8 | 298.775209 | 1.797852 | 537.153476 | 697.979912 | 0.769583 |

## Dispatch

Always pin `inputs[ref]` to the commit being tested. The workflow itself is
dispatched from `trt-bench` because this branch replaces the normal e2e
workflow.

Huawei-local-batch rack sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=rack-huawei-sweep' \
  -f 'inputs[test-name]=DSV4 GB300 NVL72 TRT Huawei-local-batch sweep' \
  -f 'inputs[global_batch_sizes]=auto' \
  -f 'inputs[salloc-time]=360' \
  -f 'inputs[worker-timeout]=18000' \
  -f 'inputs[worker-stack-period]=-1'
```

Rack maximum sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=rack-max-sweep' \
  -f 'inputs[test-name]=DSV4 GB300 NVL72 TRT max offline' \
  -f 'inputs[global_batch_sizes]=auto' \
  -f 'inputs[salloc-time]=360' \
  -f 'inputs[worker-timeout]=18000' \
  -f 'inputs[worker-stack-period]=-1'
```

Single rack point:

```bash
# Use 72, 288, or 576 with rack-huawei-sweep.
# Use 30960 or 36864 with rack-max-sweep.
-f 'inputs[global_batch_sizes]=72'
```

Single-engine Huawei sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=huawei' \
  -f 'inputs[test-name]=DSV4 GB300 TRT Huawei fixed GBS' \
  -f 'inputs[global_batch_sizes]=16,64,128' \
  -f 'inputs[salloc-time]=180' \
  -f 'inputs[worker-timeout]=7200' \
  -f 'inputs[worker-stack-period]=-1'
```

PR recipe sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=pr-max-sweep' \
  -f 'inputs[test-name]=DSV4 GB300 TRT PR max offline' \
  -f 'inputs[global_batch_sizes]=auto' \
  -f 'inputs[salloc-time]=300' \
  -f 'inputs[worker-timeout]=18000' \
  -f 'inputs[worker-stack-period]=-1'
```

## Monitor And Collect

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml --event workflow_dispatch --limit 1 \
  --json databaseId --jq '.[0].databaseId')

gh run watch "$RUN_ID" \
  --repo SemiAnalysisAI/InferenceX \
  --exit-status

gh api \
  "/repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN_ID/artifacts" \
  --jq '.artifacts[].name'

gh run download "$RUN_ID" \
  --repo SemiAnalysisAI/InferenceX \
  -n offline-trt-summary \
  -D ./offline-summary
```

Inspect concise rows:

```bash
jq -r '
  .rows[] |
  [
    .experiment_id,
    .status,
    .global_batch_size,
    .local_batch_size,
    .decode_round_tpot_ms,
    .decode_step_tput_per_gpu,
    .observed_tokens_per_step,
    .output_tput_per_gpu
  ] | @tsv
' ./offline-summary/offline_aggregate.json

jq -r '
  .[] |
  [
    .conc,
    .decode_round_tpot_ms,
    .decode_step_tput_per_gpu,
    .observed_tokens_per_step,
    .output_tput_per_gpu
  ] | @tsv
' ./offline-summary/agg_bmk.json
```

Renderer:

```text
https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=RUN_ID
```

## Debugging Order

1. Read the job's `Show result headline`.
2. Download `offline-trt-job-EXPERIMENT_ID`.
3. Inspect result `status`, `phase`, `failure_kind`, and `error`.
4. Read `offline_controller_EXPERIMENT_ID.log`.
5. On GB300, validate rank map and topology before debugging TRT.
6. For PR/rack timing, inspect `offline_timing_EXPERIMENT_ID.log`; the final
   occurrence of each selected iteration ID is authoritative.
7. Extract the debug archive and inspect `worker.log`,
   `worker_result.json`, and measured iteration stats.
8. For rack runs, inspect all `r00` through `r08` child launchers, result
   files, timing logs, and barrier records.
9. Require matching successful result and completion records before treating
   cleanup-time MPI or Slurm warnings as harmless.

For initialization stalls, use controller heartbeat rank progress. Preserve
60-second progress logs. A TensorRT fatal line, `entry_failed`, or rank
`*_error` marker is terminal even if the MPI parent remains alive.

Do not weaken the batch gate, schedule proof, fabric proof, timing source, or
MTP counter requirement to make a failing run pass.

## Local Verification

```bash
python -m pytest utils/bench_offline/ -q
python -m compileall -q utils/bench_offline
bash -n benchmarks/single_node/offline/dsv4_fp4_b300_trt.sh
bash -n benchmarks/single_node/offline/run_dsv4_trt_container.sh
bash -n benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh
bash -n benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_trt.sh
bash -n benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_replica.sh
shellcheck benchmarks/single_node/offline/dsv4_fp4_b300_trt.sh
shellcheck benchmarks/single_node/offline/run_dsv4_trt_container.sh
shellcheck benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh
shellcheck benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_trt.sh
shellcheck benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_replica.sh
actionlint .github/workflows/e2e-tests.yml
git diff --check
```
