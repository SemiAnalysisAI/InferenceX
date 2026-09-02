# Evaluation and AgentX Procedures

<div align="center">

**English** | [中文](./eval-agentx-procedures_zh.md)

</div>


Use this page to add and run graded evals, operate AgentX trace replays, preserve evidence, and decide whether a long run should continue. Commands assume the repository root and replace values in `<ANGLE_BRACKETS>`.

## 1. Pick the correct execution mode

There are two distinct layers: the matrix generator decides **which jobs exist**, while runtime variables decide **what a launched job does**.

| Need | Generator/workflow mode | Runtime behavior |
|---|---|---|
| Normal sweep | no eval option | Throughput jobs plus the selected 8k/1k eval subset |
| Throughput only | `--no-evals` | No eval jobs |
| Selected eval subset only | `--evals-only` | Jobs have `RUN_EVAL=true`, `EVAL_ONLY=true` |
| Every eligible eval only | `--all-evals` | Equivalent to `--evals-only --all-evals` and includes all fixed-sequence 8k/1k rows plus single-node and multi-node agentic GSM8K rows |
| Throughput then eval in one recipe | `RUN_EVAL=true`, `EVAL_ONLY=false` | Server starts, throughput runs, then `run_eval` runs |
| Eval against a freshly started server | `RUN_EVAL=true`, `EVAL_ONLY=true` | Launcher expands eval context, skips throughput, and runs the eval |

Default selection is scenario-aware. Single-node fixed-sequence evals use the median and highest eligible concurrency for each 8k/1k model/runner/framework/precision/parallelism group. Multi-node evals use the highest eligible concurrency per topology. Concurrency below 16 is not selected. Agentic evals are opt-in; single-node and multi-node agentic rows select their highest eligible concurrency. See [`mark_eval_entries()`](../utils/matrix_logic/generate_sweep_configs.py#L276-L396) and [`mark_all_eval_entries()`](../utils/matrix_logic/generate_sweep_configs.py#L399-L482).

On a PR, combine one primary sweep label (normally `full-sweep-fail-fast`) with eval modifiers. `all-evals` expands coverage without suppressing throughput. `evals-only` suppresses throughput. Together they run all eligible evals only. Runs with `evals-only` are not reusable, while normal full sweeps and `all-evals` full sweeps are reusable. Adding or removing a modifier restarts the active sweep ([label policy](../.github/workflows/README.md#pr-eval-modifiers)).

```bash
# Selected eval subset only
gh pr edit <PR_NUMBER> --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast --add-label evals-only

# Every eligible eval only
gh pr edit <PR_NUMBER> --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast --add-label all-evals --add-label evals-only
```

Preview the exact matrix before consuming a runner:

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py \
  test-config \
  --config-keys qwen3.5-fp8-b200-sglang-agentic \
  --conc 1 \
  --evals-only \
  --config-files configs/nvidia-master.yaml | jq .
```

A correct AgentX eval row contains `"scenario-type": "agentic-coding"`, `"run-eval": true`, and `"eval-only": true`. The workflow splits generated rows into throughput, fixed-sequence eval, and agentic eval jobs in [`.github/workflows/e2e-tests.yml`](../.github/workflows/e2e-tests.yml#L278-L293).

## 2. Add a graded eval

1. Add `utils/evals/<task>.yaml` using the lm-evaluation-harness task format. Pin the dataset/split, deterministic generation settings, prompt contract, filters, and primary metric. Use [`gsm8k.yaml`](../utils/evals/gsm8k.yaml) or [`gpqa_diamond.yaml`](../utils/evals/gpqa_diamond.yaml) as an in-tree pattern.
2. Give `task:` a stable name. That exact name is the key used by score thresholds and appears in collected rows.
3. Add the minimum accepted score to [`utils/evals/thresholds.yaml`](../utils/evals/thresholds.yaml). Put a general floor under `default`. Add `models.<model-prefix>.<task>` only when a justified model-specific floor is required.
4. If the task's primary result is not compatible with the collector's strict/extract/accuracy rules, extend [`extract_lm_metrics()`](../utils/collect_eval_results.py#L115-L197). Do not publish a row whose `score` is null.
5. Run a small explicit slice, inspect samples, then run the full split. `EVAL_LIMIT` is a smoke-test control, not a publishable score setting.

Against an already healthy OpenAI-compatible server:

```bash
source benchmarks/benchmark_lib.sh
export MODEL='<HF_MODEL_ID>'
export MODEL_NAME='<SERVED_MODEL_NAME>'
export MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>'
export EVAL_TASKS_DIR='utils/evals/<task>.yaml'
export EVAL_CONCURRENT_REQUESTS='16'
export EVAL_LIMIT='10'
run_eval --framework lm-eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py \
  --model-prefix "$MODEL_PREFIX" \
  --results-glob 'results*.json'
```

For the full eval, unset the limit and repeat against a clean, correctly configured server:

```bash
unset EVAL_LIMIT
run_eval --framework lm-eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py --model-prefix "$MODEL_PREFIX"
```

`run_lm_eval` passes concurrency through `num_concurrent` in `--model_args`. It is deliberately an environment variable, not a `run_eval` CLI option. The exact invocation is in [`run_lm_eval()`](../benchmarks/benchmark_lib.sh#L1080-L1162).

## 3. `EVAL_ONLY` is a launcher contract

Set `EVAL_ONLY=true` **before server launch**. It is not merely a switch inside `run_eval`:

1. `compute_eval_context_length`/`setup_eval_context` chooses the requested eval context capped by the model's native maximum.
2. The launcher wires it to the server (`--context-length`, `--max-model-len`, or the framework equivalent).
3. The health check still runs.
4. Throughput returns immediately or is skipped.
5. `run_eval` and artifact staging run.

Relevant implementation: [context setup](../benchmarks/benchmark_lib.sh#L1049-L1078), [eval dispatch and failure policy](../benchmarks/benchmark_lib.sh#L1789-L1923), and [workflow inputs](../.github/workflows/benchmark-tmpl.yml#L79-L97).

Do not toggle `EVAL_ONLY` after a throughput-sized server is already running and assume the context changed. Restart through the recipe. In eval-only mode an eval failure is returned after available artifacts are staged. In a workflow, upload happens with `always()` before score validation so failed evidence survives ([single-node upload and gate](../.github/workflows/benchmark-tmpl.yml#L399-L417), [multi-node upload and gate](../.github/workflows/benchmark-multinode-tmpl.yml#L466-L488)).

## 4. Batched eval concurrency

A space-separated `EVAL_CONCURRENT_REQUESTS` value runs several concurrency points **sequentially against one live engine**. It does not run several harnesses simultaneously. Within each point, the harness issues up to that point's concurrency.

```bash
source benchmarks/benchmark_lib.sh
export MODEL='<HF_MODEL_ID>' MODEL_NAME='<SERVED_MODEL_NAME>' MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>' EVAL_TASKS_DIR='utils/evals/gsm8k.yaml'
export EVAL_CONCURRENT_REQUESTS='16 32 64'
run_eval --framework lm-eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py --expected-concs '16 32 64'
```

The batch runner creates a fresh temporary output directory per point, stages files with `_conc<N>` suffixes, and writes these arrays to `meta_env.json`:

- `eval_concs`: requested points.
- `completed_eval_concs`: eval and staging both succeeded.
- `failed_eval_concs`: either eval or staging failed.

A failed point is deferred so artifacts from every attempted point can upload. The post-upload validator then fails the job. Batched mode accepts positive integers and supports only `lm-eval`. See [`run_eval` batching](../benchmarks/benchmark_lib.sh#L1839-L1900), [artifact suffixing](../benchmarks/benchmark_lib.sh#L1163-L1222), and [manifest validation](../utils/evals/validate_scores.py#L72-L171).

For multi-node `all-evals`, the workflow constructs `EVAL_CONC` by joining the topology's concurrency list ([dispatch](../.github/workflows/e2e-tests.yml#L397-L400)). Never compare a point if its `_conc<N>` result or completed-manifest entry is missing.

## 5. Validate scores, not file existence

Run:

```bash
python3 utils/evals/validate_scores.py \
  --thresholds utils/evals/thresholds.yaml \
  --meta-env meta_env.json \
  --results-glob 'results*.json'
```

For a batch, add the independently expected points:

```bash
python3 utils/evals/validate_scores.py \
  --expected-concs '16 32 64' \
  --thresholds utils/evals/thresholds.yaml
```

Validation resolves the threshold in this order: `models.<prefix>.<task>`, `default.<task>`, then `--min-score` (default `0.85`). By default it checks numeric, non-stderr metrics beginning with `exact_match,`. It fails when a score is below threshold, no metric matches, a requested concurrency is absent, metadata has duplicates/invalid values, any point is marked failed, or result suffixes do not match the manifest. Current floors are authoritative in [`thresholds.yaml`](../utils/evals/thresholds.yaml). See [threshold resolution](../utils/evals/validate_scores.py#L61-L69) and the [validation flow](../utils/evals/validate_scores.py#L174-L302).

A manual combined throughput+eval recipe uploads eval output but the template's automatic score gate is specific to eval-only jobs. Run the validator explicitly for manual or combined runs.

## 6. Collect and inspect eval artifacts

The collection workflow downloads `eval_*`, aggregates raw sets with `utils/collect_eval_results.py`, uploads `eval_results_all/agg_eval_all.json`, and writes the table to the step summary ([`collect-evals.yml`](../.github/workflows/collect-evals.yml)).

```bash
RUN_ID='<RUN_ID>'
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --name eval_results_all --dir ./evals
jq -r '.[] | [.hw, .framework, .precision, .tp, .conc, .task,
  (.score * 100 | round | . / 100)] | @tsv' \
  ./evals/agg_eval_all.json | column -t
jq '[.[] | select(.hw == "B200")]' ./evals/agg_eval_all.json
```

Download raw evidence when an aggregate is missing or suspicious:

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'eval_*' --dir ./evals/raw
```

Retain `meta_env.json`, `results*.json`, and `sample*.jsonl`. Agentic SWE-bench additionally uploads `agent_preds.json`, `predictions.jsonl`, `swebench_report_*.json`, and trajectory files in the single-node template. The aggregate is a navigation aid, not a substitute for raw samples and batch completeness.

## 7. Run AgentX: fast feedback versus canonical evidence

AgentX is AIPerf `inferencex-agentx-mvp` trace replay, not a fixed-token synthetic benchmark. The checked-in default uses ten additional warmup requests per trajectory lane and the recipe's configured profile duration. `agentx-fast` forces one warmup request per lane and a 1,200-second profile. It affects single- and multi-node AgentX throughput only. Fixed-sequence throughput and evals remain canonical. Fast runs are not eligible for artifact reuse ([workflow policy](../.github/workflows/README.md#agentx-fast-mode), [fast replay settings](../benchmarks/benchmark_lib.sh#L2104-L2128)).

Keep non-index engine or router wheels reproducible and immutable: check in the source patch and builder beside the launcher, verify the upstream wheel's digest before patching, assign an explicit local version, and install the published artifact through an exact URL with a SHA256 fragment. A local backport must not use an unreleased upstream version number.

Targeted canonical run (configured duration and warmup, with fast and duration overrides omitted):

```bash
REF='<BRANCH_OR_SHA>'
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref "$REF" \
  -f generate-cli-command='test-config --config-keys qwen3.5-fp8-b200-sglang-agentic --conc 1 --config-files configs/nvidia-master.yaml' \
  -f test-name='agentx-canonical-qwen35-c1'
```

Fast diagnostic run:

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref "$REF" \
  -f generate-cli-command='test-config --config-keys qwen3.5-fp8-b200-sglang-agentic --conc 1 --config-files configs/nvidia-master.yaml' \
  -f test-name='agentx-fast-qwen35-c1' \
  -f agentx-fast=true
```

For controlled Kimi-K3 MI355X concurrency-one experiments, the manual
`k3-perf-variant` input selects an isolated runtime arm while the generated
recipe remains unchanged. The `m7tunedgemm` arm adds only the five exact-M=7
BF16 rows that passed changed-input graph replay and strict tuner gates. Pin the
branch SHA, keep `agentx-fast=false`, use a 1,200-second duration override, and
compare the resulting raw request records with the matching baseline before
promoting the rows. The `tritonmla` arm disables only AITER MLA so target-model
decode auto-selection uses `TRITON_MLA`; it keeps the rest of the AITER ROCm
stack enabled and fails closed unless the selected backend is proven in the
server log. For sub-millisecond confirmation, `m7tunedgemmaba` retains one
exclusive Slurm allocation and runs a fresh baseline, the exact-M=7 candidate,
and a second fresh baseline for 1,200 seconds each. The candidate stays in the
standard result layout; both controls and a node/arm manifest are stored under
`results/same_node_aba/`. The structural `spec3aba` mode uses the same
baseline/candidate/baseline protocol but changes only the C1 draft count from
six to three while retaining the synthetic acceptance length of 3.75. It first
performs a baseline/candidate startup smoke, isolates the Triton,
TorchInductor, AITER JIT, and Python caches for every launch, and verifies GPU,
process, and shared-memory cleanup before starting the next server. Because the
Enroot allocation and host telemetry can own KFD before the server starts, the
harness first requires low VRAM and no existing vLLM process, records the KFD
and shared-memory baseline, and requires every arm to restore that exact state.
Baseline arms capture graph sizes 2 through 14; the three-draft candidate
captures 2 through 8. The `metadatareuseaba` mode keeps six drafts and graph
sizes 2 through 14, pins all three arms to GPU-memory utilization 0.85, and
applies the exact vLLM metadata-reuse source overlay only to the candidate.
The candidate source is restored and checksum-verified after both its startup
smoke and full benchmark; restoration failure aborts the A/B/A before a later
baseline can run.

Targeted AgentX SWE-bench smoke eval (first ten instances, real agentic generation):

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref "$REF" \
  -f generate-cli-command='test-config --config-keys qwen3.5-fp8-b200-sglang-agentic --conc 1 --evals-only --config-files configs/nvidia-master.yaml' \
  -f test-name='swebench-smoke-qwen35-c1' \
  -f eval-framework=swebench \
  -f eval-limit='10' \
  -f swebench-gen-mode='agentic'
```

For a publishable SWE-bench score, omit `eval-limit`. Do not use `single-shot`, which is only a debugging escape hatch. SWE-bench generation/scoring controls and its `0.50` full-split threshold are documented next to the implementation in [`utils/evals/EVALS.md`](../utils/evals/EVALS.md#swe-bench-lite---framework-swebench).

Treat fast results as bring-up evidence, never as a replacement for the canonical candidate. A duration below 900 seconds or `AIPERF_UNSAFE_OVERRIDE=true` adds AIPerf's `--unsafe-override` and flags the submission invalid. Use it only for smoke diagnosis ([source](../benchmarks/benchmark_lib.sh#L2266-L2268)). After a fast run is healthy, run the exact candidate canonically before claiming benchmark success.

## 8. Preserve trace and run provenance

AgentX defaults to recorded assistant-response replay. Live server outputs are measured but discarded when constructing later turns. Set `AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1` only for an explicitly different live-assistant experiment. The selected trace corpus is model-family dependent unless `WEKA_LOADER_OVERRIDE` pins it. The resolver logs both loader and Hugging Face dataset ([trace resolution](../benchmarks/benchmark_lib.sh#L2023-L2102), [replay semantics](../benchmarks/benchmark_lib.sh#L2104-L2270)).

Capture orchestration provenance immediately:

```bash
RUN_ID='<RUN_ID>'
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --json url,headSha,headBranch,event,status,conclusion,createdAt,updatedAt,jobs \
  > run-provenance.json
```

Download AgentX evidence:

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'bmk_agentic_*' --dir ./agentx/aggregate
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'agentic_*' --dir ./agentx/raw
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern '*server_logs_*' --dir ./agentx/server-logs
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'gpu_metrics_*' --dir ./agentx/gpu
```

For each concurrency retain:

- `benchmark_command.txt` (the exact AIPerf command) and `benchmark.log`.
- AIPerf `profile_export*`, `server_metrics_export.json`, plots, and distribution analysis.
- aggregate JSON and its `dataset` object (`source_type`, loader, HF dataset/split, entry count).
- server/frontend logs and every metrics endpoint represented.
- run URL/ID, attempt, head SHA, recipe/config identity, image, topology, fast flag, and any override.

The runner writes the command before replay and validates raw results after aggregation ([execution path](../benchmarks/benchmark_lib.sh#L2320-L2360)). Aggregation preserves dataset provenance and hardware/model/topology fields ([aggregate construction](../utils/agentic/aggregation/process_agentic_result.py#L194-L272)). Raw workflow uploads intentionally omit very large `inputs.json` and `profile_export_raw.jsonl`. If those are required for an investigation, preserve them from the live allocation before cleanup ([single-node artifact contract](../.github/workflows/benchmark-tmpl.yml#L349-L358), [multi-node contract](../.github/workflows/benchmark-multinode-tmpl.yml#L455-L464)).

## 9. Debug long AgentX runs from live evidence

GitHub Actions is the orchestration/final-status view. The cluster is the live diagnostic source. Obtain the SSH alias, runner user, and access-controlled paths from the InferenceX Clusters canvas. Never guess or publish private infrastructure coordinates.

Resolve the exact matrix job:

```bash
gh run view <RUN_ID> --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | select(.name | test("agentic|AgentX"; "i")) |
        [.databaseId, .status, .conclusion, .name] | @tsv'
```

On the controller, identify and verify the allocation:

```bash
squeue -u <RUNNER_USER> -o "%.8i %.8T %.10M %.20N %.100j"
scontrol show job -o <SLURM_JOB_ID> | tr " " "\n" | \
  grep -E '^(JobId|JobState|RunTime|TimeLimit|NodeList|WorkDir)='
```

Derive `<LOG_DIR>` from `WorkDir`. srt-slurm normally uses `<WorkDir>/outputs/<SLURM_JOB_ID>/logs/`. Inventory before selecting files:

```bash
find "<LOG_DIR>" -maxdepth 1 -type f -print | sort
```

Always include the custom benchmark log from the beginning, then every topology-relevant backend and frontend/router log:

```bash
ssh <CLUSTER_ALIAS> 'tail -f -n+1 "<LOG_DIR>/benchmark.out"'
tail -F -n+1 <BENCHMARK_LOG> <FRONTEND_LOG> <SERVER_LOGS...>
rg -n -i 'Phase |warmup|profiling|returned=|in_flight=|queue=|kv_usage=|prefix_cache_hit=|tput_|ERROR|Traceback|OOM|NCCL|RCCL|timeout|connection refused' <LOGS...>
```

Topology rules:

- Aggregated: inspect every aggregate backend. Attention DP can expose several metrics sources and is not disaggregation.
- Disaggregated: inspect every prefill backend, every decode backend, and the frontend/router. A healthy decode pool does not prove prefill/KV transfer health.
- Confirm the AIPerf command includes all `AIPERF_SERVER_METRICS_URLS`. Missing endpoints produce falsely healthy partial evidence.

Read each endpoint directly when summaries are ambiguous:

```bash
curl -fsS '<METRICS_URL>' | \
  rg -i 'request|queue|cache|token|prefill|decode|error|fail'
```

Track trends over repeated samples: running/waiting requests, KV usage, prefix hits, input/output token rates, completed/cancelled/errored requests, frontend routing balance, and disaggregated KV transfer. AIPerf records endpoint identity for every server series ([metrics wiring](../benchmarks/benchmark_lib.sh#L2236-L2260)).

Use phase markers, not total Slurm age:

```bash
grep -E 'Phase warmup progress|WARMUP cache pressure|Phase warmup complete|Phase profiling started|Phase profiling complete|replay_rc=' \
  "<LOG_DIR>/benchmark.out"
date -u
```

Report phase elapsed/remaining, last log update, error count, request/queue/KV trends, files and metric sources inspected, and separate expected benchmark completion from expected GitHub completion. A run is not green until required artifacts upload and the workflow accepts them.

## 10. Short-circuit rules

Recommend stopping early when direct evidence is already disqualifying:

- deterministic OOM, NCCL/RCCL failure, parser crash, or a missing worker.
- counters and log timestamps show no forward progress across repeated samples.
- persistent near-100% KV usage plus a growing queue and unusable latency.
- throughput has plateaued while more concurrency only worsens TTFT/TPOT.
- any disaggregated pool or required metrics source never registers.
- AIPerf validation shows zero completed requests or error rate above the configured `0.10` limit ([validator](../utils/agentic/validation/validate_agentic_result.py#L49-L87)).

Do **not** stop merely because model loading, dataset configuration, warmup, cutoff drain, or profiling is slow while completions advance and queues remain stable. Before any cancellation, capture timestamps, exact topology, relevant log lines, at least two metric samples showing the trend, current phase, and diagnosis.

Cancellation mutates shared infrastructure. Unless the current task explicitly authorizes it, ask first. Prefer GitHub cancellation so workflow cleanup runs:

```bash
gh run cancel <RUN_ID> --repo SemiAnalysisAI/InferenceX
```

Use `scancel` or process termination only with explicit approval and a concrete reason. They can bypass cleanup or strand the runner. After a recipe fix, dispatch one targeted fast e2e point, inspect it live, then reserve a canonical run/full sweep for the candidate that passed.

## Completion checklist

- Matrix preview matches intended scenario, topology, eval mode, and concurrency.
- Full eval has no `EVAL_LIMIT`, and every expected batch point is completed and has a suffixed result.
- `validate_scores.py` passes against the intended task/model threshold.
- Aggregate and raw eval/AgentX artifacts are downloaded and internally consistent.
- AgentX corpus, replay mode, exact command, commit, image, recipe, topology, and fast/override state are recorded.
- Every backend/frontend and metrics source is represented in live evidence.
- Fast/smoke results are labeled diagnostic. Only the canonical candidate is used for final comparison.
- Workflow and artifact collection conclude green before success is reported.
