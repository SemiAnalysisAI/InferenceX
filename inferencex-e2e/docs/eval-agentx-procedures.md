# Evaluation and AgentX Procedures


Use this page to add and run graded evals, operate AgentX trace replays, preserve evidence, and decide whether a long run should continue. Commands assume `inferencex-e2e/` as the working directory and replace values in `<ANGLE_BRACKETS>`.

To run only the replay client against an existing server, follow
[Run AgentX-Harness Standalone](agentx-standalone.md). It includes installation
and a direct `aiperf profile` command without CI or Slurm.

## 1. Pick the correct execution mode

For a throughput-only PR sweep, set `no-evals: true` on its
`perf-changelog.yaml` entries and use one primary sweep label (normally
`full-sweep-fail-fast`). This skips all eval job families for those entries without
changing benchmark duration or Prometheus artifacts. The flag defaults to false
and is retained in changelog metadata. Another entry requesting the same config
can still select its evals; mark every applicable entry to suppress them entirely.
Combining `no-evals` with `all-evals`, `evals-only`, or `eval-min-prefill-ep`
on the entry, or with either eval PR modifier, is rejected. Such a run provides
throughput evidence, not model-evaluation evidence.

There are two distinct layers: the matrix generator decides **which jobs exist**, while runtime variables decide **what a launched job does**.

| Need | Generator flag (`infx.matrix.generate`) or workflow variables | Runtime behavior |
|---|---|---|
| Normal sweep | no eval option | Throughput jobs plus the selected 8k/1k eval subset and agentic GSM8K subset |
| Throughput only | `--no-evals` | No eval jobs |
| Selected eval subset only | `--evals-only` | Jobs have `RUN_EVAL=true`, `EVAL_ONLY=true` |
| Every eligible eval only | `--all-evals` | Equivalent to `--evals-only --all-evals` and includes all fixed-sequence 8k/1k rows plus single-node and multi-node agentic GSM8K rows |
| Throughput then eval in one recipe | `RUN_EVAL=true`, `EVAL_ONLY=false` | Server starts, throughput runs, then `python3 -m infx.bench eval` runs |
| Eval against a freshly started server | `RUN_EVAL=true`, `EVAL_ONLY=true` | Launcher applies the eval-only server settings, skips throughput, and runs the eval |

The PR `all-evals` label instead goes through [`infx.matrix.plan`](../infx/matrix/plan.py), which expands eval selection and keeps throughput.

Default selection is scenario-aware. Single-node fixed-sequence evals use the median and highest eligible concurrency for each 8k/1k model/runner/framework/precision/parallelism group. Multi-node evals use the highest eligible concurrency per topology. Fixed-sequence concurrency below 16 is not selected. Every AgentX model, including Kimi K3 and MiniMax M3, gets GSM8K by default. Single-node agentic rows run it at the highest concurrency of each model/runner/framework/precision/spec-decoding/dp-attn/image group, so MTP, DP-attention and image variants each get their own eval while TP/EP and KV-offloading variants share one. Multi-node agentic rows run it at the highest eligible concurrency per topology, and a deployment with no topology at concurrency 16 or above gets one eval at its highest concurrency. Kimi K3 and MiniMax M3 rows additionally run their vendor evals at every generated point, including lower concurrencies, so their GSM8K is an extra eval-only row. Every eval runs as a separate eval-only job, so agentic throughput coverage is unchanged. See [`mark_eval_entries()` and `mark_all_eval_entries()`](../infx/matrix/generate.py).

Kimi K3 automatically runs `kimi-vendor` / `kimi_tool_call_schema_full` on AMD and NVIDIA, for single-node and multi-node recipes. This runs 204 unique schema cases in streaming and non-streaming modes, producing 408 checks. An explicit `eval-framework=kimi-vendor` and `eval-suite=kimi_tool_call_schema` workflow override retains the one-case, two-check smoke for fast diagnosis. `--trim-conc` trims deployment points, not the suite's case count. MiniMax M3 automatically runs `minimax-vendor` / `minimax_m3_full` across both vendors, covering all 102 provider cases. Its one-case `minimax_m3_smoke` is available through an explicit override. Fixed-sequence GSM8K selection is unchanged.

Read the task name and `n_eff` with the score: `kimi_tool_call_schema = 1.0, n_eff = 2` means one schema case passed both modes. It is not a full-suite result or a GSM8K score. Historical smoke artifacts retain their original identity. Full Kimi results remain diagnostic with a `0.0` quality threshold; missing checks, setup failures, and integration errors still fail the job. [Suite definitions and artifact contract](../infx/evals/EVALS.md#how).

The full Kimi vendor suite has no adapter-level whole-process timeout. Stock request timeouts, engine readiness deadlines, and workflow/scheduler allocation limits still apply. The smoke keeps its 900-second deadline; a direct Python adapter invocation can impose an explicit positive `--timeout-seconds` override on either suite.

On a PR, combine one primary sweep label (normally `full-sweep-fail-fast`) with eval modifiers. `all-evals` expands coverage without suppressing throughput. `evals-only` suppresses throughput. Together they run all eligible evals only. Runs with `evals-only` are not reusable, while normal full sweeps and `all-evals` full sweeps are reusable. Adding or removing a modifier restarts the active sweep ([label policy](ci-procedures.md#pr-primary-and-modifier-labels)).

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
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate \
  test-config \
  --config-keys qwen3.5-fp8-b200-sglang-agentic \
  --conc 1 \
  --evals-only \
  --config-files configs/nvidia-master.yaml | jq .
```

A correct AgentX eval row contains `"scenario-type": "agentic-coding"`, `"run-eval": true`, and `"eval-only": true`. The workflow splits generated rows into throughput, fixed-sequence eval, and agentic eval jobs in [`.github/workflows/e2e-tests.yml`](../../.github/workflows/e2e-tests.yml#L328-L335).

## 2. Add a graded eval

1. Add `infx/evals/<task>.yaml` using the lm-evaluation-harness task format. Pin the dataset/split, deterministic generation settings, prompt contract, filters, and primary metric. Use [`gsm8k.yaml`](../infx/evals/gsm8k.yaml) or [`gpqa_diamond.yaml`](../infx/evals/gpqa_diamond.yaml) as an in-tree pattern.
2. Give `task:` a stable name. That exact name is the key used by score thresholds and appears in collected rows.
3. Add the minimum accepted score to [`infx/evals/thresholds.yaml`](../infx/evals/thresholds.yaml). Put a general floor under `default`. Add `models.<model-prefix>.<task>` only when a justified model-specific floor is required.
4. If the task's primary result is not compatible with the collector's strict/extract/accuracy rules, extend `extract_metrics()` in [`infx.results.evals`](../infx/results/evals.py). It accepts loaded JSON and explicit source provenance; `build_rows()` applies collector score validation and metadata conversion. A successful published row must have a non-null `score`.
5. Run a small explicit slice, inspect samples, then run the full split. `EVAL_LIMIT` is a smoke-test control, not a publishable score setting.

Against an already healthy OpenAI-compatible server:

```bash
export MODEL='<HF_MODEL_ID>'
export MODEL_NAME='<SERVED_MODEL_NAME>'
export MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>'
export EVAL_ONLY=false IS_MULTINODE=false OPENAI_API_KEY=EMPTY
export EVAL_TASKS_DIR='infx/evals/<task>.yaml'
export EVAL_LIMIT='10'
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency 16 --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores \
  --model-prefix "$MODEL_PREFIX" \
  --meta-env "$EVAL_DIR/meta_env.json" \
  --results-glob "$EVAL_DIR/results*.json"
```

For the full eval, unset the limit and repeat against a clean, correctly configured server:

```bash
unset EVAL_LIMIT
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency 16 --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores --model-prefix "$MODEL_PREFIX" \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

Run these with Python 3.10 or newer, normally inside the serving container, because lm-eval installs its pinned harness into that `python3` with `uv pip`. The command copies the allow-listed artifacts into `--stage-to` and writes `meta_env.json` there. It takes the concurrency from `--concurrency`, which lm-eval receives as `num_concurrent` in `--model_args`. `EVAL_CONCURRENT_REQUESTS` is no longer read. The exact invocation is in [`infx.bench.eval.lm_eval.run`](../infx/bench/eval/lm_eval.py#L121-L150).

## 3. `EVAL_ONLY` is a launcher contract

Set `EVAL_ONLY=true` **before server launch**. It is not merely a switch inside the eval command:

1. For a single-node fixed-sequence job, the srt binder sets the server context to the matrix `MAX_MODEL_LEN` (`isl + osl + 256`) through `context-length` for SGLang, `max_seq_len` and `max_num_tokens` for TRT-LLM, or `max-model-len` for vLLM and ATOM. AgentX points and multi-node jobs keep their recipe's own context, and multi-node jobs can select a real-verification `EVAL_CONFIG_FILE`.
2. The health check still runs. In eval-only jobs, vendor frameworks also wait for the served model on the OpenAI chat route, within `EVAL_ENDPOINT_READY_TIMEOUT_SECONDS`.
3. Throughput is skipped.
4. `python3 -m infx.bench eval` sizes each lm-eval request from `EVAL_MAX_MODEL_LEN`, else from `MAX_MODEL_LEN` capped at the model's native maximum.
5. The same command stages the artifacts and writes `meta_env.json`, whether the eval passed or failed.

Relevant implementation: [server context](../infx/srt_slurm/single_node.py#L183-L194), [request budget](../infx/bench/eval/lm_eval.py#L77-L98), [eval dispatch and failure policy](../infx/bench/eval/__init__.py#L74-L173), and [workflow inputs](../../.github/workflows/benchmark-tmpl.yml#L36-L53).

Native multi-node post-eval reads the mounted checkpoint at `/model` and enables dataset downloads in the eval process, without changing worker environments. Context lookup reads numeric limits from local `config.json` before falling back to Transformers; an explicit `EVAL_MAX_MODEL_LEN` still takes precedence.

Do not toggle `EVAL_ONLY` after a throughput-sized server is already running and assume the context changed. Restart through the recipe. In eval-only mode an eval failure is returned after available artifacts are staged. In a workflow, upload happens with `always()` before score validation so failed evidence survives ([single-node upload and gate](../../.github/workflows/benchmark-tmpl.yml#L449-L472), [multi-node upload and gate](../../.github/workflows/benchmark-multinode-tmpl.yml#L477-L503)).

## 4. Batched eval concurrency

A space-separated `--concurrency` value runs several concurrency points **sequentially against one live engine**. Multi-node jobs pass `EVAL_CONC` this way. It does not run several harnesses simultaneously. Within each point, the harness issues up to that point's concurrency.

```bash
export MODEL='<HF_MODEL_ID>' MODEL_NAME='<SERVED_MODEL_NAME>' MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>' EVAL_TASKS_DIR='infx/evals/gsm8k.yaml'
export EVAL_ONLY=false IS_MULTINODE=false OPENAI_API_KEY=EMPTY
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency '16 32 64' --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores --expected-concs '16 32 64' \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

The batch runner creates a fresh temporary output directory per point, stages files with `_conc<N>` suffixes, and writes these arrays to `meta_env.json`:

- `eval_concs`: requested points.
- `completed_eval_concs`: the eval succeeded and staged at least one artifact.
- `failed_eval_concs`: the eval or its staging failed, or it staged nothing.

A failed point is deferred so artifacts from every attempted point can upload. The post-upload validator then fails the job. Batched mode accepts positive integers and supports only `lm-eval`. See [batching](../infx/bench/eval/__init__.py#L176-L211), [artifact suffixing](../infx/bench/eval/stage.py#L20-L43), and [manifest validation](../infx/evals/validate_scores.py#L119-L216).

For multi-node `all-evals`, the workflow constructs `EVAL_CONC` by joining the topology's concurrency list ([dispatch](../../.github/workflows/e2e-tests.yml#L390-L392)). Never compare a point if its `_conc<N>` result or completed-manifest entry is missing.

## 5. Validate scores, not file existence

Run:

```bash
python3 -m infx.evals.validate_scores \
  --thresholds infx/evals/thresholds.yaml \
  --meta-env meta_env.json \
  --results-glob 'results*.json'
```

For a batch, add the independently expected points:

```bash
python3 -m infx.evals.validate_scores \
  --expected-concs '16 32 64' \
  --thresholds infx/evals/thresholds.yaml
```

Validation resolves the threshold in this order: `models.<prefix>.<task>`, `default.<task>`, then `--min-score` (default `0.85`). By default it checks numeric, non-stderr metrics beginning with `exact_match,`. It fails when a score is below threshold, no metric matches, a requested concurrency is absent, metadata has duplicates/invalid values, any point is marked failed, or result suffixes do not match the manifest. Current floors are authoritative in [`thresholds.yaml`](../infx/evals/thresholds.yaml). See [threshold resolution](../infx/evals/validate_scores.py#L63-L73) and the [validation flow](../infx/evals/validate_scores.py#L219-L373).

A manual combined throughput+eval recipe uploads eval output but the template's automatic score gate is specific to eval-only jobs. Run the validator explicitly for manual or combined runs.

## 6. Collect and inspect eval artifacts

The collection workflow downloads `eval_*`, aggregates raw sets with `infx/results/collect_eval_results.py`, uploads `eval_results_all/agg_eval_all.json`, and writes the table to the step summary ([`collect-evals.yml`](../../.github/workflows/collect-evals.yml)).

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

Retain `meta_env.json`, `results*.json`, and `sample*.jsonl`. The aggregate is a navigation aid, not a substitute for raw samples and batch completeness.

## 7. Run AgentX: fast feedback versus canonical evidence

`python3 -m infx.bench agentic` builds its own client runtime with uv ([`infx/bench/agentic/venv.py`](../infx/bench/agentic/venv.py)). It creates a fresh Python 3.11 venv under `AIPERF_RUNTIME_DIR` (default `<tmp>/inferencex-agentic-<SLURM_JOB_ID or PID>`) and installs the editable `utils/aiperf` with its declared dependencies, plus the client requirements AIPerf does not declare ([`requirements.txt`](../infx/bench/agentic/requirements.txt)). It then re-runs itself under that venv's Python. Recipes reach it through [`benchmarks/srt_agentic.sh`](../benchmarks/srt_agentic.sh).

AgentX is AIPerf `agentx` trace replay, not a fixed-token synthetic benchmark. The `agentx` scenario owns the replay defaults: ten additional warmup requests per trajectory lane, a 1,800-second warmup drain limit, a 0.10 live failure threshold, and a 300-second trace idle cap. A recipe may raise the drain limit with `AGENTIC_WARMUP_GRACE_PERIOD` or loosen the live abort with `AIPERF_LIVE_FAILED_REQUEST_THRESHOLD`; the finished profile still fails validation above a 0.10 error rate ([post-run gate](../infx/bench/agentic/run.py#L33-L35)). The profile runs for the configured duration. `agentx-fast` forces one warmup request per lane and a 1,200-second profile. It affects single- and multi-node AgentX throughput only. Fixed-sequence throughput and evals remain canonical. Fast runs are not eligible for artifact reuse ([workflow policy](ci-procedures.md#pr-primary-and-modifier-labels), [fast replay settings](../infx/bench/agentic/replay.py#L64-L65)).

Every AgentX throughput concurrency runs against a fresh server deployment. The matrix creates a separate job per point. `infx.launch` rejects a multi-node AgentX throughput job whose `CONC_LIST` is not exactly its one positive `CONC`, and the replay client rejects a `CONC_LIST` that differs from `CONC`. AgentX does not flush caches or reuse a running server for another point. Warmup and profiling for the same point share the deployment. This does not change fixed-sequence sweeps or graded-eval batching.

For multi-node srt-slurm jobs, the benchmark client may run on a different host from the frontend. The replay targets `http://$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT` whenever `SRT_FRONTEND_HOST` is set, otherwise an explicit `AIPERF_SERVER_URL`, and falls back to `http://localhost:$PORT` only when neither is available ([`_server_url`](../infx/bench/agentic/replay.py#L115-L122)).

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

Treat fast results as bring-up evidence, never as a replacement for the canonical candidate. A duration below 900 seconds adds AIPerf's `--unsafe-override` and flags the submission invalid. Use it only for smoke diagnosis ([source](../infx/bench/agentic/replay.py#L105)). After a fast run is healthy, run the exact candidate canonically before claiming benchmark success.

## 8. Preserve trace and run provenance

AgentX defaults to recorded assistant-response replay. Live server outputs are measured but discarded when constructing later turns. The selected trace corpus is model-family dependent unless `WEKA_LOADER_OVERRIDE` pins `semianalysis_cc_traces_weka_062126` or `semianalysis_cc_traces_weka_062126_256k`. The resolver logs both loader and Hugging Face dataset ([trace resolution](../infx/bench/agentic/traces.py#L20-L25), [replay semantics](../infx/bench/agentic/replay.py#L150-L185)). Replays keep the model's native context. The client ignores `MAX_MODEL_LEN`, and only an explicit `AIPERF_MAX_CONTEXT_LENGTH` adds AIPerf's `--max-context-length`.

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
```

For each concurrency retain:

- `benchmark_command.txt` (the exact AIPerf command) and `benchmark.log`.
- AIPerf `profile_export*`, `server_metrics_export.json`, plots, and distribution analysis.
- aggregate JSON and its `dataset` object (`source_type`, loader, HF dataset/split, entry count).
- server/frontend logs and every metrics endpoint represented.
- run URL/ID, attempt, head SHA, recipe/config identity, image, topology, fast flag, and any override.

The runner writes the command before replay and validates raw results after aggregation ([execution path](../infx/bench/agentic/run.py#L135-L218)). Aggregation preserves dataset provenance and hardware/model/topology fields ([aggregate construction](../infx/results/agentic/__init__.py)). Raw workflow uploads intentionally omit very large `inputs.json` and `profile_export_raw.jsonl`. If those are required for an investigation, preserve them from the live allocation before cleanup ([single-node artifact contract](../../.github/workflows/benchmark-tmpl.yml#L382-L391), [multi-node contract](../../.github/workflows/benchmark-multinode-tmpl.yml#L466-L475)).

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

Track trends over repeated samples: running/waiting requests, KV usage, prefix hits, input/output token rates, completed/cancelled/errored requests, frontend routing balance, and disaggregated KV transfer. AIPerf records endpoint identity for every server series ([metrics wiring](../infx/bench/agentic/replay.py#L125-L147)). When `AIPERF_SERVER_METRICS_URLS` is unset and `SRTCTL_FRONTEND_TYPE` is not `dynamo`, the replay scrapes each worker's `/metrics` from `SRT_AGG_ENDPOINTS`, or from `SRT_PREFILL_ENDPOINTS` plus `SRT_DECODE_ENDPOINTS`.

Use phase markers, not total Slurm age:

```bash
grep -E 'Phase warmup progress|WARMUP cache pressure|Phase warmup complete|Phase profiling started|Phase profiling complete|process_agentic_result' \
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
- AIPerf validation shows zero completed requests or error rate above the configured `0.10` limit ([validator](../infx/results/agentic/validate_agentic_result.py#L48-L88)).

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
