---
name: debug-agentx-runs
description: Debug long-running AgentX benchmark jobs from live cluster logs and metrics instead of waiting for buffered GitHub Actions output. Use for AgentX bring-up, performance tuning, apparent hangs, warmup or profiling failures, aggregated or disaggregated serving runs, deciding whether to short-circuit an unproductive run, estimating phase completion, and verifying that every frontend, prefill, decode, or aggregate engine is healthy.
---

# Debug AgentX runs from the cluster

AgentX runs have long model-load, warmup, drain, and profiling phases. Treat GitHub Actions
as the orchestration and final-status view. Use the cluster as the primary source of live
diagnostic signal. Compose with `$debug-runs` when it is available for the general
full-sweep and reproduction workflow.

Cluster login addresses, users, jumpboxes, runner directories, and storage paths belong in
the access-controlled InferenceX Clusters Slack canvas, not this repository. Ask the user
for the canvas link or the intended SSH alias when access is not already configured. Never
guess or publish private infrastructure details. Read-only SSH inspection is allowed.
Before editing cluster files, restarting or killing processes, draining nodes, cancelling
Slurm jobs, or otherwise mutating shared infrastructure, stop and ask for approval unless
the user explicitly authorized that exact action in the current task.

## 1. Resolve the exact job and serving topology

Start with the workflow run and list its AgentX matrix jobs:

```bash
gh run view <RUN_ID> --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | select(.name | test("agentic|AgentX"; "i")) |
        [.databaseId, .status, .conclusion, .name] | @tsv'
```

Read the selected master-config entry, launcher, and checked-in recipe before interpreting
logs. The recipe is the source of truth for what srt-slurm launches.

- **Aggregated:** look for `resources.agg_*`, `vllm_config.aggregated`, and usually a null
  connector. One logical deployment handles both prefill and decode.
- **Disaggregated:** look for separate prefill and decode resources/configs plus a KV
  connector. Prefill and decode are separate services and must both be inspected.
- **Attention DP inside an aggregate deployment is not disaggregation.** A TP4 × DP4 =
  EP16 deployment exposes four DP engine metrics sources, but every engine handles both
  prefill and decode.

Do not infer the topology from AIPerf source labels alone. Aggregate DP engines can appear
as `srv decode 0`, `srv decode 1`, and so on even though they are not decode-only servers.
Likewise, InferenceX may encode an aggregate worker in the master YAML as one prefill
worker and zero decode workers for result labeling.

## 2. Find the live Slurm allocation and log directory

Use the access method from the InferenceX Clusters canvas or the operator's configured SSH
alias. On the controller:

```bash
squeue -u <RUNNER_USER> -o "%.8i %.8T %.10M %.20N %.100j"
scontrol show job -o <SLURM_JOB_ID> | tr " " "\n" |
  grep -E "^(JobId|JobState|RunTime|TimeLimit|NodeList|WorkDir)="
```

Derive paths from `WorkDir`. Do not guess or hardcode a cluster path. For an srt-slurm
job, logs normally live under:

```text
<WorkDir>/outputs/<SLURM_JOB_ID>/logs/
```

Inventory the directory before choosing files:

```bash
find "<LOG_DIR>" -maxdepth 1 -type f -print | sort
```

If the Slurm job ID is not obvious, correlate the GitHub run ID, runner suffix, config
name, start time, and allocated nodes. Never attach to a similarly named job without
verifying those fields.

## 3. Follow the live logs

Always stream the custom benchmark log from the beginning so phase transitions and the
current metrics snapshot have context:

```bash
ssh <CLUSTER_ALIAS> 'tail -f -n+1 "<LOG_DIR>/benchmark.out"'
```

Select additional logs by topology:

- **Aggregated:** inspect every aggregate backend log, commonly `*_agg_w*.out`. Add the
  frontend/router log, commonly `*_frontend_*.out`, when requests are not registering,
  routing is imbalanced, metrics are missing, or the frontend reports active requests
  after engines are idle.
- **Disaggregated:** inspect all prefill backend logs, all decode backend logs, and the
  frontend/router log. A healthy decode pool does not prove that prefill or KV transfer is
  healthy, and vice versa.
- **Both:** inspect infrastructure logs when etcd/NATS registration, health checks, or
  worker discovery is suspect.

After discovering the real filenames, follow them together:

```bash
tail -F -n+1 <BENCHMARK_LOG> <FRONTEND_LOG> <SERVER_LOGS...>
```

Useful signatures:

```bash
rg -n -i \
  "Phase |warmup|profiling|returned=|in_flight=|queue=|kv_usage=|prefix_cache_hit=|tput_|ERROR|Traceback|OOM|NCCL|RCCL|timeout|connection refused" \
  <LOGS...>
```

GitHub job logs are often buffered or unavailable until a long-running job finishes.
Do not wait for them when the compute-visible logs are updating.

## 4. Inspect every live metrics source

Confirm the AIPerf command contains the frontend and all applicable engine metrics URLs.

- Aggregated TP/TEP normally has one logical engine source.
- Aggregated attention-DP has one source per DP engine.
- Disaggregated serving needs every prefill source, every decode source, and the frontend.

Missing URLs create a falsely healthy partial view. Read the endpoint directly when the
benchmark summary is ambiguous:

```bash
curl -fsS "<METRICS_URL>" |
  rg -i "request|queue|cache|token|prefill|decode|error|fail"
```

Prefer the metric names actually exposed by the running image instead of assuming a
specific vLLM, SGLang, or Dynamo version. Track at least:

- active/running and waiting requests
- KV-cache usage and prefix-cache hit rate
- input and output token rates
- completed, cancelled, and errored requests
- frontend active requests and per-worker routing balance
- KV-transfer activity and failures for disaggregated runs.

Interpret trends, not one scrape:

- KV usage pinned near 100%, collapsing cache hits, and a growing queue indicate a
  capacity cliff.
- Flat or falling throughput as concurrency rises, with sharply worse TTFT/TPOT, means
  the added concurrency is not productive.
- One idle engine while peers have deep queues suggests routing, registration, or metrics
  coverage problems.
- No new log timestamps and no changing counters suggest a hang.
- Increasing completions with stable queues and zero errors is healthy even when a
  full-context AgentX phase is slow.

## 5. Determine phase progress and ETA from cluster timestamps

Use AgentX phase markers, not total Slurm runtime:

```bash
grep -E \
  "Phase warmup progress|WARMUP cache pressure|Phase warmup complete|Phase profiling started|Phase profiling complete|replay_rc=" \
  "<LOG_DIR>/benchmark.out"
date -u
```

Model loading and warmup can dominate job age. For a duration-based profiling phase,
calculate the nominal end from the `Phase profiling started` timestamp plus the configured
duration. Then allow a few minutes for cutoff drain, aggregation, staging, and artifact
upload. State separately:

1. phase elapsed and remaining time
2. whether logs are still updating
3. errors observed
4. expected benchmark completion
5. expected GitHub job completion.

For warmup drains, report returned, sent, in-flight, errors, elapsed time, and the
configured grace limit. Compare the curve with a prior failed run at the same elapsed time
when validating a timeout change.

## 6. Short-circuit clearly bad runs

Do not burn hours waiting for final JSON when direct signals already disqualify a config.
Use server logs and metrics to make the decision early. Examples include:

- deterministic OOM, NCCL/RCCL failure, parser crash, or missing worker
- no forward progress across repeated samples
- persistent KV saturation and queue growth with unusable latency
- throughput that has plateaued while added concurrency only increases latency
- a disaggregated pool or metrics source that never registered.

Before cancellation, capture the relevant log lines, timestamps, topology, and metric
trend and state the diagnosis. Unless the user explicitly authorized cancellation or
short-circuiting in the current task, ask before terminating anything.

Prefer cancelling from GitHub so workflow cleanup executes:

```bash
gh run cancel <RUN_ID> --repo SemiAnalysisAI/InferenceX
```

Use `scancel` or direct process termination only with explicit approval and a concrete
reason. Doing so can bypass cleanup or strand shared-cluster state. Never kill only the
backend and leave the workflow silently occupying a runner.

After a recipe/config fix, use a targeted e2e dispatch for fast feedback. Reserve another
official full sweep for the candidate that has passed direct cluster inspection.

## 7. Report the live diagnosis

For each active point, report:

- GitHub job and Slurm job links/IDs
- aggregate versus disaggregated topology
- phase, elapsed time, remaining time, and last log update
- log files and metrics sources inspected
- request, queue, KV-cache, cache-hit, and token-rate trends
- errors and the likely root cause
- whether to continue, short-circuit, or rerun
- which points are fully green versus merely healthy in progress.

Do not call a run successful until GitHub has accepted its result artifacts and the
required workflow concludes green.
