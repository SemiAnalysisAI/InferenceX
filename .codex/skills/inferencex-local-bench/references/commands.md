# Local Benchmark Command Reference

## Preflight

```bash
pwd
git status --short --branch
git rev-parse HEAD
```

Keep durable run roots under `/it-share/yichaozhu/...` or `/data/yichaozhu/...`, not `/tmp`.

Example run root:

```bash
export RUN_ROOT=/it-share/yichaozhu/kimi-agentx-v1/runs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT"
```

## Allocate or Reuse Slurm

Use the site-specific partition/nodelist requested by the user. Record the job ID.

```bash
squeue -u "$USER"
scontrol show job "$JOB_ID"
```

When reusing an existing allocation:

```bash
export SLURM_REUSE_JOBID=<jobid>
```

## Launch Local Runner

Prefer local runner wrappers for MIA/local replay:

```bash
KEEP_LOGS=1 \
RUN_ROOT="$RUN_ROOT" \
SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-}" \
bash benchmarks/multi_node/local_runner/kimik2.5_mori_lmcache_agentic_1p2d_mi355x.sh
```

If no wrapper exists, use the CI benchmark script plus explicit env overrides and write a local wrapper before repeated experiments.

## Monitor

```bash
squeue -j "$JOB_ID"
tail -F "$RUN_ROOT"/logs/slurm_job-"$JOB_ID"/*.log
```

Collect vLLM metrics:

```bash
curl -fsS http://<server-ip>:<port>/metrics \
  | grep -E 'vllm:num_requests_running|vllm:num_requests_waiting|vllm:prompt_tokens_cached|vllm:external_prefix_cache_hits|vllm:external_prefix_cache_queries|vllm:kv_cache_usage_perc|vllm:gpu_cache_usage_perc|vllm:prefix_cache_hits|vllm:prefix_cache_queries|vllm:num_preemptions|vllm:request_success'
```

Collect LMCache health:

```bash
curl -fsS http://127.0.0.1:8080/healthcheck || true
curl -fsS http://127.0.0.1:8080/metrics | grep -Ei 'hit|miss|evict|lookup|retrieve|store|cache|l1|token|byte' || true
```

## Smoke and Correctness

Smoke with OpenAI-compatible endpoint:

```bash
curl -sS http://<router-ip>:<router-port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K2.5-MXFP4","messages":[{"role":"user","content":"Return exactly: OK"}],"max_tokens":8,"temperature":0}'
```

Run GSM8K/eval through the repo helper when the server is already up. Do not restart prefill/decode just to run correctness.

## Agentic Pressure

Use the repository aiperf/agentic path configured by the recipe. For each run, record:

- dataset path and model name
- concurrency and duration
- success count and total count
- input/output/total throughput
- GPU prefix hit, external hit/query, LMCache hit/miss if available
- run directory and log archive path

When parsing completed runs, prefer the current run's exact result filename:

```bash
find "$RUN_ROOT" -path "*/workspace_artifacts/${RESULT_FILENAME}.json" -print
find "$RUN_ROOT" -path "*/workspace_artifacts/${RESULT_FILENAME}_conc*.json" -print
```

Do not use every `*.json` under `workspace_artifacts`: local replay workspaces can contain stale JSON copied from previous runs.

## Archive

```bash
ARCHIVE=/it-share/yichaozhu/kimi-agentic/archives/kimi_run_$(date +%Y%m%d_%H%M%S).tar.gz
mkdir -p "$(dirname "$ARCHIVE")"
tar czf "$ARCHIVE" -C "$RUN_ROOT" .
```

Write a short status file:

```bash
cat > "$RUN_ROOT/STATUS_$(date +%Y%m%d_%H%M%S).md" <<EOF
# Status

- git: $(git rev-parse HEAD)
- job: ${JOB_ID:-unknown}
- nodes:
- result:
- logs:
- next:
EOF
```

## Cleanup

Only cleanup when requested or when the time box ends:

```bash
scancel "$JOB_ID"
squeue -j "$JOB_ID"
```
