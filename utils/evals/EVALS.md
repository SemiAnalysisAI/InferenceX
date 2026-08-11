# Evals

Graded QA jobs (`gsm8k`, `gpqa`) catch accuracy regressions from parallelism,
concurrency, kernels, and other throughput optimizations. They run separately
from throughput. Selection lives in `mark_eval_entries()` in
`utils/matrix_logic/generate_sweep_configs.py`.

## Selection

- **Single-node:** 8k1k only, at the highest and median concurrency for every model,
  runner, framework, precision, TP, and decoding configuration.
- **Multi-node:** 8k1k only, with one job per parallelism topology at its highest
  eligible concurrency. Rows differing only by concurrency share a topology.
- **Agentic (SWE-bench), single-node:** highest-conc entry per (model,
  runner, framework, precision) group.
- **Agentic (SWE-bench), multi-node:** same policy as multi-node fixed-seq-len
  above (highest eligible conc per parallelism topology), since SWE-bench
  doesn't support batched concurrencies the way lm-eval does.

Generator eval modes:

- Default: throughput plus the selected eval subset.
- `--no-evals`: throughput only.
- `--evals-only`: selected evals only.
- `--all-evals`: every fixed-sequence eval only. This is equivalent to
  `--evals-only --all-evals`. Multi-node topologies run all `conc-list` values
  sequentially on one engine. Agentic-coding configs are included and run
  GSM8K (they are excluded only from the default, non-eval sweep).

Changelog entries use `evals-only: true` and `all-evals: true`. The `all-evals`
setting implies eval-only there. On PRs, the same names are modifier labels:
`all-evals` expands coverage without suppressing throughput, while `evals-only`
suppresses it. Modifier runs cannot be reused.

Deduplication is scenario-aware: fixed-sequence coverage does not suppress
agentic coverage, and `all-evals` wins over default eval coverage.

### Artifact reuse

Default full sweeps may reuse their eval subset. Source coverage is
authoritative. Raw `meta_env.json` identities must match `eval_results_all`,
and batched evals use `completed_eval_concs`. Policy drift is allowed, but
malformed metadata, duplicates, and raw/aggregate mismatches are not. See
[workflow reuse](../../.github/workflows/README.md#reusing-an-approved-pr-full-sweep).

## How?
`run_eval` in `benchmarks/benchmark_lib.sh` runs EleutherAI/lm-evaluation-harness against the server's OpenAI-compatible endpoint. Concurrency is set via `EVAL_CONCURRENT_REQUESTS` env var (not a CLI flag). Results are collected by `utils/collect_eval_results.py` and published as a summary table.

The default eval framework is [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`lm-eval`). Agentic eval-only matrix jobs inherit this default and therefore run the same GSM8K task as 8k1k. Explicit agentic runs can still select SWE-bench.

### Benchmark script flow

All benchmark scripts in `benchmarks/` follow one of two flows:

```bash
# Combined mode (benchmark + eval):
# 1. Start server (with context-length expansion if EVAL_ONLY=true)
# 2. wait_for_server_ready
# 3. run_benchmark_serving (skipped automatically when EVAL_ONLY=true)
# 4. Run evals:
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary  # Writes meta_env.json and moves artifacts
fi

# Eval-only mode (EVAL_ONLY=true):
# 1. Compute eval context via compute_eval_context_length
# 2. Start server with that context (--context-length or --max-model-len)
# 3. wait_for_server_ready
# 4. run_benchmark_serving returns immediately (skipped)
# 5. run_eval + append_lm_eval_summary
```

Key eval functions in `benchmarks/benchmark_lib.sh`:

| Function | Description |
|----------|-------------|
| `run_eval` | Unified entrypoint - dispatches to framework-specific runner |
| `run_lm_eval` | Runs lm-eval harness against the OpenAI-compatible endpoint |
| `append_lm_eval_summary` | Writes `meta_env.json` and moves eval artifacts to workspace |
| `_install_lm_eval_deps` | Installs lm-eval dependencies |
| `_patch_lm_eval` | Patches lm-eval for reasoning tokens and TRT compatibility |
| `compute_eval_context_length` | Computes eval context length (requested benchmark context, capped at model native max) |
| `get_native_max_context_length` | Extracts model's native max context length from HF config |

### Single-node
In eval-only mode (`EVAL_ONLY=true`), the benchmark script computes `EVAL_MAX_MODEL_LEN` via `compute_eval_context_length`, starts the server with that context length, skips throughput, and runs lm-eval directly. Each framework wires that context differently (`--context-length` for SGLang, `--max_seq_len` for TRT-LLM).

### Multi-node
Multi-node evals support two hardware paths:

**MI355X (AMD)** — `benchmarks/multi_node/amd_utils/server_sglang.sh`
- Skips throughput when `EVAL_ONLY=true`
- Fixed-seq-len: runs lm-eval via `run_eval --framework lm-eval` against the router on port 30000
- Agentic-coding (disaggregated, `IS_AGENTIC=1`): runs SWE-bench via `run_eval --port 30000` (no
  `--framework` override, same auto-selection as single-node agentic eval-only). Since there's no
  single "TP" for a disaggregated topology, and the workflow spells a couple of metadata fields
  differently (`PREFILL_DP_ATTN`/`DECODE_DP_ATTN`) than `append_lm_eval_summary` expects
  (`PREFILL_DP_ATTENTION`/`DECODE_DP_ATTENTION`), the agentic branch bridges those before calling
  `run_eval`; `append_lm_eval_summary` itself runs automatically inside `run_eval()` (same
  `EVAL_ONLY=true && IS_AGENTIC` auto-staging as single-node), not as a separate call.
- Concurrency uses workflow-provided `EVAL_CONC` when set, otherwise falls back to max of `BENCH_MAX_CONCURRENCY` (x-separated values)
- Eval artifacts copied to `/run_logs/slurm_job-*/eval_results/`
- `runners/launch_mi355x-amds.sh` skips benchmark result collection when `EVAL_ONLY=true` and uses `find` to locate eval results

**NVIDIA Slurm multi-node (GB200, GB300, B200, B300, H100, H200)** runs through [srt-slurm](https://github.com/NVIDIA/srt-slurm) on the `sa-submission-q2-2026` branch.
- `do_sweep.py` skips the benchmark stage when `EVAL_ONLY=true`, runs `_run_post_eval()` directly
- In eval-only mode, uses the full `wait_for_model()` health check (same as benchmark stage) since the benchmark health check was skipped
- `lm-eval` runner (`benchmarks/lm_eval.py`) is invoked by `do_sweep.py` as a post/eval-only step and sources InferenceX's `benchmark_lib.sh` from the mounted workspace (`/infmax-workspace`)
- Eval artifacts written to `/logs/eval_results/` inside the container, collected by launch scripts
- NVIDIA Slurm launch scripts always collect server logs for debugging but skip benchmark result collection when `EVAL_ONLY=true`
- Env vars threaded: `RUN_EVAL`, `EVAL_ONLY`, `IS_MULTINODE`, `FRAMEWORK`, `PRECISION`, `MODEL_PREFIX`, `RUNNER_TYPE`, `RESULT_FILENAME`, `SPEC_DECODING`, `ISL`, `OSL`, `PREFILL_TP/EP/NUM_WORKERS/DP_ATTN`, `DECODE_TP/EP/NUM_WORKERS/DP_ATTN`, `MODEL_NAME`, `EVAL_CONC`

For multi-node `all-evals`, `EVAL_CONC` is a space-separated list. When it contains multiple values, `run_eval` runs those concurrency points sequentially against the same live engine, stages each result with a `_concN` filename suffix, and records expected/completed/failed points in `meta_env.json`.

### Workflow structure
- `e2e-tests.yml`: `test-sweep-evals` (single-node fixed-seq-len), `test-sweep-multi-node-evals`
  (multi-node fixed-seq-len), `test-sweep-agentic-evals` (single-node agentic), and
  `test-sweep-multi-node-agentic-evals` (multi-node agentic)
- `run-sweep.yml`: `sweep-evals`, `sweep-multi-node-evals`, `sweep-agentic-evals`, and
  `sweep-multi-node-agentic-evals` (same four-way split)
- All four use their respective benchmark templates (`benchmark-tmpl.yml` for single-node,
  `benchmark-multinode-tmpl.yml` for multi-node) with `eval-only: true`, `run-eval: true`
- `collect-evals` depends on all four eval jobs; `collect-results` only runs when benchmark jobs ran
- `process_changelog.py` splits eval results by node count and scenario type into `evals`
  (single-node fixed-seq-len), `agentic_evals` (single-node agentic), `multinode_evals`
  (multi-node fixed-seq-len), and `multinode_agentic_evals` (multi-node agentic)

### Result collection

Eval results are collected by `.github/workflows/collect-evals.yml`:

1. Downloads all `eval_*` artifacts
2. Runs `utils/collect_eval_results.py` to aggregate results
3. Outputs `agg_eval_<exp_name>.json` with all eval metrics
4. Publishes a summary table to GitHub Step Summary

Fetch and inspect eval results:

```bash
# Download eval results artifact
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n eval_results_all -D ./evals

# View eval summary
cat ./evals/agg_eval_all.json | jq -r '
  .[] | [.hw, .framework, .precision, .tp, .conc, .task, (.score * 100 | round | . / 100)]
  | @tsv' | column -t

# Filter to specific hardware
cat ./evals/agg_eval_all.json | jq '[.[] | select(.hw == "B200")]'
```

### Metrics

| Field | Description |
|-------|-------------|
| `score` | Primary metric (exact match for GSM8K) |
| `em_strict` | Strict exact match (requires `####` format) |
| `em_flexible` | Flexible extraction (looser number matching) |
| `n_eff` | Number of samples evaluated |
| `task` | Eval task name (e.g., `gsm8k`) |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_EVAL` | `false` | Enable eval after throughput benchmark |
| `EVAL_ONLY` | `false` | Skip throughput, only run evals (set by workflow) |
| `EVAL_FRAMEWORK` | `lm-eval` | Eval framework to use |
| `EVAL_TASKS_DIR` | `utils/evals/gsm8k.yaml` | Path to lm-eval task YAML |
| `EVAL_RESULT_DIR` | `/tmp/eval_out-*` | Output directory for eval results |
| `EVAL_MAX_MODEL_LEN` | `16384` | Max context for eval (set by `compute_eval_context_length`) |
| `EVAL_CONCURRENT_REQUESTS` | `64` | Concurrent requests during eval. A space-separated list enables sequential batched evals against one live engine |
| `EVAL_LIMIT` | empty | Limit eval to first N instances (smoke tests). Empty means the full set |

### Score validation
`utils/evals/validate_scores.py` checks eval results against thresholds in `utils/evals/thresholds.yaml`. Runs as a separate workflow step after artifact upload so results are preserved even if validation fails.

### Adding a new eval task

1. Create a task YAML in `utils/evals/` following the lm-eval task format.
2. Set `EVAL_TASKS_DIR=utils/evals/<your_task>.yaml` when running benchmarks.
3. Update `utils/collect_eval_results.py` if new metrics need extraction.

### Runtime patches (`utils/evals/patches/`)

The benchmark helpers invoke these standalone scripts against pinned dependencies.
Source rewrites are anchor-checked, idempotent, and atomic.

- `lm_eval_sitecustomize.py` (`_patch_lm_eval`): reasoning-token handling
  (extracts `reasoning_content` when `message.content` is empty) and TRT
  compatibility (no `{"type": "text"}` injection for non-HF tokenizers).
  Copied into a temp dir as `sitecustomize.py` on `PYTHONPATH`.
- `patch_swebench_agent.py` (`_patch_swebench_agent`): mini-swe-agent/swe-rex
  sandbox lifecycle cleanup, budget-exhaustion submission fallback, and the
  [SWE-ReX #281](https://github.com/SWE-agent/SWE-ReX/pull/281) closed-stdin fix.
- `patch_swebench_scoring.py` (`_patch_swebench_scoring`): swebench Modal
  scorer reserved-CPU reduction + sandbox termination on instance completion.

### SWE-bench Lite (`--framework swebench`)

SWE-bench requires applying each generated patch and running repository tests.
The dedicated framework uses mini-swe-agent for agentic generation by default,
then scores predictions with the official SWE-bench harness. It emits
`exact_match,resolved` in the existing lm-eval result shape so collection and
validation remain shared with the other evals.

```bash
run_eval --framework swebench --port "$PORT"
append_lm_eval_summary
```

- Task metadata and single-shot prompt: `utils/evals/swebench_lite.yaml`.
- Scoring: `utils/evals/swebench_score.py` (diff extraction → `predictions.jsonl` →
  `python -m swebench.harness.run_evaluation` → resolved-rate → results JSON). Offline
  `--report` mode skips Docker for testing.
- Generation modes (`SWEBENCH_GEN_MODE`) include `agentic`, the default, which runs the
  mini-swe-agent loop against the local endpoint. Each instance's shell runs in a Modal
  sandbox via swe-rex, matching the real SWE-bench setting. The `single-shot` mode uses
  lm-eval with one prompt per instance. It provides a roughly 10% floor baseline and is
  kept only as an explicit debugging escape hatch. Agentic knobs include `SWEBENCH_AGENT_WORKERS`
  (default: the config's `CONC`, else 64), `SWEBENCH_AGENT_STEP_LIMIT` (250),
  `SWEBENCH_AGENT_CMD_TIMEOUT` (per command, 300s), `SWEBENCH_AGENT_TIMEOUT` (6h),
  `SWEBENCH_AGENT_SANDBOX_CPU` (unset = Modal default), and `SWEBENCH_MODAL_APP_NAME`
  (`infx-evals-swe`).
- Run size: an empty `EVAL_LIMIT` runs the full split of roughly 300 instances. A positive integer runs the
  first N as an explicit smoke-test slice. `EVAL_LIMIT=full` (or `0`) also selects the full split.
- Scoring knobs: `SWEBENCH_TASK_NAME` (selects the YAML), `SWEBENCH_MAX_WORKERS`,
  `SWEBENCH_EVAL_SANDBOX_CPU` (cores per scoring sandbox, default 2), `SWEBENCH_EVAL_TIMEOUT`
  (per-instance test timeout, default 900s), `SWEBENCH_NAMESPACE` (pass `""` on arm/Mac),
  `SWEBENCH_SKIP_SCORE=true` (generate-only), `SWEBENCH_USE_MODAL=true` (score on Modal remote
  sandboxes instead of local Docker, as used in CI). For Modal credentials, set
  `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` (e.g. from a GitHub secret) or provide `~/.modal.toml`.
  If the file is absent, the env vars are bootstrapped into it automatically. The scoring dataset
  is derived from the YAML's `dataset_path`, which keeps generation and scoring aligned.
  If `SWEBENCH_DATASET` is set, it must match or the run fails fast.
- Scoring runs on Modal remote sandboxes in CI (`SWEBENCH_USE_MODAL=true`, no Docker on the GPU
  nodes). Local Docker scoring needs about 120 GB of disk. The `thresholds.yaml` gate is `0.50`,
  calibrated from full-split runs that scored 54%. Historical 50-instance slices scored 62–76%.

## Task files
The following files are task definitions from lm-eval. More information on changes lives within the files:
- `utils/evals/gsm8k.yaml`
- `utils/evals/gpqa_diamond.yaml`
- `utils/evals/swebench_lite.yaml` (generation only, scored by `swebench_score.py`)
