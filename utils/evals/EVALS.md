# Evals

## What?
Quick graded QnA which measures model performance. Examples of test suites:
- **gsm8k**: Grade school math questions
- **gpqa**: Graduate level, Google-Proof multiple choice questions

## When?
Evals run as **separate workflow jobs** from throughput benchmarks. The selection logic is in `mark_eval_entries()` of `utils/matrix_logic/generate_sweep_configs.py`.

**Single-node**: At the highest and median concurrency levels (all TPs), per (model, runner, framework, precision, ISL, OSL, spec-decoding, dp-attn), only for 8k1k.

**Multi-node**: One entry per (model, runner, framework, precision, spec-decoding, prefill-dp-attn, decode-dp-attn) with the highest max eligible concurrency, only for 8k1k. The eval job runs at `eval-conc`, the upper median of that config's eligible concurrency list.

## Why?
To verify how model outputs are affected by throughput optimizations.
- TP/Conc might affect model outputs
- Check kernel implementations for correctness
- If there was a tradeoff in accuracy for performance

## How?
`run_eval` in `benchmarks/benchmark_lib.sh` runs EleutherAI/lm-evaluation-harness against the server's OpenAI-compatible endpoint. Concurrency is set via `EVAL_CONCURRENT_REQUESTS` env var (not a CLI flag). Results are collected by `utils/collect_eval_results.py` and published as a summary table.

The default eval framework is [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`lm-eval`).

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

**MI355X (AMD)** — `benchmarks/multi_node/amd_utils/server.sh`
- Skips `bench.sh` when `EVAL_ONLY=true`
- Runs lm-eval via `run_eval` against the router on port 30000
- Concurrency uses workflow-provided `EVAL_CONC` when set, otherwise falls back to max of `BENCH_MAX_CONCURRENCY` (x-separated values)
- Eval artifacts copied to `/run_logs/slurm_job-*/eval_results/`
- `runners/launch_mi355x-amds.sh` skips benchmark result collection when `EVAL_ONLY=true` and uses `find` to locate eval results

**NVIDIA Slurm multi-node (GB200, GB300, B200, B300, H100, H200)** — via [srt-slurm](https://github.com/NVIDIA/srt-slurm) (`sa-submission-q2-2026` branch)
- `do_sweep.py` skips the benchmark stage when `EVAL_ONLY=true`, runs `_run_post_eval()` directly
- In eval-only mode, uses the full `wait_for_model()` health check (same as benchmark stage) since the benchmark health check was skipped
- `lm-eval` runner (`benchmarks/lm_eval.py`) is invoked by `do_sweep.py` as a post/eval-only step and sources InferenceX's `benchmark_lib.sh` from the mounted workspace (`/infmax-workspace`)
- Eval artifacts written to `/logs/eval_results/` inside the container, collected by launch scripts
- NVIDIA Slurm launch scripts always collect server logs for debugging but skip benchmark result collection when `EVAL_ONLY=true`
- Env vars threaded: `RUN_EVAL`, `EVAL_ONLY`, `IS_MULTINODE`, `FRAMEWORK`, `PRECISION`, `MODEL_PREFIX`, `RUNNER_TYPE`, `RESULT_FILENAME`, `SPEC_DECODING`, `ISL`, `OSL`, `PREFILL_TP/EP/NUM_WORKERS/DP_ATTN`, `DECODE_TP/EP/NUM_WORKERS/DP_ATTN`, `MODEL_NAME`, `EVAL_CONC`

### Workflow structure
- `e2e-tests.yml`: `test-sweep-evals` (single-node) and `test-sweep-multi-node-evals` (multi-node)
- `run-sweep.yml`: `sweep-evals` (single-node) and `sweep-multi-node-evals` (multi-node)
- Both use their respective benchmark templates with `eval-only: true`, `run-eval: true`
- `collect-evals` depends on both eval jobs; `collect-results` only runs when benchmark jobs ran
- `process_changelog.py` splits eval results into `evals` (single-node) and `multinode_evals`

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
| `EVAL_CONCURRENT_REQUESTS` | `64` | Concurrent requests during eval |

### Score validation
`utils/evals/validate_scores.py` checks eval results against thresholds in `utils/evals/thresholds.json`. Runs as a separate workflow step after artifact upload so results are preserved even if validation fails.

### Adding a new eval task

1. Create a task YAML in `utils/evals/` following the lm-eval task format.
2. Set `EVAL_TASKS_DIR=utils/evals/<your_task>.yaml` when running benchmarks.
3. Update `utils/collect_eval_results.py` if new metrics need extraction.

### lm-eval patches

The codebase patches lm-eval compatibility via `_patch_lm_eval`:

1. Reasoning token handling: extracts `reasoning_content` when `message.content` is empty.
2. TRT compatibility: avoids injecting `{"type": "text"}` for non-HF tokenizers.

## Task files
The following files are task definitions from lm-eval; more information on changes lives within the files:
- `utils/evals/gsm8k.yaml`
- `utils/evals/gpqa_diamond.yaml`
