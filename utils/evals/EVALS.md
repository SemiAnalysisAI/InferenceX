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

### Score validation
`utils/evals/validate_scores.py` checks eval results against thresholds in `utils/evals/thresholds.json`. Runs as a separate workflow step after artifact upload so results are preserved even if validation fails.

## Misc
Following files are task definitions from lmeval, more info on changes within the files
- `utils/evals/gsm8k.yaml`
- `utils/evals/gpqa_diamond.yaml`
