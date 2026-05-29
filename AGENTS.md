# AGENT.md

Guidance for AI agents working with InferenceX.

> **Before debugging a failing Klaud-Cold / claude/* image-bump PR, read [`KLAUD_DEBUG.md`](KLAUD_DEBUG.md).** It captures recurring failure modes (vLLM CUDA-graph OOM, B300 sglang regressions, cluster docker/perms/disk issues), the exact workarounds, and gh-CLI gotchas — most cron-PR failures are already cataloged there.

## Project Overview

InferenceX is an open-source automated benchmarking system that tracks LLM inference performance across hardware (NVIDIA B200/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results published to https://inferencex.com/.

## Directory Structure

Run `ls` for details. Key paths:

- `perf-changelog.yaml` - benchmark trigger log; append-only; preserve whitespace.
- `benchmarks/` - `benchmark_lib.sh` (shared helpers); `single_node/` and `multi_node/` entrypoints; `*_mtp.sh` for MTP/spec-decoding; `multi_node/srt-slurm-recipes/` checked-in external recipe YAMLs.
- `runners/` - hardware launcher scripts.
- `utils/matrix_logic/` - `generate_sweep_configs.py`, `validation.py` Pydantic schemas, tests.
- `utils/bench_serving/` - `benchmark_serving.py` and backends.
- `utils/evals/` - lm-eval task configs, thresholds, `validate_scores.py` (see `EVALS.md`).
- `utils/` - `process_result.py`, `process_changelog.py` (incl. `trim_conc`), `summarize.py`, `collect_*.py`, `compare_results.py`.
- `experimental/` - non-core experiments.

## Terminology

STP (Single Token Prediction): vanilla autoregressive decoding, one token per forward pass, no speculative decoding. MTP (Multi-Token Prediction): predicts multiple tokens per forward pass via speculative decoding (EAGLE, NEXTN, etc.).

## Development Workflow

Tests: `python -m pytest utils/matrix_logic/ -v` (markers: `slow`, `integration`).

Generate configs:

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  [--model-prefix dsr1|gptoss|dsv4|...] \
  [--framework sglang|trt|vllm|atom|dynamo-trt|dynamo-sglang] \
  [--precision fp4|fp8|...] \
  [--runner-type b200|h100|h200|gb200|...]
```

Process results: `python utils/process_result.py && python utils/summarize.py`.

## Supported Configuration Values

Frameworks: `sglang`, `trt`, `vllm`, `atom`, `dynamo-trt`, `dynamo-sglang`, `sglang-disagg`.
Sequence lengths (ISL/OSL): `1k1k` (1024/1024), `8k1k` (8192/1024).

## Code Conventions

Python: type hints (`list[str]`, `Optional[int]`), Pydantic with `extra='forbid'`, field aliases `Field(alias="model-prefix")`, docstrings on functions.

YAML: kebab-case field names (`model-prefix`, `conc-start`, `dp-attn`). Master configs define all benchmark configurations. `perf-changelog.yaml` triggers which configs to benchmark and is read chronologically (oldest at top, newest at bottom) - new entries MUST be appended to the END, never inserted in the middle or prepended.

Bash: source shared utilities via `source benchmark_lib.sh` (`check_env_vars`, `wait_for_server_ready`, `run_benchmark_serving`, `run_eval`, `append_lm_eval_summary`); parameters passed via env vars. **MTP scripts MUST pass `--use-chat-template` to `run_benchmark_serving`** - EAGLE-style spec decoding is trained against chat-formatted inputs; benchmarking against raw prompts silently regresses acceptance rate. Applies to every `*_mtp.sh`.

Git: conventional commit messages. `[skip-sweep]` in commit message skips benchmarks (push-to-main only). Changes to `perf-changelog.yaml` trigger benchmark runs.

### Pull Request Sweep Labels

PRs do not run the sweep automatically - `run-sweep.yml` is gated on a label. Pick exactly one; setting multiple sweep labels is rejected by the workflow's `setup` job.

- `sweep-enabled` - runs the sweep with `--trim-conc` (each parallelism config reduced to its single lowest concurrency). Default for most PRs.
- `full-sweep-enabled` - runs the full intermediate concurrency sweep behind a sequential single-node canary gate. Use when intermediate points matter (e.g. a recipe change shifts the throughput/latency curve, not just its endpoints).
- `non-canary-full-sweep-enabled` - runs the full intermediate concurrency sweep without the canary gate. Use when the canary is flaky or not representative of the affected configuration.

**The sweep does not trigger while the PR has merge conflicts.** Even with `sweep-enabled`, `full-sweep-enabled`, or `non-canary-full-sweep-enabled` applied, the `run-sweep.yml` workflow will not start until the PR cleanly merges into main — a stale claude/* or update-* branch with a `perf-changelog.yaml` conflict (the common case) will sit in NO_SWEEP / NO_SUCCESS until rebased. Resolution recipe is documented in `KLAUD_DEBUG.md §1.1`: `git merge origin/main`, then `git checkout origin/main -- perf-changelog.yaml`, then re-append the PR's own changelog entry at the tail. Don't 3-way merge `perf-changelog.yaml`; whitespace edits silently re-trigger the deletion check.

Push-to-main always runs the full untrimmed sweep unless `[skip-sweep]` is in the commit message. Trim logic lives in `trim_conc()` in `utils/process_changelog.py`: single-node entries are grouped by every non-`conc` field and only the lowest-`conc` entry per group is kept; multi-node entries have their `conc` list collapsed to `[min(conc)]`.

## Common Tasks

### Dispatching jobs

Sweeps and one-offs dispatch against `.github/workflows/e2e-tests.yml` (`workflow_dispatch`). `run-sweep.yml` is push/PR-triggered, not dispatchable.

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=my-feature-branch' \
  -f 'inputs[test-name]=DSR1 fp8 H200 sglang smoke' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files .github/configs/nvidia-master.yaml --model-prefix dsr1 --framework sglang --runner-type h200 --min-conc 4 --max-conc 4 --seq-lens 1k1k' \
  -f 'inputs[duration-override]='
```

Inputs: top-level `ref` (required) is the workflow ref to dispatch from, almost always `main`. `inputs[ref]` is the repo ref under test (defaults to the dispatch ref's `github.sha`). `inputs[generate-cli-command]` (required) is passed verbatim to `generate_sweep_configs.py` - test locally first. `inputs[test-name]` is the display name in the Actions UI. `inputs[duration-override]` overrides per-config duration (seconds); empty = use matrix value.

The POST returns no body and no run ID - find the run with `gh run list` below.

### Monitoring jobs

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status   # block, non-zero on failure
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed     # inspect failures
gh run cancel "$RUN_ID" --repo SemiAnalysisAI/InferenceX                # cancel
```

Artifacts: see "Fetching GitHub Actions Benchmark Results" below.

### Adding a benchmark configuration

Add entry to `.github/configs/nvidia-master.yaml` or `amd-master.yaml`, append to `perf-changelog.yaml`, validate with `generate_sweep_configs.py full-sweep`.

### Adding a runner

Add to `.github/configs/runners.yaml`, create launcher in `runners/`, add the runner type to the relevant master config.

### Registering recipes from srtslurm

For `dynamo-sglang` / `dynamo-trt` disaggregated multi-node configs, see `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md` for the full mapping from srtslurm recipe YAML to `nvidia-master.yaml` entries.

Multi-node srt-slurm changes must edit the recipe yaml AND `nvidia-master.yaml` together. `srtctl` reads only the recipe (`model.container`, resources, prefill/decode workers); the sweep generator (`utils/matrix_logic/generate_sweep_configs.py`) reads `nvidia-master.yaml` for frontend labels - its prefill/decode numbers never reach `srtctl`. Recipe-only edits mislabel results, master-only edits don't take effect. For image bumps, `model.container` must equal `image:`, since the launcher uses the latter as the container-alias key.

### Updating Docker images

Update the image tag in the relevant `.github/configs/*-master.yaml` and/or `benchmarks/*.sh`, update any related env vars / config params, and append a `perf-changelog.yaml` entry (required - triggers benchmarks):

```yaml
- config-keys:
    - dsr1-fp8-*-vllm  # wildcards match multiple configs
  description:
    - "Update vLLM image from v0.11.2 to v0.13.0"
    - "Add VLLM_MXFP4_USE_MARLIN=1 environment variable"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

## Evals (Accuracy Validation)

Optional accuracy checks ensuring inference optimizations do not degrade outputs. See `utils/evals/EVALS.md` for the full reference.

Eval selection is marked by `mark_eval_entries()` in `utils/matrix_logic/generate_sweep_configs.py`; evals run by default on the 8k1k subset. Workflow jobs run separately from throughput jobs in `EVAL_ONLY=true` mode. Flags on `generate_sweep_configs.py full-sweep`: `--no-evals` to skip, `--evals-only` for the eval subset only. Aggregated output produced by `utils/collect_eval_results.py`.

## Key Files

`utils/matrix_logic/validation.py` (config schemas), `generate_sweep_configs.py` (config generation), `utils/bench_serving/benchmark_serving.py` (benchmark client), `.github/configs/nvidia-master.yaml` (NVIDIA benchmark definitions), `.github/workflows/run-sweep.yml` (main CI/CD), `.github/workflows/collect-evals.yml` (eval collection), `benchmarks/benchmark_lib.sh` (shared utilities), `utils/evals/` (eval task definitions), `utils/collect_eval_results.py` (aggregator).

## Important Notes

- No new directories in `/workspace` during a benchmark (files are fine).
- **Never delete or modify whitespace in `perf-changelog.yaml`** - CI depends on exact whitespace (including trailing spaces on blank separator lines). Altering it breaks CI.

## Fetching GitHub Actions Benchmark Results

```bash
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/<RUN_ID>/artifacts --jq '.artifacts[].name'
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
```

### Parsing results (don't dump raw JSON)

`agg_bmk.json` is large with many decimals - never `cat` raw. Use `jq` to extract and round:

```bash
cat ./results/agg_bmk.json | jq -r '
  .[] | [.hw, .infmax_model_prefix, "\(.isl)/\(.osl)", (.tput_per_gpu | round)]
  | @tsv' | column -t

cat ./results/agg_bmk.json | jq '[.[] | select(.infmax_model_prefix == "gptoss")]'
```

### Key metrics

`tput_per_gpu` (total throughput per GPU, tok/s), `output_tput_per_gpu` (output token throughput), `mean_ttft` / `p99_ttft` (time to first token), `mean_tpot` (time per output token), `mean_e2el` (end-to-end latency).

### Artifacts

`results_bmk` → `agg_bmk.json` (aggregated). `results_all` → all results aggregated (may not exist). `eval_results_all` → `agg_eval_all.json` (may not exist). `run-stats` → `run_stats.json` (which nodes ran and succeeded).
