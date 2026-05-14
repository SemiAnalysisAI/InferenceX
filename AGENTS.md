# AGENT.md

This file provides guidance for AI agents working with the InferenceX codebase.

## Project Overview

InferenceX is an open-source, automated benchmarking system that continuously tracks LLM inference performance across different hardware platforms (NVIDIA B200/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results are published to https://inferencex.com/.

## Directory Structure

Top-level layout (run `ls` for details):

- `perf-changelog.yaml` — benchmark trigger log; append-only; preserve whitespace
- `benchmarks/` — `benchmark_lib.sh` shared helpers; `single_node/` and `multi_node/` entrypoints (per model/precision/hardware/framework `.sh` scripts); `*_mtp.sh` for MTP/spec-decoding; `multi_node/srt-slurm-recipes/` checked-in external recipe YAMLs
- `runners/` — hardware launcher scripts
- `utils/matrix_logic/` — `generate_sweep_configs.py` CLI, `validation.py` Pydantic schemas, tests
- `utils/bench_serving/` — `benchmark_serving.py` serving client and backends
- `utils/evals/` — lm-eval task configs, thresholds, `validate_scores.py` (see `EVALS.md`)
- `utils/` (top level) — `process_result.py`, `process_changelog.py` (incl. `trim_conc`), `summarize.py`, `collect_*.py`, `compare_results.py`
- `experimental/` — non-core experiments

## Terminology

- **STP (Single Token Prediction)**: Standard autoregressive decoding where one token is generated per forward pass. No speculative decoding or MTP (Multi-Token Prediction) is used. When a benchmark is labeled "STP only", it means vanilla decoding without any speculation.
- **MTP (Multi-Token Prediction)**: A technique where the model predicts multiple tokens per forward pass, typically using speculative decoding methods like EAGLE or NEXTN.

## Development Workflow

### Running Tests

```bash
python -m pytest utils/matrix_logic/ -v
```

Markers: `slow`, `integration`.

### Generating Benchmark Configs

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  [--model-prefix dsr1|gptoss|dsv4|...] \
  [--framework sglang|trt|vllm|atom|dynamo-trt|dynamo-sglang] \
  [--precision fp4|fp8|...] \
  [--runner-type b200|h100|h200|gb200|...]
```

### Processing Results

```bash
python utils/process_result.py
python utils/summarize.py
```

## Supported Configuration Values

When working with benchmark configurations, use these valid values:

**Frameworks**:
- `sglang` - SGLang inference engine
- `trt` - TensorRT-LLM
- `vllm` - vLLM inference engine
- `atom` - AMD ATOM framework
- `dynamo-trt` - NVIDIA Dynamo with TensorRT-LLM backend
- `dynamo-sglang` - NVIDIA Dynamo with SGLang backend
- `sglang-disagg` - SGLang disaggregated inference

**Sequence Lengths (ISL/OSL)**:
- `1k1k` - 1024 input / 1024 output
- `8k1k` - 8192 input / 1024 output

## Code Conventions

### Python

- Use type hints: `list[str]`, `dict`, `Optional[int]`
- Pydantic models for validation with `extra='forbid'`
- Field aliases for YAML compatibility: `Field(alias="model-prefix")`
- Docstrings for functions

### YAML

- Kebab-case for field names: `model-prefix`, `conc-start`, `dp-attn`
- Master configs define all benchmark configurations
- `perf-changelog.yaml` triggers which configs to benchmark
  - **The file is read in chronological order: oldest at the top, newest at the bottom. New entries MUST be appended to the END of the file — never insert in the middle or prepend.**

### Bash

- Source shared utilities: `source benchmark_lib.sh`
- Functions: `check_env_vars()`, `wait_for_server_ready()`, `run_benchmark_serving()`, `run_eval()`, `append_lm_eval_summary()`
- Parameters passed via environment variables
- **MTP scripts MUST pass `--use-chat-template` to `run_benchmark_serving` — no exceptions.** EAGLE-style speculative decoding is trained against chat-formatted inputs, so benchmarking against raw prompts silently regresses acceptance rate and produces misleading numbers. This applies to every `*_mtp.sh` script regardless of model, precision, or runner.

### Git

- Conventional commit messages
- Use `[skip-sweep]` in commit message to skip benchmarks (push-to-main only)
- Changes to `perf-changelog.yaml` trigger benchmark runs

### Pull Request Sweep Labels

PRs do **not** run the sweep automatically — `run-sweep.yml` is gated on a label. Pick exactly one of the two; setting both is rejected by the workflow.

`sweep-enabled` - Runs the sweep with `--trim-conc`: each parallelism config is reduced to its single highest configured concurrency point. Default for most PRs — validates the change runs end-to-end without consuming the full cluster.
`full-sweep-enabled` - Runs the full intermediate concurrency sweep, identical to a push-to-main run. Use when intermediate concurrency points actually matter for the PR (e.g., a recipe change expected to shift the throughput/latency curve, not just its endpoints).

Notes:
- The two labels are mutually exclusive — `run-sweep.yml`'s `setup` job fails fast with an explicit error if both are present.
- Push-to-main always runs the full untrimmed sweep unless `[skip-sweep]` is in the commit message; the trim only applies to PR runs that opt in via `sweep-enabled`.
- The trimming logic lives in `trim_conc()` in `utils/process_changelog.py` — single-node entries are grouped by every non-`conc` field and only the highest-`conc` entry per group is kept; multi-node entries have their `conc` list collapsed to `[max(conc)]`.

## Common Tasks

### Dispatching jobs

Sweeps and one-off runs are dispatched against `.github/workflows/e2e-tests.yml` (the `workflow_dispatch` entrypoint). `run-sweep.yml` is push/PR-triggered and is not dispatchable.

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=my-feature-branch' \
  -f 'inputs[test-name]=DSR1 fp8 H200 sglang smoke' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files .github/configs/nvidia-master.yaml --model-prefix dsr1 --framework sglang --runner-type h200 --min-conc 4 --max-conc 4 --seq-lens 1k1k' \
  -f 'inputs[duration-override]='
```

Inputs:

| Input | Required | Meaning |
|---|---|---|
| `ref` (top-level, no `inputs[...]`) | yes | Workflow ref to dispatch *from*. Almost always `main` unless you're testing a workflow change. |
| `inputs[ref]` | no | Repo ref the matrix-generation job checks out (the branch/SHA under test). Defaults to `github.sha` of the dispatch ref. |
| `inputs[generate-cli-command]` | yes | Args passed verbatim to `utils/matrix_logic/generate_sweep_configs.py`. Test locally first by running that command. |
| `inputs[test-name]` | no | Display name in the Actions UI / `run-name`. |
| `inputs[duration-override]` | no | Override per-config `duration` (seconds). Empty string = use matrix value. |

The POST returns no body and no run ID. Find the run you just dispatched with the next section.

### Monitoring jobs

```bash
# 1. Find the run you just dispatched (most recent workflow_dispatch on e2e-tests.yml)
gh run list --repo SemiAnalysisAI/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 5
# Capture the ID of the run you just dispatched:
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')

# 2. Block until it finishes (non-zero exit if the run fails)
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status

# 3. Inspect jobs / failed logs
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed

# 4. Cancel if needed
gh run cancel "$RUN_ID" --repo SemiAnalysisAI/InferenceX
```

For result artifacts after a successful run, see "Fetching GitHub Actions Benchmark Results" below.

### Adding a New Benchmark Configuration

1. Add entry to `.github/configs/nvidia-master.yaml` or `amd-master.yaml`
2. Add corresponding entry to `perf-changelog.yaml` to trigger benchmark
3. Run validation: `python utils/matrix_logic/generate_sweep_configs.py full-sweep ...`

### Adding a New Runner

1. Add runner to `.github/configs/runners.yaml`
2. Create launcher script in `runners/` directory
3. Update relevant master config with new runner type

### Registering Recipes from srtslurm

For `dynamo-sglang` / `dynamo-trt` disaggregated multi-node configs, see `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md` for the full mapping from srtslurm recipe YAML to `nvidia-master.yaml` entries.

### Updating Docker Images

When upgrading Docker images in benchmark scripts and master configs .yaml:

1. Update the image tag in the relevant `.github/configs/*-master.yaml` and/or `benchmarks/*.sh` script(s)
2. Update any related environment variables or configuration parameters
3. **MUST**: Add an entry to `perf-changelog.yaml`: for example:
   ```yaml
   - config-keys:
       - dsr1-fp8-*-vllm  # Use wildcards to match multiple configs
     description:
       - "Update vLLM image from v0.11.2 to v0.13.0"
       - "Add VLLM_MXFP4_USE_MARLIN=1 environment variable"
     pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
   ```
4. This triggers benchmarks for affected configs and tracks performance changes

## Evals (Accuracy Validation)

Optional accuracy checks that ensure inference optimizations do not degrade model outputs. See `utils/evals/EVALS.md` for the full reference.

Quick pointers:
- Eval selection is marked by `mark_eval_entries()` in `utils/matrix_logic/generate_sweep_configs.py`; evals are marked by default on the 8k1k subset.
- Eval workflow jobs run separately from throughput jobs in eval-only mode (`EVAL_ONLY=true`).
- Flags on `generate_sweep_configs.py full-sweep`: `--no-evals` to skip, `--evals-only` for the eval subset only.
- Aggregated eval output is produced by `utils/collect_eval_results.py`.

## Key Files to Understand

- `utils/matrix_logic/validation.py` - Defines all configuration schemas
- `utils/matrix_logic/generate_sweep_configs.py` - Config generation logic
- `utils/bench_serving/benchmark_serving.py` - Benchmark client for measuring serving performance
- `.github/configs/nvidia-master.yaml` - NVIDIA benchmark definitions
- `.github/workflows/run-sweep.yml` - Main CI/CD workflow
- `.github/workflows/collect-evals.yml` - Eval results collection workflow
- `benchmarks/benchmark_lib.sh` - Shared benchmark/eval utilities
- `utils/evals/` - Eval task definitions (`gsm8k.yaml`, `gpqa_diamond.yaml`)
- `utils/collect_eval_results.py` - Aggregates eval results into JSON/table

## Important Notes
1. Make sure no new directories are created in `/workspace` during the benchmark. Files are ok.
2. **Never delete or modify whitespace in `perf-changelog.yaml`** — the CI pipeline depends on the exact whitespace (including trailing spaces on blank separator lines). Removing or altering whitespace will break CI and cause pipeline crashes.

## Fetching GitHub Actions Benchmark Results

When asked to analyze benchmark results from a GitHub Actions run:
```bash
# List artifacts for a run
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/<RUN_ID>/artifacts --jq '.artifacts[].name'

# Download aggregated results
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
```
### Parsing Results (IMPORTANT: avoid dumping raw JSON)

The results JSON is large with many decimals — never `cat` it raw. Use `jq` to extract and round:

```bash
# Summary table: hw, model, isl/osl, throughput (rounded)
cat ./results/agg_bmk.json | jq -r '
  .[] | [.hw, .infmax_model_prefix, "\(.isl)/\(.osl)", (.tput_per_gpu | round)]
  | @tsv' | column -t

# Filter to a specific model
cat ./results/agg_bmk.json | jq '[.[] | select(.infmax_model_prefix == "gptoss")]'
```

### Key Metrics

| Field | Description |
|-------|-------------|
| `tput_per_gpu` | Total throughput per GPU (tokens/sec) |
| `output_tput_per_gpu` | Output token throughput |
| `mean_ttft` / `p99_ttft` | Time to first token |
| `mean_tpot` | Time per output token |
| `mean_e2el` | End-to-end latency |

### Artifact Naming

| Pattern | Contents |
|---------|----------|
| `results_bmk` | Aggregated benchmark results, `agg_bmk.json` |
| `results_all` | All results aggregated , might not exist |
| `eval_results_all` | Eval results, `agg_eval_all.json`, might not exist |
| `run-stats` | `run_stats.json`, run stats, which nodes were ran and succeeded |
