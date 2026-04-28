# AGENT.md

This file provides guidance for AI agents working with the InferenceX codebase.

## Project Overview

InferenceX is an open-source, automated benchmarking system that continuously tracks LLM inference performance across different hardware platforms (NVIDIA B200/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results are published to https://inferencex.com/.

## Directory Structure

```
├── benchmarks/              # Shell scripts for running benchmarks
│   ├── benchmark_lib.sh     # Shared benchmarking/eval utilities
│   ├── dsr1_*.sh            # Deepseek R1-specific benchmark scripts
│   └── gptoss_*.sh          # gptoss-specific benchmark scripts
├── runners/                 # Launch scripts for different hardware
│   ├── launch_b200/h100/h200-*.sh     # NVIDIA launcher scripts
│   └── launch_mi*.sh                  # AMD launcher scripts
├── utils/                   # Python utilities
│   ├── matrix_logic/        # Config generation and validation
│   │   ├── generate_sweep_configs.py  # CLI for generating benchmark matrix
│   │   ├── validation.py              # Pydantic validation models
│   │   └── test_*.py                  # Unit tests
│   ├── bench_serving/       # Benchmark serving client (upstreamed from vLLM)
│   │   ├── benchmark_serving.py       # Main benchmark client script
│   │   ├── backend_request_func.py    # Backend-specific request functions
│   │   └── benchmark_utils.py         # Utility functions
│   ├── evals/               # Eval task definitions for lm-eval
│   │   ├── EVALS.md         # Evals documentation
│   │   ├── gsm8k.yaml
│   │   └── gpqa_diamond.yaml
│   ├── collect_eval_results.py  # Aggregates eval results
│   ├── process_result.py    # Post-processes benchmark results
│   ├── process_changelog.py # Processes perf-changelog.yaml
│   └── summarize.py         # Generates markdown summaries
├── .github/
│   ├── workflows/           # GitHub Actions CI/CD
│   │   ├── run-sweep.yml    # Main performance sweep
│   │   ├── e2e-tests.yml    # End-to-end testing
│   │   ├── benchmark-tmpl.yml           # Single-node benchmark job template
│   │   ├── benchmark-multinode-tmpl.yml # Multi-node benchmark job template
│   │   └── collect-evals.yml            # Eval results collection
│   └── configs/             # Master configuration files
│       ├── nvidia-master.yaml
│       ├── amd-master.yaml
│       └── runners.yaml
└── perf-changelog.yaml      # Triggers benchmarks on changes
```

## Terminology

- **STP (Single Token Prediction)**: Standard autoregressive decoding where one token is generated per forward pass. No speculative decoding or MTP (Multi-Token Prediction) is used. When a benchmark is labeled "STP only", it means vanilla decoding without any speculation.
- **MTP (Multi-Token Prediction)**: A technique where the model predicts multiple tokens per forward pass, typically using speculative decoding methods like EAGLE or NEXTN.

## Key Technologies

- **Python 3.13**: Core automation and config generation
- **Pydantic**: Configuration validation (V2 with strict mode)
- **Bash**: Benchmark execution and infrastructure orchestration
- **YAML**: Configuration files
- **GitHub Actions**: CI/CD workflows
- **Evals**: lm-eval validation of benchmark results
- **pytest**: Testing framework

## Development Workflow

### Running Tests

```bash
cd utils
python -m pytest matrix_logic/ -v
```

### Generating Benchmark Configs

```bash
# Full sweep with all configs
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml

# Filter by model prefix (dsr1 or gptoss)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --model-prefix dsr1

# Filter by framework (sglang, trt, vllm, atom, dynamo-trt, dynamo-sglang)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --framework sglang

# Filter by precision (fp4, fp8)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --precision fp8

# Filter by runner type (b200, h100, h200, gb200, mi300x, mi325x, mi355x)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --runner-type b200
```

### Processing Results

```bash
python utils/process_result.py
python utils/summarize.py
```

## Supported Configuration Values

When working with benchmark configurations, use these valid values:

**Models (model-prefix)**:
- `dsr1` - DeepSeek-R1-0528
- `dsv4` - DeepSeek-V4-Pro
- `gptoss` - GPT-OSS-120B

**Precisions**:
- `fp4`
- `fp8`

**Frameworks**:
- `sglang` - SGLang inference engine
- `trt` - TensorRT-LLM
- `vllm` - vLLM inference engine
- `atom` - AMD ATOM framework
- `dynamo-trt` - NVIDIA Dynamo with TensorRT-LLM backend
- `dynamo-sglang` - NVIDIA Dynamo with SGLang backend
- `sglang-disagg` - SGLang disaggregated inference

**Runners (NVIDIA)**:
- `b200` - NVIDIA B200 GPU
- `b200-trt` - NVIDIA B200 with TensorRT
- `h100` - NVIDIA H100 GPU
- `h200` - NVIDIA H200 GPU
- `gb200` - NVIDIA GB200 (multi-node)

**Runners (AMD)**:
- `mi300x` - AMD MI300X GPU
- `mi325x` - AMD MI325X GPU
- `mi355x` - AMD MI355X GPU

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

| Label | Behavior | When to use |
|-------|----------|-------------|
| `sweep-enabled` | Runs the sweep with `--trim-conc`: each parallelism config is reduced to its single highest configured concurrency point. | Default for most PRs — validates the change runs end-to-end without consuming the full cluster. |
| `full-sweep-enabled` | Runs the full intermediate concurrency sweep, identical to a push-to-main run. | Use when intermediate concurrency points actually matter for the PR (e.g., a recipe change expected to shift the throughput/latency curve, not just its endpoints). |

Notes:
- The two labels are mutually exclusive — `run-sweep.yml`'s `setup` job fails fast with an explicit error if both are present.
- Push-to-main always runs the full (untrimmed) sweep unless `[skip-sweep]` is in the commit message; the trim only applies to PR runs that opt in via `sweep-enabled`.
- The trimming logic lives in `trim_conc()` in `utils/process_changelog.py` — single-node entries are grouped by every non-`conc` field and only the highest-`conc` entry per group is kept; multi-node entries have their `conc` list collapsed to `[max(conc)]`.

## Common Tasks

### Adding a New Benchmark Configuration

1. Add entry to `.github/configs/nvidia-master.yaml` or `amd-master.yaml`
2. Add corresponding entry to `perf-changelog.yaml` to trigger benchmark
3. Run validation: `python utils/matrix_logic/generate_sweep_configs.py full-sweep ...`

### Adding a New Runner

1. Add runner to `.github/configs/runners.yaml`
2. Create launcher script in `runners/` directory
3. Update relevant master config with new runner type

### Registering Recipes from srtslurm

For disaggregated multi-node configurations (dynamo-sglang, dynamo-trt), recipes are stored in the external [srtslurm](https://github.com/NVIDIA/srt-slurm) repository. To stage these recipes in InferenceX:

**1. Locate source recipes in srtslurm:**
```bash
# Example: H200 sglang disagg recipes
ls /path/to/srtslurm/recipes/h200/
# 1k1k/  8k1k/
```

**2. Analyze recipe structure:**
Each recipe YAML contains:
- `name`: Recipe identifier
- `model`: Model path/container info
- `resources`: GPU type, prefill/decode node/worker counts
- `backend.sglang_config`: Prefill and decode configuration (tp-size, dp-size, ep-size, dp-attention, etc.)
- `benchmark`: ISL/OSL and concurrency settings

**3. Add config to nvidia-master.yaml:**
```yaml
dsr1-fp8-h200-dynamo-sglang:
  image: lmsysorg/sglang:v0.5.8-cu130-runtime
  model: deepseek-ai/DeepSeek-R1-0528
  model-prefix: dsr1
  runner: h200-multinode-slurm
  precision: fp8
  framework: dynamo-sglang
  multinode: true
  disagg: true
  seq-len-configs:
  - isl: 1024
    osl: 1024
    search-space:
    - conc-list: [1, 4, 16, 32, 64, 128, 256, 512]
      prefill:
        num-worker: 1
        tp: 8
        ep: 1
        dp-attn: false
        additional-settings:
        - "CONFIG_FILE=recipes/h200/1k1k/bs128-agg-tp.yaml"
      decode:
        num-worker: 0
        tp: 8
        ep: 1
        dp-attn: false
```

**4. Key mapping from srtslurm to nvidia-master.yaml:**

| srtslurm field | nvidia-master.yaml field |
|----------------|-------------------------|
| `resources.prefill_workers` | `prefill.num-worker` |
| `resources.decode_workers` | `decode.num-worker` |
| `sglang_config.prefill.tp-size` | `prefill.tp` |
| `sglang_config.prefill.ep-size` | `prefill.ep` |
| `sglang_config.prefill.enable-dp-attention` | `prefill.dp-attn` |
| `benchmark.concurrencies` (parsed) | `conc-list` |
| Recipe file path | `additional-settings: CONFIG_FILE=...` |

**5. Common patterns:**
- **Aggregated (AGG)**: Single node, `num-worker: 1` for prefill, `num-worker: 0` for decode
- **TEP (Tensor-Expert Parallel)**: `dp-attn: false`, `ep: 1`
- **DEP (Data-Expert Parallel)**: `dp-attn: true`, `ep: 8` (typically)
- **Low latency**: More decode workers (e.g., 9), lower concurrencies
- **High throughput**: Fewer decode workers, higher concurrencies

**6. Add perf-changelog entry:**
```yaml
- config-keys:
    - dsr1-fp8-h200-dynamo-sglang
  description:
    - "Add DSR1 FP8 H200 Dynamo SGLang disaggregated multinode configuration"
    - "Image: lmsysorg/sglang:v0.5.8-cu130-runtime"
    - "Recipes sourced from srtslurm repo (recipes/h200/)"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

**7. Validate configuration:**
```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --framework dynamo-sglang
```

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

### Debugging Benchmark Failures

1. Check GitHub Actions logs for the failed job
2. Look at environment variables passed to benchmark script
3. Review benchmark script in `benchmarks/` directory
4. Check `wait_for_server_ready()` logs for server startup issues

## Evals (Accuracy Validation)

Evals run optional accuracy checks to ensure model outputs aren't degraded by inference optimizations. They can run alongside benchmarks or independently in eval-only mode.

### When Evals Run

Evals run as **separate workflow jobs** from throughput benchmarks (eval-only mode). The `EVAL_ONLY` flag skips throughput benchmarking and only runs lm-eval.

**Single-node** eval selection:
- All TPs at **highest concurrency** and **median concurrency** per (model, runner, framework, precision, ISL, OSL, spec-decoding, dp-attn)
- Only on `8k1k` sequence length

**Multi-node** eval selection:
- Entry with **highest max eligible concurrency** per (model, runner, framework, precision, spec-decoding, prefill-dp-attn, decode-dp-attn)
- Only `8k1k` sequence length
- Eval runs at `eval-conc`, the upper median concurrency from the selected config

This selection logic is in `mark_eval_entries()` in `utils/matrix_logic/generate_sweep_configs.py`.

**Workflow separation**: Eval jobs are independent from benchmark jobs:
- `run-sweep.yml`: `sweep-evals` (single-node) and `sweep-multi-node-evals` (multi-node)
- `e2e-tests.yml`: `test-sweep-evals` and `test-sweep-multi-node-evals`
- Both use their respective benchmark templates with `eval-only: true`
- `collect-evals` depends only on eval jobs, not benchmark jobs

**Multi-node eval infrastructure**:
- AMD (MI355X): `server.sh` skips `bench.sh` when `EVAL_ONLY=true`, runs lm-eval directly
- NVIDIA Slurm multi-node (GB200, GB300, B200, B300, H100, H200): srt-slurm invokes its `lm-eval` runner from `do_sweep.py` as a post/eval-only step using `INFMAX_WORKSPACE`

### Eval Framework: lm-eval

The default eval framework is [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`lm-eval`).

### Running Evals via CLI

```bash
# Generate configs (evals marked by default on 8k1k subset)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml

# Generate throughput-only configs (skip evals)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --no-evals

# Generate ONLY the eval subset (excludes non-eval configs)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --evals-only
```

### Eval Integration in Benchmark Scripts

All benchmark scripts in `benchmarks/` follow one of two flows:

```bash
# Combined mode (benchmark + eval):
# 1. Start server (with --context-length expansion if EVAL_ONLY=true)
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

**Multi-node AMD** (`benchmarks/multi_node/amd_utils/server.sh`):
- Skips `bench.sh` when `EVAL_ONLY=true`
- Runs lm-eval via `run_eval` against the router on port 30000
- Copies eval artifacts to `/run_logs/slurm_job-*/eval_results/`

**Multi-node NVIDIA Slurm** (GB200, GB300, B200, B300, H100, H200 via srt-slurm):
- Uses the srt-slurm `lm-eval` runner as a post/eval-only step from `do_sweep.py`
- Mounts the InferenceX checkout from `INFMAX_WORKSPACE` at `/infmax-workspace`
- `lm-eval` runner sources `benchmark_lib.sh` from `/infmax-workspace`

### Key Eval Functions in `benchmarks/benchmark_lib.sh`

| Function | Description |
|----------|-------------|
| `run_eval` | Unified entrypoint - dispatches to framework-specific runner |
| `run_lm_eval` | Runs lm-eval harness against the OpenAI-compatible endpoint |
| `append_lm_eval_summary` | Writes `meta_env.json` and moves eval artifacts to workspace |
| `_install_lm_eval_deps` | Installs lm-eval dependencies |
| `_patch_lm_eval` | Patches lm-eval for reasoning tokens and TRT compatibility |
| `compute_eval_context_length` | Computes eval context length (requested benchmark context, capped at model native max) |
| `get_native_max_context_length` | Extracts model's native max context length from HF config |

### Eval Results Collection

Eval results are collected by `.github/workflows/collect-evals.yml`:

1. Downloads all `eval_*` artifacts
2. Runs `utils/collect_eval_results.py` to aggregate results
3. Outputs `agg_eval_<exp_name>.json` with all eval metrics
4. Publishes summary table to GitHub Step Summary

### Fetching Eval Results

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

### Eval Metrics

| Field | Description |
|-------|-------------|
| `score` | Primary metric (exact match for GSM8K) |
| `em_strict` | Strict exact match (requires `####` format) |
| `em_flexible` | Flexible extraction (looser number matching) |
| `n_eff` | Number of samples evaluated |
| `task` | Eval task name (e.g., `gsm8k`) |

### Environment Variables for Evals

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_EVAL` | `false` | Enable eval after throughput benchmark |
| `EVAL_ONLY` | `false` | Skip throughput, only run evals (set by workflow) |
| `EVAL_FRAMEWORK` | `lm-eval` | Eval framework to use |
| `EVAL_TASKS_DIR` | `utils/evals/gsm8k.yaml` | Path to lm-eval task YAML |
| `EVAL_RESULT_DIR` | `/tmp/eval_out-*` | Output directory for eval results |
| `EVAL_MAX_MODEL_LEN` | `16384` | Max context for eval (set by `compute_eval_context_length`) |
| `EVAL_CONCURRENT_REQUESTS` | `64` | Concurrent requests during eval |

### Adding a New Eval Task

1. Create a task YAML in `utils/evals/` (follow lm-eval task format)
2. Set `EVAL_TASKS_DIR=utils/evals/<your_task>.yaml` when running benchmarks
3. Update `utils/collect_eval_results.py` if new metrics need extraction

### lm-eval Patches

The codebase includes patches for lm-eval compatibility (`_patch_lm_eval`):

1. **Reasoning token handling**: Extracts `reasoning_content` when `message.content` is empty
2. **TRT compatibility**: Avoids injecting `{"type": "text"}` for non-HF tokenizers

These patches are applied via `sitecustomize.py` in `PYTHONPATH`.

## Key Files to Understand

- `utils/matrix_logic/validation.py` - Defines all configuration schemas
- `utils/matrix_logic/generate_sweep_configs.py` - Config generation logic
- `utils/bench_serving/benchmark_serving.py` - Benchmark client for measuring serving performance
- `.github/configs/nvidia-master.yaml` - NVIDIA benchmark definitions
- `.github/workflows/run-sweep.yml` - Main CI/CD workflow
- `.github/workflows/collect-evals.yml` - Eval results collection workflow
- `benchmarks/benchmark_lib.sh` - Shared benchmark/eval utilities
- `utils/evals/` - Eval task definitions (gsm8k.yaml, math500.yaml)
- `utils/collect_eval_results.py` - Aggregates eval results into JSON/table

## Testing

Tests are located in `utils/matrix_logic/`:

- `test_validation.py` - Pydantic model validation tests
- `test_generate_sweep_configs.py` - Config generation tests
- `test_process_result.py` - Result processing tests

Run with: `python -m pytest utils/matrix_logic/ -v`

Markers available: `slow`, `integration`

## Important Notes
1. Make sure no new directories are created in `/workspace` during the benchmark. Files are ok.

## Fetching GitHub Actions Benchmark Results

When asked to analyze benchmark results from a GitHub Actions run URL, use the `gh` CLI.

### Commands
```bash
# List artifacts for a run
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/<RUN_ID>/artifacts --jq '.artifacts[].name'

# Download aggregated results
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
```
### Parsing Results (IMPORTANT: avoid dumping raw JSON)

The results JSON can be large with multiple decimal places, so avoid dumping the raw JSON. Use `jq` to extract and round to see only what you need, for example:
```bash
# Count total results
cat ./results/results_bmk/*.json | jq 'length'

# List unique hardware/framework combinations
cat ./results/agg_bmk.json | jq -r '[.[] | "\(.hw)/\(.framework)"] | unique | .[]'

# Summary table: hw, model, isl/osl, throughput (rounded)
cat ./results/agg_bmk.json | jq -r '
  .[] | [.hw, .infmax_model_prefix, "\(.isl)/\(.osl)", (.tput_per_gpu | round)] 
  | @tsv' | column -t

# Filter to specific model
cat ./results/agg_bmk.json | jq '[.[] | select(.infmax_model_prefix == "gptoss")]'

# Get single best result by throughput
cat ./results/agg_bmk.json | jq 'max_by(.tput_per_gpu)'

# Compact view with rounded values
cat ./results/agg_bmk.json | jq '
  .[] | {
    hw, framework, model: .infmax_model_prefix, 
    isl, osl, tp, ep, conc,
    tput: (.tput_per_gpu | round),
    ttft_p99: (.p99_ttft | .*100 | round | ./100),
    e2e_mean: (.mean_e2el | .*100 | round | ./100)
  }'
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
