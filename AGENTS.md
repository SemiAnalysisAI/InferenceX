# AGENT.md

This file provides guidance for AI agents working with the InferenceX codebase.

## Project Overview

InferenceX is an open-source, automated benchmarking system that continuously tracks LLM inference performance across different hardware platforms (NVIDIA B200/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results are published to https://inferencex.com/.

## Directory Structure

```
.
├─AGENTS.md                         # agent instructions
├─perf-changelog.yaml               # benchmark trigger log; append-only; preserve whitespace
├─benchmarks/
│ ├─benchmark_lib.sh                # shared benchmark/eval/server helpers
│ ├─single_node/                    # single-node benchmark entrypoints
│ │ ├─agentic/                      # agentic benchmark scripts
│ │ ├─chat_templates/               # model chat templates, e.g. DeepSeek-V4 thinking
│ │ ├─*_mtp.sh                      # MTP/spec-decoding scripts
│ │ └─*.sh                          # per model/precision/hardware/framework scripts
│ └─multi_node/                     # multinode benchmark entrypoints
│   ├─agentic_srt.sh
│   ├─amd_utils/                    # AMD multinode Slurm/server/bench helpers
│   │ ├─bench.sh
│   │ ├─env.sh
│   │ ├─job.slurm
│   │ ├─models.yaml
│   │ ├─server.sh
│   │ ├─submit.sh
│   │ └─sync.py
│   ├─*_sglang-disagg.sh            # SGLang disaggregated multinode scripts
│   ├─*_dynamo-trt.sh               # Dynamo/TensorRT multinode scripts
│   └─srt-slurm-recipes/            # checked-in external recipe YAMLs
│     ├─sglang/deepseek-v4/8k1k/
│     └─vllm/deepseek-v4/8k1k/
├─runners/                          # hardware launcher scripts
├─utils/
│ ├─matrix_logic/                   # benchmark matrix generation/validation/tests
│ │ ├─generate_sweep_configs.py     # full-sweep/test-config CLI
│ │ ├─validation.py                 # Pydantic schemas
│ │ ├─test_generate_sweep_configs.py
│ │ └─test_validation.py
│ ├─bench_serving/                  # serving benchmark client
│ │ ├─benchmark_serving.py
│ │ ├─backend_request_func.py
│ │ ├─benchmark_utils.py
│ │ ├─encoding_dsv4.py
│ │ └─KNOWN_LIMITATION.md
│ ├─evals/                          # lm-eval task configs and score validation
│ │ ├─EVALS.md
│ │ ├─gsm8k.yaml
│ │ ├─gpqa_diamond.yaml
│ │ ├─thresholds.json
│ │ ├─utils.py
│ │ └─validate_scores.py
│ ├─agentic-benchmark/              # agentic benchmark collection/analysis helpers
│ ├─trace-replay/                   # trace replay utilities
│ ├─constants.py
│ ├─collect_results.py
│ ├─collect_eval_results.py
│ ├─compare_results.py
│ ├─calc_success_rate.py
│ ├─process_result.py               # benchmark aggregation/normalization
│ ├─process_agentic_result.py
│ ├─process_changelog.py            # perf-changelog parsing and trim_conc
│ ├─summarize.py                    # markdown summary generation
│ └─test_process_result.py
└─experimental/                     # non-core experiments
```

## Terminology

- **STP (Single Token Prediction)**: Standard autoregressive decoding where one token is generated per forward pass. No speculative decoding or MTP (Multi-Token Prediction) is used. When a benchmark is labeled "STP only", it means vanilla decoding without any speculation.
- **MTP (Multi-Token Prediction)**: A technique where the model predicts multiple tokens per forward pass, typically using speculative decoding methods like EAGLE or NEXTN.

## Key Technologies

- Python 3.13: Core automation and config generation
- Pydantic Configuration validation (V2 with strict mode)
- Bash**: Benchmark execution and infrastructure orchestration
- YAML: Configuration files
- GitHub Actions: CI/CD workflows
- Evals: lm-eval validation of benchmark results
- pytest: Testing framework

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

When asked to do a run or a sweep,
```
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='<ref>' \
  -f 'inputs[ref]=<input ref>' \
  -f 'inputs[test-name]=<name>' \
  -f 'inputs[generate-cli-command]=command'
```
Input meanings:

* ref: workflow ref to dispatch from; usually the branch containing the workflow.
* inputs[ref]: checkout ref used by jobs and matrix generation.
* inputs[test-name]: display name in GitHub Actions.
* inputs[generate-cli-command]: arguments passed to utils/matrix_logic/generate_sweep_configs.py. Can be tested locally.

To monitor: `gh run watch <RUN_ID> --repo SemiAnalysisAI/InferenceX --exit-status`

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
  scenarios:
    fixed-seq-len:
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

Evals are optional accuracy checks that ensure inference optimizations do not degrade model outputs. Keep detailed eval reference material in `utils/evals/EVALS.md`; this top-level file should only carry the essentials needed during routine agent runs.

Quick pointers:
- Eval selection is marked by `mark_eval_entries()` in `utils/matrix_logic/generate_sweep_configs.py`.
- Eval workflow jobs run separately from throughput jobs in eval-only mode (`EVAL_ONLY=true`).
- Generate normal configs with eval markings by default, skip evals with `--no-evals`, or generate only eval jobs with `--evals-only`.
- Benchmark/eval helpers live in `benchmarks/benchmark_lib.sh`; aggregated eval output is produced by `utils/collect_eval_results.py`.

### CLI

```bash
# Generate configs (evals marked by default on 8k1k subset)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml

# Generate throughput-only configs (skip evals)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --no-evals

# Generate only the eval subset (excludes non-eval configs)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --evals-only
```

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
