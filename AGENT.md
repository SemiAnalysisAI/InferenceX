# AGENT.md

This file provides guidance for AI agents working with the InferenceMAX codebase.

## Project Overview

InferenceMAX is an open-source, automated benchmarking system that continuously tracks LLM inference performance across different hardware platforms (NVIDIA B200/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results are published to https://inferencemax.ai/.

## Directory Structure

```
├── benchmarks/              # Shell scripts for running benchmarks
│   ├── benchmark_lib.sh     # Shared benchmarking utilities
│   └── dsr1_*.sh            # Model-specific benchmark scripts
├── runners/                 # Launch scripts for different hardware
│   ├── launch_b200-*.sh     # NVIDIA B200 launcher scripts
│   ├── launch_h100/h200-*.sh
│   └── launch_mi*.sh        # AMD MI launcher scripts
├── utils/                   # Python utilities
│   ├── matrix_logic/        # Config generation and validation
│   │   ├── generate_sweep_configs.py  # CLI for generating benchmark matrix
│   │   ├── validation.py              # Pydantic validation models
│   │   └── test_*.py                  # Unit tests
│   ├── process_result.py    # Post-processes benchmark results
│   ├── process_changelog.py # Processes perf-changelog.yaml
│   └── summarize.py         # Generates markdown summaries
├── .github/
│   ├── workflows/           # GitHub Actions CI/CD
│   │   ├── run-sweep.yml    # Main performance sweep
│   │   ├── e2e-tests.yml    # End-to-end testing
│   │   └── benchmark-tmpl.yml
│   └── configs/             # Master configuration files
│       ├── nvidia-master.yaml
│       ├── amd-master.yaml
│       └── runners.yaml
└── perf-changelog.yaml      # Triggers benchmarks on changes
```

## Key Technologies

- **Python 3.13**: Core automation and config generation
- **Pydantic**: Configuration validation (V2 with strict mode)
- **Bash**: Benchmark execution and infrastructure orchestration
- **YAML**: Configuration files
- **GitHub Actions**: CI/CD workflows
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
  --master-config .github/configs/nvidia-master.yaml \
  --runners-config .github/configs/runners.yaml

# Filter by model prefix (dsr1 or gptoss)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml \
  --runners-config .github/configs/runners.yaml \
  --model dsr1

# Filter by framework (sglang, trt, vllm, atom, dynamo-trt, dynamo-sglang)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml \
  --runners-config .github/configs/runners.yaml \
  --framework sglang

# Filter by precision (fp4, fp8)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml \
  --runners-config .github/configs/runners.yaml \
  --precision fp8

# Filter by runner type (b200, h100, h200, gb200, mi300x, mi325x, mi355x)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --master-config .github/configs/nvidia-master.yaml \
  --runners-config .github/configs/runners.yaml \
  --runner b200
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
- `1k8k` - 1024 input / 8192 output
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

### Bash

- Source shared utilities: `source benchmark_lib.sh`
- Functions: `check_env_vars()`, `wait_for_server_ready()`, `run_benchmark_serving()`
- Parameters passed via environment variables

### Git

- Conventional commit messages
- Use `[skip-sweep]` in commit message to skip benchmarks
- Changes to `perf-changelog.yaml` trigger benchmark runs

## Common Tasks

### Adding a New Benchmark Configuration

1. Add entry to `.github/configs/nvidia-master.yaml` or `amd-master.yaml`
2. Add corresponding entry to `perf-changelog.yaml` to trigger benchmark
3. Run validation: `python utils/matrix_logic/generate_sweep_configs.py full-sweep ...`

### Adding a New Runner

1. Add runner to `.github/configs/runners.yaml`
2. Create launcher script in `runners/` directory
3. Update relevant master config with new runner type

### Updating Docker Images

When upgrading Docker images in benchmark scripts:

1. Update the image tag in the relevant `benchmarks/*.sh` script(s)
2. Update any related environment variables or configuration parameters
3. **MUST**: Add an entry to `perf-changelog.yaml`:
   ```yaml
   - config-keys:
       - dsr1-fp8-*-vllm  # Use wildcards to match multiple configs
     description:
       - "Update vLLM image from v0.11.2 to v0.13.0"
       - "Add VLLM_MXFP4_USE_MARLIN=1 environment variable"
     pr-link: https://github.com/InferenceMAX/InferenceMAX/pull/XXX
   ```
4. This triggers benchmarks for affected configs and tracks performance changes

### Debugging Benchmark Failures

1. Check GitHub Actions logs for the failed job
2. Look at environment variables passed to benchmark script
3. Review benchmark script in `benchmarks/` directory
4. Check `wait_for_server_ready()` logs for server startup issues

## Key Files to Understand

- `utils/matrix_logic/validation.py` - Defines all configuration schemas
- `utils/matrix_logic/generate_sweep_configs.py` - Config generation logic
- `.github/configs/nvidia-master.yaml` - NVIDIA benchmark definitions
- `.github/workflows/run-sweep.yml` - Main CI/CD workflow
- `benchmarks/benchmark_lib.sh` - Shared benchmark utilities

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
gh api /repos/InferenceMAX/InferenceMAX/actions/runs/<RUN_ID>/artifacts --jq '.artifacts[].name'

# Download aggregated results
gh run download <RUN_ID> --repo InferenceMAX/InferenceMAX -n results_bmk -D ./results
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
