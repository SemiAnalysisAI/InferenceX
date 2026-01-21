# InferenceMAX Development Guidelines

This document provides guidelines for Claude (and other AI assistants) when working on the InferenceMAX repository.

## Performance Changelog Requirements

When making changes to benchmark configurations or scripts that affect performance benchmarks, you **MUST** follow these steps:

### When to Update perf-changelog.yaml

You must add an entry to `perf-changelog.yaml` when:

1. **Updating image tags** in `.github/configs/*-master.yaml` configuration files
2. **Updating image tags** in `benchmarks/*.sh` scripts
3. **Adding or modifying environment variables** that affect benchmark behavior
4. **Changing configuration parameters** (e.g., concurrency ranges, tensor parallelism, expert parallelism)
5. **Adding new benchmark configurations**
6. **Removing or deprecating benchmark configurations**

### perf-changelog.yaml Entry Format

Each entry in `perf-changelog.yaml` must follow this format:

```yaml
- config-keys:
    - dsr1-fp8-*-vllm  # Use wildcards to match multiple configs
  description:
    - "Update vLLM image from v0.11.2 to v0.13.0"
    - "Add VLLM_MXFP4_USE_MARLIN=1 environment variable"
  pr-link: https://github.com/InferenceMAX/InferenceMAX/pull/XXX
```

### Field Descriptions

- **config-keys**: List of configuration keys affected by this change. Use wildcards (`*`) to match multiple configurations (e.g., `dsr1-fp8-*-vllm` matches all DeepSeek R1 FP8 vLLM configs).
- **description**: List of concise descriptions explaining what changed. Each item should be a clear, actionable statement.
- **pr-link**: Link to the pull request that introduced this change.

### Examples

**Example 1: Image version update**
```yaml
- config-keys:
    - gptoss-fp4-b200-vllm
    - gptoss-fp4-h100-vllm
    - gptoss-fp4-h200-vllm
  description:
    - "Update vLLM image from v0.11.2 to v0.13.0"
    - "Add VLLM_MXFP4_USE_MARLIN=1 to H100 and H200 benchmark scripts"
  pr-link: https://github.com/InferenceMAX/InferenceMAX/pull/327
```

**Example 2: Configuration parameter changes**
```yaml
- config-keys:
    - dsr1-fp4-b200-sglang
    - dsr1-fp8-b200-sglang
    - dsr1-fp8-h200-sglang
  description:
    - "Consolidate H200 and B200 SGLang configurations to use unified v0.5.5-cu129-amd64 image tag"
    - "Update deprecated SGLang server arguments to current equivalents"
    - "Replace --enable-ep-moe with --ep-size $EP_SIZE"
  pr-link: https://github.com/InferenceMAX/InferenceMAX/pull/204
```

**Example 3: Using wildcards**
```yaml
- config-keys:
    - dsr1*
  description:
    - "Remove Llama 70B runs to make room for multi-node disagg prefill+wideEP on h100/h200/b200/mi300/mi325/mi355"
  pr-link: https://github.com/InferenceMAX/InferenceMAX/pull/149
```

## Checklist for Benchmark/Config Changes

When modifying benchmark configurations, use this checklist:

- [ ] Update the image tag in the relevant `.github/configs/*-master.yaml` file(s)
- [ ] Update the image tag in the relevant `benchmarks/*.sh` script(s) if applicable
- [ ] Update any related environment variables or configuration parameters
- [ ] **Add an entry to `perf-changelog.yaml`** with:
  - Affected config-keys (use wildcards where appropriate)
  - Clear description of all changes
  - PR link (update after PR is created)

## Repository Structure

- `.github/configs/`: Master configuration files for benchmark sweeps
  - `nvidia-master.yaml`: NVIDIA GPU configurations (H100, H200, B200, GB200)
  - `amd-master.yaml`: AMD GPU configurations (MI300X, MI325X, MI355X)
- `benchmarks/`: Benchmark launch scripts
- `perf-changelog.yaml`: Performance changelog tracking all benchmark configuration changes
