# How to Test Workflows

In order to test configurations described in `.github/configs`, the primary workflow file used is `.github/workflows/e2e-tests.yml`. As input, this workflow takes in the CLI arguments for the `utils/matrix_logic/generate_sweep_configs.py` script. The usage for this script is shown below:

```
usage: generate_sweep_configs.py [-h] {full-sweep,runner-model-sweep} ...

Generate benchmark configurations from YAML config files

positional arguments:
  {full-sweep,runner-model-sweep}
                        Available commands
    full-sweep          Generate full sweep configurations with optional
                        filtering by model, precision, framework, runner type,
                        and sequence lengths
    runner-model-sweep  Given a runner type, find all configurations matching
                        the type, and run that configuration on all individual
                        runner nodes for the specified runner type. This is
                        meant to validate that all runner nodes work on all
                        configurations for a runner type. For instance, to
                        validate that all configs that specify an h200 runner
                        successfully run across all h200 runner nodes.

options:
  -h, --help            show this help message and exit
```

## `full-sweep` Command

The `full-sweep` command generates benchmark configurations with optional filtering. It requires specifying either `--single-node` or `--multi-node`.

```
usage: generate_sweep_configs.py full-sweep
    --config-files CONFIG_FILES [CONFIG_FILES ...]
    [--runner-config RUNNER_CONFIG]
    [--model-prefix MODEL_PREFIX [MODEL_PREFIX ...]]
    [--precision PRECISION [PRECISION ...]]
    [--framework FRAMEWORK [FRAMEWORK ...]]
    [--runner-type RUNNER_TYPE [RUNNER_TYPE ...]]
    [--seq-lens {1k1k,1k8k,8k1k} [{1k1k,1k8k,8k1k} ...]]
    [--step-size STEP_SIZE]
    [--max-conc MAX_CONC]
    [--max-tp MAX_TP]
    [--max-ep MAX_EP]
    (--single-node | --multi-node)
```

### Examples

**Test all single-node gptoss configurations on B200 with 1k1k sequence lengths:**
```
full-sweep --single-node --model-prefix gptoss --runner-type b200 --seq-lens 1k1k --config-files .github/configs/nvidia-master.yaml
```

**Test all single-node fp8 precision configs for 1k8k workloads:**
```
full-sweep --single-node --precision fp8 --seq-lens 1k8k --config-files .github/configs/nvidia-master.yaml .github/configs/amd-master.yaml
```

**Test all single-node TRT configs on H200 runners:**
```
full-sweep --single-node --framework trt --runner-type h200 b200-trt --config-files .github/configs/nvidia-master.yaml
```

**Test specific single-node model on specific hardware with specific sequence lengths:**
```
full-sweep --single-node --model-prefix dsr1 --runner-type b200 --precision fp4 --framework sglang --seq-lens 1k1k 8k1k --config-files .github/configs/nvidia-master.yaml
```

**Limit concurrency and parallelism for faster testing:**
```
full-sweep --single-node --max-conc 64 --max-tp 4 --config-files .github/configs/nvidia-master.yaml
```

**Test all multi-node configurations:**
```
full-sweep --multi-node --config-files .github/configs/nvidia-master.yaml
```

## `runner-model-sweep` Command

The `runner-model-sweep` command validates that all runner nodes of a specific type work with all model configurations. It requires specifying either `--single-node` or `--multi-node`.

```
usage: generate_sweep_configs.py runner-model-sweep
    --config-files CONFIG_FILES [CONFIG_FILES ...]
    [--runner-config RUNNER_CONFIG]
    --runner-type RUNNER_TYPE
    [--runner-node-filter RUNNER_NODE_FILTER]
    (--single-node | --multi-node)
```

### Scenario: Validating Runner Infrastructure

I just upgraded the CUDA drivers on all H200 runners and need to verify that all models that use H200 still work correctly across all H200 nodes.

Go to the GitHub Actions UI, click on the `End-to-End Tests` workflow, and enter the following command as the text input:
```
runner-model-sweep --single-node --runner-type h200 --config-files .github/configs/amd-master.yaml .github/configs/nvidia-master.yaml
```

This will run a test (just the highest available parallelism and lowest available concurrency) for each configuration that specifies the `h200` runner type, across all H200 runner nodes defined in `.github/configs/runners.yaml`.

For example, if you have configs `dsr1-fp8-h200-sglang`, `dsr1-fp8-h200-trt`, and `gptoss-fp4-h200-vllm` that all use `runner: h200`, and you have 8 H200 nodes (`h200-cw_0`, `h200-cw_1`, etc.), this will run all 3 configs on all 8 nodes (24 total test runs).

This is particularly useful when:
- You've made infrastructure changes to a specific runner type (driver updates, system configuration, Docker setup)
- You've added new runner nodes and want to validate they work with all existing model configurations
- You want to verify that all models remain compatible with a specific GPU type after system updates

### Filtering Runner Nodes

Use `--runner-node-filter` to only test a subset of runner nodes:
```
runner-model-sweep --single-node --runner-type mi300x --runner-node-filter mi300x-amd --config-files .github/configs/amd-master.yaml
```

This will only include runner nodes whose names contain "mi300x-amd"

## Validation Architecture

The benchmarking system uses a strict validation methodology to ensure correctness at every stage. This is implemented in `utils/matrix_logic/validation.py` using Pydantic models.

### Validation Methodology

The system validates **both ends** of the configuration pipeline:

1. **Input Validation (Master Configs)**: Validates the structure of `.github/configs/*.yaml` files before any processing occurs
2. **Output Validation (Matrix Entries)**: Validates the generated matrix entries that are passed to workflow templates

This dual-validation approach ensures:
- No malformed configurations enter the pipeline
- No invalid parameters reach the benchmark workflows
- Workflow templates (`benchmark-tmpl.yml`, `benchmark-multinode-tmpl.yml`) can assume all inputs are valid—no runtime validation needed

### Input Validation: Master Config Files

Master config files (e.g., `nvidia-master.yaml`, `amd-master.yaml`) are validated against strict Pydantic schemas:

- **`SingleNodeMasterConfigEntry`**: Validates single-node configurations
- **`MultiNodeMasterConfigEntry`**: Validates multi-node configurations

Each config must specify:
- Required fields: `image`, `model`, `model-prefix`, `precision`, `framework`, `runner`, `multinode`
- Sequence length configs with search spaces defining TP, EP, concurrency ranges, etc.
- Optional fields like `disagg`, `spec-decoding`, `dp-attn`

Invalid or missing fields raise immediate validation errors before any matrix generation.

### Output Validation: Matrix Entries

Generated matrix entries (the actual workflow inputs) are validated against:

- **`SingleNodeMatrixEntry`**: Matches the inputs expected by `benchmark-tmpl.yml`
- **`MultiNodeMatrixEntry`**: Matches the inputs expected by `benchmark-multinode-tmpl.yml`

These Pydantic models mirror the workflow template input definitions exactly. For example, `benchmark-tmpl.yml` expects:
```yaml
inputs:
  runner: required
  image: required
  model: required
  model-prefix: required
  precision: required
  framework: required
  ...
```

The corresponding `SingleNodeMatrixEntry` enforces these same fields with appropriate types.

### Key Design Principles

1. **No defaults in output validation**: Matrix entry models don't set defaults. Missing values must fail validation rather than silently using fallbacks.

2. **`extra='forbid'`**: Unknown fields are rejected, preventing typos or deprecated fields from slipping through.

3. **Strict typing**: Fields like `spec-decoding` use `Literal["mtp", "draft_model", "none"]` to restrict values to known options.

4. **Concurrency validation**: The system ensures either `conc-list` OR `conc-start`/`conc-end` is provided, but not both.

### Validation Flow

```
.github/configs/*.yaml
        │
        ▼
┌─────────────────────────┐
│  validate_master_config │  ← Input validation (Pydantic)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  generate_sweep_configs │  ← Matrix generation
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  validate_matrix_entry  │  ← Output validation (Pydantic)
└─────────────────────────┘
        │
        ▼
  benchmark-tmpl.yml or
  benchmark-multinode-tmpl.yml
```

## Utility Scripts

### `utils/summarize.py`

Aggregates benchmark results from a directory of JSON files and outputs a markdown summary table. Used after `collect-results.yml` downloads all artifacts.

Usage:
```bash
python utils/summarize.py <results_directory>
```

Outputs GitHub-flavored markdown tables with metrics including TTFT, TPOT, interactivity, E2EL, and throughput per GPU for both single-node and multi-node results.
