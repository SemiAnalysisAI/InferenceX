# How to Test Workflows

In order to test configurations described in `.github/configs`, the primary workflow file used is `.github/workflows/e2e-tests.yml`. As input, this workflow takes in the CLI arguments for the `utils/matrix_logic/generate_sweep_configs.py` script. The usage for this script is shown below:

```
usage: generate_sweep_configs.py [-h] {full-sweep,runner-model-sweep,test-config} ...

Generate benchmark configurations from YAML config files

positional arguments:
  {full-sweep,runner-model-sweep,test-config}
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
    test-config         Generate full sweep for specific config keys.
                        Supports wildcard patterns (* and ?) for matching
                        multiple keys at once.

options:
  -h, --help            show this help message and exit
```

## `full-sweep` Command

The `full-sweep` command generates benchmark configurations with optional filtering. You can specify `--single-node`, `--multi-node`, or both. If neither is specified, both types are generated.

```
usage: generate_sweep_configs.py full-sweep
    --config-files CONFIG_FILES [CONFIG_FILES ...]
    [--runner-config RUNNER_CONFIG]
    [--no-evals | --evals-only] [--all-evals]
    [--model-prefix MODEL_PREFIX [MODEL_PREFIX ...]]
    [--precision PRECISION [PRECISION ...]]
    [--framework FRAMEWORK [FRAMEWORK ...]]
    [--runner-type RUNNER_TYPE [RUNNER_TYPE ...]]
    [--seq-lens {1k1k,8k1k} [{1k1k,8k1k} ...]]
    [--step-size STEP_SIZE]
    [--max-conc MAX_CONC]
    [--max-tp MAX_TP]
    [--max-ep MAX_EP]
    [--single-node] [--multi-node]
```

If neither `--single-node` nor `--multi-node` is specified, both types are generated.

By default, throughput runs for every generated config and eval-only jobs run for the selected 8k1k subset. `--no-evals` disables eval jobs, `--evals-only` emits only that selected subset, and adding `--all-evals` expands it to every fixed-sequence config. `--all-evals` alone is an equivalent eval-only shorthand; it cannot be combined with `--no-evals`.

`--step-size` must be greater than 1 and applies to concurrency ranges. Explicit `conc-list` values are emitted directly and are filtered by `--min-conc` / `--max-conc` when provided; when both bounds are set, `--min-conc` must not exceed `--max-conc`.

### Examples

**Generate all single-node and multi-node configurations (default):**
```
full-sweep --config-files .github/configs/nvidia-master.yaml
```

**Test all single-node gptoss configurations on B200 with 1k1k sequence lengths:**
```
full-sweep --single-node --model-prefix gptoss --runner-type b200 --seq-lens 1k1k --config-files .github/configs/nvidia-master.yaml
```

**Test all single-node fp8 precision configs for 8k1k workloads:**
```
full-sweep --single-node --precision fp8 --seq-lens 8k1k --config-files .github/configs/nvidia-master.yaml .github/configs/amd-master.yaml
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

The `runner-model-sweep` command validates that all runner nodes of a specific type work with all model configurations. You can specify `--single-node`, `--multi-node`, or both. If neither is specified, both types are generated.

```
usage: generate_sweep_configs.py runner-model-sweep
    --config-files CONFIG_FILES [CONFIG_FILES ...]
    [--runner-config RUNNER_CONFIG]
    [--no-evals | --evals-only] [--all-evals]
    --runner-type RUNNER_TYPE
    [--runner-node-filter RUNNER_NODE_FILTER]
    [--single-node] [--multi-node]
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

## `test-config` Command

The `test-config` command generates the full sweep for one or more specific config keys. This is useful for testing individual configurations without filtering by model prefix, framework, etc.

```
usage: generate_sweep_configs.py test-config
    --config-files CONFIG_FILES [CONFIG_FILES ...]
    [--runner-config RUNNER_CONFIG]
    [--no-evals | --evals-only] [--all-evals]
    --config-keys CONFIG_KEYS [CONFIG_KEYS ...]
    [--conc CONC [CONC ...]]
```

Config keys support **wildcard patterns** using `*` (matches any characters) and `?` (matches a single character). Patterns that match no keys will raise an error.

### Examples

**Test a single config by exact name:**
```
test-config --config-keys dsr1-fp4-b200-sglang --config-files .github/configs/nvidia-master.yaml
```

**Test multiple exact configs:**
```
test-config --config-keys dsr1-fp4-b200-sglang dsr1-fp8-h200-trt --config-files .github/configs/nvidia-master.yaml
```

**Use wildcard to test all B200 configs:**
```
test-config --config-keys *-b200-* --config-files .github/configs/nvidia-master.yaml
```

**Use wildcard to test all sglang configs:**
```
test-config --config-keys *-sglang --config-files .github/configs/nvidia-master.yaml .github/configs/amd-master.yaml
```

**Use wildcard to test all dsr1 model configs:**
```
test-config --config-keys dsr1* --config-files .github/configs/nvidia-master.yaml
```

**Mix exact keys and patterns:**
```
test-config --config-keys dsr1-fp4-b200-sglang gptoss* --config-files .github/configs/nvidia-master.yaml
```

**Override concurrency for targeted testing:**
```
test-config --config-keys *-b200-* --conc 4 8 --config-files .github/configs/nvidia-master.yaml
```

**Run eval-only jobs for every generated fixed-sequence config:**
```
test-config --config-keys dsr1-fp8-h200-sglang --evals-only --all-evals --config-files .github/configs/nvidia-master.yaml
```

## PR Eval Modifiers

Use `all-evals` and/or `evals-only` with one primary sweep label. `all-evals`
covers every fixed-sequence config; each multi-node topology runs all
`conc-list` values on one engine. `evals-only` suppresses throughput; together
they run all evals only. The primary label still controls canary/fail-fast.
`all-evals` full sweeps are reusable. Runs with `evals-only`, including runs
with both modifiers, are not. Default full sweeps, including default evals,
are also reusable.

## Reusing an Approved PR Full Sweep

`[skip-sweep]` skips PR benchmark setup only; changelog and reuse checks still
run. Pushes to `main` ignore it.

After an eligible full sweep (`full-sweep-enabled`,
`non-canary-full-sweep-enabled`, or either fail-fast variant), an authorized
maintainer can comment:

```
/reuse-sweep-run
```

This selects the latest successful `run-sweep.yml` PR run whose commit remains
in the PR. A run ID can pin an eligible successful or failed run:

```
/reuse-sweep-run <run_id>
```

Failed-run artifacts must still validate. The latest matching comment by an
`OWNER`, `MEMBER`, or `COLLABORATOR` wins. Comments do not trigger or cancel
sweeps; later commits skip a new sweep after changelog/matrix validation.
Remove and re-add the sweep label to force one.

`utils/merge_with_reuse.sh <pr-number>` is the supported merge path for reuse.
It merges `main`, preserves changelog bytes, fixes an appended `XXX` PR link,
pushes a synchronization commit, waits for checks, then merges.

The main run verifies the source, validates and uploads its ingest artifacts,
then ingests them with merge-run changelog metadata. Source coverage is
authoritative, so later matrix/eval policy changes do not invalidate reuse.
Validation rejects duplicate fixed rows, missing run stats, inconsistent
agentic artifacts, malformed eval metadata, and raw/aggregate eval mismatches.
Batched evals use only `completed_eval_concs`.

Reuse fails closed when authorized but ineligible or invalid; without
authorization, `main` runs the normal full sweep.

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
