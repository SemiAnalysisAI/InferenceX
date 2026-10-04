# How to Test Workflows

Run end-to-end commands from `inferencex-e2e/`, which owns `pyproject.toml`, `uv.lock`, and `.python-version`. Workflow dispatch generator arguments also resolve paths relative to that directory. The project-local `.python-version` selects its Python version.

In order to test configurations described in `inferencex-e2e/configs`, the primary workflow file used is `.github/workflows/e2e-tests.yml`. As input, this workflow takes in the CLI arguments for the `python -m infx.matrix.generate` command. The command usage is shown below:

```
usage: python -m infx.matrix.generate [-h] {full-sweep,test-config} ...

Generate benchmark configurations from YAML config files

positional arguments:
  {full-sweep,test-config}
                        Available commands
    full-sweep          Generate full sweep configurations with optional
                        filtering by model, precision, framework, runner type,
                        and sequence lengths
    test-config         Generate full sweep for specific config keys.
                        Validates that all specified keys exist before
                        generating.

options:
  -h, --help            show this help message and exit
```

## `full-sweep` Command

The `full-sweep` command generates benchmark configurations with optional filtering. You can specify `--single-node`, `--multi-node`, or both. If neither is specified, both types are generated.

```
usage: python -m infx.matrix.generate full-sweep
    --config-files CONFIG_FILES [CONFIG_FILES ...]
    [--runner-config RUNNER_CONFIG]
    [--no-evals | --evals-only] [--all-evals]
    [--smoke] [--trim-conc]
    [--runner-node-filter RUNNER_NODE_FILTER]
    [--scenario-type {fixed-seq-len,agentic-coding} [{fixed-seq-len,agentic-coding} ...]]
    [--model-prefix MODEL_PREFIX [MODEL_PREFIX ...]]
    [--precision PRECISION [PRECISION ...]]
    [--framework FRAMEWORK [FRAMEWORK ...]]
    [--runner-type RUNNER_TYPE [RUNNER_TYPE ...]]
    [--seq-lens {1k1k,8k1k} [{1k1k,8k1k} ...]]
    [--step-size STEP_SIZE]
    [--min-conc MIN_CONC]
    [--max-conc MAX_CONC]
    [--max-tp MAX_TP]
    [--max-ep MAX_EP]
    [--single-node] [--multi-node]
```

If neither `--single-node` nor `--multi-node` is specified, both types are generated.

By default, throughput runs for every generated config and eval-only jobs run for the selected 8k1k subset and the AgentX GSM8K subset. `--no-evals` disables eval jobs, `--evals-only` emits only that selected subset, and adding `--all-evals` expands it to every fixed-sequence config. `--all-evals` alone is an equivalent eval-only shorthand, but it cannot be combined with `--no-evals`.

`--step-size` must be greater than 1 and applies to concurrency ranges. Explicit `conc-list` values are emitted directly and are filtered by `--min-conc` / `--max-conc` when provided. When both bounds are set, `--min-conc` must not exceed `--max-conc`.

`--trim-conc` (the `trim-conc` input of `e2e-tests.yml`) keeps only the minimum concurrency of every generated single- and multi-node deployment shape, after eval selection, for a lowest-concurrency smoke run; in changelog-ref mode only throughput rows are trimmed, and evals keep their selected concurrency. No PR label enables trimming.

### Examples

**Generate all single-node and multi-node configurations (default):**
```
full-sweep --config-files configs/nvidia-master.yaml
```

**Test all single-node dsr1 configurations on B200 with 8k1k sequence lengths:**
```
full-sweep --single-node --model-prefix dsr1 --runner-type b200 --seq-lens 8k1k --config-files configs/nvidia-master.yaml
```

**Test all single-node fp8 precision configs for 8k1k workloads:**
```
full-sweep --single-node --precision fp8 --seq-lens 8k1k --config-files configs/nvidia-master.yaml configs/amd-master.yaml
```

**Test all single-node TRT configs on H200 runners:**
```
full-sweep --single-node --framework trt --runner-type h200 b200-trt --config-files configs/nvidia-master.yaml
```

**Test specific single-node model on specific hardware with specific sequence lengths:**
```
full-sweep --single-node --model-prefix dsr1 --runner-type b200 --precision fp4 --framework sglang --seq-lens 8k1k --config-files configs/nvidia-master.yaml
```

**Limit concurrency and parallelism for faster testing:**
```
full-sweep --single-node --max-conc 64 --max-tp 4 --config-files configs/nvidia-master.yaml
```

**Test all multi-node configurations:**
```
full-sweep --multi-node --config-files configs/nvidia-master.yaml
```

**Test agentic configurations:**
```
full-sweep --scenario-type agentic-coding --config-files configs/nvidia-master.yaml configs/amd-master.yaml
```

## `test-config` Command

The `test-config` command generates the full sweep for one or more specific config keys. This is useful for testing individual configurations without filtering by model prefix, framework, etc.

```
usage: python -m infx.matrix.generate test-config
    --config-files CONFIG_FILES [CONFIG_FILES ...]
    [--runner-config RUNNER_CONFIG]
    [--no-evals | --evals-only] [--all-evals]
    [--smoke] [--trim-conc]
    [--runner-node-filter RUNNER_NODE_FILTER]
    [--scenario-type {fixed-seq-len,agentic-coding} [{fixed-seq-len,agentic-coding} ...]]
    --config-keys CONFIG_KEYS [CONFIG_KEYS ...]
    [--conc CONC [CONC ...]]
    [--exp-names EXP_NAMES [EXP_NAMES ...]]
    [--seq-lens {1k1k,8k1k} [{1k1k,8k1k} ...]]
```

Config keys support **wildcard patterns** using `*` (matches any characters) and `?` (matches a single character). Patterns that match no keys will raise an error.

### Examples

**Test a single config by exact name:**
```
test-config --config-keys dsr1-fp4-b200-sglang --config-files configs/nvidia-master.yaml
```

**Test multiple exact configs:**
```
test-config --config-keys dsr1-fp4-b200-sglang dsr1-fp8-h200-trt --config-files configs/nvidia-master.yaml
```

**Use wildcard to test all B200 configs:**
```
test-config --config-keys *-b200-* --config-files configs/nvidia-master.yaml
```

**Use wildcard to test all sglang configs:**
```
test-config --config-keys *-sglang --config-files configs/nvidia-master.yaml configs/amd-master.yaml
```

**Use wildcard to test all dsr1 model configs:**
```
test-config --config-keys dsr1* --config-files configs/nvidia-master.yaml
```

**Mix exact keys and patterns:**
```
test-config --config-keys dsr1-fp4-b200-sglang qwen3.5* --config-files configs/nvidia-master.yaml
```

**Override concurrency for targeted testing:**
```
test-config --config-keys *-b200-* --conc 4 8 --config-files configs/nvidia-master.yaml
```

**Run eval-only jobs for every generated fixed-sequence config:**
```
test-config --config-keys dsr1-fp8-h200-sglang --evals-only --all-evals --config-files configs/nvidia-master.yaml
```

## PR Sweep Labels

`run-sweep.yml` sweeps only same-repository PRs that change
`inferencex-e2e/perf-changelog.yaml`, using the appended entries as the matrix.
Fork PRs use the [trusted dispatch](#trusted-external-fork-sweep-dispatch-poc)
instead. Apply exactly one primary sweep label; more than one fails
`check-changelog`.

| Label | Canary | Per-matrix fail-fast |
| --- | --- | --- |
| `full-sweep-fail-fast` (recommended) | Yes | Yes |
| `full-sweep-enabled` | Yes | No |
| `non-canary-full-sweep-enabled` | No | No |

No label trims concurrency. For canary selection (some sweeps have no
candidate) and label-change cancellation, see
[CI procedures](../../inferencex-e2e/docs/ci-procedures.md#pr-primary-and-modifier-labels).

## PR Eval Modifiers

Use `all-evals` and/or `evals-only` with one primary sweep label. `full-sweep-fail-fast` is the strongly recommended primary. Use `full-sweep-enabled` only when jobs must keep running past a failure. `all-evals`
covers every fixed-sequence config. Each multi-node topology runs all
`conc-list` values on one engine. `evals-only` suppresses throughput. Together
they run all evals only. The primary label still controls canary/fail-fast.
Default full sweeps, including their default evals, and `all-evals` sweeps are
reusable; [reuse is rejected](#reusing-an-approved-pr-full-sweep) while the PR
carries `evals-only`, alone or with `all-evals`. Either modifier fails
`check-changelog` when the changelog additions include `append-only: true` or
`no-evals: true` entries.

## AgentX Fast Mode

Add `agentx-fast` alongside one primary sweep label to run one additional
warmup request per AgentX lane after mandatory primers and a 20-minute profile
for single- and multi-node AgentX throughput jobs. Fixed-sequence throughput
and eval jobs retain their canonical settings. Adding or removing the modifier
restarts the active sweep. [Reuse is rejected](#reusing-an-approved-pr-full-sweep)
while the PR carries `agentx-fast`.

## Trusted External-Fork Sweep Dispatch (PoC)

Public-fork `pull_request` workflows receive no repository secrets, and every
`run-sweep.yml` job requires a same-repository head. For an external PR,
`run-sweep.yml` therefore runs no jobs: it neither validates the changelog nor
fans out onto GPU runners. A maintainer with `write`, `maintain`, or `admin`
permission can add any modifier labels first, then apply one primary sweep
label to approve the PR's exact current head SHA. The PR must be open and
non-draft, and dispatch is refused until GitHub reports `merge_commit_sha`;
resolve conflicts first. `trusted-external-sweep.yml` then dispatches
`e2e-tests.yml` from `main` and pins both the approved head and GitHub's merge
SHA. `e2e-tests.yml` plans the changelog matrix itself and runs it with the
trusted workflow's secrets.

The approval is revision-specific. A later push is not trusted automatically.
Remove and re-add the primary sweep label to approve the new SHA. The trusted
dispatcher never checks out or executes PR code itself.

This proof of concept produces benchmark and evaluation artifacts through the
End-to-End Tests workflow. Those runs are not yet eligible for
`/use`, which currently accepts only `run-sweep.yml` runs. The PoC
also fans out the selected matrix immediately. It does not reproduce
`run-sweep.yml`'s canary-first sequencing: no label runs a canary here, and
only `full-sweep-fail-fast` sets fail-fast, so the other two primary labels
behave identically.

## Reusing an Approved PR Full Sweep

`[skip-sweep]` skips PR benchmark setup only. Changelog and reuse checks still
run. The push-to-`main` `merge-ingest.yml` run ignores it.

An authorized maintainer can reuse an eligible completed sweep without keeping
a primary sweep label on the PR, although staging its results requires one:

```
/use <run_id>
```

Keep the command and required run ID on one line. This pins an eligible completed
`run-sweep.yml` PR run whose commit remains in the PR, including failed or cancelled
runs with usable results.

The legacy `/reuse-sweep-run <run_id>` remains equivalent. Bare `/reuse-sweep-run`
selects the latest successful eligible run automatically; bare `/use` is rejected.
Both names share authorization, validation, and reactions.

Source validation checks identity and artifacts, not full-matrix coverage.
Acceptance does not
certify a green full sweep. Verify coverage and pin the run ID when a full sweep
is required by the review process.

The latest matching comment across both names by an `OWNER`, `MEMBER`, or `COLLABORATOR` wins.
The bot reacts with 👍 after validating the request, or 👎 on rejection; details
are in the Actions run summary. Edits replace the bot's old reaction. The reuse
check posts no comment, but `/use <run_id>` also triggers `stage-results.yml`.
Staging requires a primary sweep label on the PR and a source run created while
one was applied. It posts a staging comment, or a rejection comment when the PR
has no primary label. Comments do not trigger or cancel GPU sweeps. Later commits
skip a new sweep after changelog/matrix and source-run validation. Merge-time
validation remains authoritative; an acknowledgment cannot override expired or
invalid artifacts. Reuse is rejected while the PR currently carries
`evals-only` or `agentx-fast`, checked when `/use` is acknowledged and at merge
(and by `merge_with_reuse`). On a push these labels skip the reuse gate, so a
primary label starts a fresh sweep. Source-run label history is not inspected,
so pin only runs produced without them. To force a fresh sweep after a reuse
command, remove and re-add the primary sweep label.

`uv run --extra workflows python -m infx.workflows.merge_with_reuse <pr-number>` is the supported merge path for reuse.
It merges `main`, preserves changelog bytes, fixes an appended `XXX` PR link,
pushes a synchronization commit, waits for checks, then merges.

At merge, `merge-ingest.yml` ("Merge Ingest") publishes the reused sweep. It
runs on pushes to `main` that change `inferencex-e2e/perf-changelog.yaml`. Its
single `ingest` job resolves the merge commit's PR, reuse command, and source
run with `infx.workflows.reuse`, computes the changelog delta with
`infx.matrix.plan`, and fails before uploading or dispatching anything unless
reuse is validly authorized. It then uploads merge-time `changelog-metadata`
and sends one `repository_dispatch` to InferenceX-app: `ingest-agentic-results`
(with `database-target: production`) when the delta has agentic entries,
otherwise `ingest-results`. The payload's `source-run-id` is the reused PR
`run-sweep.yml` run and its `merge-run-id` is the Merge Ingest run.

The app downloads source artifacts, keeps the newest upload for each exact
artifact name, and ingests them with changelog metadata from the merge run. The
normal ingestion code skips failed benchmark rows. Benchmark rows and public
links retain source-run provenance. Source coverage is authoritative, so later
matrix/eval policy changes do not invalidate reuse.

Reuse fails closed when authorized but ineligible or invalid. Pushes to `main`
never run a sweep: `run-sweep.yml` is PR-only, and without reuse authorization
the Merge Ingest run fails and nothing is benchmarked or ingested.

## Validation Architecture

The benchmarking system uses a strict validation methodology to ensure correctness at every stage. This is implemented in `inferencex-e2e/infx/matrix/validation.py` using Pydantic models.

### Validation Methodology

The system validates **both ends** of the configuration pipeline:

1. **Input Validation (Master Configs)**: Validates the structure of `inferencex-e2e/configs/*.yaml` files before any processing occurs
2. **Output Validation (Matrix Entries)**: Validates the generated matrix entries that are passed to workflow templates

This dual-validation approach ensures:
- No malformed configurations enter the pipeline
- No invalid parameters reach the benchmark workflows
- Workflow templates (`benchmark-tmpl.yml`, `benchmark-multinode-tmpl.yml`) can assume all inputs are valid, with no runtime validation needed

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
inferencex-e2e/configs/*.yaml
        │
        ▼
┌─────────────────────────┐
│  validate_master_config │  ← Input validation (Pydantic)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  infx.matrix.generate   │  ← Matrix generation
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
