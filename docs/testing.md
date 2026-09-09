<div align="center">

**English** | [中文](./testing_zh.md)

</div>

# Testing

Use the narrowest check that can falsify the change, then widen only when the changed contract crosses more layers. Local checks are fast proof of syntax, schema, generation, and transformation behavior. GPU execution is proof of allocation, serving, workload, and artifact behavior. A smoke run reduces feedback time, but it is not the full-sweep-and-eval evidence required for merge review.

## Index

- [Sources of truth](#sources-of-truth)
- [Testing layers](#testing-layers)
- [Test quality](#test-quality)
- [Local checks](#local-checks)
- [Smoke, sweep, and eval](#smoke-sweep-and-eval)
- [Evidence standard](#evidence-standard)
- [Review gates](#review-gates)
- [Stop conditions](#stop-conditions)

## Sources of truth

- [`.github/AGENT_OPERATIONS.md`](../.github/AGENT_OPERATIONS.md#sweep-labels-and-reuse) defines sweep labels and modifiers. Its [dispatch section](../.github/AGENT_OPERATIONS.md#workflow-dispatch-and-monitoring) defines manual runs and artifact inspection.
- [`docs/configuration-procedures.md`](./configuration-procedures.md#validate) is the focused configuration validation procedure.
- [`.github/workflows/README.md`](../.github/workflows/README.md) documents matrix generation, `e2e-tests.yml`, PR sweeps, and reuse.
- [`run-sweep.yml`](../.github/workflows/run-sweep.yml) is the executable PR/push gate. [`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) is the manually dispatched end-to-end path.
- [`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) is the merge-review standard. [The verifier prompt](../.github/codeowner-signoff-verify-prompt.md#check-1--a-passing-sweep--evals-ran-on-a-commit-in-this-pr) states how sweep and eval evidence is independently checked.

These sources outrank this guide when behavior changes. Update the English page first, then translate the same structure and evidence into this page's Chinese counterpart.

## Testing layers

| Layer | What it can prove | What it cannot prove |
| --- | --- | --- |
| Parse and syntax | Edited YAML loads, and edited Bash parses | Schema validity, runtime routing, or GPU behavior |
| Schema and matrix | A config key validates and emits the intended matrix fields | Runner availability, server startup, or performance |
| Focused Python tests | Changed generator, changelog, result, eval, collection, or reuse contracts behave on covered inputs | Container, accelerator, network, or Slurm behavior |
| Smoke run | One tightly filtered path allocates, starts a server, runs a workload, and emits artifacts | The complete concurrency/search space or merge eligibility |
| Trimmed PR sweep | Each selected single-node group runs its lowest concurrency (`sweep-enabled`) | Intermediate concurrency points required by a full sweep |
| Full sweep and eval | The selected untrimmed matrix and eval jobs execute on the reviewed commit | Correctness of evidence that was not inspected, or unrelated configurations |

A green later layer does not erase missing earlier evidence. For example, a green collector can aggregate an empty set, so review must inspect the underlying executed jobs and artifacts.

## Test quality

Tests protect behavior, not coverage numbers. During review, ask what plausible bug each test would catch and whether it exercises the implementation that ships.

- Prefer a small input with a hand-worked expected result, including relevant boundary, malformed-input, or failure cases. Do not copy the implementation's calculation or call the same helper to produce the expected result.
- Do not snapshot the current recipe count, model/hardware inventory, image pin, enum definition, or source text. Adding a valid recipe or refactoring equivalent code should not force unrelated assertion changes.
- Preserve genuine contracts: numerical results, rejected invalid inputs, stable artifact formats, and agreement between independently consumed configurations. Assert only the parts of the contract the consumer needs.
- Mock external services or processes when necessary, but run the actual behavior under test. A copied parser, filter, or fake implementation cannot detect a regression in the real one.
- Delete redundant tests without replacement. Extend existing fixtures only when there is a meaningful gap; do not build a new test framework to preserve a test count.

See [Randy Coulman's Tautological Tests](https://randycoulman.com/blog/2016/12/20/tautological-tests/) for the distinction between independent expectations and assertions that merely repeat the implementation.

## Local checks

Run checks from the repository root and replace placeholders with the exact changed path or key.

### Parse and syntax

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('configs/runners.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
bash -n benchmarks/<path>/<script>.sh
bash -n runners/launch_<cluster>.sh
```

Parsing is only the first gate. Do not report a YAML parse as matrix validation.

### Exact config, then filtered family

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --config-keys <exact-key>

uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --model-prefix <prefix> \
  --framework <framework> \
  --precision <precision> \
  --runner-type <runner> \
  --seq-lens 1k1k 8k1k
```

Inspect the emitted values, not only the exit code or row count: config key, model, image, runner, scenario, concurrency, `max-model-len`, TP/PP/EP/DCP/PCP, prefill/decode workers, hardware, router, KV transfer, eval flags, `additional-settings`, and `spec-decoding`. The schema lives in [`validation.py`](../utils/matrix_logic/validation.py), and the generator is [`generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py).

### Focused suites by changed contract

| Change | Focused command |
| --- | --- |
| Matrix schema or generation | `python -m pytest utils/matrix_logic/ -v` |
| Changelog content or PR gating | `python -m pytest utils/test_process_changelog.py utils/changelog_gate_tests/ -v` |
| Result processing | `python -m pytest utils/test_process_result.py utils/test_aggregate_power.py utils/test_calc_success_rate.py -v` |
| Eval dispatch, batching, or patches | `python -m pytest utils/evals/ -v` |
| Eval collection | `python -m pytest utils/test_collect_eval_results.py -v` |
| Sweep reuse or reusable artifacts | `python -m pytest utils/test_find_reusable_sweep_run.py utils/test_validate_reusable_sweep_artifacts.py -v` |

For an edited changelog, also run the same matrix-compatibility validator used by setup, with real base and head refs:

```bash
python3 utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref <base-ref> \
  --head-ref <head-ref>
```

Its contract is implemented in [`validate_perf_changelog.py`](../utils/validate_perf_changelog.py). This check validates the generated matrix and rejects prohibited content changes, but whitespace-only historical deletions can be invisible to its diff reader. Inspect the exact byte diff as a separate evidence gate. Do not rewrite or normalize historical `perf-changelog.yaml` bytes.

A local matrix cannot prove Slurm allocation or llm-d endpoint discovery. Multi-node recipe changes still require the upstream recipe checker and an execution on the intended fleet, as described in [configuration validation](./configuration-procedures.md#validate).

## Smoke, sweep, and eval

### Smoke

A smoke run is a manually dispatched [`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) run whose generator command is restricted to the changed model/framework/runner/scenario and a minimal supported concurrency. Generate that exact command locally first. Use smoke evidence to answer a narrow question such as “does this image start this server on this runner and produce a result artifact?”

A smoke run is not merge evidence: it intentionally omits configurations and concurrency points. Likewise, `agentx-fast` is an AgentX preflight, not a canonical AgentX result. The [dispatch reference](../.github/AGENT_OPERATIONS.md#workflow-dispatch-and-monitoring) defines the reduced warmup/profile behavior.

### Trimmed and full sweeps

- `sweep-enabled` trims each parallelism group to its lowest concurrency and is the default for most PR feedback.
- `full-sweep-fail-fast` is the recommended full-sweep label. It uses the sequential single-node canary and stops each matrix after that matrix's first failure while preserving completed results.
- Use a no-canary full-sweep label only when the canary is known to be flaky or unrepresentative. Use `full-sweep-enabled` instead of fail-fast only when every matrix job must continue despite a failure.
- Apply exactly one primary sweep label. Modifier-only or conflicting primary labels do not constitute a valid sweep.

The current meanings and eligibility rules are defined in the [sweep-label reference](../.github/AGENT_OPERATIONS.md#sweep-labels-and-reuse) and implemented by [`run-sweep.yml`](../.github/workflows/run-sweep.yml).

### Eval

Throughput and evals are separate jobs. The default sweep evaluates the selected 8k1k subset. `all-evals` expands eval selection, and `evals-only` suppresses throughput. Choose modifiers from the changed scope, but do not substitute an eval-only or preflight run for the required full sweep.

Eval completion is not just a green job. Preserve and inspect `meta_env.json`, the `results*.json` files, the score-validation output, the inference image, and the aggregated eval artifact. [`utils/evals/EVALS.md`](../utils/evals/EVALS.md) owns task and artifact behavior. [`validate_scores.py`](../utils/evals/validate_scores.py) rejects missing result files, below-threshold scores, and runs with no checked metrics. When expected concurrency metadata is available, it also rejects invalid/incomplete/failed batches. The workflow invokes it without `--expected-concs`, so reviewers must verify `meta_env.json` independently for single-concurrency artifacts.

## Evidence standard

Record enough information for another reviewer to reproduce the claim without guessing:

1. Exact commit SHA and whether it is still in the PR.
2. Exact local command or workflow generator command, including every filter and modifier.
3. Workflow URL, run ID, attempt, job/check name, and conclusion. Preserve failed logs before rerunning.
4. Config key plus resolved model, image, runner, framework, precision, scenario, topology, and concurrency range.
5. Artifact names and the relevant structured fields or summarized metrics. Do not paste an unbounded raw aggregate.
6. For evals, task, expected threshold, observed score, completion metadata, and proof that the evaluated image matches the PR config.
7. For a failure, record the first failing layer and the evidence that rules out earlier layers. Use the classification in [`troubleshooting.md`](./troubleshooting.md).

“CI is green,” a screenshot without a run URL, a collector success without executed jobs, or an artifact from a rebased-out commit is not sufficient evidence.

## Review gates

1. **Before GPU work:** parsing, exact-key generation, relevant focused suites, and inspection of emitted fields are green. The exact dispatch generator command has been run locally.
2. **Before broadening:** a smoke or canary proves the changed runtime path. If it fails, diagnose that layer before spending a full sweep.
3. **Before CODEOWNER sign-off:** follow [`PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md), including its code-quality, architecture, image provenance, upstream recipe, patch/waiver, chat-template, and AgentX requirements where applicable.
4. **For sweep/eval acceptance:** at least one commit currently in the PR has successful, non-skipped executed `single-node */` and `eval /` checks. A successful `collect-evals` alone is insufficient. Download the corresponding eval artifacts and confirm non-empty, passing accuracy and the same inference image. These are the executable rules in [verifier Checks 1 and 2](../.github/codeowner-signoff-verify-prompt.md#check-1--a-passing-sweep--evals-ran-on-a-commit-in-this-pr).
5. **For reuse at merge:** an authorized `OWNER`, `MEMBER`, or `COLLABORATOR` posts a whole-line `/reuse-sweep-run` command (optionally with the eligible source run ID) before the supported merge path. The verifier treats a missing or unauthorized command as a failure. See [verifier Check 4](../.github/codeowner-signoff-verify-prompt.md#check-4--reuse-sweep-command-explicitly-posted) and [the reuse procedure](../.github/workflows/README.md#reusing-an-approved-pr-full-sweep).
6. **At merge:** a CODEOWNER's exact sign-off is independently accepted by [`codeowner-signoff-verify.yml`](../.github/workflows/codeowner-signoff-verify.yml). If the PR head changes, reassess and sign the new commit evidence.
7. **After merge:** the author confirms the main-branch jobs pass, as required by [`CONTRIBUTING.md`](../CONTRIBUTING.md#after-merging).

## Stop conditions

Stop and fix or escalate instead of widening the run or approving when:

- syntax, schema, exact-key generation, changelog validation, or a focused suite fails.
- generated fields differ from the intended config even if the command exits zero.
- the runner/fleet prerequisite cannot be exercised for a multi-node change.
- a smoke or canary fails and the failure layer is still unknown.
- expected jobs are skipped, cancelled, absent, or attached only to a commit no longer in the PR.
- eval artifacts are missing, empty, incomplete, below threshold, or use a different image.
- required evidence, an upstream recipe, a waiver, or a reviewer-owned checklist fact is unknown.

Do not convert an unknown into a pass, rerun until a flake disappears without preserving the original failure, or spend a broader sweep to hide a narrower failure.
