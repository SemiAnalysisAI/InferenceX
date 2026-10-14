# InferenceX Documentation

This is the mandatory low-context router for InferenceX work. Pick the one page that owns the task, then follow only its source links. Repository source files and workflows remain authoritative.

Paths and shell commands in these guides are relative to `inferencex-e2e/` unless stated otherwise. The Python manifest, lockfile, and `.python-version` live in that project directory. Repository-wide policy and GitHub workflows remain at the repository root.

## Task routing

| Task | Open first | Then inspect |
| --- | --- | --- |
| Add a model or GPU benchmark | [Config reference](../configs/CONFIGS.md) | closest benchmark script, launcher, master YAML, changelog |
| Modify an existing config | [Config reference](../configs/CONFIGS.md) | validation schema, generator, runtime consumer |
| Add a runner | [Runner setup](../utils/runner_setup/RUNNER_SETUP.md) | `configs/runners.yaml`, launcher |
| Change srt-slurm or llm-d | [Recipe reference](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md) | Recipe YAML, master config, `srtctl` mapping, launcher |
| Change MTP | [Draft-model precision](../../CONTRIBUTING.md#draft-model-precision) | MTP sibling, draft model, chat-template path |
| Validate a matrix | [CI procedures](ci-procedures.md#local-matrix-generation) | generator CLI and Pydantic validation |
| Dispatch or monitor a run | [CI procedures](ci-procedures.md#manual-end-to-end-dispatch) | `e2e-tests.yml`, run logs, artifacts |
| Prepare a PR sweep | [CI procedures](ci-procedures.md#pr-primary-and-modifier-labels) | `run-sweep.yml`, labels, changelog delta |
| Reuse a green sweep | [CI procedures](ci-procedures.md#artifact-reuse-and-merge-with-reuse) | reuse gate, source artifacts, merge helper |
| Add or debug evals | [Eval reference](../infx/evals/EVALS.md) | eval templates, score validator |
| Run AgentX | [Standalone replay](agentx-standalone.md) | agentic config, trace source, [live-run skill](../../.agents/skills/debug-agentx-runs/SKILL.md) |
| Inspect a result or ingest | [Result inspection](../../.claude/commands/recover-failed-ingest.md#inspect-results-and-choose-the-repair) | artifact schema, collector, app ingest workflow |
| Recover failed ingest | [Recovery command](../../.claude/commands/recover-failed-ingest.md) | recovery tool, source-run artifacts, ancestry rules |
| Debug a runner or workspace | [Troubleshooting](troubleshooting.md) | launcher cleanup, `.claude/commands/` cluster playbooks, cluster logs |

## Task and page index

| Page | Open it for |
| --- | --- |
| [`index.md`](index.md) | This task router |
| [`power_model` README](../../power_model/README.md) | Installation, CLI usage, supported systems, and power-model assumptions |
| [`ci-procedures.md`](ci-procedures.md) | Matrix generation, validation, dispatch, PR sweeps, reuse, staging, and artifact downloads |
| [`agentx-standalone.md`](agentx-standalone.md) | Install the pinned AgentX client and replay traces against an existing server without CI or Slurm |
| [Recovery command](../../.claude/commands/recover-failed-ingest.md) | Result inspection, ingest verification, and artifact-reuse recovery |
| [`testing.md`](testing.md) | Local checks, smoke runs, evidence standards, and review gates |
| [`troubleshooting.md`](troubleshooting.md) | Failure-layer diagnosis, known cases, safe remediation, and stop conditions |
| [`PR_REVIEW_CHECKLIST.md`](PR_REVIEW_CHECKLIST.md) | CODEOWNER review and exact sign-off requirements |

## Authoritative references

| Reference | Owns |
| --- | --- |
| [`AGENTS.md`](../../AGENTS.md) | Mandatory low-context agent policy and benchmark invariants |
| [`CONTRIBUTING.md`](../../CONTRIBUTING.md) | PR review, CODEOWNER sign-off, sweep reuse, merge, and post-merge duties |
| [`.github/AGENT_OPERATIONS.md`](../../.github/AGENT_OPERATIONS.md) | Translation terms, sweep labels, dispatch, eval selection, power, metrics, and artifacts |
| [`configs/CONFIGS.md`](../configs/CONFIGS.md) | Master-config schema, search spaces, runners, and topology fields |
| [`.github/workflows/README.md`](../../.github/workflows/README.md) | Generator examples, workflow operation, and reuse policy |
| [`infx/evals/EVALS.md`](../infx/evals/EVALS.md) | Eval task, execution, collection, and validation contracts |
| [`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md) | Disaggregated recipe registration and master-config coupling |
| [`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md) | Runner provisioning and setup |
| [`MODELS.md`](MODELS.md) | Supported models, hardware coverage, and naming |
| [`klaud.md`](klaud.md) | Klaud Cold selection, ownership, validation and recovery |
| [`klaud-reporting.md`](klaud-reporting.md) | Klaud PR body, progress comments, numeric comparisons and final preflight |
| [`benchmarks/srt_agentic.sh`](../benchmarks/srt_agentic.sh) | AgentX trace replay client shared by single- and multi-node srt-slurm recipes |

## Context rules

1. Open only the focused page and source sections needed for the task.
2. Do not load large YAML, JSON, logs, generated matrices, or whole reference files when a filtered view answers the question.
3. Source code, workflow YAML, schemas, launchers, and collectors win over explanatory docs.
4. When behavior changes, update the nearest English guide. Only READMEs retain Chinese translations; update an existing `README_zh.md` alongside its English README, and do not create Chinese counterparts for other docs.
