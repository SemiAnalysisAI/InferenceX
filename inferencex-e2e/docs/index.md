# InferenceX Documentation

<div align="center">

**English** | [中文](index_zh.md)

</div>

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
| Add or debug evals | [Eval and AgentX procedures](eval-agentx-procedures.md#2-add-a-graded-eval) | `EVALS.md`, eval templates, score validator |
| Run AgentX | [Eval and AgentX procedures](eval-agentx-procedures.md#7-run-agentx-fast-feedback-versus-canonical-evidence) | agentic config, trace source, live-run skill |
| Inspect a result or ingest | [Recovery and results procedures](recovery-results-procedures.md#result-pipeline-know-what-should-exist) | artifact schema, collector, app ingest workflow |
| Recover failed ingest | [Recovery and results procedures](recovery-results-procedures.md#failed-ingest-recovery) | recovery tool, source-run artifacts, ancestry rules |
| Debug a runner or workspace | [Recovery and results procedures](recovery-results-procedures.md#amd-root-owned-workspace-prevention-and-recovery) | launcher cleanup, `.claude/commands/` cluster playbooks, cluster logs |

## Task and page index

| Page | Open it for |
| --- | --- |
| [`index.md`](index.md) / [`index_zh.md`](index_zh.md) | This task router and its Chinese counterpart |
| [`architecture.md`](architecture.md) / [`architecture_zh.md`](architecture_zh.md) | Config-to-result flow, ownership boundaries, artifacts, and InferenceX-app handoff |
| [`power_model` README](../../power_model/README.md) | Installation, CLI usage, supported systems, and power-model assumptions |
| [`ci-procedures.md`](ci-procedures.md) / [`ci-procedures_zh.md`](ci-procedures_zh.md) | Matrix generation, validation, dispatch, PR sweeps, reuse, staging, and artifact downloads |
| [`eval-agentx-procedures.md`](eval-agentx-procedures.md) / [`eval-agentx-procedures_zh.md`](eval-agentx-procedures_zh.md) | Eval and AgentX selection, execution, scoring, evidence, and live-run diagnosis |
| [`agentx-standalone.md`](agentx-standalone.md) / [`agentx-standalone_zh.md`](agentx-standalone_zh.md) | Install the pinned AgentX client and replay traces against an existing server without CI or Slurm |
| [`results-and-ingestion.md`](results-and-ingestion.md) / [`results-and-ingestion_zh.md`](results-and-ingestion_zh.md) | Published-result lookup, artifact identities and schemas, app ingestion, dedupe, and provenance |
| [`recovery-results-procedures.md`](recovery-results-procedures.md) / [`recovery-results-procedures_zh.md`](recovery-results-procedures_zh.md) | Result processing, ingest verification and recovery, runner cleanup, and failure classification |
| [`testing.md`](testing.md) / [`testing_zh.md`](testing_zh.md) | Local checks, smoke runs, evidence standards, and review gates |
| [`troubleshooting.md`](troubleshooting.md) / [`troubleshooting_zh.md`](troubleshooting_zh.md) | Failure-layer diagnosis, known cases, safe remediation, and stop conditions |
| [`PR_REVIEW_CHECKLIST.md`](PR_REVIEW_CHECKLIST.md) / [`PR_REVIEW_CHECKLIST_zh.md`](PR_REVIEW_CHECKLIST_zh.md) | CODEOWNER review and exact sign-off requirements |

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
| [`klaud.md`](klaud.md) / [`klaud_zh.md`](klaud_zh.md) | Klaud Cold selection, ownership, validation and recovery |
| [`klaud-reporting.md`](klaud-reporting.md) / [`klaud-reporting_zh.md`](klaud-reporting_zh.md) | Klaud PR body, progress comments, numeric comparisons and final preflight |
| [`benchmarks/srt_agentic.sh`](../benchmarks/srt_agentic.sh) | AgentX trace replay client shared by single- and multi-node srt-slurm recipes |

## Context rules

1. Open only the focused page and source sections needed for the task.
2. Do not load large YAML, JSON, logs, generated matrices, or whole reference files when a filtered view answers the question.
3. Source code, workflow YAML, schemas, launchers, and collectors win over explanatory docs.
4. When behavior changes, update the nearest English guide first and its `_zh.md` counterpart in the same change.
