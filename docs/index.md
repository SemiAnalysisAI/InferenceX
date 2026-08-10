# InferenceX Documentation

<div align="center">

**English** | [中文](./index_zh.md)

</div>

This is the mandatory low-context router for InferenceX work. Pick the one page that owns the task, then follow only its source links. Repository source files and workflows remain authoritative.

## Task and page index

| Page | Open it for |
| --- | --- |
| [`index.md`](./index.md) / [`index_zh.md`](./index_zh.md) | This task router and its Chinese counterpart |
| [`agent-guide.md`](./agent-guide.md) / [`agent-guide_zh.md`](./agent-guide_zh.md) | Agent onboarding, safe start, invariants, and verification |
| [`procedures.md`](./procedures.md) / [`procedures_zh.md`](./procedures_zh.md) | Routing from a recurring task to one focused operational checklist |
| [`architecture.md`](./architecture.md) / [`architecture_zh.md`](./architecture_zh.md) | Config-to-result flow, ownership boundaries, artifacts, and InferenceX-app handoff |
| [`configuration-procedures.md`](./configuration-procedures.md) / [`configuration-procedures_zh.md`](./configuration-procedures_zh.md) | Config, runner, image, recipe, llm-d, srt-slurm, and MTP changes |
| [`ci-procedures.md`](./ci-procedures.md) / [`ci-procedures_zh.md`](./ci-procedures_zh.md) | Matrix generation, validation, dispatch, PR sweeps, reuse, staging, and artifact downloads |
| [`eval-agentx-procedures.md`](./eval-agentx-procedures.md) / [`eval-agentx-procedures_zh.md`](./eval-agentx-procedures_zh.md) | Eval and AgentX selection, execution, scoring, evidence, and live-run diagnosis |
| [`results-and-ingestion.md`](./results-and-ingestion.md) / [`results-and-ingestion_zh.md`](./results-and-ingestion_zh.md) | Published-result lookup, artifact identities and schemas, app ingestion, dedupe, and provenance |
| [`recovery-results-procedures.md`](./recovery-results-procedures.md) / [`recovery-results-procedures_zh.md`](./recovery-results-procedures_zh.md) | Result processing, ingest verification and recovery, runner cleanup, and failure classification |
| [`testing.md`](./testing.md) / [`testing_zh.md`](./testing_zh.md) | Local checks, smoke runs, evidence standards, and review gates |
| [`troubleshooting.md`](./troubleshooting.md) / [`troubleshooting_zh.md`](./troubleshooting_zh.md) | Failure-layer diagnosis, known cases, safe remediation, and stop conditions |
| [`documentation-procedures.md`](./documentation-procedures.md) / [`documentation-procedures_zh.md`](./documentation-procedures_zh.md) | Adding, translating, indexing, reviewing, and maintaining documentation |
| [`PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) / [`PR_REVIEW_CHECKLIST_zh.md`](./PR_REVIEW_CHECKLIST_zh.md) | CODEOWNER review and exact sign-off requirements |
| [`DOCUMENTATION_PLAN.md`](./DOCUMENTATION_PLAN.md) / [`DOCUMENTATION_PLAN_zh.md`](./DOCUMENTATION_PLAN_zh.md) | Remaining documentation gaps, target information architecture, and rollout |

## Authoritative references

| Reference | Owns |
| --- | --- |
| [`AGENTS.md`](../AGENTS.md) | Mandatory low-context agent policy and benchmark invariants |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | PR review, CODEOWNER sign-off, sweep reuse, merge, and post-merge duties |
| [`.github/AGENT_OPERATIONS.md`](../.github/AGENT_OPERATIONS.md) | Translation terms, sweep labels, dispatch, eval selection, power, metrics, and artifacts |
| [`configs/CONFIGS.md`](../configs/CONFIGS.md) | Master-config schema, search spaces, runners, and topology fields |
| [`.github/workflows/README.md`](../.github/workflows/README.md) | Generator examples, workflow operation, and reuse policy |
| [`utils/evals/EVALS.md`](../utils/evals/EVALS.md) | Eval task, execution, collection, validation, and SWE-bench contracts |
| [`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md) | Disaggregated recipe registration and master-config coupling |
| [`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md) | Runner provisioning and setup |
| [`MODELS.md`](../MODELS.md) | Supported models, hardware coverage, and naming |
| [`KLAUD_DEBUG.md`](../KLAUD_DEBUG.md) | Historical Klaud-Cold, CI, image, cluster, and GitHub CLI failure signatures |
| [`benchmarks/single_node/agentic/README.md`](../benchmarks/single_node/agentic/README.md) | AgentX trace benchmark implementation |

## Context rules

1. Open only the focused page and source sections needed for the task.
2. Do not load large YAML, JSON, logs, generated matrices, or whole reference files when a filtered view answers the question.
3. Source code, workflow YAML, schemas, launchers, and collectors win over explanatory docs.
4. When behavior changes, update the nearest English guide first and its `_zh.md` counterpart in the same change.
