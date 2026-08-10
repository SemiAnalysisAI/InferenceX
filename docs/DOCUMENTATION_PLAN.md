# Documentation Improvement Plan

<div align="center">

**English** | [中文](./DOCUMENTATION_PLAN_zh.md)

</div>

## Decision

Build `docs/` as the durable navigation layer for InferenceX. Keep implementation contracts in code and workflow files, keep fast policy rules in `AGENTS.md`, and use linked docs pages to explain system boundaries, rationale, task playbooks, and operational recovery. Do not copy large source files into documentation.

This increment establishes [`index.md`](./index.md), [`agent-guide.md`](./agent-guide.md), the canonical procedure catalog, and the first focused architecture, result, testing, and troubleshooting guides. This plan tracks the remaining consolidation work and the acceptance bar for each page.

## Problem statement

Important information already exists, but it is distributed across agent instructions, contributor guidance, workflow notes, config references, eval documentation, recipe notes, runner setup, and troubleshooting files. Before this increment, an agent had to discover the right file before understanding the change, and `docs/` contained only the PR checklist rather than a navigation layer.

The main risks are:

- A source-of-truth rule is duplicated and becomes stale in one location.
- A config change updates the master YAML but not the recipe, changelog, launcher, or generated matrix that consumes it.
- A workflow failure is misdiagnosed because orchestration, runner, serving, eval, and collection failures look similar in the UI.
- Operational recovery knowledge remains trapped in a long agent-instruction file or historical debugging note.
- New contributor-facing pages violate the bilingual documentation rule or omit the matching Chinese link.
- Existing domain references are not consistently discoverable from `docs/`, and many are not paired with `_zh.md`. The migration must distinguish contributor-facing pages from internal implementation notes.
- AgentX documentation is inconsistent: `benchmarks/single_node/agentic/README.md` describes an unpublished experimental MVP while `MODELS.md` and active master configs expose agentic-coding coverage.

## Baseline knowledge inventory

This table records the repository state before the staged consolidation below.

| Topic | Source before this change | Gap to close |
| --- | --- | --- |
| Agent rules and repository map | `AGENTS.md` | Fast policy was comprehensive but not organized as task-oriented navigation |
| PR review and merge | `CONTRIBUTING.md`, `docs/PR_REVIEW_CHECKLIST.md` | Easy to find once known, but not linked from a central docs index |
| Recurring CI and cluster failures | `KLAUD_DEBUG.md` | Valuable recovery steps are mixed with historical failure context |
| Config schema and topology | `configs/CONFIGS.md`, `utils/matrix_logic/validation.py` | Schema reference exists, but change workflow and consumer path are separate |
| Matrix generation and sweep reuse | `.github/workflows/README.md`, `.github/workflows/*.yml` | Operational guide is under `.github`, outside the main docs path |
| Evals and score gates | `utils/evals/EVALS.md`, `utils/evals/thresholds.yaml` | Detailed reference exists, but its relationship to throughput and collection is not obvious |
| Multi-node recipes | `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md` | Recipe/master-config coupling should be visible before editing either file |
| Runner setup | `utils/runner_setup/RUNNER_SETUP.md`, `runners/` | Provisioning and runtime launcher concerns are separated |
| Model and hardware catalog | `MODELS.md`, `configs/*-master.yaml` | Public model list and runnable config list serve different audiences |
| AgentX and agentic coding | `benchmarks/single_node/agentic/README.md`, `MODELS.md`, `configs/*-master.yaml` | Status and publication claims conflict. The official trace-to-result path needs one maintained guide |
| Artifact schemas and app handoff | `utils/process_result.py`, `utils/collect_*.py`, `../InferenceX-app/.github/workflows/ingest-results.yml`, `../InferenceX-app/packages/db/src/etl/*` | Artifact identities, JSON contracts, and the boundary into InferenceX-app are not documented in this repository |

## Target information architecture

Pages should answer one question each and link to exact implementation sources.

| Page | Status | Primary question |
| --- | --- | --- |
| `docs/index.md` | Done in this change | Where do I start, and which guide owns my task? |
| `docs/procedures.md` | Done in this change | Where is the canonical checklist for every recurring configuration, CI, runner, eval, recovery, AgentX, and documentation operation? |
| `docs/agent-guide.md` | Done in this change | What must an agent know before editing or verifying a change? |
| `docs/configuration-procedures.md` | Done in this change | How do I change benchmark configurations, runners, images, recipes, and MTP without breaking the end-to-end contract? |
| `docs/ci-procedures.md` | Done in this change | How do I generate, dispatch, monitor, rerun, stage, reuse, and inspect a sweep? |
| `docs/eval-agentx-procedures.md` | Done in this change | How do I run and validate evals or AgentX while preserving evidence and provenance? |
| `docs/recovery-results-procedures.md` | Done in this change | How do I process results, verify app ingest, recover failures, and protect runner workspaces? |
| `docs/documentation-procedures.md` | Done in this change | How do I add, index, review, and maintain bilingual documentation? |
| `docs/architecture.md` | Done in this change | How does a config become a benchmark result and a published row? |
| `docs/configuration.md` | Planned | How do I add or change a config without breaking schema, topology, or changelog contracts? |
| `docs/benchmark-development.md` | Planned | How do benchmark scripts, shared helpers, launchers, and runtime env vars fit together? |
| `docs/agentx.md` | Planned | What is the current AgentX status, trace contract, execution path, and publication boundary? |
| `docs/workflows-and-sweeps.md` | Planned | How do I generate, dispatch, monitor, reuse, and collect a sweep? |
| `docs/evals.md` | Planned | How are evals selected, run, scored, validated, and collected? |
| `docs/runners-and-clusters.md` | Planned | How are runner labels, hardware facts, Slurm jobs, cleanup, and logs managed? |
| `docs/recipes-and-disaggregated.md` | Planned | How do srt-slurm recipes map to master configs and disaggregated topologies? |
| `docs/results-and-ingestion.md` | Done in this change | How do artifacts become aggregated results and dashboard data? |
| `docs/glossary.md` | Planned | Which benchmark terms, metric names, artifact fields, and abbreviations have stable meanings? |
| `docs/troubleshooting.md` | Done in this change | How do I classify a failure and choose a safe recovery path? |
| `docs/testing.md` | Done in this change | Which local checks prove a change before GPU or CI execution? |

Existing detailed references remain in place during migration. A new page should first link to the existing source and only move content when the target page can preserve the source contract and update its links.

## Rollout phases

### Phase 1, navigation and agent onboarding

- Add the docs index and agent guide.
- Link the index from `AGENTS.md` and both root README language variants.
- Make source-of-truth boundaries explicit.
- Keep the current detailed references unchanged.

### Phase 2, architecture and development flow

- Add an architecture page with the config-to-result data flow.
- Add a configuration page that joins schema, generator, changelog, and validation steps.
- Add a benchmark-development page covering shared Bash helpers, environment propagation, single-node and multi-node paths, and MTP chat-template requirements.
- Add diagrams only where they clarify ownership or transitions.

### Phase 3, operations and recovery

- Add the canonical procedures catalog, including labels, canary/fail-fast behavior, eval modifiers, artifact reuse, result downloads, staging, reruns, recovery, AgentX, and runner cleanup.
- Add workflow and sweep deep-dive pages when the procedure catalog needs more implementation detail.
- Add runners and clusters, including AMD root-file cleanup and no-new-directory constraints.
- Add recipes and disaggregated serving, with a synchronized edit checklist.
- Add troubleshooting by failure layer, linking recurring cases back to `KLAUD_DEBUG.md`.

### Phase 4, results and maintenance

- Add result processing, eval collection, and ingestion boundaries.
- Add a focused local-testing page with commands that exercise configuration, matrix, and result contracts.
- Add link and language-pair checks to the documentation maintenance workflow if the repository's CI conventions support them.
- Remove duplicated procedural text only after the new page is linked and verified.

## Maintenance contract

- English is the primary documentation source. Update English first, then translate the matching `_zh.md` page from that English version in the same change.
- Every operational claim names its source file, workflow, script, or configuration field.
- When implementation behavior changes, update the closest guide in the same change. If the change affects PR checklist policy, update the verifier prompt too.
- Prefer short task checklists and source links over copied command output or long generated tables.
- Keep historical incident detail in `KLAUD_DEBUG.md` or issue records. Keep repeatable recovery procedures in the relevant operational guide and link back to the incident context.
- Review the index whenever a new guide or a major source-of-truth file is added.

## Acceptance criteria

The documentation consolidation is complete when an agent can:

1. Find the correct guide from `docs/index.md` without searching the whole repository.
2. Trace a configuration change from master YAML through matrix generation, workflow execution, launcher/runtime behavior, and result collection.
3. Identify whether a failed job is a generation, orchestration, runner, serving, eval, collection, or ingestion failure.
4. Run a narrow local verification before dispatching an expensive sweep.
5. Find the bilingual counterpart and the implementation source for every contributor-facing guide.

The index, agent guide, canonical procedures catalog, architecture, result-ingestion, testing, and troubleshooting guides now provide the first complete navigation and operational coverage. The remaining pages stay intentionally staged so each can be grounded in the implementation it documents instead of becoming a second, drifting copy of the repository.
