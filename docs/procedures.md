# Operational Procedures

<div align="center">

**English** | [中文](./procedures_zh.md)

</div>

This page is a low-context index into the InferenceX procedure knowledge base. Load this page first, then open only the focused guide for the task. Do not read every procedure page by default. Each focused guide contains prerequisites, step-by-step actions, verification, stop conditions, and links to the implementation source.

The repository source, workflow YAML, launcher scripts, and result collectors are authoritative for behavior. If a guide disagrees with code, follow the code. English pages are the documentation source versions. Update English first, then translate the matching Chinese page in the same change.

## Task routing

| Task | Open first | Then inspect |
| --- | --- | --- |
| Add a model or GPU benchmark | [Configuration procedures](./configuration-procedures.md#add-a-model--hardware-recipe) | closest benchmark script, launcher, master YAML, changelog |
| Modify an existing config | [Configuration procedures](./configuration-procedures.md#change-a-master-config) | `CONFIGS.md`, validation schema, generator, runtime consumer |
| Add a runner | [Configuration procedures](./configuration-procedures.md#register-and-set-up-a-runner) | runner setup, `configs/runners.yaml`, launcher |
| Change srt-slurm or llm-d | [Configuration procedures](./configuration-procedures.md#register-an-srt-slurm-recipe) | Recipe YAML, master config, `srtctl` mapping, launcher |
| Change MTP | [Configuration procedures](./configuration-procedures.md#add-or-change-mtp) | MTP sibling, draft model, chat-template path |
| Validate a matrix | [CI procedures](./ci-procedures.md#local-matrix-generation) | generator CLI and Pydantic validation |
| Dispatch or monitor a run | [CI procedures](./ci-procedures.md#manual-end-to-end-dispatch) | `e2e-tests.yml`, run logs, artifacts |
| Prepare a PR sweep | [CI procedures](./ci-procedures.md#pr-primary-and-modifier-labels) | `run-sweep.yml`, labels, changelog delta |
| Reuse a green sweep | [CI procedures](./ci-procedures.md#artifact-reuse-and-merge-with-reuse) | reuse gate, source artifacts, merge helper |
| Add or debug evals | [Eval and AgentX procedures](./eval-agentx-procedures.md#2-add-a-graded-eval) | `EVALS.md`, eval templates, score validator |
| Run AgentX | [Eval and AgentX procedures](./eval-agentx-procedures.md#7-run-agentx-fast-feedback-versus-canonical-evidence) | agentic config, trace source, live-run skill |
| Inspect a result or ingest | [Recovery and results procedures](./recovery-results-procedures.md#result-pipeline-know-what-should-exist) | artifact schema, collector, app ingest workflow |
| Recover failed ingest | [Recovery and results procedures](./recovery-results-procedures.md#failed-ingest-recovery) | recovery tool, source-run artifacts, ancestry rules |
| Debug a runner or workspace | [Recovery and results procedures](./recovery-results-procedures.md#amd-root-owned-workspace-prevention-and-recovery) | launcher cleanup, `KLAUD_DEBUG.md`, cluster logs |
| Add or update documentation | [Documentation procedures](./documentation-procedures.md#add-a-contributor-facing-page) | nearest source file, docs index, `_zh.md` pair |

## Common sequence

Complete the [Agent Guide safe start](./agent-guide.md#safe-start), then:

1. Read only the implementation sections linked by the focused procedure. Source files, not copied prose, define behavior.
2. Run its narrowest verification before expensive CI or GPU work.
3. Record Run IDs, attempts, artifact names, source SHAs, and failure classification when it produces external evidence.
4. Update the relevant English guide and Chinese counterpart when the procedure or contract changes.

## Source playbooks

Focused guides intentionally link these existing playbooks instead of copying their full historical context:

| Playbook | Scope |
| --- | --- |
| [`add-model-hardware.md`](../.claude/commands/add-model-hardware.md) | Day-zero model/SKU recipe creation |
| [`recover-failed-ingest.md`](../.claude/commands/recover-failed-ingest.md) | Maintainer-only artifact-reuse recovery |
| [`clean-amd-mi355-runner-root-files.md`](../.claude/commands/clean-amd-mi355-runner-root-files.md) | AMD runner root-file recovery |
| [`debug-mi300-enroot-pyxis.md`](../.claude/commands/debug-mi300-enroot-pyxis.md) | MI300X cluster setup debugging |
| [`merge-prs.md`](../.claude/commands/merge-prs.md) | Maintainer PR merge coordination |
| [`fix-klaud-cron-prs.md`](../.claude/commands/fix-klaud-cron-prs.md) | Automated image-bump PR repair |
| [`nuke.md`](../.claude/commands/nuke.md) | Automated image-tag bump, one PR per recipe family |

The focused pages are the discoverable procedure layer. These source playbooks remain implementation detail or privileged maintenance paths and should be opened only when the focused guide links them.

## Context discipline

Apply the [`docs/index.md` context rules](./index.md#context-rules). Prefer a heading or line range over a whole source file, and treat large YAML, JSON, logs, generated matrices, and artifacts as targeted data sources.
