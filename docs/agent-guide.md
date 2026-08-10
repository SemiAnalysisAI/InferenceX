# Agent Guide

<div align="center">

**English** | [中文](./agent-guide_zh.md)

</div>

This is the short orientation page for agents modifying InferenceX. [`AGENTS.md`](../AGENTS.md) is the mandatory policy layer. This page explains how to enter the larger documentation set without loading it all.

## Safe start

1. Confirm the repository root, branch, worktree, and `git status`. Preserve unrelated changes. Use a dedicated `.worktrees/<name>` worktree for multi-file work.
2. Read [`AGENTS.md`](../AGENTS.md). Read [`CONTRIBUTING.md`](../CONTRIBUTING.md) before review, sweep, merge, or Runner policy work.
3. Classify the task with [`index.md`](./index.md). Open the architecture, result, testing, or troubleshooting reference that owns a question.
4. For a recurring operational change, use [`procedures.md`](./procedures.md) to open exactly one focused checklist.
5. Follow that page's source links and trace changed fields or symbols into their implementation consumers. Source files and Workflow YAML win over explanatory docs.
6. Run the narrowest proof before dispatching GPU or CI work. Record Run IDs, attempts, source SHAs, and artifact names when external evidence is produced.

## Responsibility boundaries

| Need | Owner |
| --- | --- |
| Mandatory Agent policy and benchmark invariants | [`AGENTS.md`](../AGENTS.md) |
| Task and reference routing | [`index.md`](./index.md) |
| Recurring operational checklists | [`procedures.md`](./procedures.md) |
| PR review, sweep reuse, and merge policy | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Verification depth and evidence standards | [`testing.md`](./testing.md) |
| Failure classification and safe remediation | [`troubleshooting.md`](./troubleshooting.md) |

## Context discipline

Load the one page that owns the task and only the source sections needed for the current step. Avoid dumping large YAML, JSON, logs, generated matrices, or entire reference files into context. Put durable rationale and recurring pitfalls in the focused guide so later agents can find them without repeating repository-wide exploration.
