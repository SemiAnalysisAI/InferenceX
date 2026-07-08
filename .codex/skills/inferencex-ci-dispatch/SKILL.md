---
name: inferencex-ci-dispatch
description: Dispatch, monitor, debug, and collect artifacts for InferenceX GitHub Actions benchmark workflows. Use when Codex needs to trigger e2e-tests.yml or run-sweep CI, inspect SemiAnalysisAI/InferenceX Actions runs, validate generate_sweep_configs.py commands before dispatch, fetch benchmark/eval artifacts, or diagnose missing throughput, agentic server errors, checkout EACCES, runner assignment, or failed multinode/agentic CI jobs.
---

# InferenceX CI Dispatch

Use this skill for GitHub Actions backed InferenceX benchmarks. Keep CI behavior separate from local Slurm replay knobs; local-only overrides belong in `benchmarks/multi_node/local_runner/` and the `inferencex-local-bench` skill.

## Quick Workflow

1. Confirm the repo, branch, and intended config.
   - Default repo: `SemiAnalysisAI/InferenceX`.
   - Confirm current branch with `git status --short --branch`.
   - Fetch the target branch before comparing or force-updating.

2. Validate matrix generation locally before dispatch.
   - Use `python3 utils/matrix_logic/generate_sweep_configs.py ...`.
   - For targeted config checks, prefer `test-config`.
   - For full dispatch commands, run the exact `full-sweep ...` command locally first.

3. Dispatch CI only after the generated matrix contains the expected fields.
   - For manual one-offs, use `.github/workflows/e2e-tests.yml`.
   - For PR sweeps, labels drive `run-sweep.yml`; do not try to workflow-dispatch it.

4. Monitor the run and inspect failure logs before changing code.
   - Use `gh run list`, `gh run watch`, and `gh run view --log-failed`.
   - If UI says success but benchmark has `server internal error` or missing throughput, download artifacts and inspect result JSON, raw agentic logs, server logs, and run stats.

5. Preserve useful logs.
   - Download `results_bmk`, `results_all`, `eval_results_all`, `run-stats`, and any `benchmark_artifacts` or server-log artifacts.
   - Parse JSON with `jq`; do not dump raw aggregate JSON into the conversation.

## Commands

For exact command templates and artifact parsing snippets, read [references/commands.md](references/commands.md).

## Debug Checklist

For common failure modes and what to inspect first, read [references/debug-checklist.md](references/debug-checklist.md).

## Branch Hygiene

- If history is messy, compare against `origin/main`, not just the last remote branch head.
- Preserve `perf-changelog.yaml` whitespace; append only when a changelog entry is required.
- Use `--force-with-lease` only after fetching the remote target and confirming nobody else pushed new commits.
- Keep local replay helpers out of CI paths unless the user explicitly asks to productize them.
