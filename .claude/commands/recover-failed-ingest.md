---
description: Inspect result artifacts and recover official ingest without rerunning GPU benchmarks
argument-hint: <failed-run-or-job-url | pr-number> [source-run-id]
---

Run from a clean InferenceX checkout with authenticated `gh`, `git`, `jq`, and `uv`.
`$ARGUMENTS` supplies the failed run/job URL or original PR number, optionally an
explicit source run ID. Resolve every placeholder before running its command.
Inspect first; obtain authorization before reruns, PR writes, merges, or cluster changes.
Stop on unexpected command failure. Follow [repository policy](../../AGENTS.md)
and [review/merge rules](../../CONTRIBUTING.md).

## Inspect results and choose the repair

Record the exact run ID, attempt, event, branch, SHA, failed job, and first causal error.
For a supplied PR, resolve its merge commit and the matching completed `push` run
of `merge-ingest.yml` on `main`. For a supplied PR sweep URL, resolve its original
PR first; the sweep is the candidate source, not the failed publication target.
Use `run-sweep.yml` as the target workflow only for merges predating `merge-ingest.yml`.

```bash
gh run view "$RUN_ID" -R SemiAnalysisAI/InferenceX \
  --json event,headBranch,headSha,status,conclusion,attempt,jobs
gh api --paginate "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN_ID/artifacts?per_page=100" \
  --jq '.artifacts[] | [.id,.name,.created_at,.expires_at,.expired,.size_in_bytes] | @tsv'
```

| Evidence | Required checks |
| --- | --- |
| `results_bmk/agg_bmk.json` | JSON array; filter the intended model, hardware, topology, scenario, and concurrency. The collector parses JSON but does not validate or deduplicate rows. |
| `eval_results_all/agg_eval_all.json` and `eval_*` | Aggregate rows must match task, concurrency, score, and `n_eff`; retain per-config `meta_env.json`, results, and samples where needed. Collection may keep completed points from failed batches; reuse requires valid coverage. |
| `bmk_agentic_<suffix>` and `agentic_<suffix>` | Pair exact suffixes, including `_concN`; raw `profile_export.jsonl` is required. Preserve dataset identity, request accounting, and relevant `server_logs_*`. |
| Power audits | Retained diagnostics or `power_valid` alone do not prove successful requests, answer quality, or publishable power. Inspect the outcome and each required window's validity. |

Download only unexpired artifacts into a fresh `mktemp -d` directory. Check shape
with `jq 'type, length'`, then filter by identity, not array position. Stop on
missing/malformed required artifacts, zero successful requests, failed outcome or
error-rate validation, ambiguous siblings, conflicting datasets, unexplained
duplicates, or missing/unmapped identity fields. Do not edit downloaded JSON,
invent missing metrics, manually insert/delete database rows, or suppress skips.
Use the [result processors](../../inferencex-e2e/infx/results/) for current schemas
and the [public API](https://inferencex.semianalysis.com/api/v1) for published rows;
bound API reads by model, hardware, date, and run, use `curl --compressed`, and do not cache-bust.

Distinguish missing raw output (benchmark/launcher), missing aggregates (processing),
missing collection artifacts (collector), wrong dispatch IDs (handoff), and failed
ETL or mapping. Missing output is not repaired by rerunning a database ingest.

- If the correct app ingest exists and failed partway, verify its source/merge pair,
  unexpired inputs, and diagnosed cause; rerun that app run with `gh run rerun
  "$INGEST_RUN_ID" -R SemiAnalysisAI/InferenceX-app --failed`. If failed-only is
  unavailable, a full rerun of the same app run is allowed after the same checks.
  Writes are idempotent, but inspect skips, duplicate counts, and nullable eval
  dimensions rather than assuming every apparent duplicate is deduplicated.
- If publication is missing, skipped, bogus, or points at the wrong source, use the
  recovery PR below. Never rerun the failed InferenceX target workflow or job.
- For a failed reused AgentX dispatch, the existing
  [recover-reused-ingest workflow](../../.github/workflows/recover-reused-ingest.yml)
  accepts a verified `source-run-id`/`merge-run-id` pair. It is not a generic
  fixed-sequence repair. Do not create another recovery workflow.
- Rerun PR sweeps only for diagnosed flakes, not deterministic code/config errors.
  For cluster faults, use [sweep debugging](../../.agents/skills/debug-runs/SKILL.md),
  [MI355X workspace cleanup](clean-amd-mi355-runner-root-files.md), or
  [MI300X pyxis repair](debug-mi300-enroot-pyxis.md). Preserve evidence and obtain
  explicit approval before deleting shared files or changing nodes.

## Verify recovery eligibility

The target must be completed, on `main`, with event `push` and the workflow above.
A green legacy dispatch is not proof of valid data; leave bogus historical rows alone.
Resolve `FAILED_RUN_OR_JOB_URL`, `ORIGINAL_PR`, and `ORIGINAL_MERGE_SHA`:

```bash
uv run --project inferencex-e2e --extra workflows python -m infx.workflows.recover_failed_ingest \
  inspect-target "$FAILED_RUN_OR_JOB_URL" --output /tmp/infx-recovery-target.json
jq '{run_id,job_id,pr_number,merge_sha,run_attempt}' /tmp/infx-recovery-target.json
```

The inspector handles failed targets. For skipped or other legacy conclusions,
inspect run/job metadata with `gh api` and require exactly one merged PR from
`repos/SemiAnalysisAI/InferenceX/commits/$ORIGINAL_MERGE_SHA/pulls`.
For PR-number input, use `gh pr view "$ORIGINAL_PR" --json mergeCommit` and
`gh run list --workflow merge-ingest.yml --event push --commit "$ORIGINAL_MERGE_SHA"`
with `-R SemiAnalysisAI/InferenceX`; select the exact target, not the newest unrelated run.

Find completed `pull_request` runs of `run-sweep.yml` on the original PR branch.
Skip no-op reuse-gate runs without results. Require success unless the user explicitly
pinned the exact failed/cancelled source; only its completed, valid points can be recovered.

```bash
SOURCE_JSON=$(gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID")
SOURCE_HEAD_SHA=$(jq -r .head_sha <<<"$SOURCE_JSON")
SOURCE_RUN_ATTEMPT=$(jq -r .run_attempt <<<"$SOURCE_JSON")
jq -e '.event == "pull_request" and .status == "completed" and
  .path == ".github/workflows/run-sweep.yml"' <<<"$SOURCE_JSON"
gh api --paginate "repos/SemiAnalysisAI/InferenceX/pulls/$ORIGINAL_PR/commits" \
  --jq '.[].sha' | grep -Fx "$SOURCE_HEAD_SHA"
git fetch origin main
git fetch origin "+pull/$ORIGINAL_PR/head:refs/remotes/origin/source-pr-$ORIGINAL_PR"
git merge-base --is-ancestor "$SOURCE_HEAD_SHA" "refs/remotes/origin/source-pr-$ORIGINAL_PR"
git diff --name-status "$SOURCE_HEAD_SHA" "refs/remotes/origin/source-pr-$ORIGINAL_PR"
uv run --project inferencex-e2e --extra workflows python -m infx.workflows.recover_failed_ingest \
  audit-changelog --ref "$ORIGINAL_MERGE_SHA"
git diff "${ORIGINAL_MERGE_SHA}^" "$ORIGINAL_MERGE_SHA" -- inferencex-e2e/perf-changelog.yaml
```

Compare the recovered config and its image, model, recipe, runner, launcher,
arguments, and environment at the source SHA and final original PR head.
Stop if execution semantics changed. Unrelated edits and generator/eval-selection
drift do not invalidate source coverage. Require sufficient unexpired artifacts
for the entire ingest; if historical changelog bytes were malformed, recover the
intended entries from the original PR diff or stop on ambiguity.

## Create the recovery PR

Create an empty bootstrap branch/commit from current `origin/main`, then open the
PR before changing the changelog. Use a bilingual title/body and the required AI
disclosure; set `BRANCH` and `RECOVERY_PR` to its branch and number.
An authorized `OWNER`, `MEMBER`, or `COLLABORATOR` must pin the source
with `/use <run_id>` (legacy `/reuse-sweep-run <run_id>` is also accepted).
Verify acknowledgement before pushing the changelog change.

Reuse needs no sweep label. If a primary label is present, allow only one.
Never use `evals-only` or `agentx-fast` on the recovery PR, or select a source
produced with either modifier. Do not rely on `[skip-sweep]`.

Append only to `inferencex-e2e/perf-changelog.yaml`, preserving every historical
byte. Preserve the original `config-keys`, verbatim `description`, `evals-only`,
and `scenario-type`; use the new recovery PR URL. Put recovery IDs in the PR
body, not the description shown in the app. Do not later remove the recovery entry:
it is permanent publication metadata, not a transient trigger.

Commit the append, set `RECOVERY_COMMIT=$(git rev-parse HEAD)`, and validate:

```bash
(
  cd inferencex-e2e || exit 1
  uv run --extra workflows python -m infx.workflows.validate_perf_changelog \
    --changelog-file perf-changelog.yaml --base-ref origin/main --head-ref "$RECOVERY_COMMIT" || exit 1
  uv run --extra workflows python -m infx.matrix.plan \
    --changelog-file perf-changelog.yaml --base-ref origin/main --head-ref "$RECOVERY_COMMIT" \
    > /tmp/recovery-full-config.json
)
```

Require only the intended recovery scope; current matrix row counts may differ
from the source. With no uncommitted or untracked work, attach source ancestry
without changing the recovery tree:

```bash
test -z "$(git status --porcelain)" || exit 1
TARGET_PARENT=$(git rev-parse HEAD)
TARGET_TREE=$(git rev-parse HEAD^{tree})
ATTACH_SHA=$(printf 'chore: attach reusable sweep run %s\n' "$SOURCE_RUN_ID" |
  git commit-tree "$TARGET_TREE" -p "$TARGET_PARENT" -p "$SOURCE_HEAD_SHA")
git reset --hard "$ATTACH_SHA"
test "$(git rev-parse HEAD^1)" = "$TARGET_PARENT" || exit 1
test "$(git rev-parse HEAD^2)" = "$SOURCE_HEAD_SHA" || exit 1
test "$(git rev-parse HEAD^{tree})" = "$(git rev-parse HEAD^1^{tree})" || exit 1
test "$(git diff --name-only origin/main...HEAD)" = "inferencex-e2e/perf-changelog.yaml" || exit 1
git diff --check origin/main...HEAD || exit 1
git push origin "$BRANCH"
```

Never rebase, locally squash, amend, or force-push after attaching the source.
If `main` advances or conflicts, update from it and recreate the final two-parent
carrier; preserve the verified first-parent tree. Do not silently drop ancestry.

## Verify reuse and merge

```bash
gh api --paginate "repos/SemiAnalysisAI/InferenceX/pulls/$RECOVERY_PR/commits" \
  --jq '.[].sha' | grep -Fx "$SOURCE_HEAD_SHA"
test "$(gh pr diff "$RECOVERY_PR" -R SemiAnalysisAI/InferenceX --name-only)" = \
  "inferencex-e2e/perf-changelog.yaml"
gh pr checks "$RECOVERY_PR" -R SemiAnalysisAI/InferenceX --watch --fail-fast
```

Wait for the `run-sweep.yml` run at `ATTACH_SHA`, not just early comment checks.
Require passing `check-changelog` and `reuse-sweep-gate`, with `setup` and GPU jobs
skipped. Keep `ATTACH_SHA` as the PR head. Merge only with explicit authorization;
never bypass failed or pending checks. Admin merge is allowed only when passing
checks leave repository policy as the sole blocker. GitHub squash merge is safe
because it does not rewrite the PR branch's recorded ancestry.

## Verify the exact publication

Find the recovery PR's merge SHA, then its `merge-ingest.yml` push run.
Require successful reuse resolution, merge `changelog-metadata` upload, and
dispatch of the verified source/recovery run pair. The source owns measurements
and source attempt; the merge run owns changelog/publication metadata.
Artifact preparation must choose the newest unexpired upload of each exact
source artifact name and replace only changelog metadata with the merge copy.

From an InferenceX-app checkout, a read-only selection check is:

```bash
SOURCE_RUN_ID="$SOURCE_RUN_ID" MERGE_RUN_ID="$RECOVERY_RUN_ID" \
  INGEST_REPO=SemiAnalysisAI/InferenceX bun run admin:db:prepare:ci --dry-run
```

Locate the matching app `repository_dispatch` run for `ingest-results` or
`ingest-agentic-results` (production), not the most recent run or a green dispatch
alone. Read its preparation logs and require exact `Source run:` and `Merge run:`
IDs plus the intended attempts; avoid substring matches against longer IDs.
Watch that app run to completion.

Require expected artifact/row counts, successful migrations and ingestion,
source-attributed rows, run overrides, `admin:db:verify`, and cache invalidation.
Explain skips and duplicates; resolve unmapped entities, conflicting/missing
dataset provenance, and missing AgentX trace siblings. Check nullable eval-key
dimensions and existing trace links before calling repeated rows new data.
Power diagnostics cannot substitute for benchmark outcome checks.

Report the failed run/job, original PR, source run/attempt/SHA, recovery PR and
merge run, exact app ingest run, recovered counts, and remaining verification errors.
