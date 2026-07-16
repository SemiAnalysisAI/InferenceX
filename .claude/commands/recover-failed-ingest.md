---
description: Recover a failed main-branch sweep ingest through the normal artifact-reuse path without rerunning GPU benchmarks
argument-hint: <failed-run-or-job-url | pr-number> [source-run-id]
---

Recover the official database ingest for a failed or skipped InferenceX
push-to-main `Run Sweep` workflow by creating a recovery PR that reuses validated
artifacts from an earlier PR sweep. Do not add a one-off recovery workflow.

Inputs from `$ARGUMENTS`:

- Use the first argument as `FAILED_RUN_OR_JOB_URL`.
- Use the optional second argument as `SOURCE_RUN_ID`; treat it as a candidate
  until all source, ancestry, scope, and artifact checks pass.

The most common invocation is a forgotten `/reuse-sweep-run` before merge, where
you are handed the original PR number and/or its `pull_request` sweep run (the
source) rather than a target URL. The failed target is then the push-to-main run
on that PR's merge commit — derive it in step 1. `inspect-target` needs a
run/job URL, not a bare ID.

Run from a clean InferenceX checkout with authenticated `gh`, `git`, `jq`, and
`python3`. Stop on any unexpected command failure.

## Safety rules

- Never rerun the failed target workflow or job.
- The target must be a completed `push` run of
  `.github/workflows/run-sweep.yml` on `main` whose official ingest did not
  complete.
- Reuse only a completed `pull_request` run of `run-sweep.yml`. Unpinned reuse
  requires success. A specifically pinned failed run is allowed only when
  artifact validation proves its available result set is internally consistent;
  only completed points are recovered.
- The source run must belong to the original PR being recovered.
- Stop if that PR changed the recovered configuration's execution semantics
  after the source SHA: image, model, recipe, runner, launcher, benchmark
  arguments, or config values. Unrelated edits and generator/eval-selection
  policy drift are allowed because source coverage is authoritative.
- Preserve all historical `perf-changelog.yaml` bytes. Append recovery entries
  only at the end.
- Keep exactly one full-sweep label on the recovery PR and pin the source run
  with `/reuse-sweep-run <run_id>` before pushing the changelog change.
- The final recovery branch head must have the recovery commit as its first
  parent, the source run SHA as its second parent, and the recovery commit's
  file tree unchanged.
- Never rebase, locally squash, force-push, or otherwise rewrite the recovery
  branch after attaching the source SHA. Those operations can remove the
  ancestry that makes the source run reusable. A GitHub squash merge after all
  checks pass is allowed because it does not rewrite the PR branch.
- Do not rely on `[skip-sweep]`. Reuse authorization suppresses PR benchmark
  work, and pushes to `main` ignore that marker.
- Never bypass failing or pending checks. Use admin merge only when all checks
  passed and repository policy is the sole blocker.
- Do not add co-author lines, generated-by text, bot branding, or attribution.

## 1. Inspect the target

Ensure `pydantic` and `pyyaml` are importable
(`python3 -c 'import pydantic, yaml'`); they are usually already present. If not,
install them — a plain `pip install` fails on PEP 668 managed Pythons, so use a
venv or `--break-system-packages`. Then inspect the target:

```bash
python3 -m pip install pydantic pyyaml  # only if the import check failed
python3 utils/recover_failed_ingest.py inspect-target \
  "$FAILED_RUN_OR_JOB_URL" \
  --output /tmp/infx-recovery-target.json

TARGET_RUN_ID=$(jq -r .run_id /tmp/infx-recovery-target.json)
TARGET_JOB_ID=$(jq -r .job_id /tmp/infx-recovery-target.json)
ORIGINAL_PR=$(jq -r .pr_number /tmp/infx-recovery-target.json)
ORIGINAL_MERGE_SHA=$(jq -r .merge_sha /tmp/infx-recovery-target.json)

gh run view "$TARGET_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX \
  --job "$TARGET_JOB_ID" --log \
  > "/tmp/infx-target-$TARGET_RUN_ID.log"
```

`inspect-target` handles completed failed runs. If the target workflow itself
was `skipped`, inspect it directly with `gh api`, identify the skipped setup or
reuse job, and resolve the merge SHA to exactly one merged PR:

```bash
TARGET_RUN_ID=<run-id>
TARGET_JSON=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$TARGET_RUN_ID")
ORIGINAL_MERGE_SHA=$(jq -r .head_sha <<<"$TARGET_JSON")
ORIGINAL_PR=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/commits/$ORIGINAL_MERGE_SHA/pulls" \
  --jq 'if length == 1 then .[0].number else error("expected one PR") end')
```

If you were given the original PR number or the source sweep run instead of a
target URL — the usual forgotten-`/reuse` case — derive the target push run from
the PR's merge commit:

```bash
ORIGINAL_PR=<pr-number>
ORIGINAL_MERGE_SHA=$(gh pr view "$ORIGINAL_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --json mergeCommit --jq .mergeCommit.oid)
gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow run-sweep.yml --event push \
  --commit "$ORIGINAL_MERGE_SHA" --limit 5 \
  --json databaseId,status,conclusion,createdAt
TARGET_RUN_ID=<matching-run-id>
```

Require event `push`, workflow path `.github/workflows/run-sweep.yml`, and branch
`main`; confirm the target is no longer running before recovering. The
disqualifying state is broader than `failure`/`skipped`: when `/reuse-sweep-run`
was forgotten before merge, setup leaves reuse disabled, the GPU jobs run (often
`cancelled` to save cost), and because `collect-results`/`collect-evals` are not
skipped, `trigger-ingest` still fires `always()` and lands a *bogus* ingest under
the target's own `run_id`. So a target showing
`trigger-ingest=success` (and concluding `success` or `cancelled`) can still hold
no valid benchmark data — recovery is required. That bogus row is keyed on the
target `run_id` and is superseded by the recovery ingest under a new `run_id`;
leave it alone. Record the original PR and root cause.

Fetch history and inspect the exact original changelog delta:

```bash
git fetch origin main
git cat-file -e "${ORIGINAL_MERGE_SHA}^{commit}"
ORIGINAL_BASE_SHA=$(git rev-parse "${ORIGINAL_MERGE_SHA}^")
python3 utils/recover_failed_ingest.py audit-changelog \
  --ref "$ORIGINAL_MERGE_SHA"
git diff "$ORIGINAL_BASE_SHA" "$ORIGINAL_MERGE_SHA" -- \
  perf-changelog.yaml
```

## 2. Select and validate the source run

Find candidates from the original PR branch when no run ID was supplied:

```bash
SOURCE_BRANCH=$(gh pr view "$ORIGINAL_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --json headRefName --jq .headRefName)

gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow run-sweep.yml --event pull_request \
  --branch "$SOURCE_BRANCH" --status completed --limit 100 \
  --json databaseId,attempt,conclusion,headSha,url
```

Skip no-op reuse-gate runs with no result artifacts. Resolve the selected
candidate:

```bash
SOURCE_RUN_ID=<candidate-run-id>
SOURCE_JSON=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID")
SOURCE_HEAD_SHA=$(jq -r .head_sha <<<"$SOURCE_JSON")
SOURCE_RUN_ATTEMPT=$(jq -r .run_attempt <<<"$SOURCE_JSON")
SOURCE_CONCLUSION=$(jq -r .conclusion <<<"$SOURCE_JSON")
jq -e '
  .event == "pull_request" and
  .status == "completed" and
  .path == ".github/workflows/run-sweep.yml"
' <<<"$SOURCE_JSON" >/dev/null
SOURCE_ARTIFACTS=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID/artifacts?per_page=100" \
  --paginate --jq '.artifacts[]
    | select(.expired == false)
    | .name')
grep -Eq '^(results_bmk|eval_results_all|bmk_agentic_)' \
  <<<"$SOURCE_ARTIFACTS"
```

Require `SOURCE_CONCLUSION=success` unless this exact run ID was explicitly
supplied; a pinned failure may recover only its completed points.

Verify that membership and fetch the final PR head:

```bash
gh api \
  "repos/SemiAnalysisAI/InferenceX/pulls/$ORIGINAL_PR/commits" \
  --paginate --jq '.[].sha' |
  grep -Fx "$SOURCE_HEAD_SHA"

SOURCE_PR=$ORIGINAL_PR
git fetch origin \
  "+pull/$SOURCE_PR/head:refs/remotes/origin/source-pr-$SOURCE_PR"
git cat-file -e "${SOURCE_HEAD_SHA}^{commit}"
SOURCE_PR_HEAD=$(git rev-parse "refs/remotes/origin/source-pr-$SOURCE_PR")
git merge-base --is-ancestor "$SOURCE_HEAD_SHA" "$SOURCE_PR_HEAD"
git diff --name-status "$SOURCE_HEAD_SHA" "$SOURCE_PR_HEAD"
```

Classify the changed paths, then compare the recovered master-config object at
both refs plus any referenced recipe, runner, launcher, benchmark script, image,
model, and environment values. Stop only if execution semantics changed.

## 3. Confirm the recovery scope

From the exact changelog diff in step 1, record each original `config-keys`,
`evals-only`, and `scenario-type` value. If the merge touched historical bytes
or is not a clean append, inspect the original PR diff to recover its intended
entries. Stop if the intended scope is ambiguous; do not copy malformed
historical changelog bytes into the recovery PR.

## 4. Bootstrap the recovery PR

Create an empty bootstrap branch from current `main`. Opening and labeling the
PR before it changes `perf-changelog.yaml` avoids starting a sweep before reuse
is authorized:

```bash
git fetch origin main
BRANCH="recovery/reuse-pr-$ORIGINAL_PR"
git switch -c "$BRANCH" origin/main
git commit --allow-empty -m "chore: prepare PR $ORIGINAL_PR ingest recovery"
git push -u origin "$BRANCH"

RECOVERY_PR_URL=$(gh pr create \
  --repo SemiAnalysisAI/InferenceX \
  --base main \
  --head "$BRANCH" \
  --title "fix: recover PR $ORIGINAL_PR ingest via sweep reuse" \
  --body "Recover the missing official ingest from source run $SOURCE_RUN_ID.")
RECOVERY_PR=$(gh pr view "$RECOVERY_PR_URL" \
  --repo SemiAnalysisAI/InferenceX \
  --json number --jq .number)

gh pr edit "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast
gh pr comment "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --body "/reuse-sweep-run $SOURCE_RUN_ID"
```

Keep exactly one of `full-sweep-enabled`,
`non-canary-full-sweep-enabled`, `full-sweep-fail-fast`, or
`full-sweep-fail-fast-no-canary`.

## 5. Append the recovery changelog and validate source artifacts

Append recovery entries to the end of `perf-changelog.yaml`. Preserve the
original entries' `config-keys`, `description`, `evals-only`, and
`scenario-type` values so the recovery targets the same scope and the
`InferenceX-app` changelog UI shows the meaningful configuration change. Copy
each original description verbatim; put source-run IDs and recovery/retrigger
details in the recovery PR body and final audit comment instead. Use the new
recovery PR URL. The current generator may produce a different matrix; that does
not invalidate reuse.

This is not a transient trigger: `InferenceX-app` persists and displays the
entry, so it must remain in the append-only changelog. Changing a description
after ingest does not update the UI: historical changelog entries are immutable,
and app ingest leaves an existing `(workflow_run_id, base_ref, head_ref)` row
unchanged. Existing UI text requires an explicit database correction.

Commit without `[skip-sweep]`:

```bash
git add perf-changelog.yaml
git commit -m "fix: recover PR $ORIGINAL_PR ingest"
RECOVERY_COMMIT=$(git rev-parse HEAD)
```

Validate the recovery changelog and inspect the matrix generated by current
`main`:

```bash
python3 utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT"
python3 utils/process_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT" \
  > /tmp/recovery-full-config.json
```

Confirm the generated config contains only the intended recovery scope. Its row
counts may differ from the source run.

Download only the result artifacts needed for local validation. This avoids the
large server-log artifacts retained in the official ingest bundle. Raw per-config
`bmk_<model>_*` artifacts are intentionally not selected — they fall through the
`case` below; the aggregate `results_bmk` is what the validator reads:

```bash
rm -rf /tmp/source-artifacts
ARTIFACT_ARGS=()
while IFS= read -r name; do
  case "$name" in
    results_bmk|eval_results_all|run-stats|bmk_agentic_*|agentic_*)
      ARTIFACT_ARGS+=(-n "$name")
      ;;
    eval_server_logs_*|eval_gpu_metrics_*)
      ;;
    eval_*)
      ARTIFACT_ARGS+=(-n "$name")
      ;;
  esac
done < <(
  gh api \
    "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID/artifacts?per_page=100" \
    --paginate --jq '.artifacts[] | select(.expired == false) | .name' |
    sort -u
)

((${#ARTIFACT_ARGS[@]})) || {
  echo "No unexpired result artifacts found" >&2
  exit 1
}
gh run download "$SOURCE_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX \
  -D /tmp/source-artifacts \
  "${ARTIFACT_ARGS[@]}"
```

Validate the source artifacts:

```bash
python3 utils/validate_reusable_sweep_artifacts.py \
  --artifacts-dir /tmp/source-artifacts
```

This local preflight uses the same validator that InferenceX-app runs after it
selects and downloads the reusable artifact set.

The validator first collapses reran (flaky) eval duplicates in place — keeping
the latest result per eval identity when a retried eval left duplicate raw dirs
/ aggregate rows — so a legitimate rerun does not fail validation. It only
collapses identities with a clear latest result; genuinely ambiguous duplicates
are still rejected.

The validator does not compare source coverage with
`/tmp/recovery-full-config.json`. It rejects duplicate fixed rows, missing run
stats, inconsistent agentic artifacts, malformed eval metadata, raw/aggregate
eval mismatches, or an empty result set. For a pinned failed batched eval run,
only `completed_eval_concs` are recovered.

## 6. Attach the source SHA without changing the tree

Make the ancestry carrier the final branch commit. `git commit-tree` guarantees
the required parent order and preserves the recovery tree:

```bash
TARGET_PARENT=$(git rev-parse HEAD)
TARGET_TREE=$(git rev-parse "${TARGET_PARENT}^{tree}")
ATTACH_SHA=$(
  printf 'chore: attach reusable sweep run %s\n' "$SOURCE_RUN_ID" |
    git commit-tree "$TARGET_TREE" \
      -p "$TARGET_PARENT" \
      -p "$SOURCE_HEAD_SHA"
)
git reset --hard "$ATTACH_SHA"

test "$(git rev-parse HEAD^1)" = "$TARGET_PARENT"
test "$(git rev-parse HEAD^2)" = "$SOURCE_HEAD_SHA"
test "$(git rev-parse HEAD^{tree})" = "$(git rev-parse HEAD^1^{tree})"
test "$(git diff --name-only origin/main...HEAD)" = "perf-changelog.yaml"
git diff --check origin/main...HEAD
```

Push once the branch is based on the current `main`:

```bash
git push origin "$BRANCH"
```

Do not rebase, locally squash, amend, or force-push after this point.

## 7. Verify the PR reuse gate

Require GitHub to list `SOURCE_HEAD_SHA` in the recovery PR commit list while
the Files tab contains only the recovery changelog append:

```bash
gh api \
  "repos/SemiAnalysisAI/InferenceX/pulls/$RECOVERY_PR/commits" \
  --paginate --jq '.[].sha' |
  grep -Fx "$SOURCE_HEAD_SHA"

test "$(gh pr diff "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --name-only)" = "perf-changelog.yaml"
```

Wait for `check-changelog` and `reuse-sweep-gate` to pass. `setup` and all GPU
jobs must be skipped:

```bash
gh pr checks "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --watch --fail-fast
```

`reuse-sweep-gate` appears only once the `pull_request` `run-sweep.yml` run for
the new head SHA registers; immediately after pushing, `gh pr checks` may list
only CodeQL/`check-changelog`/`comment`. Confirm that run exists and carries
`reuse-sweep-gate` before trusting a green result, or watch it directly:

```bash
gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow run-sweep.yml --event pull_request \
  --branch "$BRANCH" --limit 5 \
  --json databaseId,status,conclusion,headSha
```

On the PR (`pull_request`) gate, `setup` is itself skipped and `reuse-sweep-gate`
does the validation; `setup` only runs on the push-to-main run in step 8.

## 8. Merge and verify official ingest

Keep the verified carrier commit as the PR head through merge. This repository
allows squash merges: squashing into `main` creates a new main commit but does
not rewrite the PR branch or its recorded commit list, so source ancestry remains
available to the push workflow. Confirm the head, then squash-merge:

```bash
test "$(gh pr view "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --json headRefOid --jq .headRefOid)" = "$ATTACH_SHA"

gh pr merge "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --squash
```

If all checks passed and repository policy is the sole blocker, repeat the merge
with `--admin`. If `main` advances or the PR conflicts, update from `main` first
and recreate the final two-parent carrier commit before pushing again.

Locate and watch the push run for the squash commit:

```bash
RECOVERY_MERGE_SHA=$(gh pr view "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --json mergeCommit --jq .mergeCommit.oid)

gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow run-sweep.yml --event push \
  --commit "$RECOVERY_MERGE_SHA" --limit 5

RECOVERY_RUN_ID=<matching-run-id>
gh run watch "$RECOVERY_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX --exit-status
```

The push-to-main `Run Sweep` must:

- run `setup` even if the merge message contains `[skip-sweep]`;
- resolve the recovery PR and pinned source run;
- set `reuse-enabled=true`;
- upload recovery changelog metadata;
- dispatch `source-run-id` and `merge-run-id` from `trigger-ingest`.

Then locate the resulting `repository_dispatch` run in
`SemiAnalysisAI/InferenceX-app`. In the forgotten-`/reuse` case the target's
bogus ingest is also a recent successful `ingest-results` run, so do not pick by
recency — pick the run whose `Prepare artifacts from InferenceX` step logs both
the expected source run and recovery merge run. That app workflow must prepare
the artifact family, pass reusable-artifact validation, and complete ingestion:

```bash
gh run list --repo SemiAnalysisAI/InferenceX-app \
  --workflow "Ingest Benchmark Results" \
  --event repository_dispatch --limit 10 \
  --json databaseId,status,conclusion,createdAt

INGEST_RUN_ID=<candidate-run-id>
gh run view "$INGEST_RUN_ID" --repo SemiAnalysisAI/InferenceX-app --log \
  | grep -m1 "Source run: $SOURCE_RUN_ID" # must match before you trust this run

gh run view "$INGEST_RUN_ID" --repo SemiAnalysisAI/InferenceX-app --log \
  | grep -m1 "Merge run:  $RECOVERY_RUN_ID" # must also match

gh run watch "$INGEST_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX-app --exit-status
```

The ingest's first step is a `sleep 300` "wait for source run to finish", so the
run idles ~5 minutes before doing work — that is normal, not a hang.

Verify its logs identify `RECOVERY_RUN_ID` as the trigger and `SOURCE_RUN_ID`
plus `SOURCE_RUN_ATTEMPT` as the reused source. Require successful artifact
download, flattening, database ingest, run overrides, database verification,
cache invalidation, and unmapped-entity checks.

Post a final recovery PR comment with the original failed or skipped run/job,
source run/attempt/SHA, recovery merge run, downstream ingest run, recovered
artifact counts, and verification outcome.
