---
description: Recover a failed main-branch sweep ingest through the normal artifact-reuse path without rerunning GPU benchmarks
argument-hint: <failed-run-or-job-url> [source-run-id]
---

Recover the official database ingest for a failed or skipped InferenceX
push-to-main `Run Sweep` workflow by creating a recovery PR that reuses validated
artifacts from an earlier PR sweep. Do not add a one-off recovery workflow.

Inputs from `$ARGUMENTS`:

- `failed-run-or-job-url` is required.
- `source-run-id` is optional and remains a candidate until every source,
  ancestry, matrix, and artifact check passes.

## Safety rules

- Never rerun the failed target workflow or job.
- The target must be a completed `push` run of
  `.github/workflows/run-sweep.yml` on `main` whose official ingest did not
  complete.
- Reuse only a completed `pull_request` run of `run-sweep.yml`. Unpinned reuse
  requires success. A specifically pinned failed run is allowed only when exact
  artifact validation proves the recovery matrix is complete.
- Stop if execution-relevant config, image, model, runner, launcher, benchmark,
  or matrix files changed between the source SHA and the final source PR head.
- Preserve all historical `perf-changelog.yaml` bytes. Append recovery entries
  only at the end.
- Keep exactly one full-sweep label on the recovery PR and pin the source run
  with `/reuse-sweep-run <run_id>` before pushing the changelog change.
- The final recovery branch head must have the recovery commit as its first
  parent, the source run SHA as its second parent, and the recovery commit's
  file tree unchanged.
- Never rebase, squash, force-push, or otherwise rewrite the recovery branch
  after attaching the source SHA. Those operations can remove the ancestry that
  makes the source run reusable.
- Do not rely on `[skip-sweep]`. Reuse authorization suppresses PR benchmark
  work, and pushes to `main` ignore that marker.
- Never bypass failing or pending checks. Use admin merge only when all checks
  passed and repository policy is the sole blocker.
- Do not add co-author lines, generated-by text, bot branding, or attribution.

## 1. Inspect the target

Install helper dependencies and inspect the target:

```bash
python -m pip install pydantic pyyaml
python utils/recover_failed_ingest.py inspect-target \
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

Require event `push`, status `completed`, workflow path
`.github/workflows/run-sweep.yml`, branch `main`, and a failed or skipped state
that explains the missing ingest. Record the original PR and root cause.

Fetch history and inspect the exact original changelog delta:

```bash
git fetch origin main
git cat-file -e "${ORIGINAL_MERGE_SHA}^{commit}"
ORIGINAL_BASE_SHA=$(git rev-parse "${ORIGINAL_MERGE_SHA}^")
python utils/recover_failed_ingest.py audit-changelog \
  --ref "$ORIGINAL_MERGE_SHA"
git diff "$ORIGINAL_BASE_SHA" "$ORIGINAL_MERGE_SHA" -- \
  perf-changelog.yaml
```

## 2. Select and validate the source run

Resolve the candidate source run and record:

```bash
SOURCE_RUN_ID=<validated-run-id>
SOURCE_JSON=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID")
SOURCE_HEAD_SHA=$(jq -r .head_sha <<<"$SOURCE_JSON")
SOURCE_RUN_ATTEMPT=$(jq -r .run_attempt <<<"$SOURCE_JSON")
```

Require:

- event `pull_request`, status `completed`, and path
  `.github/workflows/run-sweep.yml`;
- a successful conclusion unless this exact run ID was explicitly supplied;
- an unexpired benchmark, eval, or agentic result artifact;
- source-head membership in its original PR commit list.

Find the source PR and fetch its history:

```bash
SOURCE_PR=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/commits/$SOURCE_HEAD_SHA/pulls" \
  --jq 'if length == 1 then .[0].number else error("expected one source PR") end')
git fetch origin "pull/$SOURCE_PR/head:refs/remotes/origin/source-pr-$SOURCE_PR"
git cat-file -e "${SOURCE_HEAD_SHA}^{commit}"
```

Compare `SOURCE_HEAD_SHA` through the final source PR head. Stop if any
execution-relevant target configuration changed after the source run.

## 3. Reconstruct and validate the intended matrix

Use a detached worktree at the original merge to reconstruct its intended
changelog entries when historical formatting or links need repair:

```bash
WORKTREE=$(mktemp -d /tmp/infx-recovery-worktree.XXXXXX)
rmdir "$WORKTREE"
python utils/recover_failed_ingest.py create-worktree \
  --ref "$ORIGINAL_MERGE_SHA" \
  --directory "$WORKTREE"
```

Edit only `$WORKTREE/perf-changelog.yaml`. Preserve the original base bytes and
append only the original PR's intended entries with canonical links. Then build
the historical matrix:

```bash
python utils/recover_failed_ingest.py build-config \
  --worktree "$WORKTREE" \
  --base-ref "$ORIGINAL_BASE_SHA" \
  --merge-ref "$ORIGINAL_MERGE_SHA" \
  --pr-number "$ORIGINAL_PR" \
  --config-output /tmp/original-full-config.json \
  --metadata-output /tmp/original-changelog-metadata.json
```

Record the fixed-sequence, agentic, and eval counts. Remove the worktree when
finished:

```bash
git worktree remove "$WORKTREE"
```

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
  --add-label full-sweep-enabled
gh pr comment "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --body "/reuse-sweep-run $SOURCE_RUN_ID"
```

Keep exactly one of `full-sweep-enabled`,
`non-canary-full-sweep-enabled`, `full-sweep-fail-fast`, or
`full-sweep-fail-fast-no-canary`.

## 5. Append the recovery changelog and validate artifacts

Append recovery entries to the end of `perf-changelog.yaml`. Preserve the
original entries' `config-keys`, `evals-only`, and `scenario-type` shape so the
recovery PR generates the same logical matrix. Use the new recovery PR URL and
state clearly that this is an artifact-only ingest recovery.

Commit without `[skip-sweep]`:

```bash
git add perf-changelog.yaml
git commit -m "fix: recover PR $ORIGINAL_PR ingest"
RECOVERY_COMMIT=$(git rev-parse HEAD)
```

Generate the exact matrix that the merge-to-main run will validate:

```bash
python utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT"
python utils/process_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT" \
  > /tmp/recovery-full-config.json
```

Download the source run to a fresh directory, remove its
`changelog-metadata`, and retain only ingest-relevant artifact classes:

```bash
rm -rf /tmp/source-artifacts
gh run download "$SOURCE_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX \
  -D /tmp/source-artifacts
rm -rf /tmp/source-artifacts/changelog-metadata

for artifact_dir in /tmp/source-artifacts/*; do
  [ -e "$artifact_dir" ] || continue
  name=$(basename "$artifact_dir")
  case "$name" in
    results_bmk|eval_results_all|run-stats|bmk_*|eval_*|agentic_*|server_logs_*|multinode_server_logs_*)
      ;;
    *)
      rm -rf "$artifact_dir"
      ;;
  esac
done
```

The retained classes are:

```text
results_bmk
eval_results_all
run-stats
bmk_*
eval_*
agentic_*
server_logs_*
multinode_server_logs_*
```

Validate exact equality against the recovery PR matrix:

```bash
python utils/validate_reusable_sweep_artifacts.py \
  --config-json /tmp/recovery-full-config.json \
  --artifacts-dir /tmp/source-artifacts
```

Stop on any missing, unexpected, or duplicate fixed-sequence, agentic, or eval
identity.

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

Do not rebase, squash, amend, or force-push after this point.

## 7. Verify the PR reuse gate

Require GitHub to list `SOURCE_HEAD_SHA` in the recovery PR commit list while
the Files tab contains only the recovery changelog append:

```bash
gh api \
  "repos/SemiAnalysisAI/InferenceX/pulls/$RECOVERY_PR/commits" \
  --paginate --jq '.[].sha' |
  grep -Fx "$SOURCE_HEAD_SHA"

gh pr diff "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --name-only
```

Wait for `check-changelog` and `reuse-sweep-gate` to pass. `setup` and all GPU
jobs must be skipped:

```bash
gh pr checks "$RECOVERY_PR" \
  --repo SemiAnalysisAI/InferenceX \
  --watch --fail-fast
```

Also invoke `validate_reusable_run` locally against the recovery PR so source
ancestry, workflow path, conclusion, and artifacts are checked before merge.

```bash
GH_TOKEN="$(gh auth token)" \
SOURCE_RUN_ID="$SOURCE_RUN_ID" \
RECOVERY_PR="$RECOVERY_PR" \
python3 - <<'PY'
import os

from utils import find_reusable_sweep_run as reuse

repo = "SemiAnalysisAI/InferenceX"
token = os.environ["GH_TOKEN"]
run_id = int(os.environ["SOURCE_RUN_ID"])
pr_number = int(os.environ["RECOVERY_PR"])
run = reuse.github_api(repo, f"/actions/runs/{run_id}", token)
reuse.validate_reusable_run(
    repo,
    "run-sweep.yml",
    pr_number,
    run,
    token,
    allow_failed=True,
)
print(f"Validated reusable run {run_id} for PR #{pr_number}")
PY
```

## 8. Merge and verify official ingest

Merge the current branch head without rewriting its commits. If `main` advances
or the PR conflicts, update from `main` first and recreate the final two-parent
carrier commit before pushing again.

The push-to-main `Run Sweep` must:

- run `setup` even if the merge message contains `[skip-sweep]`;
- resolve the recovery PR and pinned source run;
- set `reuse-enabled=true`;
- pass `reuse-ingest-artifacts` exact validation;
- upload recovery changelog metadata;
- run `trigger-ingest`.

Then locate the resulting `repository_dispatch` run in
`SemiAnalysisAI/InferenceX-app`. Require successful artifact download,
flattening, database ingest, run overrides, database verification, cache
invalidation, and unmapped-entity checks.

Post a final recovery PR comment with the original failed run/job, source
run/attempt/SHA, recovery merge run, downstream ingest run, exact matrix counts,
and verification outcome.
