---
description: Recover a failed main-branch sweep ingest from validated PR artifacts without rerunning GPU benchmarks
argument-hint: <failed-run-or-job-url> [source-run-id]
---

Recover the official database ingest for a failed InferenceX push-to-main `Run Sweep`
workflow by reusing artifacts from the corresponding PR sweep. Execute the diagnosis,
recovery PR, dispatch, and downstream verification end to end.

Inputs from `$ARGUMENTS`:

- `failed-run-or-job-url` is required. A run-only URL is accepted only when it has exactly
  one completed failed job; otherwise require a `/job/<job-id>` URL.
- `source-run-id` is optional and is only a candidate until every source and artifact check
  passes.

## Safety rules

- Never rerun the failed target workflow or job.
- The target must be a completed failed `push` run of
  `.github/workflows/run-sweep.yml` on `main`.
- Reuse only a completed PR run of that workflow whose head SHA remains in the merged PR.
  Unpinned runs must be successful. A specifically supplied failed run is allowed only when
  exact artifact validation proves the target matrix is complete.
- Stop if execution-relevant files changed between the source SHA and final PR head.
- The recovery workflow must contain one `ubuntu-latest` job, only
  `workflow_dispatch`, and an exact confirmation input. It must not call benchmark reusable
  workflows or use a matrix. Its GitHub token permissions must be explicitly read-only.
- Preserve historical `perf-changelog.yaml` bytes. Reconstruct only the target PR's appended
  entries, with canonical links to that PR.
- Use a detached temporary worktree for reconstruction. Every `git add`, `write-tree`, and
  processor command must run through `git -C "$WORKTREE"` or the recovery helper.
- Never bypass failing or pending checks. Use admin merge only when all checks passed and
  repository policy is the sole blocker.
- Put `[skip-sweep]` in the recovery commit and merge subject.
- Do not add co-author lines, generated-by text, bot branding, or author attribution.

## 1. Inspect the exact target

Install local helper dependencies, then let the tested helper parse the URL, resolve an
omitted job ID only when unambiguous, validate target invariants, and resolve the merged PR:

```bash
python -m pip install pydantic pyyaml
python utils/recover_failed_ingest.py inspect-target \
  "$FAILED_RUN_OR_JOB_URL" \
  --output /tmp/infx-recovery-target.json

TARGET_RUN_ID=$(jq -r .run_id /tmp/infx-recovery-target.json)
TARGET_JOB_ID=$(jq -r .job_id /tmp/infx-recovery-target.json)
PR=$(jq -r .pr_number /tmp/infx-recovery-target.json)
ORIGINAL_MERGE_SHA=$(jq -r .merge_sha /tmp/infx-recovery-target.json)

gh run view "$TARGET_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX \
  --job "$TARGET_JOB_ID" --log \
  > "/tmp/infx-target-$TARGET_RUN_ID.log"
```

Fetch history and require the merge's first parent:

```bash
git fetch origin main
git cat-file -e "${ORIGINAL_MERGE_SHA}^{commit}"
ORIGINAL_BASE_SHA=$(git rev-parse "${ORIGINAL_MERGE_SHA}^")
python utils/recover_failed_ingest.py audit-changelog \
  --ref "$ORIGINAL_MERGE_SHA"
git diff --check "$ORIGINAL_BASE_SHA" "$ORIGINAL_MERGE_SHA" \
  -- perf-changelog.yaml
```

The audit reads `perf-changelog.yaml` from `ORIGINAL_MERGE_SHA`, never from the current
checkout. Its `errors` list records repairable historical defects such as a missing final
newline; duplicate keys or an unparseable schema still abort. Record the failed job's root
cause and stop unless it is an ingest/reuse failure that can be repaired without GPU work.

## 2. Select the source PR sweep

Enumerate every current PR commit and every `run-sweep.yml` PR run for those SHAs. Validate
the candidate through GitHub's API:

- workflow path, event, status, conclusion, attempt, and source head SHA;
- source-head membership in the PR commit list;
- all retained artifacts are unexpired;
- the candidate produced `results_bmk`, `eval_results_all`, or at least one
  `bmk_agentic_*` point artifact.

Do not require `results_bmk` or `run-stats` for eval-only or agentic-only matrices. The exact
matrix validator in step 4 decides which aggregates and point artifacts are required.

Compare the source SHA through the final PR head. Any target-specific config, image, model,
runner, launcher, benchmark script, or matrix change disqualifies the source.

## 3. Reconstruct in a detached worktree

Read the two prior examples, inspect the historical diff, and create a worktree at the exact
failed merge:

```bash
sed -n '1,260p' .github/workflows/recover-pr-1767-ingest.yml
sed -n '1,280p' .github/workflows/recover-pr-1798-ingest.yml
git diff "$ORIGINAL_BASE_SHA" "$ORIGINAL_MERGE_SHA" -- perf-changelog.yaml

WORKTREE=$(mktemp -d /tmp/infx-recovery-worktree.XXXXXX)
rmdir "$WORKTREE"
python utils/recover_failed_ingest.py create-worktree \
  --ref "$ORIGINAL_MERGE_SHA" \
  --directory "$WORKTREE"
```

Edit only `$WORKTREE/perf-changelog.yaml`. Repair malformed target entries and reverse any
unrelated historical repair so the resulting file is exactly the base bytes followed by the
target PR's intended entries. Do not autoformat or deduplicate.

Build the synthetic commit and config through the helper. It stages only inside the detached
worktree, rejects unrelated changes, enforces exact historical bytes and canonical PR links,
and verifies that the synthetic commit changes only `perf-changelog.yaml`:

```bash
python utils/recover_failed_ingest.py build-config \
  --worktree "$WORKTREE" \
  --base-ref "$ORIGINAL_BASE_SHA" \
  --merge-ref "$ORIGINAL_MERGE_SHA" \
  --pr-number "$PR" \
  --config-output /tmp/full-config.json \
  --metadata-output /tmp/changelog-metadata/changelog_metadata.json
```

Inspect the reported fixed-sequence, agentic, and eval counts before continuing.

## 4. Download and validate exact artifacts

Download the selected source run to a fresh directory. Remove its changelog metadata and
retain only ingest-relevant classes:

```text
results_bmk
eval_results_all
run-stats
bmk_*
eval_*
agentic_*
server_logs_*
multinode_server_logs_*
agentic_aggregated
```

Run the authoritative exact validator:

```bash
python utils/validate_reusable_sweep_artifacts.py \
  --config-json /tmp/full-config.json \
  --artifacts-dir /tmp/source-artifacts
```

It requires equality for fixed-sequence identities, agentic point/raw/aggregate identities,
and eval-only raw identities. It also rejects unexpected rows and requires `run-stats` only
when fixed-sequence collection should have produced it. Stop on any mismatch.

## 5. Create the guarded recovery workflow

Create `workflow/recover-pr-<PR>-ingest` from current `origin/main` and add
`.github/workflows/recover-pr-<PR>-ingest.yml`, based on the closest prior example.
Hard-code the validated source run/attempt/SHA, target run/job, PR, original base, and merge
SHAs. Revalidate those values in the workflow before downloading artifacts.

The workflow must reproduce the locally tested reconstruction, upload
`reused-ingest-artifacts` and corrected `changelog-metadata`, then dispatch
`ingest-results` to `SemiAnalysisAI/InferenceX-app`.

Validate it before committing:

```bash
python utils/recover_failed_ingest.py validate-workflow \
  ".github/workflows/recover-pr-$PR-ingest.yml" \
  --pr-number "$PR"
actionlint ".github/workflows/recover-pr-$PR-ingest.yml"
yq eval '.' ".github/workflows/recover-pr-$PR-ingest.yml" >/dev/null
git diff --check
```

Commit as `fix: recover PR <PR> ingest [skip-sweep]`, push, and open a PR. The body and a PR
comment must record the failed target/job, root cause, source run/attempt/SHA, exact counts,
and CPU-only safeguards. Do not add attribution.

Wait for all checks, then merge with a `[skip-sweep]` subject. Use admin merge only for the
policy-only case described above.

## 6. Dispatch and verify

Dispatch from `main` with the exact confirmation string:

```bash
gh workflow run "recover-pr-$PR-ingest.yml" \
  --repo SemiAnalysisAI/InferenceX \
  --ref main \
  -f confirm="recover-pr-$PR"
```

Require the carrier workflow to succeed and confirm it created no GPU or benchmark jobs.
Then locate the resulting `repository_dispatch` run in `SemiAnalysisAI/InferenceX-app`.
Require successful artifact download/flattening, database ingest, run overrides, database
verification, cache invalidation, and unmapped-entity checks.

Post a final recovery PR comment with carrier and downstream run links plus exact counts.
Remove the temporary worktree with `git worktree remove "$WORKTREE"` and report the recovery
PR, merge SHA, source attempt, carrier run, downstream run, row counts, and verification
outcome.
