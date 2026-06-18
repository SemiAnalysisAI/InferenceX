---
description: Recover a failed main-branch sweep ingest from validated PR artifacts without rerunning GPU benchmarks
argument-hint: <failed-run-or-job-url> [source-run-id]
---

Recover the official database ingest for a failed InferenceX push-to-main `Run Sweep`
workflow by reusing artifacts from the corresponding successful PR sweep. Execute the
recovery end to end: diagnose, validate, open and merge a guarded recovery PR, dispatch it,
and verify the downstream InferenceX-app ingest.

Inputs from `$ARGUMENTS`:

- **failed-run-or-job-url** (required) - an
  `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/<run-id>` URL, optionally ending
  in `/job/<job-id>`.
- **source-run-id** (optional) - a known successful PR sweep run. Treat this only as a
  candidate and apply every validation below.

## Non-negotiable safety rules

- Never rerun the failed target workflow or job. A rerun can launch the GPU matrix.
- The target must be a completed, failed `push` run of
  `.github/workflows/run-sweep.yml` on `main`.
- Reuse only a completed, successful `pull_request` run of the same workflow whose head SHA
  belongs to the merged PR and whose required artifacts are unexpired.
- Confirm that the source recipe and matrix match the merged target. Stop if a later PR
  commit changed a selected config, launcher, benchmark script, image, model, or search space.
- The recovery workflow must contain one `ubuntu-latest` job. It must not reference GPU
  runners, benchmark reusable workflows, matrix jobs, or `workflow_call`.
- Require an explicit workflow-dispatch confirmation string.
- Put `[skip-sweep]` in the recovery commit and merge subject.
- Do not add `Co-Authored-By`, generated-by text, bot branding, or author attribution to the
  commit, PR, comments, or workflow.
- Never bypass a failing or pending check. `--admin` is allowed only after every check has
  completed successfully and the repository policy is the sole remaining merge blocker.
- Preserve unrelated local changes. Use a fresh branch or worktree from `origin/main`; do not
  reset or overwrite an existing user branch.
- Stop rather than guessing if changelog reconstruction is ambiguous or artifact validation
  fails.

## Step 1 - inspect the failed target

Parse the run ID and optional job ID from the URL. Fetch the run through the API rather than
trusting the URL text:

```bash
REPO=SemiAnalysisAI/InferenceX
gh api "repos/$REPO/actions/runs/$TARGET_RUN_ID" \
  --jq '{id,event,status,conclusion,path,head_branch,head_sha,run_attempt,html_url}'
gh api "repos/$REPO/actions/runs/$TARGET_RUN_ID/jobs?per_page=100" \
  --jq '.jobs[] | [.id,.name,.status,.conclusion,.html_url] | @tsv'
```

Validate the target invariants from the safety rules. If a job ID was supplied, require that
exact job to be completed and failed. Read its log and record the root cause:

```bash
gh run view "$TARGET_RUN_ID" --repo "$REPO" --job "$TARGET_JOB_ID" --log \
  > "/tmp/infx-target-$TARGET_RUN_ID.log"
```

Resolve the merged PR associated with `head_sha` using the commit-to-pulls API, then verify
the PR's `mergeCommit.oid` equals the target SHA. Fetch the commit and record its first parent
as `ORIGINAL_BASE_SHA`:

```bash
gh api -H "Accept: application/vnd.github+json" \
  "repos/$REPO/commits/$ORIGINAL_MERGE_SHA/pulls"
gh pr view "$PR" --repo "$REPO" \
  --json number,title,mergedAt,mergeCommit,headRefName,headRefOid,commits,url
git fetch origin main
git cat-file -e "${ORIGINAL_MERGE_SHA}^{commit}"
ORIGINAL_BASE_SHA=$(git rev-parse "${ORIGINAL_MERGE_SHA}^")
```

Do not proceed if the failed run is unrelated to merge-time artifact reuse or ingest.

## Step 2 - find the reusable PR sweep

Enumerate every commit in the merged PR, then query `run-sweep.yml` runs for each SHA. Prefer
the newest candidate that actually produced benchmark artifacts; a successful reuse-gate-only
run is not a source sweep.

```bash
gh api "repos/$REPO/pulls/$PR/commits" --paginate --jq '.[].sha'
gh api "repos/$REPO/actions/runs?head_sha=$SHA&per_page=100" \
  --jq '.workflow_runs[]
        | select(.path == ".github/workflows/run-sweep.yml")
        | [.id,.event,.status,.conclusion,.run_attempt,.head_sha,.created_at,.html_url]
        | @tsv'
gh api "repos/$REPO/actions/runs/$SOURCE_RUN_ID/artifacts?per_page=100" \
  --jq '.artifacts[] | [.name,.expired,.created_at,.expires_at,.id] | @tsv'
```

The source run must:

- have `event == "pull_request"`, `status == "completed"`, and `conclusion == "success"`;
- have a head SHA listed in the PR's commits;
- have unexpired `results_bmk`, `run-stats`, and at least one `bmk_*` artifact;
- have `eval_results_all` and the expected raw `eval_*` artifacts when the reconstructed
  target matrix contains eval jobs.

Use the source run's current `run_attempt` in reuse metadata. Rerun attempts can leave
artifacts from multiple attempts under one run ID; `gh run download` is expected to collect
the retained point artifacts and the newest aggregate. The later validator is authoritative.

Inspect changes from the source SHA through the final PR head. Compare the selected
top-level config objects and every target-specific launcher or benchmark script. A merge from
`main` may add unrelated diffs, but any execution-relevant difference for the selected config
disqualifies the source.

## Step 3 - reconstruct only the target PR changelog delta

Read both prior examples before implementing the new recovery:

```bash
sed -n '1,260p' .github/workflows/recover-pr-1767-ingest.yml
sed -n '1,280p' .github/workflows/recover-pr-1798-ingest.yml
git diff "$ORIGINAL_BASE_SHA" "$ORIGINAL_MERGE_SHA" -- perf-changelog.yaml
```

Audit the full changelog before reconstructing anything:

```bash
python3 - <<'PY'
from pathlib import Path
import re
import sys

import yaml

sys.path.insert(0, "utils")
from matrix_logic.validation import ChangelogEntry

path = Path("perf-changelog.yaml")
raw = path.read_bytes()
errors: list[str] = []
warnings: list[str] = []

if not raw.endswith(b"\n"):
    errors.append("file does not end with a newline")
if b"\r" in raw:
    errors.append("file contains CR characters")
if b"\t" in raw:
    errors.append("file contains tabs")
if b"\0" in raw:
    errors.append("file contains NUL bytes")

text = raw.decode("utf-8")
trailing_whitespace = [
    line_number
    for line_number, line in enumerate(text.splitlines(), start=1)
    if line != line.rstrip()
]
if trailing_whitespace:
    warnings.append(
        "trailing whitespace on lines "
        + ", ".join(str(line_number) for line_number in trailing_whitespace)
    )
for line_number, line in enumerate(text.splitlines(), start=1):
    if line.startswith("  pr-link:") and not line.removeprefix("  pr-link:").strip():
        warnings.append(f"line {line_number} has a multiline or empty pr-link")

data = yaml.safe_load(text)
if not isinstance(data, list):
    errors.append("root value is not a YAML list")
    data = []

top_level_entries = sum(
    line.startswith("- config-keys:") for line in text.splitlines()
)
if top_level_entries != len(data):
    errors.append(
        f"found {top_level_entries} top-level config entries but parsed {len(data)}"
    )

for index, entry in enumerate(data, start=1):
    try:
        parsed = ChangelogEntry.model_validate(entry)
    except Exception as exc:
        errors.append(f"entry {index} fails ChangelogEntry validation: {exc}")
        continue

    if not parsed.pr_link.strip():
        warnings.append(f"entry {index} has an empty pr-link")
    elif parsed.pr_link == "XXX":
        warnings.append(f"entry {index} still has the XXX pr-link placeholder")
    elif not re.fullmatch(
        r"https://github\.com/SemiAnalysisAI/InferenceX/pull/\d+",
        parsed.pr_link,
    ):
        warnings.append(f"entry {index} has a noncanonical pr-link: {parsed.pr_link}")

seen: dict[tuple[object, ...], int] = {}
for index, entry in enumerate(data, start=1):
    if not isinstance(entry, dict):
        continue
    identity = (
        tuple(entry.get("config-keys", [])),
        tuple(entry.get("description", [])),
        entry.get("pr-link"),
        entry.get("evals-only", False),
        tuple(entry.get("scenario-type") or []),
    )
    if identity in seen:
        warnings.append(
            f"entry {index} exactly duplicates entry {seen[identity]}"
        )
    else:
        seen[identity] = index

for warning in warnings:
    print(f"WARNING: {warning}")
for error in errors:
    print(f"ERROR: {error}", file=sys.stderr)

if errors:
    raise SystemExit(1)

print(f"Validated {len(data)} changelog entries")
PY

git diff --check "$ORIGINAL_BASE_SHA" "$ORIGINAL_MERGE_SHA" -- perf-changelog.yaml
```

Treat exact duplicate entries as warnings, not automatic errors. A duplicate may be an
intentional sweep retrigger. Confirm its commit or PR context before changing it. Do not
autoformat or deduplicate historical entries: `process_changelog.py` rejects non-whitespace
deletions, and unrelated edits can change which benchmarks run.

For entries introduced by the target PR, require all of the following:

- They were appended at the end of the file.
- `config-keys` and `description` are non-empty lists.
- `pr-link` is a single-line URL for the target PR, not blank or `XXX`.
- Added lines have no trailing whitespace.
- Separator lines are truly empty.

Create a detached temporary worktree at `ORIGINAL_MERGE_SHA`. Modify only its
`perf-changelog.yaml` so the synthetic diff against `ORIGINAL_BASE_SHA` contains exactly the
target PR's intended changelog additions:

- If the target PR added a malformed entry, repair that entry in the synthetic tree.
- If the target PR also fixed an older malformed entry, reverse that unrelated repair back to
  its base form so `process_changelog.py` does not see a content deletion.
- Permit whitespace-only deletions, but leave no non-whitespace deletion in the synthetic
  diff.
- Use a context-specific edit with an exact replacement-count assertion. Never use a broad
  substitution based only on a config key; config keys can appear in older entries.

Write the tree and create a synthetic commit whose only parent is `ORIGINAL_BASE_SHA`, then
run the repository processor:

```bash
git add perf-changelog.yaml
fixed_tree=$(git write-tree)
fixed_sha=$(printf '%s\n' "Synthetic PR #$PR recovery tree" \
  | git -c user.name='InferenceX Recovery' \
        -c user.email='actions@users.noreply.github.com' \
        commit-tree "$fixed_tree" -p "$ORIGINAL_BASE_SHA")

pip install pydantic
python3 utils/process_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref "$ORIGINAL_BASE_SHA" \
  --head-ref "$fixed_sha" \
  > /tmp/full-config.json
```

Require all reconstructed `changelog_metadata.entries[].pr-link` values to point to the
target PR. Inspect the generated benchmark/eval counts and config keys. For the metadata
artifact, replace the synthetic refs with the real `ORIGINAL_BASE_SHA` and
`ORIGINAL_MERGE_SHA`.

## Step 4 - download and validate artifacts locally

Download the source run into a temporary directory. Remove its `changelog-metadata`; retain
only the artifact classes consumed by the official ingest path:

```text
results_bmk
eval_results_all
run-stats
bmk_*
eval_*
server_logs_*
multinode_server_logs_*
agentic_aggregated
```

Run the repository validator against the reconstructed target config:

```bash
python3 utils/validate_reusable_sweep_artifacts.py \
  --config-json /tmp/full-config.json \
  --artifacts-dir /tmp/source-artifacts
```

The validator rejects missing fixed-sequence identities but does not reject unexpected ones.
Also compare its `expected_benchmark_keys()` and `actual_benchmark_keys()` sets directly and
require equality:

```bash
python3 - <<'PY'
from pathlib import Path
from utils.validate_reusable_sweep_artifacts import (
    actual_benchmark_keys,
    expected_benchmark_keys,
    load_json,
)

config = load_json(Path("/tmp/full-config.json"))
expected = expected_benchmark_keys(config)
actual = actual_benchmark_keys(Path("/tmp/source-artifacts"))
missing = expected - actual
extra = actual - expected
if missing or extra:
    raise SystemExit(
        f"benchmark identity mismatch: missing={len(missing)} extra={len(extra)}"
    )
print(f"Exact benchmark identity match: {len(expected)} row(s)")
PY
```

Record the exact validated benchmark-row and eval-job counts. Stop on any mismatch, missing
aggregate, or missing raw eval artifact.

## Step 5 - create the guarded recovery workflow

Create a fresh branch from current `origin/main`, named
`workflow/recover-pr-<PR>-ingest`, and add
`.github/workflows/recover-pr-<PR>-ingest.yml`. Start from the closest prior recovery example
and hard-code the validated identifiers:

- source repository, run ID, run attempt, PR number, and source head SHA;
- failed target run ID and job ID;
- original base and merge SHAs.

The workflow must perform these steps:

1. Checkout with full history.
2. Revalidate the failed target run/job and successful source run through `gh api`.
3. Verify source-head membership in the PR and required unexpired artifacts.
4. Reproduce the exact, locally tested synthetic changelog reconstruction.
5. Generate `full-config.json` and corrected `changelog_metadata.json`.
6. Download and filter source artifacts.
7. Add `reused-ingest-metadata/reuse_source_run.json`, including source, target, and carrier
   run identifiers.
8. Run `validate_reusable_sweep_artifacts.py`.
9. Upload `reused-ingest-artifacts` and `changelog-metadata`.
10. Dispatch `ingest-results` to `SemiAnalysisAI/InferenceX-app` using
    `INFX_FRONTEND_PAT`.

Use the existing pinned action SHAs and secret names. Run:

```bash
actionlint ".github/workflows/recover-pr-$PR-ingest.yml"
yq eval '.' ".github/workflows/recover-pr-$PR-ingest.yml" >/dev/null
git diff --check
```

Also execute the workflow's target/source guard expressions and reconstruction commands
locally. Re-run the artifact validator from the exact generated config.

## Step 6 - open, comment on, and merge the recovery PR

Commit with:

```text
fix: recover PR <PR> ingest [skip-sweep]
```

Push the branch and open a PR with a neutral title such as:

```text
Recover PR <PR> ingest without rerunning sweep
```

The PR body must include:

- failed target run/job and root cause;
- reusable source run, attempt, and SHA;
- reconstructed benchmark/eval counts;
- CPU-only and explicit-confirmation safeguards;
- local validation performed.

Do not add attribution text. Add a PR comment summarizing the validated target, source,
counts, and CPU-only path. Wait for every check to finish successfully. Merge with a
`[skip-sweep]` subject. If normal merge is blocked solely by repository policy after all
checks are green, use the authenticated user's admin merge permission.

## Step 7 - dispatch and verify both workflows

After the workflow is present on `main`, dispatch it with its exact confirmation input:

```bash
gh workflow run "recover-pr-$PR-ingest.yml" \
  --repo "$REPO" --ref main \
  -f confirm="recover-pr-$PR"
```

Watch the carrier run to completion. Require success for target/source validation,
reconstruction, artifact validation, both uploads, and database dispatch. Confirm the log
reports the expected benchmark/eval counts and that no GPU or benchmark jobs were created.

Locate the new `repository_dispatch` run in `SemiAnalysisAI/InferenceX-app` created after the
carrier dispatch. Watch it through completion and require these steps to succeed:

- Download artifacts from InferenceX run
- Flatten reused ingest artifact bundle
- Ingest results to DB
- Apply run overrides
- Verify database
- Invalidate Vercel cache
- Check for unmapped entities

Inspect the ingest and verification logs for row counts, errors, and unmapped entities. Do
not report success based only on the dispatch or carrier workflow.

## Step 8 - report and clean up

Post a final comment on the recovery PR with links to the successful carrier and downstream
ingest runs plus the validated row/eval counts. Remove temporary worktrees and files without
touching unrelated user state.

Report:

- recovery PR URL and merge SHA;
- source run and attempt;
- carrier run URL;
- downstream InferenceX-app run URL;
- benchmark/eval counts;
- database verification, cache invalidation, and unmapped-entity outcome.
