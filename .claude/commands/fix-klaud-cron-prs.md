---
description: Triage failing Claude-authored PRs — read sweep logs, debug, and push a candidate fix per PR
---

For each open Claude-authored PR (`claude/*` branch) whose full-sweep validation produced at least one **FAILED** check, fetch the failing run's logs, diagnose the root cause, and push a candidate fix to the PR's branch.

This command modifies remote PR branches. **Pause for user confirmation** after listing the candidate PRs and again before pushing each fix.

## Step 1 — find failing `claude/*` PRs whose sweep actually ran

A PR qualifies only if:
- `headRefName` starts with `claude/`
- At least one `Run Sweep` check has conclusion `SUCCESS` **or** `FAILURE` (i.e. the sweep was enabled and produced real results — not all skipped)
- At least one check has conclusion `FAILURE`, `CANCELLED`, or `TIMED_OUT`

`gh pr list --json statusCheckRollup` truncates rollups, so enumerate candidates first, then re-query each PR individually.

```bash
gh pr list --repo SemiAnalysisAI/InferenceX --state open --limit 200 \
  --json number,title,headRefName \
  --jq '.[] | select(.headRefName | startswith("claude/")) | "\(.number)\t\(.headRefName)\t\(.title)"' \
  > /tmp/claude_pr_candidates.tsv

: > /tmp/claude_prs_failing.tsv
while IFS=$'\t' read -r pr branch title; do
  qualifies=$(gh pr view "$pr" --repo SemiAnalysisAI/InferenceX --json statusCheckRollup --jq '
    def state: if (.conclusion // "") != "" then .conclusion else .status end;
    . as $p
    | ($p.statusCheckRollup | any(.workflowName == "Run Sweep" and (state == "SUCCESS" or state == "FAILURE"))) as $swept
    | ($p.statusCheckRollup | any(state == "FAILURE" or state == "CANCELLED" or state == "TIMED_OUT")) as $failed
    | ($swept and $failed)')
  if [ "$qualifies" = "true" ]; then
    printf '%s\t%s\t%s\n' "$pr" "$branch" "$title" >> /tmp/claude_prs_failing.tsv
  fi
done < /tmp/claude_pr_candidates.tsv
cat /tmp/claude_prs_failing.tsv
```

Render the candidates as a markdown table with clickable PR links and **stop**. Confirm with the user which PRs to attempt fixes on (default: all). If none qualify, print a short no-results message and exit.

## Step 2 — per-PR diagnosis & fix loop

For each confirmed PR, run the following loop. Do **not** parallelize — keep state local and obvious.

### 2a. Check out the PR branch in a worktree

Use a worktree so the loop never disturbs the user's working tree:

```bash
git fetch origin "$BRANCH"
WT="/tmp/fix-pr-$PR"
rm -rf "$WT"
git worktree add "$WT" "origin/$BRANCH"
```

### 2b. Identify and download the failing run's logs

The `Run Sweep` workflow produces many matrix jobs. Find the failing job(s) on the PR's head SHA, then download just the failing step logs to keep context small:

```bash
HEAD_SHA=$(gh pr view "$PR" --repo SemiAnalysisAI/InferenceX --json headRefOid --jq .headRefOid)

# Find the most recent Run Sweep run for this commit
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX --workflow "Run Sweep" \
  --commit "$HEAD_SHA" --limit 1 --json databaseId --jq '.[0].databaseId')

# Failing job IDs + names
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | select(.conclusion == "failure") | "\(.databaseId)\t\(.name)"' \
  > /tmp/failed_jobs_$PR.tsv

# Failure-only log dump (uses --log-failed)
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed \
  > /tmp/sweep_failed_log_$PR.txt
wc -l /tmp/sweep_failed_log_$PR.txt
```

If the log file is very large (>2000 lines), grep it for the actual error signatures before reading — common patterns: `Error`, `Traceback`, `RuntimeError`, `CUDA`, `HIP`, `OOM`, `assert`, `KeyError`, `ModuleNotFound`, `connection refused`, `exit code`, `failed to launch`. Read the surrounding context (~50 lines) around each hit.

### 2c. Diagnose

Inspect the PR diff (`git -C "$WT" diff origin/main...HEAD`) and the failing-log excerpts together. Most `claude/issue-1154-*` PRs are image-bump PRs that touch a `*.yaml` recipe — failures are usually:

- Image tag typo / unavailable tag → fix the image reference.
- Engine arg incompatibility with new image version → add/remove the affected flag in the recipe.
- New required env var or container path → patch the recipe.
- Resource ask too high for the runner → drop concurrency or tp.
- Flaky infra (network, runner pickup) → not a code fix; flag and skip.

State the suspected root cause in one or two sentences before proposing any edit.

### 2d. Apply a minimal fix, then push

Make the smallest possible edit that addresses the diagnosed cause. Run any local validation that's cheap (e.g. `python -c "import yaml; yaml.safe_load(open('<file>'))"`). Then:

```bash
git -C "$WT" add -A
git -C "$WT" -c user.name="claude-fix-bot" -c user.email="claude-fix-bot@local" commit -m "fix(<recipe>): <one-line root cause summary>"
```

**Show the diff to the user and ask for confirmation before pushing.** On confirm:

```bash
git -C "$WT" push origin "HEAD:$BRANCH"
git worktree remove "$WT"
```

If diagnosis is inconclusive (e.g. infra flake, unclear log, or the fix would be too large to be a one-shot patch), do **not** push. Record the PR as "needs human triage" with a short note on why.

## Step 3 — final report

Print a summary table:

| PR | Action | Note |
|---|---|---|
| [#NNNN](https://github.com/SemiAnalysisAI/InferenceX/pull/NNNN) | fix pushed (`<sha>`) | one-line diagnosis |
| [#NNNN](https://github.com/SemiAnalysisAI/InferenceX/pull/NNNN) | skipped | reason (flake / unclear / too large) |

Do **not** merge anything. The pushed commit will re-trigger sweep on the PR; review results via `/list-claude-pr-status` later.
