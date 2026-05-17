---
description: Find Claude-authored PRs with all-green full-sweep validation and confirm before merging
---

Find open PRs authored by Claude (branches starting with `claude/`) whose full-sweep validation has completed all-green, then prompt the user before merging.

## Step 1 — fetch open PR status

```bash
gh pr list --repo SemiAnalysisAI/InferenceX --state open --limit 200 \
  --json number,title,headRefName,statusCheckRollup > /tmp/claude_prs.json
```

## Step 2 — filter for fully-validated `claude/*` PRs

A PR qualifies only if **all** of the following hold:

- `headRefName` starts with `claude/`
- No check has conclusion `FAILURE`, `CANCELLED`, or `TIMED_OUT`
- No check has status `QUEUED`, `IN_PROGRESS`, or `PENDING` (i.e. sweep is finished, not still running)
- At least one `Run Sweep` check has conclusion `SUCCESS` (i.e. the sweep actually ran — not all skipped)

```bash
jq -r '.[]
  | select(.headRefName | startswith("claude/"))
  | . as $p
  | ([$p.statusCheckRollup[] | (.conclusion // .status)]) as $s
  | select(($s | any(. == "FAILURE" or . == "CANCELLED" or . == "TIMED_OUT" or . == "QUEUED" or . == "IN_PROGRESS" or . == "PENDING")) | not)
  | select([$p.statusCheckRollup[] | select(.workflowName == "Run Sweep" and (.conclusion // .status) == "SUCCESS")] | length > 0)
  | "\(.number)\thttps://github.com/SemiAnalysisAI/InferenceX/pull/\(.number)\t\(.title)"' /tmp/claude_prs.json
```

## Step 3 — present the list and ask for confirmation

Show the matching PRs as a table with PR number, link, and title. Then **stop and ask the user to confirm** before doing anything else. Do not auto-merge.

If the user confirms, invoke `/merge-prs <pr-numbers...>` with the confirmed PR numbers.
If the user declines or wants a subset, run `/merge-prs` only on the subset they specify.
