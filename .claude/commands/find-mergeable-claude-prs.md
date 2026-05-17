---
description: Find Claude-authored PRs with all-green full-sweep validation and confirm before merging
---

Find open PRs authored by Claude (branches starting with `claude/`) whose full-sweep validation has completed all-green, then prompt the user before merging.

## Step 1 — list candidate `claude/*` PRs

`gh pr list --json statusCheckRollup` truncates each PR's rollup, so it can't be trusted for the per-check filter. Use it only to get the candidate numbers, then re-query each PR individually.

```bash
gh pr list --repo SemiAnalysisAI/InferenceX --state open --limit 200 \
  --json number,title,headRefName \
  --jq '.[] | select(.headRefName | startswith("claude/")) | .number' \
  > /tmp/claude_pr_candidates.txt
```

## Step 2 — per-PR all-green check

For each candidate, fetch the full rollup with `gh pr view`. A PR qualifies only if **all** of the following hold:

- No check has conclusion `FAILURE`, `CANCELLED`, or `TIMED_OUT`
- No check has status `QUEUED`, `IN_PROGRESS`, or `PENDING` (sweep finished, not still running)
- At least one `Run Sweep` check has conclusion `SUCCESS` (sweep actually ran — not all skipped)

Note: `gh` returns `conclusion: ""` (empty string, not `null`) for in-flight checks, so jq's `//` operator does **not** fall through to `.status`. Each check's effective state must be computed as `if conclusion is non-empty then conclusion else status`.

```bash
: > /tmp/claude_prs_green.txt
while read -r pr; do
  is_green=$(gh pr view "$pr" --repo SemiAnalysisAI/InferenceX --json statusCheckRollup --jq '
    def state: if (.conclusion // "") != "" then .conclusion else .status end;
    . as $p
    | ([$p.statusCheckRollup[] | state]) as $s
    | ($s | any(. == "FAILURE" or . == "CANCELLED" or . == "TIMED_OUT" or . == "QUEUED" or . == "IN_PROGRESS" or . == "PENDING")) as $bad
    | ([$p.statusCheckRollup[] | select(.workflowName == "Run Sweep" and (state) == "SUCCESS")] | length > 0) as $swept
    | (($bad | not) and $swept)')
  if [ "$is_green" = "true" ]; then
    title=$(gh pr view "$pr" --repo SemiAnalysisAI/InferenceX --json title --jq '.title')
    printf '%s\thttps://github.com/SemiAnalysisAI/InferenceX/pull/%s\t%s\n' "$pr" "$pr" "$title" >> /tmp/claude_prs_green.txt
  fi
done < /tmp/claude_pr_candidates.txt
cat /tmp/claude_prs_green.txt
```

## Step 3 — present the list and ask for confirmation

Show the matching PRs as a table with PR number, link, and title. Then **stop and ask the user to confirm** before doing anything else. Do not auto-merge.

If the user confirms, invoke `/merge-prs <pr-numbers...>` with the confirmed PR numbers.
If the user declines or wants a subset, run `/merge-prs` only on the subset they specify.
