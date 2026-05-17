---
description: List Claude-authored PRs that haven't failed (ready or still running) with their actual check state
---

List open PRs authored by Claude (branches starting with `claude/`) that have **not** had any check fail. Show each PR's actual state (READY when all checks finished green, RUNNING when sweeps are still queued/in-progress) along with a per-status check breakdown, rendered as a markdown table.

## Step 1 — list candidate `claude/*` PRs

`gh pr list --json statusCheckRollup` truncates each PR's rollup, so it can't be trusted for the per-check filter. Use it only to get the candidate numbers, then re-query each PR individually.

```bash
gh pr list --repo SemiAnalysisAI/InferenceX --state open --limit 200 \
  --json number,title,headRefName \
  --jq '.[] | select(.headRefName | startswith("claude/")) | "\(.number)\t\(.title)"' \
  > /tmp/claude_pr_candidates.tsv
```

## Step 2 — per-PR state classification

For each candidate, fetch the full rollup with `gh pr view`. Compute each check's effective state as `if (.conclusion // "") != "" then .conclusion else .status end` — `gh` returns `conclusion: ""` for in-flight checks, so jq's `//` does not fall through to `.status`.

Classify the PR as:
- **FAILED** — any check is `FAILURE`, `CANCELLED`, or `TIMED_OUT`. Skip these.
- **RUNNING** — no failed checks, but at least one check is `QUEUED`, `IN_PROGRESS`, or `PENDING`.
- **READY** — no failed checks, no pending checks, and at least one `Run Sweep` check is `SUCCESS` (sweep actually ran — not all skipped).
- **NO_SWEEP** — no failed, no pending, but the sweep never produced a `SUCCESS` (all skipped or never ran). Skip these.

```bash
: > /tmp/claude_pr_status.tsv
while IFS=$'\t' read -r pr title; do
  rollup=$(gh pr view "$pr" --repo SemiAnalysisAI/InferenceX --json statusCheckRollup)
  classification=$(printf '%s' "$rollup" | jq -r '
    def state: if (.conclusion // "") != "" then .conclusion else .status end;
    . as $p
    | ([$p.statusCheckRollup[] | state]) as $s
    | ($s | any(. == "FAILURE" or . == "CANCELLED" or . == "TIMED_OUT")) as $failed
    | ($s | any(. == "QUEUED" or . == "IN_PROGRESS" or . == "PENDING")) as $pending
    | ([$p.statusCheckRollup[] | select(.workflowName == "Run Sweep" and (state) == "SUCCESS")] | length > 0) as $swept
    | if $failed then "FAILED"
      elif $pending then "RUNNING"
      elif $swept then "READY"
      else "NO_SWEEP" end')
  if [ "$classification" = "READY" ] || [ "$classification" = "RUNNING" ]; then
    breakdown=$(printf '%s' "$rollup" | jq -r '
      def state: if (.conclusion // "") != "" then .conclusion else .status end;
      [.statusCheckRollup[] | state] | group_by(.) | map("\(.[0])=\(length)") | join(" ")')
    printf '%s\t%s\t%s\t%s\n' "$pr" "$classification" "$breakdown" "$title" >> /tmp/claude_pr_status.tsv
  fi
done < /tmp/claude_pr_candidates.tsv
```

## Step 3 — render markdown table

Print the result directly as a markdown table. READY rows first, then RUNNING. Each PR is a clickable link.

```bash
{
  printf '| PR | State | Check breakdown | Title |\n'
  printf '|---|---|---|---|\n'
  # READY first, then RUNNING; within each group keep input order (descending PR number from gh)
  awk -F'\t' '$2 == "READY"'   /tmp/claude_pr_status.tsv
  awk -F'\t' '$2 == "RUNNING"' /tmp/claude_pr_status.tsv
} | awk -F'\t' 'NR<=2 {print; next}
                {printf "| [#%s](https://github.com/SemiAnalysisAI/InferenceX/pull/%s) | %s | `%s` | %s |\n", $1, $1, $2, $3, $4}'
```

If `/tmp/claude_pr_status.tsv` is empty, print: `_No claude/* PRs are currently READY or RUNNING — all open Claude PRs have failures or no sweep results._`

Output the resulting markdown table to the user verbatim. This command is informational only — do **not** auto-merge.
