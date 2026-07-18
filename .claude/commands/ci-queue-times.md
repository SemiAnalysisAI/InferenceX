---
description: Show CI cluster pressure right now — active jobs per GPU pool, the live queue with wait-so-far, and a per-pool active/queued summary
argument-hint: [pool-filter-regex]   # optional, e.g. "h200" or "b200|mi355x" to restrict output
---

GitHub has no queue-metrics API for self-hosted runners, so this derives cluster
pressure from the Actions API: a job's **wait so far = now − `created_at`** (queued
jobs) and **runtime so far = now − `started_at`** (active jobs). The cluster is the
job's runner label; the pool → runner-name mapping lives in `configs/runners.yaml`.

`$ARGUMENTS` (optional) is a regex restricting which pools are shown in Steps 2–4
(e.g. `h200`). Run all steps and report the tables as-is.

## Step 1 — Take one snapshot of all GPU jobs (active + queued)

A run can be `in_progress` while its matrix jobs are still `queued` waiting for
runners (the sweep fan-out), so scan both statuses in a single pass:

```bash
SNAP=$(mktemp /tmp/ci_snapshot.XXXXXX.ndjson)
{ gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=queued&per_page=100" --jq '.workflow_runs[].id'
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=in_progress&per_page=100" --jq '.workflow_runs[].id'
} | sort -u | while read -r RUN; do
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN/jobs?per_page=100" --paginate \
    --jq '.jobs[] | select(.status=="in_progress" or .status=="queued")
      | {c: ([.labels[] | select(test("^(cluster:)?(b200|b300|h100|h200|gb200|gb300|mi300x|mi325x|mi355x)")) | sub("^cluster:";"") | sub("_\\d+$";"")] | first // "other"),
         status, name, created_at, started_at, run_id}
      | select(.c != "other")'
done > "$SNAP"
```

Sanity-check coverage: compare against
`gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=in_progress&per_page=1" --jq .total_count`
(and the same for `queued`) — if either exceeds 100, paginate the runs list too.

## Step 2 — Per-pool summary (active vs queued)

```bash
jq -s -r 'group_by(.c) | map({c: .[0].c,
    active: ([.[] | select(.status=="in_progress")] | length),
    queued: ([.[] | select(.status=="queued")] | length)})
  | sort_by(-.queued, -.active) | .[] | [.c, (.active|tostring), (.queued|tostring)] | @tsv' "$SNAP" \
  | (printf "pool\tactive\tqueued\n"; cat) | column -t -s$'\t'
```

A pool with **0 active but a growing queue** means its runners are offline, stuck
"busy" (cancelled-job artifact), or busy with another repo's jobs (runners may be
org-scoped) — investigate the fleet, not the workflow.

## Step 3 — Active jobs (what is running, and for how long)

```bash
jq -s -r '[.[] | select(.status=="in_progress")] | sort_by(.c, .started_at) | .[]
  | [.c, (((now - (.started_at|fromdateiso8601))/60 | floor | tostring) + " min"), (.name[0:58]), (.run_id|tostring)] | @tsv' "$SNAP" \
  | column -t -s$'\t'
```

## Step 4 — Queued jobs (what is waiting, and for how long)

```bash
jq -s -r '[.[] | select(.status=="queued")] | sort_by(.c, .created_at) | .[]
  | [.c, (((now - (.created_at|fromdateiso8601))/60 | floor | tostring) + " min"), (.name[0:58]), (.run_id|tostring)] | @tsv' "$SNAP" \
  | column -t -s$'\t'
```

To apply `$ARGUMENTS`, pipe Steps 2–4 through `grep -E "<pattern>"` (keep the
header line of Step 2).

## Caveats

- **Label shapes:** single-node jobs carry pool labels (`b200`, `mi355x`);
  multi-node jobs carry `cluster:*` labels (`cluster:gb200-nv`); runners also carry
  unique per-runner labels (`h100-dgxc-slurm_00`). The `sub()` calls normalize all
  of these — do not drop the `cluster:` prefix handling or multi-node jobs silently
  vanish from the results.
- **Re-run artifacts:** re-run jobs can report `started_at` from an earlier
  attempt, inflating runtime in Step 3. Cross-check an outlier against the job's
  page before believing it.
- **SLURM second queue:** for SLURM-backed pools (`*-dgxc`, `*-nv`), once the
  GitHub job starts there is a second queue inside SLURM that GitHub-side numbers
  do not capture. Check it on the cluster with `squeue -u <runner-user>` (see
  `.claude/skills/debug-runs/SKILL.md`).
- Queue time measured this way excludes time spent behind the sweep canary gate:
  matrix jobs are only *created* after the gate passes, so their `created_at` marks
  genuine runner-wait start.
- Runner capacity per pool (online/busy/offline) is a separate question — use
  `gh api repos/SemiAnalysisAI/InferenceX/actions/runners?per_page=100 --paginate`
  and bucket labels the same way as Step 1 if needed.
