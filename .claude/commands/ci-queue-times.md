---
description: Show CI cluster pressure right now — per-pool runners online/busy vs jobs active/queued, plus the live queue with wait-so-far
argument-hint: [pool-filter-regex]   # optional, e.g. "h200" or "b200|mi355x" to restrict output
---

GitHub has no queue-metrics API for self-hosted runners, so this derives cluster
pressure from the Actions API: a job's **wait so far = now − `created_at`** (queued
jobs) and **runtime so far = now − `started_at`** (active jobs). The cluster is the
job's runner label; the pool → runner-name mapping lives in `configs/runners.yaml`.

`$ARGUMENTS` (optional) is a regex restricting which pools are shown in Steps 2–4
(e.g. `h200`). Run all steps and report the tables as-is.

**Key fact for interpreting the numbers:** a self-hosted runner agent executes
**exactly one job at a time**, so per pool `jobs_active` should always be ≤
`runners_busy`. Parallelism comes from multiple registered agents per pool
(`h100-dgxc-slurm_00` … `_19`). A multi-node SLURM job still occupies exactly one
agent (the orchestrator) — the nodes it allocates inside SLURM are invisible here.

## Step 1 — Take one snapshot of jobs (active + queued) and runners

A run can be `in_progress` while its matrix jobs are still `queued` waiting for
runners (the sweep fan-out), so scan both statuses in a single pass:

```bash
SNAP=$(mktemp /tmp/ci_snapshot.XXXXXX.ndjson)
RUNNERS=$(mktemp /tmp/ci_runners.XXXXXX.ndjson)

{ gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=queued&per_page=100" --jq '.workflow_runs[].id'
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=in_progress&per_page=100" --jq '.workflow_runs[].id'
} | sort -u | while read -r RUN; do
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN/jobs?per_page=100" --paginate \
    --jq '.jobs[] | select(.status=="in_progress" or .status=="queued")
      | {c: ([.labels[] | select(test("^(cluster:)?(b200|b300|h100|h200|gb200|gb300|mi300x|mi325x|mi355x)")) | sub("^cluster:";"") | sub("_\\d+$";"")] | first // "other"),
         status, name, created_at, started_at, run_id}
      | select(.c != "other")'
done > "$SNAP"

gh api "repos/SemiAnalysisAI/InferenceX/actions/runners?per_page=100" --paginate \
  --jq '.runners[] | . as $r | [.labels[].name | select(test("^(cluster:)?(b200|b300|h100|h200|gb200|gb300|mi300x|mi325x|mi355x)")) | sub("^cluster:";"") | sub("_\\d+$";"")] | unique | .[] | {c: ., status: $r.status, busy: $r.busy}' \
  > "$RUNNERS"
```

Sanity-check coverage: compare against
`gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=in_progress&per_page=1" --jq .total_count`
(and the same for `queued`) — if either exceeds 100, paginate the runs list too.

## Step 2 — Per-pool summary: runners online/busy vs jobs active/queued

```bash
jq -n -r --slurpfile jobs "$SNAP" --slurpfile runners "$RUNNERS" '
  ($jobs | group_by(.c) | map({key: .[0].c, value: {
      active: ([.[] | select(.status=="in_progress")] | length),
      queued: ([.[] | select(.status=="queued")] | length)}}) | from_entries) as $j
  | ($runners | group_by(.c) | map({key: .[0].c, value: {
      online: ([.[] | select(.status=="online")] | length),
      busy:   ([.[] | select(.busy == true)] | length)}}) | from_entries) as $r
  | [([$j, $r] | map(keys) | add | unique | .[]) as $c
      | {c: $c, on: ($r[$c].online // 0), busy: ($r[$c].busy // 0),
         active: ($j[$c].active // 0), queued: ($j[$c].queued // 0)}]
  | map(select(.on + .busy + .active + .queued > 0))
  | sort_by(-.queued, -.active)
  | .[] | [.c, (.on|tostring), (.busy|tostring), (.active|tostring), (.queued|tostring)] | @tsv' \
  | (printf "pool\trunners_online\trunners_busy\tjobs_active\tjobs_queued\n"; cat) | column -t -s$'\t'
```

How to read it (one job per runner agent, so these patterns are diagnostic):

| pattern | meaning |
|---|---|
| `busy == active`, `queued == 0` | healthy, pool has spare capacity if `busy < online` |
| `busy == online`, `queued > 0` | pool saturated — jobs waiting for a free runner (normal queue) |
| `online == 0`, `queued > 0` | **pool is down** — no live runners; investigate the fleet, not the workflow |
| `busy > active` | runners stuck "busy" (cancelled-job artifact) or busy with another repo's jobs (runners may be org-scoped) |
| `active > busy` | impossible in steady state — treat as snapshot skew between the two API calls |

Note a physical runner counts toward **every** pool label it carries (e.g. the
same machine appears in `h200`, `h200-dgxc`, and `h200-dgxc-slurm`), so summing
rows over-counts physical machines.

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
- Snapshot skew: jobs and runners are fetched a few seconds apart; a job that
  starts/ends in between can briefly violate `active <= busy`.
