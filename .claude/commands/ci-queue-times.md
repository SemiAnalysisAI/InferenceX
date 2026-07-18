---
description: Show CI cluster pressure right now — per-pool runners online/busy vs jobs active/queued, plus the live queue with wait-so-far
argument-hint: [pool-filter-regex]   # optional, e.g. "h200" or "b200|mi355x" to restrict output
---

GitHub has no queue-metrics API for self-hosted runners, so this derives cluster
pressure from the Actions API: a job's **wait so far = now − `created_at`** (queued
jobs) and **runtime so far = now − `started_at`** (active jobs). Pool membership
comes from the repo's own `configs/runners.yaml` (the authoritative pool →
runner-name mapping), so run this **from the repo root**.

`$ARGUMENTS` (optional) is a regex restricting which pools are shown in Steps 2–4
(e.g. `h200`). Run all steps and report the tables as-is.

**Key facts for interpreting the numbers:**

- A self-hosted runner agent executes **exactly one job at a time**, so per pool
  `jobs_active` should always be ≤ `runners_busy`. Parallelism comes from multiple
  registered agents per pool (`h100-dgxc-slurm_00` … `_19`).
- Pools overlap **by design** in `configs/runners.yaml` (e.g. `h100-multinode` ⊂
  `h100`, and the same `b200-dgxc_*` machines serve `b200`, `b200-dsv4`, and
  `b200-dgxc` jobs), so one physical runner can count toward several pool rows.
- A multi-node SLURM job still occupies exactly one agent (the orchestrator) — the
  nodes it allocates inside SLURM are invisible here.

## Step 1 — Take one snapshot of jobs (active + queued) and runners

A run can be `in_progress` while its matrix jobs are still `queued` waiting for
runners (the sweep fan-out), so scan both statuses. Every list call uses
`--paginate` — without it the runs lists are silently capped at the first page.

```bash
SNAP=$(mktemp /tmp/ci_snapshot.XXXXXX.ndjson)
RUNNERS=$(mktemp /tmp/ci_runners.XXXXXX.ndjson)
POOLS=$(mktemp /tmp/ci_pools.XXXXXX.tsv)

# Pool → runner-name membership, from the repo's own config (bare pool keys and
# cluster:* keys are normalized to the same bucket: cluster:b200-dgxc → b200-dgxc).
awk '/^  [A-Za-z0-9:-]+:$/ { k=$1; sub(":$","",k); sub("^cluster:","",k); pool=k; next }
     /^    - / { print pool "\t" $2 }' configs/runners.yaml > "$POOLS"

# Jobs, bucketed by the pool label they requested (demand side).
{ gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=queued&per_page=100" --paginate --jq '.workflow_runs[].id'
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=in_progress&per_page=100" --paginate --jq '.workflow_runs[].id'
} | sort -u | while read -r RUN; do
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN/jobs?per_page=100" --paginate \
    --jq '.jobs[] | select(.status=="in_progress" or .status=="queued")
      | {c: ([.labels[] | select(test("^(cluster:)?(b200|b300|h100|h200|gb200|gb300|mi300x|mi325x|mi355x)")) | sub("^cluster:";"") | sub("_\\d+$";"")] | first // "other"),
         status, name, created_at, started_at, run_id}
      | select(.c != "other")'
done > "$SNAP"

# Runners by name (capacity side) — names match configs/runners.yaml entries.
gh api "repos/SemiAnalysisAI/InferenceX/actions/runners?per_page=100" --paginate \
  --jq '.runners[] | {name, status, busy}' > "$RUNNERS"
```

## Step 2 — Per-pool summary: runners online/busy vs jobs active/queued

Joins demand (job labels) against capacity (`configs/runners.yaml` membership):

```bash
jq -n -r --slurpfile jobs "$SNAP" --slurpfile runners "$RUNNERS" --rawfile pools "$POOLS" '
  ($pools | split("\n") | map(select(contains("\t")) | split("\t"))
    | group_by(.[1]) | map({key: .[0][1], value: [.[].[0]]}) | from_entries) as $name2pools
  | ($jobs | group_by(.c) | map({key: .[0].c, value: {
      active: ([.[] | select(.status=="in_progress")] | length),
      queued: ([.[] | select(.status=="queued")] | length)}}) | from_entries) as $j
  | ([ $runners[] as $r | ($name2pools[$r.name] // [])[] as $p | {c: $p, status: $r.status, busy: $r.busy} ]
     | group_by(.c) | map({key: .[0].c, value: {
         online: ([.[] | select(.status=="online")] | length),
         busy:   ([.[] | select(.busy==true)] | length)}}) | from_entries) as $r
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
| `busy > active` | the pool's runners are busy serving **another pool's** jobs (shared membership in `configs/runners.yaml`, e.g. `b200` vs `b200-dgxc`), stuck "busy" after a cancellation, or busy with another repo's jobs (runners may be org-scoped) |
| `active > busy` | impossible in steady state — treat as snapshot skew between the API calls |
| pool absent entirely | no online runners and no jobs — either idle-and-offline, or (if jobs *do* target it) the pool is not defined in `configs/runners.yaml` |

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

- **Demand vs capacity bucketing:** jobs are bucketed by the label they *request*
  (single-node jobs use pool labels like `b200`, multi-node jobs use `cluster:*`
  labels — the `sub()` normalization merges them). Runners are bucketed strictly by
  `configs/runners.yaml` membership, so bogus rows from per-runner name labels
  (e.g. `h100-dgxc-slurm`) cannot appear.
- **`gh api --jq` does not accept jq `--arg`** — extra arguments are parsed as API
  parameters. Interpolate values into the `--jq` string via the shell instead.
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
