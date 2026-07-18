---
description: Check CI queue times across all GPU clusters — live queued jobs with wait so far, historical per-cluster queue latency, and per-pool runner availability
argument-hint: [history-run-limit]   # optional, default 150 recent runs
---

GitHub has no queue-metrics API for self-hosted runners, so queue time is derived
from the Actions API: **queue time per job = `started_at − created_at`** (time spent
waiting for a free runner), and the cluster is the job's runner label. The pool →
runner-name mapping lives in `configs/runners.yaml`.

Run all three sections and report them as compact tables. `$ARGUMENTS` (optional)
overrides the history window in Step 2 (default 150 recent runs).

## Step 1 — Live queue: who is waiting right now, and for how long

A run can be `in_progress` while its matrix jobs are still `queued` waiting for
runners (the sweep fan-out), so scan both statuses:

```bash
{ gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=queued&per_page=50" --jq '.workflow_runs[].id'
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs?status=in_progress&per_page=50" --jq '.workflow_runs[].id'
} | sort -u | while read -r RUN; do
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN/jobs?per_page=100" --paginate \
    --jq '.jobs[] | select(.status=="queued") | {labels: [.labels[] | sub("^cluster:";"") | select(test("^(self-hosted|Linux|X64|ARM64|slurm)$|_\\d+$") | not)], name, created_at, run_id}'
done | jq -s -r 'sort_by(.created_at) | .[]
  | [((now - (.created_at|fromdateiso8601))/60 | floor | tostring) + " min",
     (.labels | join(",")), (.name[0:60]), (.run_id|tostring)] | @tsv' | column -t -s$'\t'
```

## Step 2 — Historical queue latency per cluster

Uses the last N runs (default 150; `$ARGUMENTS` overrides). **Filter `.q >= 0` is
mandatory** — re-run jobs keep `started_at` from the first attempt and produce
negative values. For a wider window raise the limit or add
`--created ">YYYY-MM-DD"` to `gh run list`.

```bash
LIMIT="${ARGUMENTS:-150}"
gh run list --repo SemiAnalysisAI/InferenceX --limit "$LIMIT" --json databaseId --jq '.[].databaseId' | while read -r RUN; do
  gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN/jobs?per_page=100" --paginate \
    --jq '.jobs[] | select(.started_at != null)
      | {c: ([.labels[] | select(test("^(cluster:)?(b200|b300|h100|h200|gb200|gb300|mi300x|mi325x|mi355x)")) | sub("^cluster:";"") | sub("_\\d+$";"")] | first // empty),
         q: (((.started_at|fromdateiso8601) - (.created_at|fromdateiso8601))/60)}
      | select(.c != "" and .q >= 0)'
done | jq -s -r 'group_by(.c)
  | map({cluster: .[0].c, n: length, avg: (map(.q)|add/length),
         p50: (map(.q)|sort|.[(length*0.5|floor)]), max: (map(.q)|max)})
  | sort_by(-.p50) | .[]
  | [.cluster, (.n|tostring), ((.avg*10|round)/10|tostring), ((.p50*10|round)/10|tostring), ((.max|round)|tostring)] | @tsv' \
  | (printf "cluster\tjobs\tavg_min\tp50_min\tmax_min\n"; cat) | column -t -s$'\t'
```

Note: the sample only covers clusters that actually ran within the window — a
missing cluster means no recent jobs, not zero queue.

## Step 3 — Runner availability per pool (the "why" behind queue times)

```bash
gh api "repos/SemiAnalysisAI/InferenceX/actions/runners?per_page=100" --paginate \
  --jq '.runners[] | . as $r | [.labels[].name | select(test("^(cluster:)?(b200|b300|h100|h200|gb200|gb300|mi300x|mi325x|mi355x)")) | sub("^cluster:";"") | sub("_\\d+$";"")] | unique | .[] | {c: ., status: $r.status, busy: $r.busy}' \
| jq -s -r 'group_by(.c) | map({c: .[0].c, total: length,
    online: ([.[]|select(.status=="online")]|length),
    busy: ([.[]|select(.busy==true)]|length)}) | sort_by(.c) | .[]
  | [.c, (.total|tostring), (.online|tostring), (.busy|tostring)] | @tsv' \
| (printf "pool\ttotal_runners\tonline\tbusy\n"; cat) | column -t -s$'\t'
```

A pool with `online == busy` (or many runners `offline`) fully explains long queues
for jobs targeting it.

## Caveats

- **Label shapes:** single-node jobs carry pool labels (`b200`, `mi355x`);
  multi-node jobs carry `cluster:*` labels (`cluster:gb200-nv`); runners also carry
  unique per-runner labels (`h100-dgxc-slurm_00`). The `sub()` calls normalize all
  of these — do not drop the `cluster:` prefix handling or multi-node jobs silently
  vanish from the results.
- **Re-run artifacts:** always keep the `.q >= 0` filter (see Step 2).
- **SLURM second queue:** for SLURM-backed pools (`*-dgxc`, `*-nv`), once the
  GitHub job starts there is a second queue inside SLURM that GitHub-side numbers
  do not capture. Check it on the cluster with `squeue -u <runner-user>` (see
  `.claude/skills/debug-runs/SKILL.md`).
- Queue time measured this way excludes time spent behind the sweep canary gate:
  matrix jobs are only *created* after the gate passes, so their `created_at` marks
  genuine runner-wait start.
