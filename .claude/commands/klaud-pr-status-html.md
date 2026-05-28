---
description: Render an HTML dashboard of Claude/Klaud-Cold PR states (state + check breakdown per PR) and open it in the browser
---

Render an HTML dashboard for every open PR in `SemiAnalysisAI/InferenceX` that was opened by Claude (either a `claude/*` branch OR a title containing `[Klaud Cold]`). Each row shows the PR's current state, a check-status breakdown, the title, and empty "Reason"/"Suggested fix" cells you can fill in afterward by reading failed-run logs.

The dashboard lives at `/tmp/klaud_pr_status.html` and is opened with `open` (macOS) at the end.

## Step 1 — list candidate PRs (`claude/*` OR title containing `[Klaud Cold]`)

The title check uses `contains` (not `startswith`) so it picks up PRs whose titles embed `[Klaud Cold]` after a prefix like `[Handoff to @Oseltamivir Claude /loop]` — handoff-style PRs from a /loop run still belong on the dashboard.

```bash
gh pr list --repo SemiAnalysisAI/InferenceX --state open --limit 200 \
  --json number,title,headRefName,createdAt \
  --jq '.[] | select((.headRefName | startswith("claude/")) or (.title | contains("[Klaud Cold]"))) | "\(.number)\t\(.headRefName)\t\(.createdAt)\t\(.title)"' \
  > /tmp/klaud_pr_candidates.tsv
wc -l /tmp/klaud_pr_candidates.tsv
```

## Step 2 — per-PR state classification

`gh pr list --json statusCheckRollup` truncates rollups, so enumerate candidates first then re-query each PR individually.

Each check's effective state is `if (.conclusion // "") != "" then .conclusion else .status end` — `gh` returns `conclusion: ""` (not `null`) for in-flight checks, so jq's `//` does not fall through to `.status`.

State buckets:
- **FAILED** — at least one check is `FAILURE` / `CANCELLED` / `TIMED_OUT`, AND no checks are still pending.
- **FAILED+RUNNING** — at least one failed check AND at least one pending check (sweep partially failed; some matrix jobs still running).
- **RUNNING** — no failed checks; at least one is `QUEUED` / `IN_PROGRESS` / `PENDING`.
- **READY** — no failed, no pending, and at least one `Run Sweep` check is `SUCCESS`.
- **NO_SUCCESS** — sweep ran but never produced a `SUCCESS` (e.g. all matrix jobs got SKIPPED).
- **NO_SWEEP** — no `Run Sweep` check exists for this head SHA at all (sweep never triggered — usually missing a sweep label such as `full-sweep-enabled` or `non-canary-full-sweep-enabled`).

```bash
: > /tmp/klaud_pr_status.tsv
: > /tmp/klaud_pr_jobs.tsv   # per-job (pr, pool, state) for queue-aware ETA
while IFS=$'\t' read -r pr branch created title; do
  rollup=$(gh pr view "$pr" --repo SemiAnalysisAI/InferenceX --json statusCheckRollup,headRefOid)
  classification=$(printf '%s' "$rollup" | jq -r '
    def state: if (.conclusion // "") != "" then .conclusion else .status end;
    . as $p
    | ([$p.statusCheckRollup[] | state]) as $s
    | ($s | any(. == "FAILURE" or . == "CANCELLED" or . == "TIMED_OUT")) as $failed
    | ($s | any(. == "QUEUED" or . == "IN_PROGRESS" or . == "PENDING")) as $pending
    | ([$p.statusCheckRollup[] | select(.workflowName == "Run Sweep" and (state) == "SUCCESS")] | length > 0) as $swept
    | ([$p.statusCheckRollup[] | select(.workflowName == "Run Sweep")] | length > 0) as $hasweep
    | if $failed and $pending then "FAILED+RUNNING"
      elif $failed then "FAILED"
      elif $pending then "RUNNING"
      elif $swept then "READY"
      elif $hasweep then "NO_SUCCESS"
      else "NO_SWEEP" end')
  breakdown=$(printf '%s' "$rollup" | jq -r '
    def state: if (.conclusion // "") != "" then .conclusion else .status end;
    [.statusCheckRollup[] | state] | group_by(.) | map("\(.[0])=\(length)") | join(" ")')
  head_sha=$(printf '%s' "$rollup" | jq -r '.headRefOid')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pr" "$classification" "$breakdown" "$branch" "$created" "$head_sha" "$title" >> /tmp/klaud_pr_status.tsv

  # Per-job pool + state for queue-aware ETA (only Run Sweep jobs that are pending).
  # Pool extracted from job name via regex; "unknown" if no match.
  printf '%s' "$rollup" | jq -r --arg pr "$pr" '
    def state: if (.conclusion // "") != "" then .conclusion else .status end;
    .statusCheckRollup[]
    | select(.workflowName == "Run Sweep")
    | state as $s
    | select($s == "QUEUED" or $s == "IN_PROGRESS")
    | ((.name | capture("(?<p>b200|b300|h100|h200|mi300x|mi325x|mi355x)").p) // "unknown") as $pool
    | "\($pr)\t\($pool)\t\($s)"
  ' >> /tmp/klaud_pr_jobs.tsv
done < /tmp/klaud_pr_candidates.tsv
```

## Step 3 — render HTML and open

State render order (action items first): `FAILED` → `FAILED+RUNNING` → `NO_SWEEP` → `NO_SUCCESS` → `RUNNING` → `READY`. Within each bucket, descending PR number.

If you have per-PR diagnoses to inject (e.g. after running `/fix-klaud-cron-prs`), write them as a JSON map `{ "1461": {"reason": "...", "fix": "..."}, ... }` to `/tmp/klaud_pr_diag.json` BEFORE running this step — the generator will pick them up. HTML may contain inline `<code>` tags.

```bash
cat > /tmp/gen_klaud_pr_status_html.py <<'PYEOF'
#!/usr/bin/env python3
"""Render the Claude/Klaud-Cold PR status HTML dashboard.

Per-PR ETA is computed as a *pool drain time* — for each runner pool the PR has
pending jobs in, ETA_pool = ceil(global_pool_pending / pool_runners) × per-job
time. The PR's overall ETA is max(ETA_pool) across the pools it touches.

This is an upper bound, not an SLA — GitHub Actions does not guarantee FIFO
dispatch outside `concurrency:` groups and does not expose queue position
(see docs.github.com/en/actions). Treat ETAs as ordering hints only.
"""
import html, json, math, re, datetime as dt
from collections import defaultdict
from pathlib import Path

tsv = Path("/tmp/klaud_pr_status.tsv").read_text().strip().splitlines()
jobs_tsv_path = Path("/tmp/klaud_pr_jobs.tsv")
jobs_tsv = jobs_tsv_path.read_text().strip().splitlines() if jobs_tsv_path.exists() else []
diag_path = Path("/tmp/klaud_pr_diag.json")
diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}

state_counts = {}
state_order = {"FAILED": 0, "FAILED+RUNNING": 1, "NO_SWEEP": 2, "NO_SUCCESS": 3, "RUNNING": 4, "READY": 5}
state_class = {
    "READY": "state-READY", "RUNNING": "state-RUNNING",
    "FAILED": "state-FAILED", "FAILED+RUNNING": "state-FAILED",
    "NO_SWEEP": "state-NOSWEEP", "NO_SUCCESS": "state-NOSWEEP",
}

# Active self-hosted runner counts per pool (from GHA registrations as of the
# session this was last touched). If a pool isn't listed, falls back to 4 — a
# conservative guess. To refresh:
#   gh api --paginate repos/SemiAnalysisAI/InferenceX/actions/runners \
#     --jq '.runners[] | select(.status == "online") | .labels[].name' \
#     | grep -xE 'b200|b300|h200|h100|mi300x|mi325x|mi355x' | sort | uniq -c
POOL_RUNNERS = {"b200": 12, "b300": 18, "h100": 19, "h200": 18,
                "mi300x": 9, "mi325x": 9, "mi355x": 9}
DEFAULT_POOL_RUNNERS = 4
AVG_JOB_MIN = 7  # rough sweep-job median; eval+1k1k ~5min, 8k1k+agentic ~10-15min

def parse_breakdown(breakdown):
    counts = {}
    for kv in breakdown.split():
        m = re.match(r"^([A-Z_]+)=(\d+)$", kv)
        if m: counts[m.group(1)] = int(m.group(2))
    return counts

# Aggregate per-pool global pending across all open Klaud-Cold/claude PRs +
# per-PR per-pool pending (so we know which pools each PR's queue traffic hits).
pool_global_pending = defaultdict(int)         # pool → total pending across all PRs
pr_pool_pending = defaultdict(lambda: defaultdict(int))  # pr → pool → pending count
for line in jobs_tsv:
    parts = line.split("\t")
    if len(parts) < 3: continue
    pr, pool, state = parts[0], parts[1], parts[2]
    pool_global_pending[pool] += 1
    pr_pool_pending[pr][pool] += 1

def eta_min_for_pr(pr):
    """Per-pool pessimistic ETA: pool drain time = ceil(global_pending / runners) × avg_job_min.
    PR's ETA is max across pools (pools run in parallel; slowest one bounds completion)."""
    pools = pr_pool_pending.get(pr, {})
    if not pools: return 0
    eta_min = 0
    for pool in pools:
        runners = POOL_RUNNERS.get(pool, DEFAULT_POOL_RUNNERS)
        drain = math.ceil(pool_global_pending[pool] / runners) * AVG_JOB_MIN
        if drain > eta_min: eta_min = drain
    return eta_min

def eta_label_from_min(m):
    if m == 0:   return ""
    if m < 5:    return "&lt;5m"
    if m <= 15:  return "5-15m"
    if m <= 30:  return "15-30m"
    if m <= 60:  return "30-60m"
    if m <= 120: return "1-2h"
    return "2h+"

ALMOST_READY_THRESHOLD = 5  # RUNNING PRs with <= this many pending jobs are "almost ready"

rows = []
almost_ready = []  # (pending_total, pr, title, breakdown, eta_min, eta_label) — for RUNNING rows
for line in tsv:
    parts = line.split("\t")
    if len(parts) < 7: continue
    pr, state, breakdown, branch, created, sha, title = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], "\t".join(parts[6:])
    state_counts[state] = state_counts.get(state, 0) + 1
    d = diag.get(pr, {})
    bc = parse_breakdown(breakdown)
    ip, q = bc.get("IN_PROGRESS", 0), bc.get("QUEUED", 0)
    em = eta_min_for_pr(pr) if state == "RUNNING" else 0
    eta = eta_label_from_min(em)
    rows.append((state_order.get(state, 99), -int(pr), pr, state, breakdown, title, d.get("reason", ""), d.get("fix", ""), eta))
    if state == "RUNNING" and 0 < (ip + q) <= ALMOST_READY_THRESHOLD:
        almost_ready.append((em, ip + q, ip, q, pr, title, breakdown, eta))

rows.sort()
almost_ready.sort()  # primary by eta-min ascending → closest to ready first
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
total_global_pending = sum(pool_global_pending.values())
pool_pressure_summary = " · ".join(
    f"{pool}={cnt}/{POOL_RUNNERS.get(pool, DEFAULT_POOL_RUNNERS)}r"
    for pool, cnt in sorted(pool_global_pending.items()) if cnt > 0
)

out = ['<!doctype html>',
'<html lang="en"><head><meta charset="utf-8"><title>Claude / Klaud Cold PR status — InferenceX</title>',
'<style>',
'  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1700px; margin: 24px auto; padding: 0 16px; color:#222; }',
'  h1 { font-size: 20px; margin-bottom: 6px; }',
'  .meta { color:#666; font-size:12px; margin-bottom: 18px; }',
'  table { border-collapse: collapse; width: 100%; font-size: 13px; }',
'  th, td { padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }',
'  th { background:#f7f7f7; position: sticky; top: 0; z-index:1; }',
'  tr:hover { background:#fafafa; }',
'  .state-READY    { color:#0a7; font-weight: 600; }',
'  .state-RUNNING  { color:#06c; font-weight: 600; }',
'  .state-FAILED   { color:#c33; font-weight: 600; }',
'  .state-NOSWEEP  { color:#a60; font-weight: 600; }',
'  .pr { font-family: ui-monospace, "SF Mono", Menlo, monospace; }',
'  .breakdown { font-family: ui-monospace, "SF Mono", Menlo, monospace; color:#444; white-space: nowrap; font-size:11px; }',
'  .reason, .fix { font-size: 12px; max-width: 460px; }',
'  .fix { color:#444; }',
'  .eta { font-family: ui-monospace, "SF Mono", Menlo, monospace; color:#0a7; white-space: nowrap; font-size:11px; }',
'  code { background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:11px; }',
'  a { color:#06c; text-decoration: none; } a:hover { text-decoration: underline; }',
'  .summary { display:flex; gap:10px; margin-bottom: 12px; flex-wrap:wrap; }',
'  .pill { padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }',
'  .pill.ready   { background:#d9f5e6; color:#0a7; }',
'  .pill.running { background:#dfeeff; color:#06c; }',
'  .pill.failed  { background:#fde0e0; color:#c33; }',
'  .pill.noswp   { background:#fbe9c8; color:#a60; }',
'  .pill.almost  { background:#e6f5e0; color:#3a3; }',
'  .almost-section { background:#f5fbf2; border:1px solid #d6ebcc; border-radius:6px; padding:10px 14px; margin-bottom:16px; font-size:13px; }',
'  .almost-section h2 { font-size:13px; margin:0 0 8px; color:#3a3; text-transform: uppercase; letter-spacing:0.04em; }',
'  .almost-section ul { margin:0; padding-left:0; list-style:none; }',
'  .almost-section li { padding: 3px 0; }',
'  .almost-section .pen { display:inline-block; min-width:48px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:11px; color:#666; }',
'  .almost-section .eta { display:inline-block; min-width:56px; color:#3a3; font-weight:600; }',
'</style></head><body>',
'<h1>Claude / [Klaud Cold] PR status &mdash; InferenceX</h1>',
f'<div class="meta">Generated {now}. Source: <code>gh pr view --json statusCheckRollup</code> for every <code>claude/*</code> or <code>[Klaud Cold]</code>-titled open PR. Diagnoses (if any) loaded from <code>/tmp/klaud_pr_diag.json</code>. <strong>ETA</strong> = pool-drain pessimistic estimate (<code>ceil(global_pool_pending / runners) × ~{AVG_JOB_MIN}m/job</code>); GHA dispatch isn\'t guaranteed FIFO and queue position isn\'t exposed (see <a href="https://docs.github.com/en/actions">docs.github.com/en/actions</a>) so this is an upper-bound ordering hint, not a SLA.</div>',
f'<div class="meta">Global pending across all open Klaud/claude sweeps: <strong>{total_global_pending}</strong> jobs &mdash; per-pool: {html.escape(pool_pressure_summary) if pool_pressure_summary else "—"}</div>']

pill_specs = [("READY", "ready"), ("RUNNING", "running"),
              ("FAILED", "failed"), ("FAILED+RUNNING", "failed"),
              ("NO_SWEEP", "noswp"), ("NO_SUCCESS", "noswp")]
pills = [f'<span class="pill {cls}">{name}: {state_counts[name]}</span>'
         for name, cls in pill_specs if state_counts.get(name, 0)]
if almost_ready:
    pills.append(f'<span class="pill almost">ALMOST READY: {len(almost_ready)}</span>')
out.append('<div class="summary">' + "".join(pills) + '</div>')

# Almost-ready section (RUNNING PRs with <= 5 pending), sorted by pool-aware ETA asc
if almost_ready:
    out.append('<div class="almost-section"><h2>Almost ready (closest to mergeable — sorted by ETA)</h2><ul>')
    for em, total, ip, q, pr, title, breakdown, eta in almost_ready:
        pool_str = ", ".join(f"{p}:{c}" for p, c in pr_pool_pending.get(pr, {}).items())
        out.append(
            f'<li>'
            f'<span class="pen">pending={total} (q={q} ip={ip}; pools={pool_str or "?"})</span> '
            f'<span class="eta">ETA {eta or "—"}</span> '
            f'<a href="https://github.com/SemiAnalysisAI/InferenceX/pull/{pr}" target="_blank">#{pr}</a> &mdash; {html.escape(title)}'
            f'</li>'
        )
    out.append('</ul></div>')

out.append('<table><thead><tr>'
           '<th>PR</th><th>State</th><th>Check breakdown</th>'
           '<th>ETA</th><th>Reason</th><th>Suggested fix</th><th>Title</th>'
           '</tr></thead><tbody>')

for _, _, pr, state, breakdown, title, reason, fix, eta in rows:
    cls = state_class.get(state, "state-RUNNING")
    out.append(
        f'<tr><td class="pr"><a href="https://github.com/SemiAnalysisAI/InferenceX/pull/{pr}" target="_blank">#{pr}</a></td>'
        f'<td class="{cls}">{state}</td>'
        f'<td class="breakdown">{html.escape(breakdown)}</td>'
        f'<td class="eta">{eta or "&mdash;"}</td>'
        f'<td class="reason">{reason or "&mdash;"}</td>'
        f'<td class="fix">{fix or "&mdash;"}</td>'
        f'<td>{html.escape(title)}</td></tr>'
    )

out.append('</tbody></table></body></html>')
Path("/tmp/klaud_pr_status.html").write_text("\n".join(out))
print(f"Wrote /tmp/klaud_pr_status.html — {len(rows)} rows, states: {state_counts}, almost-ready: {len(almost_ready)}")
PYEOF
python3 /tmp/gen_klaud_pr_status_html.py
open /tmp/klaud_pr_status.html 2>/dev/null || true
```

Output the path (`/tmp/klaud_pr_status.html`) and the per-state counts to the user. The command is informational only — it does **not** modify any PR.

### Adding diagnoses to the dashboard

To populate the Reason / Suggested fix columns for failing PRs, write a JSON file like this **before** Step 3:

```bash
cat > /tmp/klaud_pr_diag.json <<'EOF'
{
  "1461": {
    "reason": "vLLM v0.21 CUDA-graph profiler OOM at <code>--gpu-memory-utilization 0.90</code>.",
    "fix": "Add <code>export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0</code> before vllm serve."
  },
  "1422": {
    "reason": "Upstream sglang v0.5.12 <code>flash_attn</code> SM-arch regression on B300 (<code>sm_120</code>).",
    "fix": "Pin to <code>v0.5.11-cu130</code>."
  }
}
EOF
```

See `KLAUD_DEBUG.md` for the canonical catalog of recurring failure modes to draw diagnoses from.
