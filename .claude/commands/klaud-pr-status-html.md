---
description: Render an HTML dashboard of Claude/Klaud-Cold PR states (state + check breakdown per PR) and open it in the browser
---

Render an HTML dashboard for every open PR in `SemiAnalysisAI/InferenceX` that was opened by Claude (either a `claude/*` branch OR a title prefixed with `[Klaud Cold]`). Each row shows the PR's current state, a check-status breakdown, the title, and empty "Reason"/"Suggested fix" cells you can fill in afterward by reading failed-run logs.

The dashboard lives at `/tmp/klaud_pr_status.html` and is opened with `open` (macOS) at the end.

## Step 1 — list candidate PRs (`claude/*` OR `[Klaud Cold]` title)

```bash
gh pr list --repo SemiAnalysisAI/InferenceX --state open --limit 200 \
  --json number,title,headRefName,createdAt \
  --jq '.[] | select((.headRefName | startswith("claude/")) or (.title | startswith("[Klaud Cold]"))) | "\(.number)\t\(.headRefName)\t\(.createdAt)\t\(.title)"' \
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
- **NO_SWEEP** — no `Run Sweep` check exists for this head SHA at all (sweep never triggered — usually missing `full-sweep-enabled` label).

```bash
: > /tmp/klaud_pr_status.tsv
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
done < /tmp/klaud_pr_candidates.tsv
```

## Step 3 — render HTML and open

State render order (action items first): `FAILED` → `FAILED+RUNNING` → `NO_SWEEP` → `NO_SUCCESS` → `RUNNING` → `READY`. Within each bucket, descending PR number.

If you have per-PR diagnoses to inject (e.g. after running `/fix-klaud-cron-prs`), write them as a JSON map `{ "1461": {"reason": "...", "fix": "..."}, ... }` to `/tmp/klaud_pr_diag.json` BEFORE running this step — the generator will pick them up. HTML may contain inline `<code>` tags.

```bash
cat > /tmp/gen_klaud_pr_status_html.py <<'PYEOF'
#!/usr/bin/env python3
import html, json, datetime as dt
from pathlib import Path

tsv = Path("/tmp/klaud_pr_status.tsv").read_text().strip().splitlines()
diag_path = Path("/tmp/klaud_pr_diag.json")
diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}

state_counts = {}
state_order = {"FAILED": 0, "FAILED+RUNNING": 1, "NO_SWEEP": 2, "NO_SUCCESS": 3, "RUNNING": 4, "READY": 5}
state_class = {
    "READY": "state-READY", "RUNNING": "state-RUNNING",
    "FAILED": "state-FAILED", "FAILED+RUNNING": "state-FAILED",
    "NO_SWEEP": "state-NOSWEEP", "NO_SUCCESS": "state-NOSWEEP",
}

rows = []
for line in tsv:
    parts = line.split("\t")
    if len(parts) < 7:
        continue
    pr, state, breakdown, branch, created, sha, title = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], "\t".join(parts[6:])
    state_counts[state] = state_counts.get(state, 0) + 1
    d = diag.get(pr, {})
    rows.append((state_order.get(state, 99), -int(pr), pr, state, breakdown, title, d.get("reason", ""), d.get("fix", "")))

rows.sort()
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
'  code { background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:11px; }',
'  a { color:#06c; text-decoration: none; } a:hover { text-decoration: underline; }',
'  .summary { display:flex; gap:10px; margin-bottom: 12px; flex-wrap:wrap; }',
'  .pill { padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }',
'  .pill.ready   { background:#d9f5e6; color:#0a7; }',
'  .pill.running { background:#dfeeff; color:#06c; }',
'  .pill.failed  { background:#fde0e0; color:#c33; }',
'  .pill.noswp   { background:#fbe9c8; color:#a60; }',
'</style></head><body>',
'<h1>Claude / [Klaud Cold] PR status &mdash; InferenceX</h1>',
f'<div class="meta">Generated {now}. Source: <code>gh pr view --json statusCheckRollup</code> for every <code>claude/*</code> or <code>[Klaud Cold]</code>-titled open PR. Diagnoses (if any) loaded from <code>/tmp/klaud_pr_diag.json</code>.</div>']

pill_specs = [("READY", "ready"), ("RUNNING", "running"),
              ("FAILED", "failed"), ("FAILED+RUNNING", "failed"),
              ("NO_SWEEP", "noswp"), ("NO_SUCCESS", "noswp")]
pills = [f'<span class="pill {cls}">{name}: {state_counts[name]}</span>'
         for name, cls in pill_specs if state_counts.get(name, 0)]
out.append('<div class="summary">' + "".join(pills) + '</div>')

out.append('<table><thead><tr>'
           '<th>PR</th><th>State</th><th>Check breakdown</th>'
           '<th>Reason</th><th>Suggested fix</th><th>Title</th>'
           '</tr></thead><tbody>')

for _, _, pr, state, breakdown, title, reason, fix in rows:
    cls = state_class.get(state, "state-RUNNING")
    out.append(
        f'<tr><td class="pr"><a href="https://github.com/SemiAnalysisAI/InferenceX/pull/{pr}" target="_blank">#{pr}</a></td>'
        f'<td class="{cls}">{state}</td>'
        f'<td class="breakdown">{html.escape(breakdown)}</td>'
        f'<td class="reason">{reason or "&mdash;"}</td>'
        f'<td class="fix">{fix or "&mdash;"}</td>'
        f'<td>{html.escape(title)}</td></tr>'
    )

out.append('</tbody></table></body></html>')
Path("/tmp/klaud_pr_status.html").write_text("\n".join(out))
print(f"Wrote /tmp/klaud_pr_status.html — {len(rows)} rows, states: {state_counts}")
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
