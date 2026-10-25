---
name: result-report
description: Turn InferenceX benchmark runs into one standard report. Input is GitHub Actions run links/IDs or downloaded artifact/aiperf folders; output is report.md (+ optional report.html with charts) and data.json with throughput, cache hit by tier (HBM/DRAM/L3/router), TTFT/ITL/E2E avg-p50-p90-p95-p99, max CCU under SLA for search runs, and A/B or grid comparisons. Use when asked to report, summarize, compare or read the results of a benchmark run, an AgentX/tau2/livekit ladder, a sweep-isl*/fixed-isl* grid, or an aiperf output folder.
---

# result-report

One command turns benchmark results into a report the team can paste into Redmine or an MR. Tables
come from the script, so every report has the same columns and units. You write only the Key findings.

## Run it

Run from the repo root (or prefix the path with `$(git rev-parse --show-toplevel)/`):

    python3 .agents/skills/result-report/scripts/result_report.py <input>... [--html] [--title T] [--baseline ARM] [--out DIR] [--findings FILE]

`<input>` is any mix of:
- a run URL `https://github.com/vngcloud/InferenceX/actions/runs/<id>` or a bare run id (repo `--repo`, default vngcloud/InferenceX);
- a folder of downloaded artifacts (`bmk_agentic_*`, `agentic_*`) or hand-named copies;
- a raw aiperf output folder (contains `profile_export_aiperf.json`).

Run links need the `gh` CLI, logged in. The script reads single files out of the artifact zips with
HTTP Range requests and skips the heavy ones (`profile_export.jsonl`, `inputs.json`, timeslices,
`server_metrics_export.json` of fixed runs). A search run reads the aiperf export of every probe,
`search_history.json`, and the winning probe's `server_metrics_export.json` only if it is ≤ 40 MB
compressed (`--max-member-mb`; real ones are 25-35 MB each, 350-400 MB raw). Typical total: a few MB
to ~40 MB. Remote search run: ~1-2 min; grids with 20+ artifacts: ~5-15 min. Run it in the background.
Progress (`fetching ...`) goes to stderr; filter with `2>&1 | grep -v '^fetching'`. If output is piped, read the
exit code from `$?` / `PIPESTATUS`, not the pipe. Over the cap, the winner's cache tiers are skipped with a Note; this applies to local search folders too (size on disk), raise `--max-member-mb` to read them.

Output goes to `~/bench-reports/<date>-<slug>/` (or `--out DIR`): `report.md`, `data.json`,
`report.html` with `--html`. Remote downloads are cached in `<out>/.cache/` (re-runs are fast); local
folder inputs create no `.cache` and report 0.00 MB. `<out>/.cache` is much larger than the "downloaded" figure (members are stored decompressed: 18 MB downloaded ~ 360 MB cache, mostly the winner's `server_metrics_export.json`); safe to delete once the report is final. The header shows the cumulative download size
(`<out>/.cache/fetched.json`); the CLI summary prints "this run X MB (total Y MB)". To check a number,
cached files sit at `<out>/.cache/<artifact_id>/files/<path inside artifact>`.

## Several inputs: arms and grids

Inputs are grouped automatically:
- **Arm** = `serving.yaml` minus `name`, `scenario`, `conc` (comments ignored). Every other field
  counts, **including `runner`**: two halves of one logical grid that ran on different runners
  (e.g. 8x vs 4x) become 2 arms with disjoint scenarios, so no Δ% cells and matrix cells show
  `arm1 / arm2`. Tell the user the grid is split by runner; do not claim a comparison.
- **Scenario** = the workload. `sweep-isl*-osl*-cache*` / `fixed-isl*-osl*-cache*` names become grid axes.
  A missing scenario shows as `(unknown)`.
- Framework and TP/EP in the Runs table stay `—` when the aggregate lacks them (serving.yaml gives only model/image/precision/GPUs/runner). The `Config` column and arm guidance here apply only to multi-config inputs; a single run needs none of it.
- Local folders with several configs of the same arm and CCU get a `Config` column (folder name) so rows are distinguishable.

Same arm + many scenarios → a grid summary. Several arms + the same scenario + CCU → deltas: every
non-baseline cell shows `value (Δ%)` against the baseline arm (first arm, or `--baseline <label>`). Arms on different scenarios get separate scenario blocks, no deltas.
Cache-hit and Error % deltas are absolute percentage points (`20.0% (+19.95 pp)`, shown when ≥ 1 pp for cache, ≥ 0.1 pp for errors); other metrics use relative Δ%.
The `(baseline)` marker appears only in multi-arm reports, on the baseline arm's rows in the Runs table and every per-scenario table (throughput, TTFT, ITL, E2E), not in the matrices.
Arm labels are the `serving.yaml` `name` with trailing `_c<N>` / `_search` / `[-sweep|-fixed]-isl*-osl*-cache*` stripped (the `sweep-`/`fixed-` part is optional; the strip is end-anchored, so `-armB` after a grid suffix keeps it all); with no `name` the label falls back to the aggregate's `infmax_model_prefix`, then the source basename; the full config is in each arm's `config` in `data.json`.
Matrix cells: `—` = scenario not run by that arm, `✗` = infeasible / no CCU passed, `*` = cliff / boundary estimate.

Folder layout: aggregate-only folders work (aggregate JSON under `bmk_agentic_*` or `results_bmk/agg_bmk.json`);
look-alike dirs (`power_audit_*`, `gpu_metrics_*`, `server_logs_*`, `results_bmk`, `run-stats`) are ignored;
folders with no results are listed in Notes ("skipped N folder(s) ...").

A failed-job Note may start with `…` (GitHub truncates job names); the job URL in it is the reliable pointer.

## Write the Key findings

1. Run the command once.
2. Read `data.json` (`points[]`: one per config; `arms[]`; `notes[]`; `failed_jobs[]`) and the tables in `report.md`.
3. Write 3-6 bullets (plain markdown `- ...` lines, English) to a file anywhere outside the repo, then
   re-run the same command plus `--findings <file>` (cached, seconds). That re-run replaces the
   `<!-- KEY_FINDINGS ... -->` placeholder. Rules:
   - Lead with the decision: the best arm per scenario, or the max CCU under SLA. Non-search ladders (no SLA):
     lead with throughput and latency per CCU and cache hit; deltas exist only with ≥2 arms on the same scenario+CCU.
   - Quote a number from the tables. Name arm, scenario and CCU; a single-arm or single-scenario run may omit
     arm/scenario (scenario may be `(unknown)`). Bullets about stop conditions or SLA need no table number.
   - Quote deltas only when |Δ| ≥ 5 %.
   - For search runs, name the binding SLA and where it broke (`ITL p90 < 50 ms, 50.85 ms at c61`).
   - Repeat any warning from Notes that changes the conclusion: errors, `⚠` cache mismatch,
     aggregate-only rows (no p99), failed or expired jobs.
   - No adjectives without a number.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | report written | — |
| 1 | no configs found | check the run has `bmk_agentic_*`/`agentic_*` artifacts, or the folder layout |
| 2 | bad input or unknown `--baseline` | fix the argument; the error lists the arm labels |
| 3 | Range reads failed and the full zip is over `--max-full-zip-mb` (100) | ask the user, then re-run with a bigger `--max-full-zip-mb` |
| 4 | `gh` missing or failing | `gh auth status`; install / log in |

If the process is killed, nothing is written to `<out>` except `.cache/`; re-run the same command and the partial cache is reused.

## Reading the tables

- Latency is ms; columns avg/p50/p90/p95/p99. `—` means the value does not exist, never zero.
- Rows built from the aggregate JSON only (no aiperf export) have no p99, no Out tok/s/user and no Error % (`—`); Notes says so.
- Cache columns: overall, GPU (HBM), CPU (DRAM), external (L3), router. "Cached tokens by tier" uses engine tier names: `device` = GPU HBM (SGLang naming), `host` = CPU DRAM, `storage*` = external L3. Columns empty for every row are hidden.
  vLLM cannot split CPU from L3: its offload hits show as external.
- Grid tables compare the nominal cache % in the scenario name against the measured **overall** hit; `⚠` when they differ by more than 10 points.
- Search probe table `Hit overall` is the client-side aiperf cache read % of each probe; tiers exist only for the winner (see Run it). The nominal-vs-measured table may show `—` per row when server metrics are absent.
- Search runs: the headline CCU is the largest CCU that passed every SLA (`feasible_max`), not the highest-throughput probe; the probe table marks both.

## Tests

    python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v
