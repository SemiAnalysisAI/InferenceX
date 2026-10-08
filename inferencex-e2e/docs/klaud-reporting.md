# Klaud Cold reports

<div align="center">

**English** | [中文](klaud-reporting_zh.md)

</div>

[`infx/klaud/reporting.py`](../infx/klaud/reporting.py) owns the schemas, arithmetic and rendering. The agent supplies concise observations and verified evidence, not hand-calculated deltas. The PR body contains only the goal and baseline. Comments own attempts; the lifecycle receipt owns verified completion. Keep English visible and put Simplified Chinese in one collapsed `<details><summary>中文</summary>` section. Numeric tables appear once; Chinese prose refers to those tables. Apply this layout to lifecycle comments too. No mentions, review requests, raw logs, private telemetry or limitations section.

## Commands

Run from the checkout with the candidate environment provided by the workflow:

```bash
KLAUD=(uv run --no-project --exclude-newer PT12H --python 3.12 \
  --with 'pydantic>=2.10,<3' --with pyyaml python -m infx.klaud)

# goal.json: {"en":"Update ENGINE image from `OLD` to `NEW`.","zh":"将 ENGINE 镜像从 `OLD` 更新为 `NEW`。"}
# Resolve the display model name from the public OpenAPI document first.
"${KLAUD[@]}" prepare-baseline --model 'DISPLAY MODEL NAME' \
  --goal-file "$KLAUD_EVIDENCE/goal.json" --output "$KLAUD_EVIDENCE/baseline.json"

# After creating the real-change draft PR, publish the baseline into its body before GPU work.
"${KLAUD[@]}" report --kind baseline --file "$KLAUD_EVIDENCE/baseline.json"

# Read the schema once and reuse the local record for each attempt's updates.
"${KLAUD[@]}" report-schema --kind attempt > "$KLAUD_EVIDENCE/attempt-schema.json"
"${KLAUD[@]}" report --kind attempt --file "$KLAUD_EVIDENCE/attempt.json"
```

`prepare-baseline` verifies the candidate/base/source/model binding in the planner-produced `baseline-preflight.json` and reuses its public benchmark roster; it stops if this file is missing or invalid. The planner builds that roster from public `benchmarks` with the selected date, `exact=true`, and no calculator view; `workflow-info` establishes producer IDs, heads and attempts. Each producer's family is regenerated with that producer revision's own generator from a Git snapshot of the producer head, fetched from the repository when the local clone lacks it, so later config-format changes cannot break or reinterpret the baseline; this includes the historical `.github/configs` layout and flat runner-label format. The snapshot's own `infx.matrix.fingerprint` recomputes the recipe fingerprints its planner published, and revisions older than that module hashed the generated row, so that hash stands in for them. Producer heads are official runs of this repository, so this executes reviewed repository history, never candidate code. It still runs only in the planner's credential-free `regenerate-producers` step, which writes the regenerated rows to `producers.json`; `select` holds the credentials and only reads that file. Producers older than the generator's `--no-evals` option cannot be regenerated and defer the candidate. Matching requires the old image and full public workload/topology/concurrency identity; supplied recipe fingerprints must also match. Legacy rows without fingerprints require a unique match and a producer changelog selecting the family. Rows outside the reconstructed family are ignored even when their producer metadata is incomplete. Missing provenance on a matching current or historical point, ambiguous identities and duplicate points stop preparation instead of publishing an incomplete baseline. Before the first publication, supplement verified published evals and dataset provenance using [the public API routes](klaud.md#public-api-investigation). `BenchmarkRow` has no dataset identity; unproven AgentX datasets still produce N/A deltas. Never run the old image to fill a gap.

The baseline file is created once; retries do not refetch it. The PR body freezes the complete typed record in a compressed hidden marker and renders the complete baseline once. Conflicting replacement records are rejected. Existing PRs with the older baseline-comment format remain readable for recovery. A correction requires a maintainer to review the evidence and make the correction explicit; do not silently revise the baseline during repairs.

Freeze **every point in the original selected public baseline**. The helper unions the current family with the original producer families; it does not filter the public feed to the selected observation's ISL/OSL. Historical 1k/1k points therefore stay in the roster even if today's family only contains 8k/1k, and remain required unless a recorded retirement covers them. Missing published metrics remain N/A without deleting those points. Never shrink the baseline to overlapping points or the body preview. Preserve each recipe/workload/topology/concurrency/dataset identity; equal counts or extra points elsewhere do not replace missing points.

Only the planner records retirements, from [`MODELS.md`](MODELS.md) at the candidate's base. It retires a frozen point only when all of these hold: the point is fixed-seq-len and the current family no longer generates any point with the single-turn ISL/OSL in its frozen label; the only Scenarios row for that ISL/OSL starts with `Deprecated since YYYY-MM-DD ([#N](PR link))` (or the bold `**Deprecated for all models** since` form of it), linking a PR of this repository and dated after the baseline date; and the model's row in the Model support matrix lists that scenario under deprecated and not active scenarios. A repeated table or model row, a second row for the same ISL/OSL (however it is spaced) or a table row with the wrong number of cells retires nothing. Absence from the current family, a removal commit or a perf-changelog entry is not a retirement, and a partial topology or concurrency removal has no such statement, so it still fails coverage. Each `retirements` entry stores the scenario, date, PR link, model prefix, ISL/OSL and retired point keys. Records without retirements omit the field, so code that predates it can still read them. Retired points keep their published values in the roster, and the body and final report note them with that evidence. The agent can edit the published record and the preflight, so `check-final`, `finish` and recovery never trust recorded retirements: they re-run this rule on the record's roster and baseline date against the family and `MODELS.md` at the candidate's base, never the PR head, and fail unless it derives exactly the recorded retirements. Selection, `check-final`, `finish` and recovery exempt only those points. Agents never add, change or remove retirements.

Create the draft body with `<!-- klaud-baseline -->`. The helper replaces that placeholder once with the frozen record and rendered baseline, preserving anything other bots append outside it. Baseline publication creates no comment. Retrying the same record is idempotent, and later attempt reports never rewrite the body. Large attempt reports write immutable content-addressed comment parts before updating their index, so an interrupted update cannot mix revisions.

Comparison keys come from `reporting.point_key()` on the canonical generated point, excluding only image, point name, producer fingerprint and queue metadata. Preserve workload, topology, concurrency and all remaining settings. When the planner freezes the baseline, a published point whose producer-revision key is absent from the current family takes the key of the single current point with the same public benchmark identity (model, hardware, framework, precision, speculation, workload, topology, concurrency and image). Routing labels and recipe paths added since the producer ran therefore do not split one point into two. Two current matches are ambiguous and defer the candidate. A point with no current match keeps its producer key, so a shrunk family still fails coverage unless a recorded retirement covers that point. `reporting.values()` reads collector/API metrics and converts seconds to milliseconds. Do not invent an alias or key to make two points match. Dataset identity must match independently for AgentX. Missing/zero baselines, failed points and mismatched datasets/statistics produce N/A. Throughput change is `(new / old - 1) * 100`; eval scores are normalized to 0–1 and differences shown in percentage points, matching suite, metric, shape and sample counts.

Each attempt records its owned run ID, exact head and run attempt, kind/number, status, en/zh sentences for change/next, separate expected/passed benchmark and eval counts, all points and evals. The goal also uses en/zh sentences naming the engine and exact old/new images. Optional finding text stores diagnostic evidence; neither finding nor coverage counters are rendered. Keep failures, cancelled points and request errors visible. Initial update is number 0; repairs are 1–5. Confirmed transient infrastructure retries have their own kind and do not consume a recipe repair; the prompt bounds these to two per attempt. Do not call a failed eval a passed smoke because throughput passed.

Publish each attempt immediately after dispatch and before waiting. Update that run/attempt's comment on material changes or after 30 minutes; retain completed attempt history. Large attempt records are split into numbered comment parts without dropping points or limiting the family size. The PR body contains the complete baseline and provenance; comments contain attempts only. Reports are idempotent by parent/candidate/run/attempt. Only the typed public record is persisted, never the full scratch directory or execution transcript.

Use compact metadata lines, exact `8k/1k` shorthand and shared settings above the tables. Only collapse point labels to concurrency when the displayed workload/topology/statistic is shared and concurrencies are unique; otherwise retain distinct full labels. Cells contain the new value and parenthesized delta. Eval samples show `N each` only when both counts match; otherwise show old/new counts. Keep failures, request errors and reasons for unavailable comparisons as short notes. No Result column, Coverage/Finding paragraphs, legends, storage boilerplate or redundant status summary. Next names only the next subgoal. English content stays visible, including all result tables; only Chinese is collapsed.

## Body layout

```markdown
**Goal:** Update ENGINE image from `OLD_IMAGE` to `NEW_IMAGE`.\
**Baseline:** DATE · `OLD_IMAGE`\
8k/1k · TP8/EP1 · Mean latency · Sources: API links

| Concurrency | Total tok/s/GPU ↑ | Output tok/s/GPU ↑ | TTFT ms ↓ | TPOT ms ↓ |
| ---: | ---: | ---: | ---: | ---: |
| C | value | value | value | value |

| Eval | Score ↑ | Samples |
| --- | ---: | ---: |
| SUITE/METRIC · cN | SCORE% | N |

<details>
<summary>中文</summary>

**目标：**将 ENGINE 镜像从 `OLD_IMAGE` 更新为 `NEW_IMAGE`。\
**基线：**DATE · `OLD_IMAGE`\
8k/1k · TP8/EP1 · 平均延迟 · 来源：API 链接；数值及异常说明见上表。

</details>
```

## Attempt layout

```markdown
**Repair N/5 · STATUS** · [Run ID / attempt N](RUN_URL) · UTC_TIMESTAMP\
`IMAGE` · `HEAD` · 8k/1k · TP8/EP1 · Mean latency\
**Change:** One sentence with relevant source links.

| Concurrency | Output tok/s/GPU ↑ | TTFT ms ↓ | TPOT ms ↓ |
| ---: | ---: | ---: | ---: |
| C | 110 (+10%) | 180 (-10%) | 19 (-5%) |

| Eval | Score ↑ | Samples |
| --- | ---: | ---: |
| SUITE/METRIC · cN | 97% (+0.50 pp) | 1,000 each |

**Next:** Run the final full sweep.

<details>
<summary>中文</summary>

**修复 N/5 · 状态** · [Run ID / attempt N](RUN_URL) · UTC_TIMESTAMP\
`IMAGE` · `HEAD` · 8k/1k · TP8/EP1 · 平均延迟\
**变更：**简短中文翻译；数值及异常说明见上表。\
**下一步：**运行最终完整 sweep。

</details>
```

`finish` generates the final report from verified artifacts and the frozen baseline for both normal execution and interrupted-session recovery. It publishes the report **before** marking ready. Missing historical baseline data is explicitly N/A; it never invents deltas or launches a replacement baseline. A successful sweep may contain regressions; readiness means work and validation are complete, not that every metric improved. After `finish` returns `validated`, Klaud posts `/use <verified-final-run-id>` once so the completed sweep can be reused. It never posts reuse for another outcome, stages results or merges the PR.

## Final preflight

Before adding `full-sweep-fail-fast`, validate the exact pushed head's full matrix:

```bash
head_sha=$(git rev-parse HEAD)
uv run --no-project --exclude-newer PT12H --python 3.12 --with 'pydantic>=2.10,<3' --with pyyaml \
  python -m infx.matrix.plan --base-ref origin/main --head-ref "$head_sha" \
  --changelog-file perf-changelog.yaml > "$KLAUD_EVIDENCE/final-matrix.json"
"${KLAUD[@]}" check-final --matrix-file "$KLAUD_EVIDENCE/final-matrix.json"
```

The verifier independently generates the unfiltered family from exact-head YAML with trusted helper code. It compares full recipe settings using the workflow's matrix schemas, which account for defaults added after fingerprinting; changed settings with a copied fingerprint still fail. An equivalent scenario filter can pass; an omitted/changed point or default eval cannot. It does not execute downloaded PR code. Generator-policy drift on an old run requires inspection rather than silently weakening validation.

`check-final` also checks **every frozen baseline point** against the canonical final family before dispatch, exempting only recorded retirements that it re-derives exactly at the candidate's base. `finish` and interrupted-session recovery apply the same checks after validating full artifact coverage, before readiness. A missing baseline, a recorded retirement that differs from the re-derived ones, or an omitted/changed original point that no retirement covers fails validation even if the smaller current-family sweep is green. Report the affected points and call `finish` with `outcome: failed` to clean up owned runs, close the PR and delete its unchanged exact-head branch; never mark it ready or validated. `N/A` permits an unproven delta, not a missing updated-image result. Targeted smoke subsets remain allowed. Explicit maintainer handoff still preserves the PR and branch.
