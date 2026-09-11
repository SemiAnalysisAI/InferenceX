# Klaud Cold reports

<div align="center">

**English** | [中文](./klaud-reporting_zh.md)

</div>

[`utils/klaud/reporting.py`](../utils/klaud/reporting.py) owns the schemas, arithmetic and rendering. The agent supplies concise observations and verified evidence, not hand-calculated deltas. The PR body contains only the goal and baseline. Comments own attempts; the lifecycle receipt owns verified completion. All generated PR bodies and comments, including tables and lifecycle reports, are English-only. This Klaud-specific exception overrides shared bilingual guidance; do not add Chinese translations or language dividers. No mentions, review requests, raw logs, private telemetry or limitations section.

## Commands

Run from the checkout with the candidate environment provided by the workflow:

```bash
KLAUD=(uv run --no-project --exclude-newer PT12H --python 3.12 \
  --with 'pydantic>=2.10,<3' --with pyyaml python -m utils.klaud)

# goal.json is a JSON string: "Update the selected image."
# Resolve the display model name from the public OpenAPI document first.
"${KLAUD[@]}" prepare-baseline --model 'DISPLAY MODEL NAME' \
  --goal-file "$KLAUD_EVIDENCE/goal.json" --output "$KLAUD_EVIDENCE/baseline.json"

# After creating the real-change draft PR, publish the baseline before GPU work.
"${KLAUD[@]}" report --kind baseline --file "$KLAUD_EVIDENCE/baseline.json"

# Read the schema once and reuse the local record for each attempt's updates.
"${KLAUD[@]}" report-schema --kind attempt > "$KLAUD_EVIDENCE/attempt-schema.json"
"${KLAUD[@]}" report --kind attempt --file "$KLAUD_EVIDENCE/attempt.json"
```

`prepare-baseline` queries the public `benchmarks` endpoint with the selected date, `exact=true`, and no calculator view; it uses `workflow-info` for producer IDs, heads and attempts. It accepts only unique points matching the old image and full recipe fingerprint generated from the selected base. Missing fingerprints/provenance are unavailable, never an approximate SKU/concurrency match. Before the first publication, supplement verified published evals and dataset provenance using the public routes described in [the API investigation guide](./klaud.md#public-api-investigation). Public `BenchmarkRow` alone has no dataset identity; an unproven AgentX dataset produces N/A deltas. Never run the old image to fill a gap.

The baseline file is created once; retries do not refetch it. The first baseline comment freezes the typed record. Conflicting replacement records are rejected. A correction requires a maintainer to review the evidence and make the correction explicit; do not silently revise the baseline during repairs.

Create the draft body with `<!-- klaud-baseline -->`. The helper replaces that placeholder once, preserving anything other bots append outside it. If publication is interrupted after the baseline comment but before the body update, retrying the same record completes the body update. Later attempt reports never rewrite the body. Large reports write immutable content-addressed parts before updating their index, so an interrupted update cannot mix revisions.

Comparison keys come from `reporting.point_key()` on the canonical generated point, excluding only image, point name, producer fingerprint and queue metadata. Preserve workload, topology, concurrency and all remaining settings. `reporting.values()` reads collector/API metrics and converts seconds to milliseconds. Do not invent an alias or key to make two points match. Dataset identity must match independently for AgentX. Missing/zero baselines, failed points and mismatched datasets/statistics produce N/A. Throughput change is `(new / old - 1) * 100`; eval scores are normalized to 0–1 and differences shown in percentage points, matching suite, metric, shape and sample counts.

Each attempt records its owned run ID, exact head and run attempt, kind/number, status, plain English strings for change/finding/next, separate expected/passed benchmark and eval counts, all points and evals. Keep failures, cancelled points and request errors visible. Initial update is number 0; repairs are 1–5. Confirmed transient infrastructure retries have their own kind and do not consume a recipe repair; the prompt bounds these to two per attempt. Do not call a failed eval a passed smoke because throughput passed.

Publish immediately after dispatch and before waiting. Update that run/attempt's comment on material changes or after 30 minutes; retain completed attempt history. Large records are split into numbered comment parts without dropping points or limiting the family size. The compact body shows up to 12 baseline rows; all rows and provenance persist in baseline comments. Reports are idempotent by parent/candidate/run/attempt. Only the typed public record is persisted, never the full scratch directory or execution transcript.

## Body layout

```markdown
Update FAMILY from OLD_IMAGE to NEW_IMAGE.

Baseline: PUBLISHED_DATE · OLD_IMAGE · public API sources

| Point | Total tok/s/GPU | Output tok/s/GPU | TTFT ms | TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| shape and concurrency | value | value | value | value |

Eval baseline: suite/metric, score, sample count; N/A where unavailable.
```

## Attempt layout

```markdown
### Initial attempt / Repair N/5 / Infrastructure retry N / Final full sweep

Status: STATUS · UTC timestamp
Measured: IMAGE · HEAD · run link and attempt
Change: one sentence
Coverage: benchmarks passed/expected; evals passed/expected

| Point | Result | Output tok/s/GPU | Δ output | Δ TTFT | Δ TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| shape and concurrency | status, errors or N/A reason | value | signed % | signed % | signed % |

| Eval suite / metric | Baseline % | Updated % | Δ pp | n (old/new) | Result |
| --- | ---: | ---: | ---: | --- | --- |
| suite/metric | value | value | signed pp | counts | result |

Finding: observed result; distinguish hypotheses from evidence
Next: specific action or verified final disposition
```

`finish` generates the final report from verified artifacts and the frozen baseline for both normal execution and interrupted-session recovery. It publishes the report **before** marking ready. Missing historical baseline data is explicitly N/A; it never invents deltas or launches a replacement baseline. A successful sweep may contain regressions; readiness means work and validation are complete, not that every metric improved. Klaud neither authorizes reuse nor merges the PR.

## Final preflight and maintainer retry

Before adding `full-sweep-enabled`, validate the exact pushed head's full matrix:

```bash
head_sha=$(git rev-parse HEAD)
uv run --no-project --python 3.12 --with 'pydantic>=2.10,<3' --with pyyaml \
  python utils/process_changelog.py --base-ref origin/main --head-ref "$head_sha" \
  --changelog-file perf-changelog.yaml > "$KLAUD_EVIDENCE/final-matrix.json"
"${KLAUD[@]}" check-final --matrix-file "$KLAUD_EVIDENCE/final-matrix.json"
```

The verifier independently generates the unfiltered family from exact-head YAML with trusted helper code. An equivalent scenario filter can pass; an omitted/changed point or default eval cannot. It does not execute downloaded PR code. Generator-policy drift on an old run requires inspection rather than silently weakening validation.

After a confirmed blocker is fixed, a repository maintainer may explicitly release a closed candidate's retained branch:

```bash
"${KLAUD[@]}" release-candidate --parent-run-id PARENT_RUN_ID \
  --candidate-file ORIGINAL_CANDIDATE_JSON --head REVIEWED_CLOSED_PR_SHA
```

This rejects the Klaud account and non-maintainers. It requires a completed parent, verified cleanup receipt, closed/unmerged exact-head PR and terminal owned children, rechecks the branch, records approval, then deletes only that retained branch. It does not relaunch, erase historical results or take over an open PR. Ordinary capacity/readiness deferrals already release their branches through `finish`.
