# Klaude auto-sweep

<div align="center">

**English** | [中文](./klaude_zh.md)

</div>

[`klaude-plan.yml`](../.github/workflows/klaude-plan.yml) prepares candidates with Python, uses a short read-only Claude review to exclude overlapping open PRs, then calls [`klaude-candidate.yml`](../.github/workflows/klaude-candidate.yml). One autonomous Klaude Cold session owns each selected candidate's branch, draft PR, benchmarks, repairs and reporting. Its action is the last step; there is no attempt workflow, Python execution controller or downstream reporting job.

Both Claude steps explicitly use `claude-fable-5-1` (Fable 5.1) with fast mode disabled (`fastMode: false`).

## Selection

[`api.py`](../utils/klaude/api.py) handles public and private reads through one HTTP implementation. The [unified CLI](../utils/klaude/__main__.py) is `python -m utils.klaude`: `plan --directory DIR` prepares candidates and open PRs; `select --max-candidates-per-run N --directory DIR` validates Claude's structured `KLAUDE_PR_REVIEW` output and applies the workflow's candidate limit; `check-capacity --cluster ID` rechecks capacity before dispatch. Use `--root PATH` before `plan` to select the checkout whose HEAD becomes the candidate base SHA; no config file is loaded.

- Public reads use `/api/v1/latest-images` and `/api/v1/framework-releases`. Candidate identities and release keys come from those responses. Release mismatches and unstable images are review hints; Klaude Cold verifies compatible replacements.
- Private reads use only `/api/status/clusters` and require fresh, valid schema-v6 node counts. They use the API's telemetry cluster IDs directly and never print private status. Node utilization is the only capacity gate; queued work, scheduler reservations and priority overrides do not block candidate selection.
- `plan` intersects public hardware classes with available telemetry clusters, deduplicates public candidate identities and skips known Klaude branches/PRs. It also retrieves all open PRs with pagination, including drafts and arbitrary branch names. Repository identity comes from `GITHUB_REPOSITORY`.
- Claude resolves current recipe families and inspects open PR changed files, then relevant bodies/diffs. An existing image refresh or overlapping compatibility effort blocks the candidate even if it targets another image tag; unrelated work on the same hardware does not. Its structured decisions include `candidate-id`, `decision` (`proceed`, `duplicate` or `uncertain`), `family`, overlapping `pull-requests` numbers and an evidence `reason`.
- `select` accepts only validated `proceed` decisions, deduplicates resolved families and applies the total cap after review, preserving candidate order. A family marked duplicate or uncertain is excluded even if another observation says proceed. Missing/unreviewed candidates are deferred; malformed output, unknown IDs or repeated IDs fail selection without launching agents. Claude can stop once enough distinct families qualify, letting a later candidate replace a duplicate within the cap.

The private gate requires **node utilization strictly below 20%**: `(summary.allocatedNodes + summary.mixedNodes) * 5 < summary.totalNodes`, without rounding. Both fully allocated and partially used nodes count as in use; exactly 20% is rejected. There is no reserved-node deduction or idle-node minimum. Missing, invalid, inconsistent or stale data is rejected. The planner matches a public hardware key to the same telemetry ID or its hyphen-delimited cluster variants. This establishes utilization eligibility for the hardware class; the scheduler handles actual node availability. Klaude Cold must resolve the current recipe, exact cluster and full node demand before dispatching.

The `klaude-plan` artifact holds the prepared `candidates.json`, a compact `open-prs.json` index, and `selection.json` with reviewed decisions and selected IDs. Each selection gets its own `candidate.json`: the public observation, release hint, review reasons, branch, base SHA, public API discovery URLs and validated `pr-review`. It contains no private telemetry. Selected candidates run in parallel, each with its own Klaude Cold session. A failed candidate does not cancel the others. There is no copied model/runner catalog or `recipes.py`; live recipe interpretation belongs to the agent using existing InferenceX configuration and tooling.

## Klaude Cold owns execution

After checkout and context preparation, the candidate workflow hands control to Klaude Cold with `CLAUDE_PAT`, `ANTHROPIC_API_KEY` and the private API read key. Klaude Cold resolves the public observation to one live master-config family, checks the current image and existing PRs, claims its branch before spending GPU time, then makes a real change and opens a draft PR. It rechecks open PRs immediately before claiming the branch because the planner's review is a snapshot, not a lock against new human PRs. Ambiguous, retired or already updated candidates stop without a sweep. It owns commits, pushes, dispatches, monitoring, diagnosis, repairs and bilingual PR updates within the same session.

The prompt asks Klaude Cold to measure both the original and updated images with existing `e2e-tests.yml` on `main`, passing the exact measured SHA through `inputs.ref` and the complete generated `test-config` family command through `generate-cli-command`. It reads current `configs/*-master.yaml`, `configs/runners.yaml` and the existing matrix-generator CLI instead of maintaining another recipe catalog. Default evals, all recipe points, physical `nodes:N` labels, MTP chat templates and artifact contracts stay intact. It must recheck node utilization for the exact telemetry cluster before every dispatch. The `check-capacity` command returns only exit status: 0 permits dispatch below 20% node utilization; any other result means defer. It does not print capacity details.

Klaude Cold reads the run artifacts and logs, calculates matched performance deltas and publishes evidence in its PR. A working updated image with passing selected benchmarks and default evals is a successful outcome; performance improvements are best effort, with regressions reported rather than rejected by a percentage threshold. The PR and final report must include a bilingual Markdown summary table with a row for the baseline and every update or repair attempt: attempt, image/SHA, run URLs, benchmark/eval results, matched throughput and latency deltas against the baseline, and diagnosis. Label per-point deltas clearly and use `N/A` with a reason for missing or incomparable measurements, including failed or cancelled attempts. Exclude invalid, duplicate and unmatched points from improvement claims and disclose limitations. A successful selected benchmark is separate from complete performance evidence and global PR approval. The public API investigation guide below directs additional reads as needed. Public datapoints provide context; fresh old/new runs provide the comparison. Raw benchmark artifacts remain on their e2e runs. Local candidate scratch files have no automatic upload after Klaude Cold finishes.

The complete selected family runs regardless of point count. Individual benchmark runs can take up to three hours; Klaude Cold monitors their progress within the candidate job's overall time limit. There is no additional Klaude per-run timeout.

The two settings live in the workflows: `MAX_CANDIDATES_PER_RUN` in [`klaude-plan.yml`](../.github/workflows/klaude-plan.yml) controls the planner’s selection limit, and `MAX_REPAIRS` in [`klaude-candidate.yml`](../.github/workflows/klaude-candidate.yml) supplies the repair budget directly to the agent prompt.

Workflow concurrency, selection rules and the job's 360-minute timeout are enforced mechanically. Klaude Cold has a maximum of 200 turns and must finish reporting and cleanup within that job limit. **Execution rules are agent instructions, not a separate enforcement service.** Klaude Cold must stop on success, exhausted repairs, repeated failure without progress, capacity loss or an ambiguous mutation; cancel unfinished child runs and confirm their completion. Job timeout or abrupt runner/agent termination can leave an unfinished run or report, because there is no follow-up job.

Benchmark dispatch names start with `klaude-`. The shared benchmark templates add `klaude | ` to each throughput and eval job name so the InferenceX Dash priority scheduler can identify background work. Its matching scheduler change puts Klaude behind ordinary human jobs regardless of arrival time or recipe score, disables aging and node reservations for Klaude, and ignores skip-queue requests. Klaude uses remaining capacity; running work is not preempted. Explicit administrator overrides retain their existing precedence, and Klaude must not request them. Deploy the matching dashboard scheduler change before enabling auto-sweep.

## Public API investigation

Read the current [API reference](https://inferencex.semianalysis.com/api) and [OpenAPI document](https://inferencex.semianalysis.com/api/openapi.json) for parameters, response shapes and limits. The planner needs only its two public feeds; Klaude Cold chooses additional reads for its candidate. Keep API schemas and model/runner inventories in their existing sources.

All paths in this table are relative to `/api/v1/`:

| Investigation | Useful reads |
| --- | --- |
| Published coverage and recipe provenance | `availability` finds actual scenarios and dates; `workflow-info` provides run attempts, SHAs, changelog config keys and per-run coverage; `submissions` summarizes topology and point counts by date. |
| Image and performance history | `benchmarks` and `benchmarks/history` provide raw metrics, images, topology, recipe fingerprints and producer URLs. Omit the calculator view to retain all available metrics. |
| Sibling points | `benchmark-siblings` finds related points and their concurrency/topology, source GitHub run and dataset slug. Filter to the intended workload and topology before comparing. |
| Published failures and quality | `evaluations` provides task scores and run provenance; `reliability` provides hardware/date success counts, not per-image attribution. |
| Runtime diagnosis | `log-availability` checks retained logs; `server-log-files` discovers filenames; `server-log-search` searches all retained files with bounded snippets; `server-log` reads a named file in chunks. Follow `nextOffset` and respect search truncation. |
| AgentX cache, latency and workload diagnosis | `trace-availability`, `agentic-aggregates`, `derived-agentic-metrics`, `trace-histograms`, `trace-server-metrics` and `request-timeline` expose retained traces, percentiles, token samples, cache/queue/throughput series and request phases/cancellations. Start with aggregates and fetch detailed traces only when needed. |
| AgentX dataset context | `datasets`, `datasets/{slug}`, `datasets/{slug}/conversations` and `datasets/{slug}/conversations/{convId}` provide dataset metadata, distributions and conversation structure. Use the run's dataset slug rather than assuming a dataset. |
| Communication context | `collectivex/latest`, `collectivex/runs` and `collectivex/runs/{runId}` provide versioned communication measurements when relevant to a multinode failure. They require an explicit supported version and can serve stored fallback data. |

Observe these source-level details when interpreting responses:

- `latest-images` and `availability` return raw model keys; benchmark queries require display names from the current OpenAPI enum. Verify the returned model and full candidate identity. `latest-images` is an observation, not a registry of current repository families or compatible replacement images. Release-feed coverage can be incomplete or null.
- Supply the observation's date to `workflow-info`: the current query implementation casts it to a SQL date, despite the reference describing omission as an all-date lookup. Changelog config keys and run coverage narrow the search but do not prove a unique family.
- `exactRun=true` can include a same-image predecessor chain for append-only runs. Preserve each point's producer `run_url` and distinguish it from logical `curve_*` snapshot metadata. Benchmark-row `workflow_run_id` is a database ID; `runId` parameters and `workflow-info`'s run identifiers are GitHub IDs. Diagnostics take positive database benchmark-result `id` values.
- `submissions` groups points by configuration/date and picks a non-null image without ordering; it cannot establish an exact image baseline. Public caches and ingestion can lag new runs. Missing diagnostics mean unavailable evidence, not zero values or a passing result.
- Raw benchmark time metrics use seconds; request timelines have nanosecond offsets and explicitly named millisecond latency fields. Server time-series offsets use seconds. Physical chip counts are independent of logical TP; sum prefill/decode counts only for disaggregated deployments.

The app also has useful **unpublished UI readers**: `/api/unofficial-run` normalizes un-ingested run artifacts, `/api/gpu-metrics` reads GPU metric artifacts, `/api/v1/eval-samples` and `/api/v1/eval-samples-live` drill into failed samples, and `/api/v1/trace-server-metric-source` retrieves a selected worker/source's series. Check their current handlers before optional use; their contracts are page-owned. Unofficial benchmark rows have synthetic `id: 0`, so stored diagnostic lookups do not apply. Keep fresh comparison evidence tied to original run artifacts.

`tco-feed` provides interpolated chart/TCO context rather than an exact recipe comparison. `overview`, `request-chart-data` and `resident-sequence-lengths` are UI projections with no additional selection data needed here. Feedback and administration routes have no role in candidate investigation.

## Change scope and PR policy

Klaude Cold is instructed to change only the selected master family's image and its already referenced, unshared srt-slurm recipe YAML images/backend compatibility settings. Model, precision, topology, speculative decoding, workload, commands, resources and recipe references stay fixed. `model.container` and any `identity.container.image` must match the master image. Shared scripts, launchers, libraries, workflow/control files and unrelated families stay unchanged. Klaude Cold runs focused checks using **uv**; there is no separate patch validator after the agent.

The explicit task rule keeps **`perf-changelog.yaml` byte-for-byte unchanged**. This conflicts with [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`claude-pr-review.yml`](../.github/workflows/claude-pr-review.yml), which require changelog entries for these changes. Klaude Cold must explain the conflict and keep every PR draft. Existing checks and human review remain in force; no automatic merge or policy bypass is authorized. Required branch-protection checks have not been inspected.

## Workflow operation and credentials

The entry workflow is manual-only (`workflow_dispatch`). Its commented six-hour cron (`0 */6 * * *`, UTC) remains disabled. The gate `github.ref == 'refs/heads/main' && github.run_attempt == 1` skips feature branches, tags and reruns. Only the candidate exposes `workflow_call`. One non-canceling `klaude-auto-sweep` concurrency group prevents overlapping auto-sweep invocations; candidates within an invocation run in parallel without a workflow-imposed parallelism cap.

The planner's Python fetch step uses the read-only workflow token and dashboard key. Its bounded Claude PR review uses `ANTHROPIC_API_KEY` and the read-only workflow token with `pull-requests: read`; it receives neither `CLAUDE_PAT` nor the private dashboard key and makes no GitHub mutations. The candidate receives `CLAUDE_PAT` for branch/PR writes and e2e dispatch/cancellation, `ANTHROPIC_API_KEY` for Klaude Cold, and an expiring `status:read` `KLAUDE_DASHBOARD_API_KEY` covering clusters. Klaude Cold has access to these credentials and must not publish them or private API responses. The shared HTTP reader uses fixed origins, bounded GETs and no redirects. No new controller, database or environment configuration is required.

All external actions use full commit SHAs verified against upstream release/tag metadata and implementation tests on 2026-09-04. The internal call uses GitHub's [`$/` syntax](https://github.blog/changelog/2026-07-30-reference-same-repository-actions-with-self-repository-syntax/) to resolve the caller's exact commit (runner 2.336.0 or newer), with the three required secrets explicitly forwarded.

| Action | Version | Commit |
| --- | --- | --- |
| `anthropics/claude-code-action` | `v1.0.216` | [`d75b94d5ad42`](https://github.com/anthropics/claude-code-action/commit/d75b94d5ad426cb8546e6628b6f5f19b84e5cce1) |
| `actions/checkout` | `v7.0.1` | [`3d3c42e5aac5`](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1) |
| `actions/upload-artifact` | `v7.0.1` | [`043fb46d1a93`](https://github.com/actions/upload-artifact/commit/043fb46d1a93c77aae656e7c1c64a875d1fc6a0a) |
| `actions/download-artifact` | `v8.0.1` | [`3e5f45b2cfb9`](https://github.com/actions/download-artifact/commit/3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c) |
| `astral-sh/setup-uv` | `v10.0.1` | [`20cfd1bf945f`](https://github.com/astral-sh/setup-uv/commit/20cfd1bf945f4377ade1205e4dbc17946fc9a30d) |

## Local verification

```bash
uv run --no-project --python 3.12 --with "pydantic>=2.10,<3" python -m utils.klaude --help
uvx zizmor==1.30.0 --offline --no-config --no-ignores .github/workflows/klaude-plan.yml .github/workflows/klaude-candidate.yml
```

CLI and workflow checks do not establish GPU workingness. Klaude Cold uses the existing InferenceX validation and e2e workflows for its candidate changes. No live model, benchmark, PR creation or deployment is part of local verification.

The regular zizmor scan is clean. Auditor mode reports permission-documentation notes and use of the repository dashboard secret without a dedicated GitHub environment; no ignores are added. `actionlint` 1.7.12 does not understand `$/`; lint a temporary copy with only that prefix normalized to `./` to retain its other checks. Do not claim that its raw scan passes.
