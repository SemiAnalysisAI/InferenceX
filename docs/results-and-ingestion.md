# Results and Ingestion

<div align="center">

**English** | [中文](./results-and-ingestion_zh.md)

</div>

Use this page to identify benchmark artifacts, inspect their contracts, and decide whether a run is safe to hand to InferenceX-app. The producers and ingest code remain authoritative. InferenceX-app links below are pinned to commit [`3be1c34`](https://github.com/SemiAnalysisAI/InferenceX-app/tree/3be1c34a174f62fea2194f1133210e692e5bf415).

## Source map

| Source of truth | What it controls |
| --- | --- |
| [`benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml), [`benchmark-multinode-tmpl.yml`](../.github/workflows/benchmark-multinode-tmpl.yml) | Per-config names, files, and upload rules for throughput, eval, and AgentX artifacts |
| [`utils/process_result.py`](../utils/process_result.py) | Fixed-sequence throughput aggregate schema and derived per-GPU metrics |
| [`utils/collect_results.py`](../utils/collect_results.py), [`collect-results.yml`](../.github/workflows/collect-results.yml) | Recursive benchmark collection into `agg_<prefix>.json` and `results_<prefix>` |
| [`utils/collect_eval_results.py`](../utils/collect_eval_results.py), [`collect-evals.yml`](../.github/workflows/collect-evals.yml) | Eval discovery, metric extraction, batched-concurrency selection, and `eval_results_<prefix>` |
| [`process_agentic_result.py`](../utils/agentic/aggregation/process_agentic_result.py), [`request_metrics.py`](../utils/agentic/aggregation/request_metrics.py) | AgentX aggregate schema, raw-record filtering, request accounting, and derived metrics |
| [`validate_agentic_result.py`](../utils/agentic/validation/validate_agentic_result.py) | AgentX pre-upload error-rate gate |
| [`run-sweep.yml`](../.github/workflows/run-sweep.yml), [`recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml) | App dispatch payload and source/merge run identities |
| [InferenceX-app `prepare-ci-artifacts.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/prepare-ci-artifacts.ts), [`ci-artifact-preparation.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/lib/ci-artifact-preparation.ts) | Cross-run artifact selection, attempts, and reuse provenance |
| [InferenceX-app `ingest-ci-run.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/ingest-ci-run.ts) | End-to-end ingest ordering, pairing, skips, summaries, and refresh |
| [InferenceX-app `benchmark-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/benchmark-mapper.ts), [`eval-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/eval-mapper.ts), [`agentic-v3-flatten.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/agentic-v3-flatten.ts) | Artifact-to-database schemas and normalization |
| [InferenceX-app schema](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/migrations/001_initial_schema.sql), [AgentX migration](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/migrations/008_agentic.sql) | Persisted natural keys, JSONB metrics, and trace-replay sidecars |
| [InferenceX-app benchmark API](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/app/src/app/api/v1/benchmarks/route.ts), [model constants](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/constants/src/models.ts) | Published-result queries and frontend model names |

## Index

1. [Identity at each layer](#identity-at-each-layer)
2. [Throughput artifacts](#throughput-artifacts)
3. [Eval artifacts](#eval-artifacts)
4. [AgentX artifacts](#agentx-artifacts)
5. [App handoff and reused runs](#app-handoff-and-reused-runs)
6. [Ingestion stages](#ingestion-stages)
7. [Dedupe and failed-row behavior](#dedupe-and-failed-row-behavior)
8. [Provenance invariants](#provenance-invariants)
9. [Published results API](#published-results-api)
10. [Safe inspection](#safe-inspection)
11. [Verification and stop conditions](#verification-and-stop-conditions)

## Identity at each layer

Do not use one identifier as a substitute for another.

| Layer | Identity | Meaning |
| --- | --- | --- |
| GitHub workflow | `(github_run_id, run_attempt)` | One execution attempt. InferenceX-app stores this as the `workflow_runs` natural key. |
| Uploaded artifact | Artifact `name` within a workflow run | A downloadable bundle. A rerun can upload the same exact name again. |
| Per-config result | `RESULT_FILENAME`, plus a `_concN` suffix where the multi-point AgentX path emits one file per concurrency | Runtime configuration identity used to name files and sibling bundles. It is not the database key. |
| Normalized config | Model, hardware, framework, precision, speculative method, disaggregation, topology, and related config fields | InferenceX-app resolves or creates a `configs` row after normalization. |
| Throughput or AgentX database point | `(workflow_run_id, config_id, benchmark_type, isl, osl, conc, offload_mode)`, with nulls treated as equal | Idempotent benchmark identity. AgentX uses `benchmark_type=agentic_traces`, null `isl`/`osl`, and concurrency as users. |
| Eval database point | `(workflow_run_id, config_id, task, isl, osl, conc)` | Aggregate and per-config eval artifacts converge when all nullable dimensions are populated identically. The current unique constraint uses ordinary PostgreSQL null semantics, so a null `isl`, `osl`, or `conc` does not conflict with another null. |
| Eval sample | `(eval_result_id, doc_id)` | Per-document sample identity. |
| AgentX raw sidecar | `benchmark_results.trace_replay_id` | Link from a normalized AgentX point to retained and precomputed trace data. |

The distinction matters during reuse. Artifact bytes can come from a PR sweep while changelog metadata and the ingest trigger come from a later main run. The stored benchmark row still belongs to the source run and source attempt.

## Throughput artifacts

### Producer and collector

For a single-node fixed-sequence job, [`benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml) builds a `RESULT_FILENAME` from the experiment, precision, framework, TP/PP/DCP/PCP/EP/DP-attention, disaggregation, speculative mode, concurrency, and concrete runner. The benchmark first writes `<RESULT_FILENAME>.json`. [`utils/process_result.py`](../utils/process_result.py) reads that file and writes `agg_<RESULT_FILENAME>.json`. The workflow uploads it as:

```text
artifact: bmk_<RESULT_FILENAME>
file:     agg_<RESULT_FILENAME>.json
```

The multinode template encodes prefill and decode topology, worker counts, mode, concurrency, and runner in its base name. It can place several `agg_<RESULT_FILENAME>_*.json` files in one `bmk_<RESULT_FILENAME>` artifact.

[`collect-results.yml`](../.github/workflows/collect-results.yml) normally receives `result-prefix: bmk`. It downloads `bmk_*`, and [`utils/collect_results.py`](../utils/collect_results.py) recursively loads every JSON file into one array. The handoff identity is then:

```text
artifact: results_bmk
file:     agg_bmk.json
shape:    array of benchmark row objects
```

The collector does not validate rows, sort them, or deduplicate them. A successful JSON parse is its only content check. Treat `results_bmk` as a transport aggregate, not as proof that every row is usable.

### Throughput row schema

The fixed-sequence transformer requires runner, framework, precision, speculative mode, result filename, ISL, OSL, disaggregation, model prefix, and image metadata. Its base output includes:

| Field group | Important fields |
| --- | --- |
| Config and routing | `hw`, `model`, `infmax_model_prefix`, `framework`, `precision`, `spec_decoding`, `image`, `disagg`, `is_multinode`, `isl`, `osl`, `conc` |
| Single-node topology | `tp`, `pp`, `dcp_size`, `pcp_size`, `ep`, `dp_attention` |
| Multinode topology | `prefill_tp`, `prefill_pp`, `prefill_dcp_size`, `prefill_pcp_size`, `prefill_ep`, `prefill_dp_attention`, `prefill_num_workers`, matching `decode_*` fields, `num_prefill_gpu`, `num_decode_gpu`, and optional `prefill_hw`/`decode_hw` |
| Primary derived metrics | `tput_per_gpu`, `input_tput_per_gpu`, `output_tput_per_gpu` |
| Latency and interactivity | Each benchmark input key ending in `ms` is converted from milliseconds to seconds with `_ms` removed. Keys containing `tpot` also produce an `intvty` reciprocal. |
| Optional runtime metadata | `router` as exactly `{name, version}`, `kv_p2p_transfer`, and measured power patched from `gpu_metrics.csv` when available |

Single-node GPU count is `tp * pp * pcp_size`. DCP does not multiply the physical GPU count. Multinode per-GPU denominators use the declared prefill and decode GPU counts. Invalid or missing required metadata fails transformation. Power aggregation is explicitly best effort and cannot fail the benchmark aggregate.

InferenceX-app treats routing fields as columns or config dimensions and stores numeric measurements in `benchmark_results.metrics` JSONB. The mapper supports v1 shared topology, v2 split prefill/decode topology, and nested v3 AgentX metrics. Unknown numeric metrics are retained and warned about, which permits schema growth without silently losing numeric data.

### Power telemetry stop and validation

The single-node AMD monitor in [`benchmark_lib.sh`](../benchmarks/benchmark_lib.sh) waits for every observed GPU to have a positive, numeric power sample at or beyond the first whole second after the stop request. `AMD_MONITOR_STOP_TIMEOUT_S` bounds the wait (default `30`; `0` skips it). Streams without usable epoch timestamps use the legacy fixed tail wait. A timeout does not certify coverage: the aggregator still rejects an unbracketed benchmark window. AgentX cancellation skips the coverage wait and stops the monitor, including when cancellation arrives during a normal stop.

[`aggregate_power.py`](../utils/aggregate_power.py) excludes unusable power rows outside the formal window from integration and bracketing. Within the boundary search band, these rows are counted per GPU in `power_validation.json` under `boundary_degenerate_rows`, even when no usable samples remain. In-window validation is unchanged; these counters are diagnostic evidence, not replacement measurements.

## Eval artifacts

### Per-config identity and collection

Each eval upload is named `eval_<EXP_NAME>_<RESULT_FILENAME>`. Its current allowed payload includes `meta_env.json`, `results*.json`, sample JSONL, predictions, SWE-bench reports, and trajectory files. The collector uses only the metadata and lm-eval result JSON for aggregate rows.

[`utils/collect_eval_results.py`](../utils/collect_eval_results.py) applies these rules:

1. An eval set is a root or immediate child directory containing `meta_env.json`.
2. Candidate result files must parse as objects and contain `lm_eval_version`.
3. A legacy set contributes the newest candidate by modification time.
4. A batched set is identified by a list-valued `eval_concs`. It contributes the newest `_concN` result for each allowed concurrency, restricted by `completed_eval_concs` when present.
5. Each task in `results` becomes one logical row.
6. Primary `score` prefers strict or resolved exact match, then accuracy. Standard error is kept separately.

The collection handoff is:

```text
artifact: eval_results_all
file:     agg_eval_all.json
shape:    array of one row per config, concurrency, and task
```

### Eval aggregate schema

| Field group | Important fields |
| --- | --- |
| Config | `is_multinode`, `model_prefix`, `model`, `hw`, `framework`, `precision`, `spec_decoding`, `isl`, `osl`, `conc` |
| Topology | `tp`, `ep`, `dp_attention`, plus split `prefill_*` and `decode_*` worker fields |
| Evaluation | `task`, `score`, `score_name`, `score_se`, `em_strict`, `em_strict_se`, `em_flexible`, `em_flexible_se`, `n_eff` |
| Traceability | `source`, the selected result JSON path inside the downloaded artifact tree |

The app also reads every unaggregated `eval_*` directory. `meta_env.json` supplies config identity, while `results_*.json` supplies `lm_eval_version`, tasks, raw numeric metrics, and effective sample count. It normalizes strict and flexible exact-match names. Sample files are attached to the resolved eval row by task. This dual path is intentional. Aggregate rows serve summary ingestion, while per-config files retain sample details.

A collector parse failure is skipped by `load_json`. It is not represented as a failed eval row. Missing `meta_env.json`, no recognized lm-eval result, an empty `results` object, or a concurrency absent from `completed_eval_concs` means no aggregate row is emitted.

## AgentX artifacts

### Three related identities

AgentX uses the same `RESULT_FILENAME` family but produces two sibling artifact classes:

```text
aggregate artifact: bmk_agentic_<RESULT_FILENAME>
aggregate file:     <RESULT_FILENAME>.json
raw artifact:       agentic_<RESULT_FILENAME>
raw tree:           results/**, excluding inputs.json and profile_export_raw.jsonl
```

The aggregate artifact matches the `bmk_*` collection pattern and therefore also appears as a row in `results_bmk/agg_bmk.json`. The raw sibling is not fed to `collect_results.py`. InferenceX-app pairs `bmk_agentic_<suffix>` with `agentic_<suffix>` after stripping `bmk_` and `agentic_`. For files named with `_concN.json`, the concurrency is part of trace-sibling lookup.

Server logs are separate `server_logs_<RESULT_FILENAME>` artifacts. The app uses the fully stripped suffix fallback so AgentX rows can find a server log even though the log artifact has no `agentic_` prefix.

### Raw inputs and aggregate schema

[`process_agentic_result.py`](../utils/agentic/aggregation/process_agentic_result.py) resolves the current `results/aiperf_artifacts` layout and a one-child nested layout. It requires `profile_export.jsonl`. It reads these inputs when present:

| Input | Role |
| --- | --- |
| `profile_export.jsonl` | Per-request metrics and lifecycle metadata. Required. |
| `profile_export_aiperf.json` | AIPerf aggregate metadata, including dataset provenance when emitted. Optional. |
| `server_metrics_export.json` | Server cache, KV-cache, and token metrics. Missing data produces empty or warning-backed server metrics rather than replacing the request source. |
| Server logs | Framework-specific fallback for server metrics and capacity. |

Every nonblank JSONL record increments `records_total`. Records with `metadata.benchmark_phase` other than `profiling` are warmup diagnostics and are excluded. Records with a truthy `error` are also excluded and categorized. Older records with no phase are treated as profiling. The retained count becomes `num_requests_successful`. The full accounting is preserved in `request_accounting` with profiled, total dropped, warmup dropped, error dropped, and `error_categories` fields.

The AgentX aggregate has top-level identity and topology fields compatible with benchmark ingestion. Important AgentX fields include:

| Field group | Important fields |
| --- | --- |
| Scenario and request counts | `scenario_type: agentic-coding`, `num_requests_total`, `num_requests_successful`, `request_accounting` |
| Cache configuration | `kv_offloading`, `kv_offload_backend`, optional `kv_p2p_transfer`, `allocated_cpu_dram_gb`, optional `router` |
| Provenance | `dataset`, copied from AIPerf `metadata.dataset` |
| Request metrics | `request_metrics.qps`, `latency` blocks for TTFT/E2EL/ITL/TPOT/interactivity, token distributions, throughput, cache, and per-GPU throughput |
| Server metrics | `server_metrics.cache`, `kv_cache`, token totals, source details, and any `warnings` |
| Compatibility | `kv_cache_pool_tokens` mirrors `server_metrics.kv_cache.gpu_total_tokens` |

The app flattens nested AgentX v3 values into canonical metric keys. Examples include `median_ttft`, `p95_e2el`, `total_tput_tps`, `tput_per_gpu`, `server_gpu_cache_hit_rate`, and `gpu_kv_cache_usage_pct`. It maps p50 to `median`. Full-response ITL fields take precedence when present, and interactivity percentiles are derived as the reciprocal of the matching ITL percentile so historical and current rows use one definition.

Before normal upload, the single-node workflow runs [`validate_agentic_result.py`](../utils/agentic/validation/validate_agentic_result.py). It requires an aggregate object, a numeric non-negative `request_count.avg`, positive completed requests, and an error rate at or below the configured threshold. Passing this gate does not mean no requests failed. Failed request records remain visible through `request_accounting` but do not contribute to performance metrics.

## App handoff and reused runs

[`run-sweep.yml`](../.github/workflows/run-sweep.yml) dispatches either `ingest-results` or `ingest-agentic-results` to InferenceX-app.

- `source-run-id` identifies the workflow run whose benchmark, eval, AgentX, log, and stats artifacts supply measured data.
- `merge-run-id` identifies the main-branch workflow run that authorized the ingest and supplies current changelog metadata.
- For an ordinary main run, both IDs equal `github.run_id`.
- For a reused PR sweep, `source-run-id` is the selected PR run and `merge-run-id` is the current main run.

InferenceX-app's artifact preparation keeps the newest unexpired upload for each exact artifact name from the source run. In reuse mode it excludes source-run `changelog-metadata`, requires an unexpired changelog artifact from the merge run, and adds that merge-run artifact to the plan. No unexpired source artifacts, or no merge-run changelog during reuse, is a hard failure.

It writes `reused-ingest-metadata/reuse_source_run.json` only when the IDs differ. The file records source ID, source attempt, URL, PR, head SHA, plus ingest ID, ingest attempt, and URL. During ingest, this metadata switches row attribution back to the source run and source attempt. The source PR branch, SHA, and public run URL remain attached to measured rows. Trigger timing can come from the merge run. This avoids presenting reused PR measurements as if the merge run produced them.

Both app workflows pass the merge run ID and attempt into initial CI setup because that is the trigger bundle. `ingest-ci-run.ts` then reads reuse metadata and deliberately replaces them with source identity before it creates or resolves `workflow_runs`.

## Ingestion stages

InferenceX-app runs the following order. The fixed-sequence workflow has a 30-minute timeout. The blob-heavy AgentX workflow has a 180-minute timeout and can target staging or production.

1. **Prepare artifacts.** Fetch source and merge run metadata, select unexpired artifacts, download each into an empty directory, and write reuse metadata when needed.
2. **Run database migrations.** Ingest never assumes the target schema is current.
3. **Resolve provenance.** Read reuse metadata, fetch GitHub metadata, preload normalized config rows, and create or resolve the source `(github_run_id, run_attempt)` workflow row. A source run or attempt listed in run overrides stops here without writing results.
4. **Read changelog metadata.** Detect `evals-only`. If no valid changelog artifact exists, synthesize a fallback description from workflow metadata. Evals-only runs skip benchmark and run-stat stages.
5. **Ingest benchmark rows.** Read `results_bmk` and every `bmk_*` or `results_*` directory. Normalize model, hardware, framework, precision, topology, scenario, offload mode, image, and numeric metrics. Resolve configs, skip point overrides, bulk upsert rows, then build availability only from rows whose bulk write succeeded.
6. **Attach large side data.** Link server logs. Pair AgentX aggregate and raw artifacts. For rows without an existing trace link, worker processes compress raw profile/server JSON, compute aggregate stats, chart series, and request timelines, then persist one sidecar and link it to the benchmark rows.
7. **Link dataset provenance.** Extract an AgentX dataset slug from aggregate rows and upsert one `run_datasets` mapping. More than one slug in one workflow is a hard conflict. A slug absent from the dataset table is reported because timeline deep links will fail until dataset ingest occurs.
8. **Ingest run stats.** Upsert per-hardware success and total counts for non-evals-only runs.
9. **Ingest eval summaries and samples.** Process `eval_results_all`, then per-config `eval_*` directories. Both resolve to the same eval natural key. Attach sample JSONL by task and document ID.
10. **Ingest changelog and summarize.** Upsert changelog rows, print new/duplicate/skip counts and unmapped values, and write the unmapped-entities report used by workflow alerts.
11. **Refresh and verify.** Refresh `latest_benchmarks`. The workflow then applies audited point overrides, runs `admin:db:verify`, and invalidates the app cache.

This ordering is intentional. Availability is not created for a benchmark row before its write succeeds. Trace blobs are not inserted when every target row already has a sidecar. Cache invalidation happens only after migrations, ingest, overrides, and database verification.

## Dedupe and failed-row behavior

Dedupe exists at several boundaries. Check the boundary before diagnosing a duplicate.

| Boundary | Behavior |
| --- | --- |
| Artifact preparation in CI | Newest unexpired upload per exact artifact name. Reuse replaces only changelog metadata with the merge-run copy. |
| Direct app download mode | [`dedupeArtifactsByLogicalName`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/lib/github-artifacts.ts) strips a trailing runner-pool and attempt token and keeps the newest logical artifact. This prevents a retry artifact from overwriting good metrics. |
| Benchmark collection | `collect_results.py` appends every parsed JSON. It has no row-level dedupe. |
| Benchmark database write | `ON CONFLICT` on the benchmark natural key updates metrics, image, power workers, and related fields. Server-derived `kv_cache_pool_tokens` is preserved when a fresh artifact lacks it. |
| Eval database write | Aggregate and per-config rows with fully populated matching dimensions conflict on the eval natural key. The later write refreshes metrics and returns the same row ID for sample attachment. If any nullable key dimension is null, PostgreSQL's current ordinary unique constraint does not deduplicate the rows. |
| Eval samples | Conflict on `(eval_result_id, doc_id)` prevents duplicate documents. |
| AgentX trace sidecar | Existing links are queried first. Blob preparation and insertion are skipped when all benchmark rows are already linked. |

Failed data is handled separately from dedupe:

- The app benchmark mapper drops any row where numeric `num_requests_successful` is zero and `num_requests_total` is present, including zero-total server-start failures. This protects a good natural-key row from being replaced by a dataless retry.
- AgentX aggregation drops warmup and error request records from metric computation but retains counts and categories in `request_accounting`.
- Unknown models or hardware, missing fixed-sequence ISL/OSL/concurrency, bad JSON, point overrides, and database errors are tracked as skips. They do not become placeholder rows.
- Eval collector parse failures and missing recognized result files simply emit no row. App-side malformed per-config files produce warnings or tracked skips.
- A partial ingest is safe to rerun because database writes are idempotent. It is not safe to ignore new skip counts, conflicting dataset provenance, missing raw AgentX siblings, or failed database verification.

## Provenance invariants

1. **Run provenance is two-dimensional.** Record the run ID and attempt. A run URL without an attempt can point at the newest rerun rather than the bytes that were ingested.
2. **Reuse never changes measurement authorship.** Metrics and source Git metadata belong to `source-run-id`. Only the merge-run changelog and trigger context come from `merge-run-id`.
3. **Artifact provenance is inspectable.** Keep the exact artifact name, upload timestamp, expiry state, and selected file. `source` on eval aggregates is a path clue, not a replacement for the workflow run and artifact identity.
4. **Dataset provenance is carried, not inferred.** AgentX copies AIPerf `metadata.dataset` to top-level `dataset`. The app derives the dashboard slug from the declared Hugging Face dataset name. It does not guess from a benchmark name.
5. **One AgentX workflow maps to at most one dataset.** Conflicting slugs stop ingest. Missing legacy provenance leaves an existing run mapping untouched.
6. **Normalization is part of provenance.** Keep raw `model`, `infmax_model_prefix`, `hw`, `framework`, precision, image, topology, and scenario fields available while diagnosing a mapping. The database config ID is a normalized identity, not the original spelling.
7. **Raw AgentX data and aggregate metrics must pair.** A benchmark point can be ingested without a raw sidecar, but that is a reported missing-sibling condition and detail views will be incomplete.

## Published results API

Use the public API for results already ingested into the dashboard. Use GitHub Actions artifacts for un-ingested runs, raw output, logs, or debugging evidence.

```bash
INFERENCEX_API=https://inferencex.semianalysis.com/api/v1
curl --fail --compressed \
  "$INFERENCEX_API/benchmarks?model=DeepSeek-V4-Pro" \
  | jq 'first(.[] | select(.benchmark_type == "single_turn" and .isl == 8192 and .osl == 1024)) | {date, model, hardware, framework, precision, isl, osl, tput_per_gpu: .metrics.tput_per_gpu, run_url}'
```
The example intentionally returns one bounded 8k1k record. Add exact hardware, framework, precision, and date predicates before treating the selected row as evidence.

`model=` takes the frontend display name defined by the linked InferenceX-app model constants. Fixed-sequence rows use numeric `isl` and `osl`. `agentic_traces` rows use null lengths, so do not filter them out accidentally. `view=calculator&sequence=8k/1k` returns compact interpolation data. `date`, `runId`, `exact`, and `exactRun` scope historical or run-specific reads. Discovery endpoints include `/availability`, `/workflow-info`, `/evaluations`, and `/reliability`.

Always use `--compressed` and filter with `jq`. Do not dump raw benchmark JSON, cache-bust, or repeatedly poll CDN-cached results.

## Safe inspection

All commands below are read-only against GitHub. Downloads go to a new temporary directory. Replace IDs and artifact names only after listing the run.

### Inspect run metadata and artifact inventory

```bash
RUN_ID=<github-run-id>
REPO=SemiAnalysisAI/InferenceX

gh api "repos/$REPO/actions/runs/$RUN_ID" \
  --jq '{id,run_attempt,status,conclusion,event,head_branch,head_sha,html_url}'

gh api --paginate "repos/$REPO/actions/runs/$RUN_ID/artifacts?per_page=100" \
  --jq '.artifacts[] | [.id,.name,.created_at,.expires_at,.expired,.size_in_bytes] | @tsv'
```

Stop if the ID is not numeric, the repository is wrong, the run attempt is not the intended one, required artifacts are expired, or several uploads with the same name cannot be distinguished by timestamp.

### Preview a source/merge artifact plan

From an InferenceX-app checkout at the pinned or reviewed revision:

```bash
SOURCE_RUN_ID=<measurement-run-id> \
MERGE_RUN_ID=<main-trigger-run-id> \
INGEST_REPO=SemiAnalysisAI/InferenceX \
bun run admin:db:prepare:ci --dry-run
```

`--dry-run` fetches metadata and prints the selected artifact plan without downloading artifacts or connecting to the database. For reuse, confirm that benchmark data comes from the source run and `changelog-metadata` comes from the merge run.

### Download and inspect aggregate schemas

```bash
tmp="$(mktemp -d "${TMPDIR:-/tmp}/infx-results.XXXXXX")"

gh run download "$RUN_ID" -R "$REPO" -n results_bmk -D "$tmp/results_bmk"
jq 'type, length' "$tmp/results_bmk/agg_bmk.json"
jq '.[0] | {hw,model,infmax_model_prefix,framework,precision,scenario_type,isl,osl,conc,is_multinode,disagg}' \
  "$tmp/results_bmk/agg_bmk.json"

# Run this only when eval_results_all appeared in the inventory.
gh run download "$RUN_ID" -R "$REPO" -n eval_results_all -D "$tmp/eval_results_all"
jq 'type, length' "$tmp/eval_results_all/agg_eval_all.json"
jq '.[0] | {model_prefix,model,hw,framework,task,conc,score,score_name,n_eff,source}' \
  "$tmp/eval_results_all/agg_eval_all.json"
```

Do not assume row zero represents the configuration under investigation. After checking shape, filter by stable fields:

```bash
jq --arg model '<model-prefix>' \
  '[.[] | select(.infmax_model_prefix == $model or .model_prefix == $model)] |
   map({scenario_type,hw,framework,precision,isl,osl,conc,num_requests_total,num_requests_successful,request_accounting})' \
  "$tmp/results_bmk/agg_bmk.json"
```

### Inspect one raw AgentX sibling

Choose the exact `agentic_<RESULT_FILENAME>` name from the inventory:

```bash
AGENTIC_ARTIFACT='agentic_<RESULT_FILENAME>'
gh run download "$RUN_ID" -R "$REPO" -n "$AGENTIC_ARTIFACT" -D "$tmp/agentic"

jq '{metadata,request_count,error_request_count,completed_request_count}' \
  "$tmp/agentic/results/aiperf_artifacts/profile_export_aiperf.json"

jq -s '{rows:length,
        profiling:map(select((.metadata.benchmark_phase // "profiling") == "profiling"))|length,
        warmup:map(select((.metadata.benchmark_phase // "profiling") != "profiling"))|length,
        errors:map(select(.error))|length}' \
  "$tmp/agentic/results/aiperf_artifacts/profile_export.jsonl"
```

Some historical raw artifacts use a one-child nested AIPerf directory. If the exact path above is absent, stop and inspect the downloaded directory names before choosing a file. Do not flatten, rename, or combine files by guesswork.

Remove only the temporary directory you just created after inspection:

```bash
printf 'temporary inspection directory: %s\n' "$tmp"
rm -rf -- "$tmp"
```

## Verification and stop conditions

A handoff is verified only when all applicable checks pass.

### Verify before ingest

- The intended `github_run_id` and `run_attempt` are explicit.
- `results_bmk` contains a JSON array when throughput or AgentX points are expected.
- `eval_results_all` contains a JSON array when aggregate evals are expected, and the matching per-config `eval_*` bundles still exist when sample details matter.
- Every expected AgentX aggregate has its `agentic_<suffix>` raw sibling. Server logs are present when server-derived metrics are required.
- Fixed-sequence rows have positive `isl`, `osl`, and `conc`. AgentX rows have an agentic scenario, positive concurrency, request counts, and expected offload and dataset metadata.
- The source/merge dry-run selects source measurements and the merge changelog exactly as intended.
- Artifact expiry leaves enough time for the complete app workflow, especially the longer AgentX trace processing path.

### Verify after ingest

Use the app workflow logs and database verifier, not artifact existence alone. Confirm:

- the source run and source attempt shown by `ingest-ci-run` are correct.
- new and duplicate counts are plausible for a first ingest or rerun.
- skip counters, unmapped entities, missing datasets, and missing trace siblings are understood.
- no conflicting dataset provenance occurred.
- trace-replay links were created or explicitly already existed for AgentX points.
- `admin:db:verify` passed after overrides.
- cache invalidation ran after verification.

### Stop conditions

Stop the ingest or recovery investigation when any of these holds:

- source and merge IDs or attempts are ambiguous.
- the selected source has no unexpired result artifacts, or reuse has no unexpired merge-run changelog.
- aggregate JSON is malformed, has the wrong top-level shape, or contains no expected rows.
- a fixed-sequence point lacks identity dimensions, or an AgentX aggregate reports zero successful requests.
- the AgentX validation error rate exceeds its configured threshold.
- aggregate and raw AgentX suffixes cannot be paired deterministically.
- multiple dataset slugs appear in one workflow run, or required dataset provenance is absent.
- unexpected duplicate natural keys would rely on input ordering to choose the surviving values.
- unmapped model, hardware, or precision values would drop the target row.
- app ingestion reports database errors, missing required sidecars, an override mismatch, or failed database verification.

Do not repair these conditions by editing downloaded JSON, changing IDs, suppressing skips, or manually inserting rows. Fix the producer, mapping, provenance, or artifact selection at its source, then rerun the idempotent ingest.
