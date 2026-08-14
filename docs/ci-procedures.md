# CI Procedures

<div align="center">

**English** | [中文](./ci-procedures_zh.md)

</div>

Use this page for matrix generation, CI dispatch, PR sweeps, result staging, artifact reuse, and post-merge publication. English is the source version. Keep the Chinese page structurally synchronized with it. Record the repository, commit SHA, workflow run ID, run attempt, and artifact name whenever CI output is used as evidence.

## Procedure index

| Task | Go to |
| --- | --- |
| Identify the authoritative implementation | [Source map](#source-map) |
| Avoid applying the wrong branch's workflow contract | [Source-snapshot warning](#source-snapshot-warning) |
| Generate and inspect a matrix locally | [Local matrix generation](#local-matrix-generation) |
| Validate YAML and `perf-changelog.yaml` | [YAML and changelog validation](#yaml-and-changelog-validation) |
| Launch a targeted GPU run | [Manual end-to-end dispatch](#manual-end-to-end-dispatch) |
| Operate the weekly overview snapshot | [Weekly overview snapshot](#weekly-overview-snapshot) |
| Select PR sweep labels | [PR primary and modifier labels](#pr-primary-and-modifier-labels) |
| Understand early cancellation | [Canary and fail-fast semantics](#canary-and-fail-fast-semantics) |
| Diagnose or rerun a workflow | [Monitoring and reruns](#monitoring-and-reruns) |
| Publish a PR run to staging | [Stage results](#stage-results) |
| Merge without repeating an approved sweep | [Artifact reuse and merge-with-reuse](#artifact-reuse-and-merge-with-reuse) |
| Recover an append-only changelog conflict | [Changelog conflict recovery](#changelog-conflict-recovery) |
| Inspect result JSON without loading everything | [Artifact downloads and parsing](#artifact-downloads-and-parsing) |
| Verify publication after merge | [Post-merge expectations](#post-merge-expectations) |

## Source map

These files are the contract. Follow the target ref's source rather than copying a command from an old run:

| Concern | Exact source |
| --- | --- |
| Generator CLI, filtering, and eval marking | [`utils/matrix_logic/generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py) |
| Strict master-config and matrix schemas | [`utils/matrix_logic/validation.py`](../utils/matrix_logic/validation.py) |
| Generator examples and reuse policy | [`.github/workflows/README.md`](../.github/workflows/README.md) |
| Manual end-to-end inputs and matrix fan-out | [`.github/workflows/e2e-tests.yml`](../.github/workflows/e2e-tests.yml) |
| PR/main sweep gates, canary, collection, and ingest dispatch | [`.github/workflows/run-sweep.yml`](../.github/workflows/run-sweep.yml) |
| Single- and multi-node artifact uploads | [`.github/workflows/benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml), [`.github/workflows/benchmark-multinode-tmpl.yml`](../.github/workflows/benchmark-multinode-tmpl.yml) |
| Throughput and eval aggregation | [`.github/workflows/collect-results.yml`](../.github/workflows/collect-results.yml), [`.github/workflows/collect-evals.yml`](../.github/workflows/collect-evals.yml), [`utils/collect_results.py`](../utils/collect_results.py), [`utils/collect_eval_results.py`](../utils/collect_eval_results.py) |
| Changelog byte/diff/matrix gate | [`utils/validate_perf_changelog.py`](../utils/validate_perf_changelog.py), [`utils/process_changelog.py`](../utils/process_changelog.py) |
| Reuse authorization and source-run selection | [`utils/find_reusable_sweep_run.py`](../utils/find_reusable_sweep_run.py) |
| Supported reuse merge and conflict preparation | [`utils/merge_with_reuse.sh`](../utils/merge_with_reuse.sh), [`utils/prepare_perf_changelog_merge.py`](../utils/prepare_perf_changelog_merge.py) |
| Staging request and callback | [`.github/workflows/stage-results.yml`](../.github/workflows/stage-results.yml), [`.github/workflows/stage-results-callback.yml`](../.github/workflows/stage-results-callback.yml) |
| Reused agentic-ingest redispatch | [`.github/workflows/recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml) |
| Post-merge responsibility reminder | [`.github/workflows/pr-recipe-reminder.yml`](../.github/workflows/pr-recipe-reminder.yml) |

## Source-snapshot warning

This page was authored from branch commit `0c28706b33d4a796b82f6f9c3594c19c46365575`. At that time, local `origin/main` was `de493d8597035e6692833de6189b567887968460`, and the relevant CI sources were not identical:

- The branch-local [`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) requires `generate-cli-command` and hard-codes each matrix to `fail-fast: false`. The audited [`origin/main` version](https://github.com/SemiAnalysisAI/InferenceX/blob/de493d8597035e6692833de6189b567887968460/.github/workflows/e2e-tests.yml) makes that command conditionally optional and adds trusted-changelog dispatch, `fail-fast`, and power-validation inputs.
- The branch-local [`run-sweep.yml`](../.github/workflows/run-sweep.yml) lacks the same-repository-head guard that `origin/main` adds before PR GPU setup. The [`trusted-external-sweep.yml` workflow](https://github.com/SemiAnalysisAI/InferenceX/blob/de493d8597035e6692833de6189b567887968460/.github/workflows/trusted-external-sweep.yml) exists on that `origin/main` snapshot but not on this branch. Do not infer external-fork secret or GPU behavior from the branch-local workflow.
- Agentic eval comments in the branch-local generator identify SWE-bench, while the audited `origin/main` generator identifies GSM8K. Inspect the target ref before describing the agentic dataset selected by `all-evals` or `evals-only`.

A `workflow_dispatch` request uses the workflow definition from its dispatch `--ref`. The separate `inputs.ref` controls what the jobs check out. Before using inputs beyond the common example below, inspect the deployed definition:

```bash
gh workflow view e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref main --yaml
```

If the target ref differs from this source snapshot, its workflow and script source wins. Do not guess that a branch-local input or fork policy is deployed.

## Local matrix generation

The generator loads every named master config and `configs/runners.yaml`, validates the input with Pydantic, generates entries, validates the output shape, applies eval policy, and prints one JSON array. A zero exit status proves generation and schema validation, not that containers, models, or GPU runners work.

### Generate one exact configuration

Use `test-config` for exact keys or quoted `*`/`?` patterns. `--conc` must be present in the config's concurrency range/list, and `--seq-lens` must match a scenario that exists.

```bash
MATRIX=/tmp/inferencex-matrix.json
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml \
  --config-keys dsr1-fp8-h200-sglang \
  --seq-lens 8k1k \
  --conc 4 \
  --no-evals > "$MATRIX"
python3 -m json.tool "$MATRIX" >/dev/null
```

For multiple keys, pass each key after `--config-keys`. Quote wildcard patterns so the shell does not expand them:

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml \
  --config-keys '*-b200-*' \
  --conc 4 \
  --no-evals > /tmp/inferencex-b200.json
```

### Generate a filtered sweep

`full-sweep` does not necessarily mean every configuration. Narrow it by model, precision, framework, runner, sequence length, topology, concurrency, TP/EP, or scenario type:

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --framework sglang \
  --runner-type h200 \
  --seq-lens 8k1k \
  --min-conc 4 \
  --max-conc 4 \
  --no-evals > /tmp/inferencex-filtered.json
```

If neither `--single-node` nor `--multi-node` is supplied, both are generated. `--step-size` must be greater than 1. If both concurrency bounds are supplied, `--min-conc` must not exceed `--max-conc`.

### Inspect, do not dump

Check the count and the execution-critical fields instead of loading the full JSON into context:

```bash
jq 'length' "$MATRIX"
jq -r '
  .[] |
  [.["model-prefix"], .framework, .precision, .runner,
   .isl, .osl, (.conc | tostring), (.tp // "-"), (.ep // "-"),
   (.prefill.hardware // "-"), (.decode.hardware // "-"),
   (.["run-eval"] // false), (.["eval-only"] // false)] |
  @tsv
' "$MATRIX"
```

Confirm the intended image, model, hardware/cluster label, single- versus multi-node topology, input/output lengths, concurrency, TP/EP, decoding mode, and eval flags. Empty output is not a successful preflight.

Every generated multi-node row must contain a strict positive integer
`node-count`. When node-slot scheduling is enabled, the reusable workflow
publishes that value as the `nodes:N` request label; missing or invalid demand
fails matrix validation instead of silently entering the one-node queue.
Direct priority-scheduled workflows that do not use the master-config generator
must publish their own exact demand (for example, CollectiveX uses each shard's
generated `nodes` value).

Eval switches are exact:

- Default: throughput entries plus the selected default fixed-sequence eval subset.
- `--no-evals`: throughput only. It cannot be combined with `--all-evals`.
- `--evals-only`: only the selected eval subset.
- `--all-evals`: expands to every generated fixed-sequence config and, by itself, is also eval-only.
- `--evals-only --all-evals`: every expanded eval and no throughput.

## YAML and changelog validation

### Parse touched YAML

Run a syntax parse on every touched YAML file. This catches malformed YAML but does not validate GitHub expressions or workflow dependency wiring:

```bash
uv run --no-project --with pyyaml --python 3.12 \
  python -c 'import sys, yaml; [yaml.safe_load(open(path, encoding="utf-8")) for path in sys.argv[1:]]' \
  configs/nvidia-master.yaml perf-changelog.yaml .github/workflows/e2e-tests.yml
```

For master configs, matrix generation is the strict validation: [`validation.py`](../utils/matrix_logic/validation.py) forbids unknown fields and validates both master entries and emitted matrix entries. Run the smallest exact `test-config` or filtered `full-sweep` that exercises the change.

### Validate the append-only changelog contract

`perf-changelog.yaml` is not ordinary YAML. Existing bytes are immutable except narrowly validated PR-link corrections. New entries must be appended, separated by exactly one empty line, end with one newline, contain no CR/tab/NUL bytes, pass the changelog schema, and generate a valid matrix.

The validator reads Git objects, not uncommitted working-tree bytes. Commit the candidate first, then compare it with the real base:

```bash
git fetch origin main
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref HEAD
```

Add `--all-evals` and/or `--evals-only` when those PR modifiers will be active. [`run-sweep.yml`](../.github/workflows/run-sweep.yml) passes the same flags. Never use a formatter to rewrite `perf-changelog.yaml`, and never treat `yaml.safe_load` alone as sufficient changelog validation.

## Manual end-to-end dispatch

Use [`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) for a bounded one-off run only after the identical generator command succeeds locally. Make the test name unique. In the common pattern, `--ref main` selects the deployed workflow definition while input `ref` selects the branch or SHA checked out by matrix generation and benchmark jobs.

```bash
REPO=SemiAnalysisAI/InferenceX
TEST_NAME="manual-dsr1-h200-$(date -u +%Y%m%dT%H%M%SZ)"
TARGET_REF='<branch-or-full-sha>'

gh workflow run e2e-tests.yml \
  --repo "$REPO" \
  --ref main \
  -f ref="$TARGET_REF" \
  -f test-name="$TEST_NAME" \
  -f generate-cli-command='full-sweep --config-files configs/nvidia-master.yaml --single-node --model-prefix dsr1 --framework sglang --runner-type h200 --seq-lens 8k1k --min-conc 4 --max-conc 4 --no-evals' \
  -f duration-override=''
```

Use only inputs shown by `gh workflow view ... --ref main --yaml`. Inputs can differ across dispatch refs, so do not pass target-ref-specific options by assumption.

Dispatch is asynchronous and may not immediately appear. Find the run by exact display title instead of assuming the newest run belongs to you:

```bash
gh run list \
  --repo "$REPO" \
  --workflow e2e-tests.yml \
  --event workflow_dispatch \
  --limit 30 \
  --json databaseId,displayTitle,headBranch,headSha,status,conclusion,url

RUN_ID=$(gh run list \
  --repo "$REPO" \
  --workflow e2e-tests.yml \
  --event workflow_dispatch \
  --limit 30 \
  --json databaseId,displayTitle \
  | jq -r --arg title "e2e Test - $TEST_NAME" \
      '[.[] | select(.displayTitle == $title)][0].databaseId // empty')
```

Do not continue if `RUN_ID` is empty. Run metadata describes the dispatch workflow ref, which may not equal input `ref`. Verify the unique title, generator command, and checkout ref in `get-jobs` before interpreting GPU results.

## Weekly overview snapshot

[`weekly-overview-snapshot.yml`](../.github/workflows/weekly-overview-snapshot.yml) runs a curated best-config sweep every Saturday 06:00 UTC so each `/overview` model×hardware cell gets a same-batch data point at least weekly (#2304, #2586). It calls `e2e-tests.yml` with a fixed `test-config` key list, then dispatches `ingest-results` to InferenceX-app with its own run ID, so results publish to the production database without a merge to `main`.

Operational facts:

- Scope lives in the workflow's default `config-keys` list; change it there, or override per run via `workflow_dispatch` input `config-keys`.
- `workflow_dispatch` input `skip-ingest: true` runs the sweep without touching the production database (smoke test).
- Priority: `schedule` events score `-10.0` in [`configs/ci-priority.yaml`](../configs/ci-priority.yaml) and the scorer floors at 0, so snapshot jobs queue at `0.000` — below every PR and main-push job.
- Partial failures still ingest: the app-side ingestion skips failed benchmark rows, so a half-fresh snapshot publishes rather than dropping the week.
- Recovery: a failed ingest for a completed sweep goes through the normal recovery path in [`recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml); a failed sweep can simply be re-dispatched manually with the same config keys.
- The overview page shows only the latest run per serving series, so a snapshot run replaces the displayed number for its configs (accepted trade-off; see #2586).

## PR primary and modifier labels

[`run-sweep.yml`](../.github/workflows/run-sweep.yml) rejects more than one primary label. Apply exactly one:

| Primary label | Matrix scope | Canary | Matrix fail-fast |
| --- | --- | --- | --- |
| `sweep-enabled` | Changelog matrix trimmed to the minimum concurrency per configuration | No | No |
| `full-sweep-fail-fast` | Full changelog matrix | Yes | Yes. Recommended full-sweep default |
| `full-sweep-enabled` | Full changelog matrix | Yes | No. Use when every matrix point must continue |
| `full-sweep-fail-fast-no-canary` | Full changelog matrix | No | Yes |
| `non-canary-full-sweep-enabled` | Full changelog matrix | No | No |

Optional modifiers do not replace a primary label:

| Modifier | Effect | Reusable after merge? |
| --- | --- | --- |
| `all-evals` | Expand eval selection to every generated fixed-sequence configuration. Alone, it is an eval-only shorthand | Yes, if the run otherwise satisfies full-sweep reuse rules |
| `evals-only` | Suppress throughput and run only the selected eval entries. Combine with `all-evals` for all evals only | No |
| `agentx-fast` | For AgentX throughput lanes, use one additional warmup request after mandatory primers and a 20-minute profile. Fixed-sequence and eval settings stay canonical | No |

Changing a recognized primary or modifier label shares the active sweep concurrency group and normally cancels/restarts the active run. `skip_queue`, patchwork, waiver, and checklist labels are gating/priority inputs, not primary sweep modes. A head commit containing `[skip-sweep]` skips PR benchmark setup only. Changelog/reuse checks still run, and pushes to `main` ignore it.

## Canary and fail-fast semantics

Canary and fail-fast solve different problems:

1. A canary is created only for `full-sweep-enabled` or `full-sweep-fail-fast` PRs. No-canary labels and `sweep-enabled` skip it.
2. Canary selection considers single-node fixed-sequence `1k1k` and `8k1k` entries, excludes entries whose primary purpose is eval, and chooses the lowest-concurrency candidate. That one entry is removed from the later single-node matrix.
3. If there is no eligible candidate, the canary is skipped. Otherwise all benchmark/eval matrices require the canary to succeed. A failed canary prevents their fan-out.
4. `full-sweep-fail-fast` and `full-sweep-fail-fast-no-canary` set `strategy.fail-fast: true` separately on each matrix job family. The first failing point cancels queued/in-progress siblings in that matrix family. It is not one global kill switch for every independent family.
5. Non-fail-fast labels leave matrix fail-fast false so other points continue and preserve broader diagnostic coverage.
6. A fail-fast run can conclude `cancelled` because sibling points were cancelled after a failure. Classify the first real failure before treating cancellation as an infrastructure event.

Manual `e2e-tests.yml` has no canary. Its `fail-fast` input defaults to false and is passed to every matrix job family. Always use the definition from the dispatch ref.

## Monitoring and reruns

### Monitor the selected run

```bash
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed
gh api "/repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN_ID" \
  --jq '[.id, .run_attempt, .event, .head_sha, .status, .conclusion, .html_url] | @tsv'
```

Watch the first canary or matrix failure, then classify it before rerunning:

- **Configuration/runtime failure:** reproducible launcher, validation, model, image, readiness, OOM, or result error. Fix the source and generate a new run.
- **Runner/infrastructure flake:** runner loss, transient network/storage/service failure, or unrelated cancellation. Preserve logs and rerun only after confirming the change itself is not responsible.
- **Policy/gate failure:** conflicting labels, invalid changelog, missing authorization, merge conflict, or ineligible artifacts. Correct the gate. GPU reruns will not fix it.
- **Superseded run:** a later commit or recognized label change cancelled it through workflow concurrency. Monitor the replacement run rather than reviving stale evidence.

### Rerun safely

Do not rerun an in-progress run blindly. A completed failed run can rerun only failed jobs and their dependents:

```bash
gh run rerun <RUN_ID> --failed --repo SemiAnalysisAI/InferenceX
```

For a completed `cancelled` fail-fast run, rerun the whole attempt so cancelled matrix points are recreated:

```bash
gh run rerun <RUN_ID> --repo SemiAnalysisAI/InferenceX
```

A rerun remains the same workflow run ID with a higher attempt. Artifact APIs can contain uploads from multiple attempts, so preserve `run_attempt` and inspect artifact timestamps. `run-stats` intentionally counts jobs from all attempts. If the source must change, do not rerun old code. Push the fix and monitor the new run. Removing and re-adding the primary sweep label forces a fresh labeled run. A later commit can invalidate reuse eligibility.

## Stage results

[`stage-results.yml`](../.github/workflows/stage-results.yml) is a maintainer-only publication path for PR results, not a substitute for merge or production ingestion.

A request is stageable only when all of the following hold:

- The commenter has `write`, `maintain`, or `admin` repository permission.
- The PR currently has one of the four full-sweep labels. `sweep-enabled` is not enough.
- The candidate is a completed `pull_request` run of `run-sweep.yml`, created while a full-sweep label was active, with conclusion `success`, `failure`, or `cancelled`.
- The candidate is associated with the PR under the workflow's current-head/historical-pin rules.
- Unexpired `changelog-metadata` and at least one of `results_bmk`, `eval_results_all`, or `bmk_agentic_*` exist. Failed/cancelled runs may therefore stage useful partial data, but empty or metadata-only runs cannot.

An authorized maintainer comments exactly one of:

```text
/stage-results
/stage-results <run-id>
```

Without an ID, the workflow selects the latest stageable completed run on the PR branch whose head SHA remains in the PR commit list. A pinned ID permits an explicitly associated historical run. The workflow acknowledges the run, dispatches the `stage-results` event to InferenceX-app, and [`stage-results-callback.yml`](../.github/workflows/stage-results-callback.yml) replaces the acknowledgement with a success chart or failure link.

Staging preserves earlier staged runs. Staging the same run ID again updates that run's staged data. Always keep the source run ID and downstream app workflow link. A staging success does not prove production reuse eligibility or post-merge ingestion.

## Artifact reuse and merge-with-reuse

Reuse prevents an approved full PR sweep from being rerun on `main`. It is not a way to bypass changelog validation.

### Eligibility and authorization

1. The PR must retain exactly one full-sweep primary label: `full-sweep-enabled`, `non-canary-full-sweep-enabled`, `full-sweep-fail-fast`, or `full-sweep-fail-fast-no-canary`.
2. `evals-only` and `agentx-fast` make the run ineligible. A default full sweep and a full sweep with `all-evals` remain eligible.
3. The source must be a completed PR `run-sweep.yml` run whose head SHA is still in the PR commit list and which has an unexpired `results_bmk`, `eval_results_all`, or `bmk_agentic_*` result artifact.
4. An `OWNER`, `MEMBER`, or `COLLABORATOR` authorizes reuse with `/reuse-sweep-run` or `/reuse-sweep-run <run_id>`. The newest authorized matching command determines whether source selection is automatic or pinned.
5. Unpinned selection requires the latest eligible source run to be successful. A pinned run is an explicit maintainer decision and may have conclusion `success`, `failure`, or `cancelled`. Downstream ingestion keeps only available/valid rows, so report it as partial rather than green.

The comment does not trigger a run. On a later PR `synchronize` event, the reuse gate skips another PR sweep after changelog validation. On `main`, authorization that maps ambiguously, points to an invalid run, or conflicts with labels fails closed. With no authorization, `main` performs the normal sweep.

### Supported merge path

Run from a clean checkout with authenticated `gh`, `git`, `jq`, and Python:

```bash
utils/merge_with_reuse.sh <pr-number>
```

[`merge_with_reuse.sh`](../utils/merge_with_reuse.sh) verifies an eligible successful source artifact, posts the authorization, merges `origin/main` into the PR branch, resolves only a `perf-changelog.yaml` conflict, canonicalizes appended `XXX` links, creates/pushes a synchronization commit when needed, waits for `check-changelog` and all PR checks, verifies the head did not move, and admin squash-merges. It refuses forks, dirty worktrees, multiple primary labels, incompatible modifiers, unexpected conflicts, missing artifacts, failed checks, or a moving PR head.

Do not manually reproduce only half of this sequence. In particular, posting the comment and squash-merging without the synchronization/check phase can leave the merge run unable to select the intended source.

On the `main` run, [`run-sweep.yml`](../.github/workflows/run-sweep.yml) sends two distinct IDs to InferenceX-app:

- `source-run-id`: the PR run containing benchmark/eval artifacts.
- `merge-run-id`: the `main` run containing merge-time `changelog-metadata`.

Public rows and links retain source-run provenance. Source artifact coverage is authoritative. Later matrix-policy changes do not manufacture missing points.

## Changelog conflict recovery

`perf-changelog.yaml` commonly conflicts because every PR appends at the tail. Never resolve it by accepting only `ours` or `theirs`, and never reformat or hand-merge historical blocks.

For an ordinary PR synchronization, first record the PR number, fetch `main`, and merge:

```bash
PR=<pr-number>
git fetch origin main
git merge origin/main
```

If and only if `perf-changelog.yaml` is the unresolved file, use the byte-preserving helper while the three conflict stages are still present:

```bash
python3 utils/prepare_perf_changelog_merge.py resolve-conflict \
  --changelog-file perf-changelog.yaml \
  --pr-number "$PR" \
  --repo SemiAnalysisAI/InferenceX
git add perf-changelog.yaml
git commit --no-edit
```

The helper reads merge-base/PR/main bytes from index stages 1/2/3, validates the PR-side delta, starts from the current `main` bytes, re-appends only unique PR contributions, canonicalizes the PR link, and validates the resulting raw-byte contract. If it refuses, stop. An unexpected historical edit, conflicting PR-link correction, missing contribution, or non-changelog conflict requires maintainer review. Do not guess a three-way resolution.

After committing, run the exact gate against `origin/main`:

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref HEAD
```

When reuse is authorized, prefer [`utils/merge_with_reuse.sh`](../utils/merge_with_reuse.sh). It performs this conflict preparation and the required synchronization/check sequence together.

## Artifact downloads and parsing

### List provenance before downloading

Use the REST endpoint so expired status, size, and timestamps are visible across all pages:

```bash
REPO=SemiAnalysisAI/InferenceX
RUN_ID=<run-id>
gh api --paginate "/repos/$REPO/actions/runs/$RUN_ID/artifacts?per_page=100" \
  --jq '.artifacts[] | [.name, .expired, .size_in_bytes, .created_at, .updated_at] | @tsv'
gh api "/repos/$REPO/actions/runs/$RUN_ID" \
  --jq '[.id, .run_attempt, .event, .head_sha, .status, .conclusion, .html_url] | @tsv'
```

Download only the named artifact needed for the question:

```bash
OUT="/tmp/inferencex-run-$RUN_ID"
mkdir -p "$OUT"
gh run download "$RUN_ID" --repo "$REPO" -n results_bmk -D "$OUT/results_bmk"
gh run download "$RUN_ID" --repo "$REPO" -n eval_results_all -D "$OUT/eval_results_all"
gh run download "$RUN_ID" --repo "$REPO" -n run-stats -D "$OUT/run-stats"
gh run download "$RUN_ID" --repo "$REPO" -n changelog-metadata -D "$OUT/changelog-metadata"
```

Do not assume every run has every artifact. Important contracts are:

| Artifact | Typical file | Produced from |
| --- | --- | --- |
| `results_bmk` | `agg_bmk.json` | `bmk_*` throughput/agentic JSON found by the result collector |
| `eval_results_all` | `agg_eval_all.json` | `eval_*` artifacts summarized by the eval collector |
| `run-stats` | `run_stats.json` | Hardware success counts across all attempts |
| `changelog-metadata` | `changelog_metadata.json` | Search-space metadata from sweep setup |
| `bmk_agentic_*` | Per-job JSON | Raw AgentX result upload used by agentic ingestion/staging |
| `server_logs_*`, `multinode_server_logs_*`, `gpu_metrics_*`, `agentic_*` | Logs, metrics, or diagnostic payloads | `always()`/diagnostic uploads. Names vary by template and mode |

### Parse bounded fields

Throughput aggregate fields come from [`utils/process_result.py`](../utils/process_result.py):

```bash
jq -r '
  .[] |
  [.hw, .infmax_model_prefix, .framework, .precision,
   "\(.isl)/\(.osl)", .tp, .conc,
   (if .tput_per_gpu == null then "" else ((.tput_per_gpu * 100 | round) / 100) end)] |
  @tsv
' "$OUT/results_bmk/agg_bmk.json"
```

Eval aggregate fields come from [`utils/collect_eval_results.py`](../utils/collect_eval_results.py):

```bash
jq -r '
  .[] |
  [.hw, .model_prefix, .framework, .precision,
   .tp, .conc, .task, .score_name,
   (if .score == null then "" else ((.score * 10000 | round) / 10000) end)] |
  @tsv
' "$OUT/eval_results_all/agg_eval_all.json"
```

Inspect run statistics without conflating skipped jobs with attempted jobs:

```bash
jq -r 'to_entries[] | [.key, .value.n_success, .value.total] | @tsv' \
  "$OUT/run-stats/run_stats.json"
```

For an unfamiliar or raw agentic artifact, start with `jq 'type, keys'` and read its producing script before selecting fields. Never load or paste a multi-megabyte artifact merely to answer a narrow question. Report the run ID, attempt, exact artifact name, and filters alongside extracted values.

## Post-merge expectations

A merge is not complete operationally until the `main` publication path and downstream ingest are verified.

1. A push to `main` triggers [`run-sweep.yml`](../.github/workflows/run-sweep.yml) only when `perf-changelog.yaml` changed. Confirm the merge commit produced that run. Do not assume an unrelated merge invokes it.
2. Without valid reuse authorization, the `main` run processes the changelog delta and runs its normal matrix. PR canary logic does not run on `push`, and PR label-driven fail-fast is not available on that event.
3. With reuse, benchmark jobs are skipped and the run uses source artifacts plus merge-run changelog metadata. Confirm the setup outputs selected the intended source run rather than inferring reuse from skipped jobs alone.
4. `upload-changelog-metadata` must produce `changelog-metadata`. For search spaces without agentic entries, `trigger-ingest` dispatches `ingest-results`. Agentic search spaces follow the separate `trigger-agentic-ingest` conditions and dispatch `ingest-agentic-results` with `database-target: production`.
5. A green dispatch step proves only that GitHub accepted the InferenceX-app repository dispatch. Follow the downstream InferenceX-app run and verify the expected rows/links and source-run provenance. Its ingest implementation is outside this checkout.
6. Check the final `main` run conclusion, every rerun attempt, aggregate artifacts, metadata, and publication result. The PR author remains responsible for all post-merge Actions jobs passing, including flakes that require a justified rerun.
7. Do not rerun GPU benchmarks merely because downstream ingestion failed while valid artifacts exist. For a failed reused **agentic** ingest, an authorized maintainer can use [`recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml) with the original source and merge IDs. That workflow dispatches only `ingest-agentic-results`, so it is not a generic fixed-sequence recovery tool:

   ```bash
   gh workflow run recover-reused-ingest.yml \
     --repo SemiAnalysisAI/InferenceX \
     --ref main \
     -f source-run-id='<source-pr-run-id>' \
     -f merge-run-id='<main-merge-run-id>'
   ```

Stop and escalate when the source run, merge run, artifact coverage, changelog metadata, or downstream event is ambiguous. Never substitute a convenient run ID or claim publication from an Actions dispatch alone.
