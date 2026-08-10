<div align="center">

**English** | [中文](./troubleshooting_zh.md)

</div>

# Troubleshooting

Classify a failure by the first layer that did not establish its contract. Preserve evidence before rerunning, repair only the owning layer, and stop when the evidence is insufficient for a scoped and reversible action. Downstream jobs may run under `always()` or aggregate an empty set, so their green status does not prove that an upstream benchmark or eval succeeded.

## Index

- [Sources of truth](#sources-of-truth)
- [Evidence before remediation](#evidence-before-remediation)
- [Failure-layer matrix](#failure-layer-matrix)
- [Changelog and matrix](#changelog-and-matrix)
- [Runner and AMD root files](#runner-and-amd-root-files)
- [Server](#server)
- [Eval and collection](#eval-and-collection)
- [Ingest](#ingest)
- [Known KLAUD cases](#known-klaud-cases)
- [Verification and stop conditions](#verification-and-stop-conditions)

## Sources of truth

- [`KLAUD_DEBUG.md`](../KLAUD_DEBUG.md) records recurring Klaud-Cold/image-bump incidents and their observed signatures. It is incident knowledge, not a substitute for current workflow or review policy.
- [`run-sweep.yml`](../.github/workflows/run-sweep.yml), [`benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml), and [`benchmarks/benchmark_lib.sh`](../benchmarks/benchmark_lib.sh) define orchestration, artifact upload, server readiness, benchmark, and eval behavior.
- [`validate_perf_changelog.py`](../utils/validate_perf_changelog.py), [`generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py), and [`validation.py`](../utils/matrix_logic/validation.py) own changelog, matrix, and schema failures.
- [`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md) and [`runners/`](../runners/) own provisioning and launcher routing. [`CONTRIBUTING.md`](../CONTRIBUTING.md#amd-cluster-never-leave-root-owned-files-in-runner-workspaces) owns AMD workspace safety.
- [`utils/evals/EVALS.md`](../utils/evals/EVALS.md), [`validate_scores.py`](../utils/evals/validate_scores.py), and [`collect_eval_results.py`](../utils/collect_eval_results.py) own eval execution, validation, and collection.
- [The failed-ingest recovery command](../.claude/commands/recover-failed-ingest.md) is the guarded recovery procedure. The downstream source is InferenceX-app's [`ingest-results.yml`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/.github/workflows/ingest-results.yml), [`prepare-ci-artifacts.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/prepare-ci-artifacts.ts), [`ingest-ci-run.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/ingest-ci-run.ts), and [`benchmark-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/etl/benchmark-mapper.ts).

## Evidence before remediation

Before rerunning or editing, record:

1. Workflow URL, run ID, attempt, event, branch, head SHA, failing job/check, runner name, and timestamps.
2. Exact config key and generated model, image, runner, framework, scenario, topology, and concurrency.
3. The first error plus enough preceding log context to identify which process emitted it. Save server and Slurm logs separately from workflow orchestration logs.
4. Artifact names, expiration state, and the small structured fields relevant to the failure (`meta_env.json`, result metadata, run stats, score summary). Do not dump large aggregate JSON.
5. What already succeeded. A matrix that never emitted a job, a job that never acquired a runner, and a server that never became healthy are different failures even if all end with “no result.”

Do not rerun first: reruns can replace logs, change runner/node placement, or make a deterministic failure look transient.

## Failure-layer matrix

| First failing layer | Typical symptom | Safe evidence | First safe action |
| --- | --- | --- | --- |
| Changelog/setup | `Deletions are not allowed`, malformed YAML, trigger scope wrong | Setup log, exact base/head changelog bytes and delta | Restore current main bytes and append only this PR's intended entry, then rerun the validator |
| Matrix/orchestration | Generator exception, empty/wrong matrix, jobs skipped, label conflict | Exact generated CLI, generator/Pydantic error, PR labels, and head commit | Reproduce locally and fix master config/generator/labels. Do not bypass setup |
| Runner/allocation | Long queue, checkout `EACCES`, Slurm never starts, Docker socket or disk error | GitHub runner name, launcher output, `squeue`/`sacct`/node state, and workspace ownership | Repair or escalate runner infrastructure. Do not tune the recipe for an unhealthy node |
| Server | Process exits, log never appears, health never passes, OOM/kernel/port error | Server log, PID exit, `/health`, image digest/tag, and GPU/Slurm logs | Match the exact signature. Change one evidence-backed runtime/image setting or roll back |
| Eval | `eval /` fails, batch incomplete, score below threshold, result missing | `meta_env.json`, each `results*.json`, validator output, image and task | Fix eval/server/task cause and rerun the exact eval config |
| Collection | “No eval results found,” empty aggregate, collector green after skipped evals | Underlying `eval /` conclusions, artifact tree and metadata, and collector output | Restore or fix the upstream artifact contract. Do not diagnose serving from collector output alone |
| Ingest | Dashboard rows absent/wrong, with target `trigger-ingest` green but no valid data | Target and source run metadata, unexpired artifacts, app workflow/ETL logs, and changelog scope | Use the guarded recovery procedure. Never rerun the failed target workflow |

## Changelog and matrix

### Changelog

A setup-stage deletion error usually means a stale branch or whitespace-changing merge made historical bytes appear deleted. Follow the canonical repair in [`KLAUD_DEBUG.md` §1.1](../KLAUD_DEBUG.md#11-perf-changelogyaml-deletion-not-allowed): take the current main version verbatim, then append only this PR's entry at the tail. Validate against the real base and head with [`validate_perf_changelog.py`](../utils/validate_perf_changelog.py).

Do not 3-way merge or normalize `perf-changelog.yaml`. Stop if the intended config keys, eval flags, scenario scope, or historical delta is ambiguous. Changelog additions and permitted `pr-link` correction behavior are enforced by the validator, not by a visually valid YAML parse.

### Matrix and orchestration

Re-run the exact workflow `generate-cli-command` locally. Start with exact-key `test-config`, then the same filtered `full-sweep` command. Inspect emitted image, runner, scenario, topology, concurrency, eval flags, and additional settings. The commands and inspection list are in [`testing.md`](./testing.md#exact-config-then-filtered-family).

If no GPU jobs start, inspect setup before investigating a runner. [`run-sweep.yml`](../.github/workflows/run-sweep.yml) rejects conflicting primary labels, does not run from eval modifiers alone, honors `[skip-sweep]` only for PR heads, and waits for merge conflicts to be resolved. Fix the input or label state. Do not manually dispatch a different matrix and call it equivalent evidence.

Stop when the local matrix does not exactly match the intended PR scope. A successful generator with the wrong config is not a recovery.

## Runner and AMD root files

Classify queue/allocation failures before reading server logs:

- Map the generated runner label through [`configs/runners.yaml`](../configs/runners.yaml) and the launch script selected from [`runners/`](../runners/). Confirm the intended fleet, node, and Slurm job.
- Use the workflow job's runner name and allocation timestamps to correlate `squeue`/`sacct` and node state. A drained node, broken Pyxis, Docker socket permission, disk exhaustion, or runner checkout failure is infrastructure evidence.
- Rerun only after the runner is healthy. Do not change model parallelism, memory flags, or image merely to escape a bad node.
- Escalate access, drained-node, socket, storage, and permanent Slurm configuration changes to cluster operators.

[`KLAUD_DEBUG.md` §5](../KLAUD_DEBUG.md#5-cluster-infrastructure-amd-mi355x--mi300x--mi325x) lists known AMD node, Docker socket, disk, and port incidents. Treat named-node state as historical until current node evidence confirms it.

### AMD root-owned workspace files

The signature is checkout cleanup failing with `EACCES` on `benchmark_logs/logs/slurm_job-*`. Root-running Slurm containers may leave root-owned directories when cancellation skips teardown, blocking every later job on that runner. The prevention contract in [`CONTRIBUTING.md`](../CONTRIBUTING.md#amd-cluster-never-leave-root-owned-files-in-runner-workspaces) is:

1. Never write as root inside the runner `_work` workspace. Use scratch outside `_work`.
2. If unavoidable, install cancellation-safe cleanup that changes ownership or removes every root-owned output.
3. Cancel a test run and verify the workspace remains clean.

For recovery, follow [`.claude/commands/clean-amd-mi355-runner-root-files.md`](../.claude/commands/clean-amd-mi355-runner-root-files.md): use the documented sudo hop, perform a read-only scan scoped to runner `_work`, review every path, stop if any match is outside `_work`, delete only confirmed matches with the same scope, then rescan for zero entries before rerunning. Never run an unscoped `rm -rf` on `/it-share`.

## Server

[`wait_for_server_ready`](../benchmarks/benchmark_lib.sh) distinguishes “server died before log,” “server died before healthy,” and a live process whose `/health` endpoint has not passed. Preserve the server log and PID status. The workflow's final timeout alone is not a diagnosis.

Use the earliest specific signature:

- **Image pull/tag failure:** verify the exact registry tag or digest exists before touching runtime flags. [`KLAUD_DEBUG.md` §6](../KLAUD_DEBUG.md#6-docker-image-tag-gotchas) warns against deriving release tags from dated nightlies.
- **Weight/KV/CUDA-graph OOM:** capture free memory, configured utilization, per-rank concurrency, graph limits, and where startup failed. Apply only the setting supported by the matching known case. Confirm startup and workload afterward.
- **Kernel/architecture assertion or illegal address:** preserve the complete stack and GPU architecture. Prefer a fixed/pinned upstream image or supported backend over an unreviewed local engine patch.
- **Address in use:** identify the owning process and cluster owner before terminating it. Do not kill an unverified PID or unrelated service.
- **Healthy server dies during benchmark:** [`run_benchmark_serving`](../benchmarks/benchmark_lib.sh) monitors the server PID. Preserve both client and server logs and classify the server's first error, not the client's downstream connection failure.

Stop if the proposed workaround changes model semantics, reduces model FLOPs, patches the serving stack, or lacks an exact-source guard. The current [PR checklist](./PR_REVIEW_CHECKLIST.md) prohibits inference-engine patches unless the documented waiver path is satisfied.

## Eval and collection

### Eval

Read the individual `eval /` job, not only `collect-evals`. For each expected concurrency, inspect `meta_env.json`, completion/failure metadata, and its `results*.json`. [`validate_scores.py`](../utils/evals/validate_scores.py) rejects missing result files, below-threshold scores, and runs with zero checked metrics. With expected concurrency metadata it also rejects invalid manifests, duplicate/unexpected/missing concurrency, and failed batches. Because the workflow invokes it without `--expected-concs`, inspect unbatched `meta_env.json` separately. Missing or invalid metadata remains a failure even when the score validator exits successfully.

Confirm the task and image match the generated config. If the server failed during eval, return to the server layer. If the task, threshold, or manifest is wrong, fix that source and rerun the exact eval. Do not accept a green job with skipped, empty, or mismatched results.

### Collection

[`collect_eval_results.py`](../utils/collect_eval_results.py) discovers both flat and nested artifact layouts, chooses the latest legacy result or latest result per batched concurrency, and filters batched results to completed concurrencies. Its `> No eval results found to summarize.` message means no rows were collected. It is not proof that evals passed.

When collection is empty:

1. Check the underlying `eval /` jobs were executed and succeeded, not skipped.
2. Confirm each downloaded artifact has `meta_env.json` and recognized lm-eval JSON with `lm_eval_version`.
3. For batched evals, compare requested, completed, failed, and filename concurrency suffixes.
4. Reproduce collection on the downloaded artifact tree.

Repair the producer or artifact layout. Do not add fake metadata, mark failed concurrencies complete, or modify the collector to ingest incomplete output.

## Ingest

A successful `trigger-ingest` does not prove valid benchmark rows exist: [`run-sweep.yml`](../.github/workflows/run-sweep.yml) can trigger downstream handling after collection, and a cancelled/no-result target may still reach that job. Verify the target run's event, workflow, branch, head SHA, changelog delta, result artifacts, and downstream InferenceX-app logs.

The app workflow prepares, migrates, ingests, applies overrides, and verifies data in [`ingest-results.yml`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/.github/workflows/ingest-results.yml). [`prepare-ci-artifacts.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/prepare-ci-artifacts.ts) selects and downloads source/merge artifacts and writes reuse metadata. [`ingest-ci-run.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/ingest-ci-run.ts) owns database ingestion. The failed-row guard in [`benchmark-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/etl/benchmark-mapper.ts) skips a row only when numeric `num_requests_successful` is zero and `num_requests_total` is numeric.

Use the guarded [failed-ingest recovery procedure](../.claude/commands/recover-failed-ingest.md), not a rerun of the failed target. It requires a completed pull-request `run-sweep.yml` source, unexpired result artifacts, source membership in the original PR, unchanged execution semantics, unambiguous changelog scope, and preserved recovery ancestry.

Stop recovery if the source run or artifacts are ineligible, source ancestry cannot be proved, config/recipe/image semantics changed after the source SHA, or the intended changelog scope is ambiguous. Never bypass pending/failing checks, rewrite the recovery branch after attaching source ancestry, or trust the target's green trigger without inspecting data.

## Known KLAUD cases

Use [`KLAUD_DEBUG.md`](../KLAUD_DEBUG.md) to recognize an exact signature, then verify it against the current image, recipe, hardware, and policy before applying its remediation.

| Signature | Known case and safe boundary |
| --- | --- |
| Setup says changelog history was deleted | [§1.1](../KLAUD_DEBUG.md#11-perf-changelogyaml-deletion-not-allowed): restore main bytes and append only the PR entry. Never 3-way merge history |
| vLLM weight load/KV allocation OOM | [§2](../KLAUD_DEBUG.md#2-vllm-v021x--v020x-gpu-oom-at-model-load): confirm the memory-profiler signature before reducing utilization or using the recorded profiler setting |
| DEP decoder fails during large CUDA-graph capture | [§2.1](../KLAUD_DEBUG.md#21-dep-cuda-graph-capture-oom-on-gb300): size sequence/graph limits from per-DP-rank load, not global concurrency |
| DSV4 works on custom digest but OOMs on generic SGLang | [§3](../KLAUD_DEBUG.md#3-custom-dsv4-image--generic-v0512-ooms): generic release was not a drop-in. Retain or return to a proven compatible image |
| B300 DeepGemm illegal address, EAGLE trtllm GEMM failure, or flash-attn architecture assertion | [§4](../KLAUD_DEBUG.md#4-upstream-sglang-v0512-b300-regressions): distinguish the three stacks. Use a supported backend/cap or fixed/pinned upstream image |
| AMD drained/Pyxis, Docker socket, disk-full, or occupied port | [§5](../KLAUD_DEBUG.md#5-cluster-infrastructure-amd-mi355x--mi300x--mi325x): confirm current node state and escalate infrastructure. There is no recipe-level fix for unhealthy infrastructure |
| Guessed Docker tag returns 404 | [§6](../KLAUD_DEBUG.md#6-docker-image-tag-gotchas): verify the exact tag at the registry. Do not infer naming patterns |
| `gh run rerun --failed` is refused | [§7](../KLAUD_DEBUG.md#7-ci-rerun-mechanics): inspect run status/conclusion. Only completed failures support failed-only rerun, while cancelled runs require a full rerun |
| MiniMax M3 B300 MSA says `q2k_indices` is non-contiguous | [§11](../KLAUD_DEBUG.md#11-minimax-m3-b300-msa-top-k-slice-is-non-contiguous): recognize TP1/data-parallel-attention exposure and prefer an upstream-fixed image. Shipping the recorded engine patch requires current checklist/waiver compliance |

Historical KLAUD label or merge advice does not override the current [sweep-label policy](../.github/AGENT_OPERATIONS.md#sweep-labels-and-reuse), [`CONTRIBUTING.md`](../CONTRIBUTING.md), or the [PR checklist](./PR_REVIEW_CHECKLIST.md).

## Verification and stop conditions

A remediation is complete only when the same failing path is rerun and the original signature is absent, the expected job actually executes, and the expected structured artifact/data appears. **Ingest is the exception:** never rerun the failed target workflow or job. Verify the new recovery push and its downstream app ingest instead. Record the new run/attempt and compare it with the preserved failure evidence.

Stop and escalate instead of acting when:

- the first failing layer cannot be identified from preserved logs and metadata.
- a deletion, cleanup, PID kill, node change, or branch rewrite is not narrowly scoped and independently verifiable.
- a workaround changes model semantics, architecture FLOPs, image provenance, or patches the serving stack without an approved waiver.
- a cluster fix requires privileges or ownership you do not have.
- eval completion, scores, image identity, or artifact provenance is missing.
- collection has no upstream artifacts.
- ingest source eligibility, execution equivalence, changelog scope, or ancestry is uncertain.

Do not call repeated reruns a fix. If the same input succeeds only after runner/node placement changes, preserve both runs and classify the original as infrastructure or a suspected flake with evidence.
