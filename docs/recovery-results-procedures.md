# Results, Ingest, and Recovery Procedures

<div align="center">

**English** | [中文](./recovery-results-procedures_zh.md)

</div>

Use this page after a throughput or eval job starts producing output, or when a sweep, runner, cluster, handoff, or database ingest fails. It separates evidence collection from repair so an infrastructure problem is not mistaken for a benchmark regression and an ingest problem does not trigger another expensive GPU sweep.

## Safety gates

1. **Identify the exact run, attempt, job, event, branch, head SHA, and runner before changing anything.** A green `trigger-ingest` job is not proof that valid benchmark rows reached the database.
2. **Inspect first, then mutate.** Reading GitHub logs, `sinfo`, `squeue`, sysctls, ownership, and artifacts is safe. Deleting shared files, changing a sysctl, draining nodes, restarting services, or editing cluster state requires explicit operator approval.
3. **Never rerun a failed push-to-`main` target merely to repair official ingest.** Use the validated artifact-reuse recovery path. It avoids GPU work and preserves source-run provenance.
4. **Rerun only diagnosed flakes.** A retry does not repair a bad image, recipe, model, launcher, config, missing artifact, expired artifact, or changed execution semantics.
5. **Treat `source-run-id` and `merge-run-id` as a pair.** The source run owns benchmark/eval artifacts. The merge run owns publication/changelog metadata. Verify both in downstream logs before trusting an ingest.
6. **Do not choose an InferenceX-app ingest by recency alone.** A stale or bogus ingest can be newer than the recovery ingest.
7. **Preserve evidence before cleanup.** Record run/job URLs, run attempt, runner name, source and merge IDs, artifact names/counts, the first causal error, and any node or Slurm job IDs.

Sources: [sweep debugging guardrails](../.agents/skills/debug-runs/SKILL.md#L78-L152), [failed-ingest safety rules](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md#L22-L54).

## Result pipeline: know what should exist

### Throughput results

For a normal single-node throughput job:

1. The launcher must leave `${RESULT_FILENAME}.json` in the workspace. The workflow waits briefly and fails if it never appears.
2. `utils/process_result.py` reads the raw JSON plus topology/runtime environment variables, normalizes metadata and per-GPU throughput, converts millisecond fields to seconds, derives interactivity, and writes `agg_${RESULT_FILENAME}.json`.
3. The job uploads that aggregate as artifact `bmk_${RESULT_FILENAME}`.
4. `collect-results.yml` downloads `bmk_*`, runs `python3 utils/collect_results.py results/ bmk`, and uploads `results_bmk`, whose payload is `agg_bmk.json`.

For multi-node throughput, every `${RESULT_FILENAME}_*.json` is processed separately. The workflow derives total, prefill, and decode GPU counts from each filename and invokes:

```bash
RESULT_FILENAME=${result_file%.json} \
IS_MULTINODE=true \
PREFILL_GPUS="$prefill_gpus" \
DECODE_GPUS="$decode_gpus" \
python3 utils/process_result.py
```

The uploaded `bmk_${RESULT_FILENAME}` artifact contains `agg_${RESULT_FILENAME}_*.json`. Missing source files indicate a benchmark/launcher failure. Missing `agg_` files indicate a processing failure. Missing `results_bmk` indicates a collection failure. Do not classify any of those as a database failure.

Sources: [single-node process/upload](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-tmpl.yml#L289-L323), [multi-node process/upload](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-multinode-tmpl.yml#L345-L387), [`process_result.py` contract](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/utils/process_result.py#L43-L75), [throughput collector](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/collect-results.yml#L25-L38).

### Eval results

Eval jobs upload per-config artifacts named `eval_${EXP_NAME}_${RESULT_FILENAME}`. They contain the files that exist for that evaluator, including `meta_env.json`, `results*.json`, `sample*.jsonl`, and, for supported agentic evaluators, predictions, reports, or trajectories. The workflow behavior is deliberate:

- an eval-only job errors when no eval files are found.
- eval files upload under `always()`, preserving partial evidence from a failed job.
- `utils/evals/validate_scores.py` validates eval-only score coverage after upload.
- `collect-evals.yml` downloads `eval_*`, runs `collect_eval_results.py`, prints a summary, and uploads `eval_results_all/agg_eval_all.json`.

The app can ingest both the aggregate rows and the per-config eval directories. They converge on the same natural key, while sample files attach detail to the resolved eval row. Therefore, an aggregate alone proves collection, not sample completeness. Verify the per-config artifact when sample-level output matters.

Sources: [single-node eval upload/validation](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-tmpl.yml#L362-L385), [multi-node eval upload/validation](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/benchmark-multinode-tmpl.yml#L416-L434), [eval collector](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/collect-evals.yml#L24-L46), [app eval artifact handling](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/ingest-ci-run.ts#L738-L792).

### Inspect result artifacts without changing the run

```bash
RUN_ID=<InferenceX-run-id>

# Run identity and jobs
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --json event,headBranch,headSha,status,conclusion,attempt,jobs

# Exact unexpired artifact inventory
gh api "repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN_ID/artifacts?per_page=100" \
  --paginate --jq '.artifacts[] | select(.expired == false) | [.name,.created_at,.size_in_bytes] | @tsv'

# Download only the two collection products when they exist
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --name results_bmk --dir /tmp/infx-results-$RUN_ID
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --name eval_results_all --dir /tmp/infx-evals-$RUN_ID

jq 'length' /tmp/infx-results-$RUN_ID/agg_bmk.json
jq 'length' /tmp/infx-evals-$RUN_ID/agg_eval_all.json
```

**Gate:** zero rows, absent aggregate artifacts, or an expired source artifact requires classification before any retry. Artifact existence does not prove valid rows: the app deliberately skips a benchmark row when both request counts exist and `num_requests_successful == 0`.

Sources: [failed-row guard](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/benchmark-mapper.ts#L203-L216), [artifact inventory checks used by recovery](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md#L149-L174).

## InferenceX-app handoff and ingest verification

On a push to `main`, `run-sweep.yml` dispatches `ingest-results` only after setup and the applicable collection paths have resolved. Its payload is:

```json
{
  "event_type": "ingest-results",
  "client_payload": {
    "source-run-id": "<artifact-owning-run>",
    "merge-run-id": "<publication-run>"
  }
}
```

A normal run uses the current run for both IDs. Reuse points `source-run-id` to the validated PR sweep and keeps `merge-run-id` on the new push-to-main recovery run. InferenceX-app selects the newest unexpired upload for each exact artifact name from the source run. For reuse, it replaces source changelog metadata with `changelog-metadata` from the merge run. Preparation fails if the source has no unexpired artifacts or the merge run has no changelog artifact.

The app workflow then runs, in order: artifact preparation, migrations, database ingest, run overrides, database verification, cache invalidation, and unmapped-entity inspection. Database writes are idempotent (`ON CONFLICT DO UPDATE` or `DO NOTHING`), so a correctly targeted ingest can safely resume after a partial failure.

Sources: [dispatch payload and gates](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.github/workflows/run-sweep.yml#L978-L1021), [artifact selection](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/lib/ci-artifact-preparation.ts#L10-L48), [app ingest stages](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/.github/workflows/ingest-results.yml#L50-L124), [idempotency rationale](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/docs/data-pipeline.md#L26-L34).

### Verify the exact downstream ingest

```bash
gh run list --repo SemiAnalysisAI/InferenceX-app \
  --workflow "Ingest Benchmark Results" \
  --event repository_dispatch --limit 10 \
  --json databaseId,status,conclusion,createdAt

INGEST_RUN_ID=<candidate-run-id>
SOURCE_RUN_ID=<expected-artifact-run-id>
MERGE_RUN_ID=<expected-publication-run-id>

gh run view "$INGEST_RUN_ID" --repo SemiAnalysisAI/InferenceX-app --log \
  | grep -m1 "Source run: $SOURCE_RUN_ID"
gh run view "$INGEST_RUN_ID" --repo SemiAnalysisAI/InferenceX-app --log \
  | grep -m1 "Merge run:  $MERGE_RUN_ID"

gh run watch "$INGEST_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX-app --exit-status
```

Require all of the following, not merely a green conclusion:

- both log matches identify the intended source and merge runs and attempts.
- artifact preparation reports the expected selected artifact names/counts.
- reused ingest says it took changelog metadata from the merge run.
- database ingest reports sensible benchmark/eval/changelog counts and explains any skipped rows.
- run overrides and database verification succeed.
- cache invalidation is attempted.
- unmapped models, hardware, or precisions are absent or explicitly triaged.

The preparation script emits the authoritative `Source run:` and `Merge run:` lines. Reused ingests do not wait five minutes because their source run is already complete.

Sources: [authoritative preparation logs](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/prepare-ci-artifacts.ts#L99-L152), [official verification sequence](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md#L391-L429).

## Failed ingest recovery

First split the incident into two cases.

### Case A: the correct app ingest exists and failed partway

If its preparation logs match the intended source and merge pair and artifacts are still unexpired, rerunning that **InferenceX-app ingest run** is data-safe because the ETL is idempotent:

```bash
gh run rerun "$INGEST_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX-app --failed
```

If GitHub refuses a failed-only rerun, a full rerun of the same app run is still data-idempotent:

```bash
gh run rerun "$INGEST_RUN_ID" \
  --repo SemiAnalysisAI/InferenceX-app
```

**Do not rerun yet** when preparation shows the wrong IDs, artifacts are expired/missing, changelog metadata is absent, or execution semantics changed after the candidate source SHA. A retry would faithfully repeat the wrong operation.

### Case B: official ingest is missing, skipped, bogus, or points at the wrong source

Use the repository's recovery-PR procedure. Do not rerun the failed target, do not add a one-off workflow, and do not manually copy rows into the database.

#### 1. Inspect and prove the target

Run from a clean InferenceX checkout with authenticated `gh`, `git`, `jq`, and Python dependencies available:

```bash
python3 utils/recover_failed_ingest.py inspect-target \
  "$FAILED_RUN_OR_JOB_URL" \
  --output /tmp/infx-recovery-target.json

TARGET_RUN_ID=$(jq -r .run_id /tmp/infx-recovery-target.json)
TARGET_JOB_ID=$(jq -r .job_id /tmp/infx-recovery-target.json)
ORIGINAL_PR=$(jq -r .pr_number /tmp/infx-recovery-target.json)
ORIGINAL_MERGE_SHA=$(jq -r .merge_sha /tmp/infx-recovery-target.json)

gh run view "$TARGET_RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --job "$TARGET_JOB_ID" --log > "/tmp/infx-target-$TARGET_RUN_ID.log"

python3 utils/recover_failed_ingest.py audit-changelog \
  --ref "$ORIGINAL_MERGE_SHA"
```

**Target gate:** require a completed push event for `.github/workflows/run-sweep.yml` on `main`. A target can conclude `success` or `cancelled` and still have a bogus ingest: if reuse was forgotten, GPU jobs may be cancelled while `trigger-ingest` still succeeds against the target's own run ID. Leave that row alone. The correctly recovered publication uses a new run ID.

#### 2. Validate the source, ancestry, and artifacts

```bash
SOURCE_RUN_ID=<candidate-pr-sweep-run-id>
SOURCE_JSON=$(gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID")
SOURCE_HEAD_SHA=$(jq -r .head_sha <<<"$SOURCE_JSON")
SOURCE_RUN_ATTEMPT=$(jq -r .run_attempt <<<"$SOURCE_JSON")

jq -e '
  .event == "pull_request" and
  .status == "completed" and
  .path == ".github/workflows/run-sweep.yml"
' <<<"$SOURCE_JSON" >/dev/null

gh api "repos/SemiAnalysisAI/InferenceX/pulls/$ORIGINAL_PR/commits" \
  --paginate --jq '.[].sha' | grep -Fx "$SOURCE_HEAD_SHA"

gh api \
  "repos/SemiAnalysisAI/InferenceX/actions/runs/$SOURCE_RUN_ID/artifacts?per_page=100" \
  --paginate --jq '.artifacts[] | select(.expired == false) | .name' \
  | grep -Eq '^(results_bmk|eval_results_all|bmk_agentic_)'
```

Use an unpinned source only when it succeeded. A specifically pinned failed run may recover only its completed points because failed benchmark rows are skipped. Compare the source SHA with the original PR's final head and **stop** if later edits changed the recovered image, model, recipe, runner, launcher, benchmark arguments, or config values.

#### 3. Authorize reuse before the changelog change

Create an empty recovery PR from current `main`, give it exactly one full-sweep label, then pin the source before pushing the recovery changelog commit:

```bash
gh pr edit "$RECOVERY_PR" --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast
gh pr comment "$RECOVERY_PR" --repo SemiAnalysisAI/InferenceX \
  --body "/reuse-sweep-run $SOURCE_RUN_ID"
```

Append recovery entries to the end of `perf-changelog.yaml`. Never modify historical bytes. Preserve the original `config-keys`, `description`, `evals-only`, and `scenario-type`, but use the recovery PR URL. Validate both the changelog and generated scope:

```bash
python3 utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT"
python3 utils/process_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref "$RECOVERY_COMMIT" \
  > /tmp/recovery-full-config.json
```

#### 4. Preserve source ancestry without changing the tree

The final recovery branch head must use the recovery commit as first parent and the source SHA as second parent. The file tree must remain unchanged:

```bash
TARGET_PARENT=$(git rev-parse HEAD)
TARGET_TREE=$(git rev-parse "${TARGET_PARENT}^{tree}")
ATTACH_SHA=$(
  printf 'chore: attach reusable sweep run %s\n' "$SOURCE_RUN_ID" |
    git commit-tree "$TARGET_TREE" \
      -p "$TARGET_PARENT" \
      -p "$SOURCE_HEAD_SHA"
)
git reset --hard "$ATTACH_SHA"

test "$(git rev-parse HEAD^1)" = "$TARGET_PARENT"
test "$(git rev-parse HEAD^2)" = "$SOURCE_HEAD_SHA"
test "$(git rev-parse HEAD^{tree})" = "$(git rev-parse HEAD^1^{tree})"
test "$(git diff --name-only origin/main...HEAD)" = "perf-changelog.yaml"
git diff --check origin/main...HEAD
```

After pushing, never rebase, locally squash, amend, or force-push this carrier commit. Require the source SHA in the PR commit list, only `perf-changelog.yaml` in Files, passing `check-changelog` and `reuse-sweep-gate`, and skipped PR GPU jobs. Merge only with explicit authorization and never bypass failing or pending checks. Then verify the new push run and downstream app run with the handoff procedure above.

Canonical source: [complete failed-ingest recovery command](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/recover-failed-ingest.md).

## H100 offload cleanup stalls

An H100 AgentX benchmark can finish and write all result files while the GitHub job remains in `Launch job script`. Check the launcher process tree on the login node. A long-running child matching either of these commands means teardown, rather than the benchmark, is stalled:

```text
srun --jobid=<id> bash -c find /dev/shm ... vllm_offload_*.mmap ... -delete
srun --jobid=<id> bash -c rm -rf -- /mnt/numa0/.../inferencex-kv-<id>
```

Preserve and verify the result files before intervening. The H100 launcher bounds each pre-run and post-run offload cleanup step with `H100_OFFLOAD_CLEANUP_TIMEOUT_S` (120 seconds by default), followed by a 15-second forced-termination window. A post-run timeout is best-effort: the launcher cancels the allocation and preserves the benchmark exit status so artifact uploads can proceed. A pre-run timeout fails closed because stale shared memory or NVMe state can invalidate the next benchmark.

For an already-running job created before this guard existed, terminate only the stuck login-node `srun` client after confirming the benchmark outputs are complete. The launcher's cleanup trap then continues to `scancel` the allocation. Do not cancel the GitHub job or delete its workspace; doing so can discard uploadable evidence.

## AMD root-owned workspace prevention and recovery

### Prevent recurrence

Containers can run as root while the GitHub workspace is bind-mounted. The shared benchmark library prevents root-owned Python cache directories by setting:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
```

Do not override these paths back into the workspace. The MI355X launcher also deletes stale benchmark logs before launch and installs an EXIT trap that copies Slurm output/error evidence, prints the error tail, then runs scoped `sudo rm -rf "$BENCHMARK_LOGS_DIR"`. Keep `KEEP_LOGS=1` for deliberate local debugging only. It disables the cleanup trap. Cancellation can still bypass teardown, so use the recovery scan below after an `EACCES` cleanup failure.

Sources: [Python-cache prevention](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/benchmarks/benchmark_lib.sh#L5-L10), [MI355X cleanup trap](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/runners/launch_mi355x-amds.sh#L49-L76).

### Recover an MI355X TW runner workspace

Canonical signature:

```text
Deleting the contents of '.../actions-runner/_work/InferenceX/InferenceX'
Error: File was unable to be removed Error: EACCES: permission denied, rmdir '.../benchmark_logs/logs/slurm_job-<id>'
```

The jumpbox has no sudo. Use agent forwarding to the hop host that has passwordless sudo on `/it-share`.

1. **Read-only scan first:**

   ```bash
   ssh -A -o BatchMode=yes amd-tw-mi355 "ssh -o BatchMode=yes mia1-vm-amd-prj3-slog-001 \
     'sudo find /it-share/gharunners*/gharunner*/actions-runner/_work -user root 2>/dev/null'"
   ```

2. Review every result. Every path must be below `actions-runner/_work/`, normally in `InferenceX/InferenceX/benchmark_logs/`. If any path is outside `_work`, **stop**.
3. With explicit approval, delete only the verified matches:

   ```bash
   ssh -A -o BatchMode=yes amd-tw-mi355 "ssh -o BatchMode=yes mia1-vm-amd-prj3-slog-001 \
     'sudo find /it-share/gharunners*/gharunner*/actions-runner/_work -user root -print0 2>/dev/null \
      | xargs -0 -r sudo rm -rf'"
   ```

4. Run the read-only scan again and require zero results.
5. Only after cleanup, rerun diagnosed failed sweeps. `slurm_job-<id>` can be correlated with `sacct -j <id>`. `CANCELLED` supports the skipped-teardown diagnosis.

Never run an unscoped `rm -rf` against `/it-share`.

Canonical source: [MI355X root-owned file recovery](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/clean-amd-mi355-runner-root-files.md).

## MI300X cluster debugging: enroot/pyxis user-namespace failures

Canonical signature on `mi300x-amds_*` / `chi-mi300x-*`:

```text
error: pyxis:     enroot-nsenter: failed to create user namespace: Permission denied
error: pyxis: couldn't start container
error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
srun: error: chi-mi300x-0XX: task 0: Exited with exit code 1
```

Known July 2026 cause: Ubuntu 24.04 provisioning drift leaves `kernel.apparmor_restrict_unprivileged_userns=1` on some nodes, blocking the actual enroot path. `unshare -U` is not a valid discriminator because its AppArmor profile may still allow it.

1. Confirm the exact signature and record failing nodes from GitHub logs:

   ```bash
   gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed
   ```

2. Access compute nodes through Slurm from the root controller. Direct root SSH to compute nodes is not available:

   ```bash
   ssh amd-vultr-mi300 \
     'srun -w chi-mi300x-043 -N1 --immediate=30 bash -c "<read-only-command>"'
   ```

3. Survey every visible node without changing it:

   ```bash
   ssh amd-vultr-mi300 'for n in $(sinfo -N -h -o "%N" | sort -u); do
     v=$(srun -w $n -N1 --immediate=20 sysctl -n kernel.apparmor_restrict_unprivileged_userns 2>&1 | tail -1)
     echo "$n: $v"
   done'
   ```

   A split of failing nodes at `1` and working nodes at `0` confirms drift. If all nodes are `0`, stop treating this as the known issue. Compare enroot versions, pyxis plugin state, and AppArmor coverage of `/usr/local/bin/enroot-nsenter` against a working node.

4. **Only with explicit approval**, change drifted nodes to the working baseline and persist it:

   ```bash
   ssh amd-vultr-mi300 'for n in <drifted-nodes>; do
     srun -w $n -N1 --immediate=30 bash -c \
       "sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 && \
        echo kernel.apparmor_restrict_unprivileged_userns=0 > /etc/sysctl.d/99-enroot-userns.conf"
   done'
   ```

   This disables a kernel security mitigation. Verify each live value is `0` and the persistent file exists. Escalate the durable fix to the node provisioning image. Otherwise, reprovisioned nodes will regress.

5. Rerun only affected flaky jobs after the cluster baseline is restored.

Canonical source: [MI300X enroot/pyxis recovery](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/debug-mi300-enroot-pyxis.md).

## Safe workflow reruns

Start by listing failed jobs and capturing failure-only logs:

```bash
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | select(.conclusion=="failure") | "\(.databaseId)\t\(.name)"'
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed \
  > /tmp/sweep_failed.txt
```

| Diagnosis | Safe action | Do not do |
| --- | --- | --- |
| Transient runner pickup, network, or confirmed infrastructure flake on a PR sweep | `gh run rerun "$RUN_ID" --repo SemiAnalysisAI/InferenceX --failed` | Dispatch a fresh sweep and lose the original run context |
| Cancelled run after MI355X workspace cleanup or MI300X repair | Try failed-only rerun. If GitHub refuses because the run was cancelled, `gh run rerun "$RUN_ID" --repo SemiAnalysisAI/InferenceX` | Full-rerun before fixing shared state can fail identically and spend GPU time |
| Reproducible OOM, HIP/CUDA/RCCL/NCCL error, invalid result, bad score, image/recipe/config bug | Fix or validate the root cause first, then rerun the narrowest affected path | Label the incident a flake because a retry happens to pass once |
| Correct InferenceX-app ingest, partial DB/migration/verification failure | Rerun failed jobs on that app run. A full app rerun is data-idempotent if needed | Rerun the GPU-producing InferenceX target |
| Wrong/missing/expired source artifacts or wrong source/merge pair | Repair selection or use the recovery PR procedure | Repeatedly rerun the same wrong ingest |
| Missing, skipped, bogus, or failed official push-to-main ingest | Create the validated recovery PR and reuse the original PR artifacts | Rerun the failed target workflow/job or create a one-off ingest workflow |

If a cancelled InferenceX run refuses partial rerun, a full rerun repeats all eligible work and can be expensive. Removing and re-adding the PR's sweep label is a last resort because it creates a fresh sweep rather than preserving the failed run.

Sources: [flake rerun rule](../.agents/skills/debug-runs/SKILL.md#L146-L153), [cancelled-run fallback](https://github.com/SemiAnalysisAI/InferenceX/blob/0c28706b33d4a796b82f6f9c3594c19c46365575/.claude/commands/clean-amd-mi355-runner-root-files.md#L52-L57), [app ingest idempotency](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/docs/data-pipeline.md#L26-L34).

## Failure classification

Classify by the earliest broken boundary, not the final red job.

| Class | Evidence | Owner/action |
| --- | --- | --- |
| **Benchmark/runtime** | server never becomes ready, OOM, HIP/CUDA/RCCL/NCCL, request failures, raw result missing, or zero successful requests | Reproduce the exact config, compare with a working node/SKU, and repair the recipe/image/runtime. Do not touch ingest |
| **Eval** | eval-only has no `results*.json`, score validation fails, `meta_env.json` coverage/concurrency mismatch, or sample files are incomplete | Inspect evaluator output and requested concurrencies. Preserve uploaded partial artifacts. Fix the evaluator/config before rerun |
| **Processing/collection** | raw JSON exists but `agg_*.json` is missing, collector cannot parse, or `results_bmk` or `eval_results_all` is absent | Inspect `process_result.py`/collector logs and artifact layout. Rerun failed workflow jobs only after the format problem is understood |
| **Runner workspace** | checkout cleanup `EACCES` under `_work/.../benchmark_logs/logs/slurm_job-*` | Read-only ownership scan, approved scoped deletion, zero-result verification, then rerun |
| **MI300X provisioning** | pyxis/enroot namespace signature, with failing nodes reading sysctl `1` and working nodes reading `0` | Apply the approved node repair and escalate the provisioning-image fix, then rerun affected jobs |
| **Dispatch/handoff** | no app run after `trigger-ingest`, curl/auth failure, or app preparation logs showing wrong IDs or missing/expired artifacts | Repair dispatch credentials/selection or use recovery. Do not rerun GPU work |
| **ETL/database** | correct pair and artifacts prepared, followed by migration/ingest/verification failure | Rerun the same app ingest after diagnosing the database/service cause. Rely on idempotency, not manual deletes |
| **Mapping/data quality** | app reports skipped failed rows, unmapped model/hardware/precision, missing eval rows, or implausible counts | Add or fix entity mapping or source metadata. A green workflow with unmapped data is not complete verification |
| **Publication/provenance** | wrong changelog, source-run provenance, merge-run identity, or bogus run with no valid benchmark data | Use the append-only recovery PR. Preserve source ancestry and verify the exact downstream ingest |

For an actionable incident report, record:

```text
InferenceX run / attempt / event / SHA:
Failed job / runner / node / Slurm job:
Failure class and first causal signature:
Raw, per-config, and aggregate artifact names/counts:
Expected source run / attempt / SHA:
Expected merge run:
InferenceX-app ingest run:
Repair performed and approval obtained:
Rerun command and resulting run:
DB verification, skipped rows, and unmapped entities:
Remaining durable fix:
```

This evidence is the completion gate. “Workflow green” without artifact identity, source/merge identity, and ingest counts is not a verified result recovery.
