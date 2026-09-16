# OperatorX GitHub Actions

**English** | [中文](CI_zh.md)

[OperatorX Sweep](../../.github/workflows/operatorx-sweep.yml) runs manually on
`h100-dgxc` (default) or `h200-dgxc`. Pull requests only run the hosted planner;
GPU work requires `workflow_dispatch`. The first validation target is H100.

## Dispatch

Once GitHub has registered the workflow, select **OperatorX Sweep → Run workflow**,
choose the source branch, and keep the initial defaults: `pool=h100-dgxc`,
`backends=torch`, `testlists=gemm_perf`, `world_sizes=1`, `chunk_size=50`.
This schedules one shard containing 11 BF16 GEMM shapes. A newly added workflow
may need to reach the default branch before GitHub accepts manual dispatch.

```bash
gh workflow run operatorx-sweep.yml --repo SemiAnalysisAI/InferenceX \
  --ref <branch> -f pool=h100-dgxc -f backends=torch \
  -f testlists=gemm_perf -f world_sizes=1 -f chunk_size=50
```

Other NVIDIA backends and testlists are explicit selections, not validated
Hopper coverage. Unsupported operations remain visible in results. Backend
import errors, benchmark errors, and zero successful rows fail the shard.
Start with BF16 GEMM, then bounded collectives and compatible MoE combinations.
Do not infer that Blackwell-specific FP4 kernels work on Hopper.

## Execution contract

- Hosted planning validates inputs, groups backends by container image, separates
  world sizes and MoE parallelism triples, and splits shapes into bounded chunks.
  At most 256 shards are accepted. World sizes are restricted to 1, 2, 4, and 8;
  shapes outside the requested sizes are counted in `excluded_shapes`.
- Each Actions shard holds exactly one exclusive eight-GPU Slurm node. The GPU
  process count is the selected world size. Admission uses the existing priority
  scorer and `ci-job-*`, `ci-attempt-*`, and exactly one `nodes:1` label. Initial
  concurrency is two shards. Both scheduler switches must remain enabled.
- Runner settings come from CollectiveX's tracked platform registry. Source is
  checked out at the workflow SHA and copied into a private, compute-visible
  directory below the configured shared squash parent. Results never depend on
  a submit-host `/tmp` mount being visible to compute nodes.
- The planner resolves each image digest. Imports are locked and cached by image
  plus digest, with a second digest check after import. A moved or unresolvable
  tag fails rather than claiming the planned image was measured. Images must be
  anonymously readable from the planning and import hosts.
- The launcher remains active through allocation, import, and execution. The
  allocation time limit is 45 minutes; Actions permits 70 minutes including
  queueing and cleanup. Slurm job names match the Actions runner name.
- Signals and the workflow's `always()` recovery step cancel recorded allocations,
  stop writers, recover partial results, and remove staged sources. A failed
  cleanup retains staging for investigation. Slurm's time limit is the last
  bound if the runner host disappears.
- Strict CI runs atomically checkpoint rank-zero rows after each operation,
  outside kernel timing. The existing non-CI timing loop is unchanged.

## Artifacts and reruns

`operatorx-manifest-<run_id>` records requested cases, image digests and source
SHA. It remains available to failed-job reruns. Each attempt uploads separate
`operatorx-shard-<run_id>-<attempt>-<shard>` artifacts containing execution
metadata, allocation/import/benchmark logs, status and any raw result JSONs.
Startup failures can have logs without results; cancellation checkpoints are
partial coverage. A successful shard requires successful measurements, not just
successful Slurm submission. Result environments record the workflow run,
attempt, shard, source SHA and image digest.

Download artifacts through `gh run download`. Preserve raw files and provenance;
`scripts/consolidate_results.py` is not part of CI. Dashboard ingestion is separate.

## Local validation

Planning requires Python 3.11 or newer. Compute-side control code uses the existing
Python 3.10+ Slurm-host environment. CPU tests run the real planner, benchmark
orchestration and launcher with external GPU/Slurm collaborators substituted.

```bash
uv run --no-project --python 3.12 --with pytest --with pyyaml \
  python -m pytest experimental/operatorx/tests/ -q
```

Real acceptance additionally requires a Hopper smoke run with artifacts, a
failed-shard rerun, and cancellation with confirmed allocation release. CPU
checks alone do not establish GPU compatibility or cluster storage visibility.

The final coverage job selects the newest artifact attempt for each requested
shard, preserves successful shards from previous attempts, and fails if any shard
is missing or failed. Its summary separates requested shapes from result rows
(one shape may run on multiple backends).
