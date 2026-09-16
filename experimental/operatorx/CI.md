# OperatorX GitHub Actions

**English** | [中文](CI_zh.md)

[OperatorX Sweep](../../.github/workflows/operatorx-sweep.yml) runs manually on
`h100-dgxc` (default), `h200-dgxc`, `b200-nscale`, `b300`, `gb200`, or `gb300`.
Pull requests only run the hosted planner; GPU work requires `workflow_dispatch`.

| GPU | Pool | GPUs per physical node | Image platform | Result cluster |
| --- | --- | ---: | --- | --- |
| H100 | `h100-dgxc` | 8 | `linux/amd64` | `h100_dgxc_8x` |
| H200 | `h200-dgxc` | 8 | `linux/amd64` | `h200_dgxc_8x` |
| B200 | `b200-nscale` | 8 | `linux/amd64` | `b200_nscale_8x` |
| B300 | `b300` | 8 | `linux/amd64` | `b300_dsxe_8x` |
| GB200 | `gb200` | 4 | `linux/arm64` | `gb200_nvl72_4x` |
| GB300 | `gb300` | 4 | `linux/arm64` | `gb300_nvl72_4x` |

GB200/GB300 runs use one four-GPU tray, not the full NVL72 rack. Dense GEMM uses
`world_sizes=1` on every pool; reported TFLOPS remains per GPU. Hardware facts come
from CollectiveX's platform registry, and both planning and execution validate them.

## Dispatch

Once GitHub has registered the workflow, select **OperatorX Sweep → Run workflow**,
choose the source branch, and keep the initial defaults: `pool=h100-dgxc`,
`backends=torch`, `testlists=gemm`, `world_sizes=1`, `chunk_size=500`.
This schedules the complete checked-in GEMM catalog in bounded shards (currently
7,212 cases in 15 shards). The catalog includes formats unsupported by a selected
backend and shapes that can exceed device memory. Unsupported rows remain visible;
actual kernel and allocation errors fail CI. A full catalog run is not a promise
that every case fits or is supported on H100. A newly added workflow
may need to reach the default branch before GitHub accepts manual dispatch.

```bash
gh workflow run operatorx-sweep.yml --repo SemiAnalysisAI/InferenceX \
  --ref <branch> -f pool=h100-dgxc -f backends=torch \
  -f testlists=gemm -f world_sizes=1 -f chunk_size=500
```

For a quick infrastructure smoke check, explicitly select `testlists=gemm_perf`
and `chunk_size=50` (11 BF16 cases). Other NVIDIA backends and testlists are explicit
selections, not validated Hopper coverage. Unsupported operations remain visible in results. Backend
import errors, benchmark errors, and zero successful rows fail the shard.
Start with BF16 GEMM, then bounded collectives and compatible MoE combinations.
Do not infer that Blackwell-specific FP4 kernels work on Hopper.

## Execution contract

- Hosted planning validates inputs, groups backends by container image, separates
  world sizes and MoE parallelism triples, and splits shapes into bounded chunks.
  At most 256 shards are accepted. World sizes are restricted to 1, 2, 4, and 8,
  and must fit one physical node (GB200/GB300 reject 8). Shapes outside the
  requested sizes are counted in `excluded_shapes`.
- Each Actions shard holds exactly one exclusive physical Slurm node (four or eight GPUs). The GPU
  process count is the selected world size. Admission uses the existing priority
  scorer and `ci-job-*`, `ci-attempt-*`, and exactly one `nodes:1` label. Initial
  concurrency is two shards. Both scheduler switches must remain enabled.
- Runner settings come from CollectiveX's tracked platform registry. Source is
  checked out at the workflow SHA and copied into a private, compute-visible
  directory below the configured shared squash parent or a writable configured
  `storage_roots` entry (GB200). B300 uses the compute-visible account home from
  the password database, matching CollectiveX; an explicit `stage_dir` takes
  precedence. Results never depend on
  a submit-host `/tmp` mount being visible to compute nodes.
- The planner resolves each image digest. Imports are locked and cached by image
  plus digest and CPU architecture, with a second digest check after import. A moved or unresolvable
  tag fails rather than claiming the planned image was measured. Images must be
  anonymously readable from the planning and import hosts. Imports verify the host
  CPU architecture; B300 imports on its submit host, matching CollectiveX, while
  other pools import inside their allocation. Enroot uses explicit registry URLs,
  private temporary directories, and any pool-configured cache path. Allocation
  forwards account, QoS, and quarantined nodes; B300/GB pools retain their existing
  remap-root and memory settings. B300 leaves QoS selection to its partition/account,
  matching the inference launcher; the former `batch_1_qos` override is rejected
  by the current cluster. Its former excluded node names also do not exist in
  this pool and have been removed; Slurm still honors drained nodes. GB300 retains
  its configured QoS and exclusions.
- The launcher remains active through allocation, import, and execution. The
  allocation time limit is 45 minutes; Actions permits 70 minutes including
  queueing and cleanup. Slurm job names match the Actions runner name.
- Signals and the workflow's `always()` recovery step cancel recorded allocations,
  stop writers, recover partial results, and remove staged sources. The workflow
  explicitly allows 180 seconds for Slurm epilog/node release, including on H200. A failed
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

Real acceptance additionally requires a smoke run with artifacts on each selected pool, a
failed-shard rerun, and cancellation with confirmed allocation release. CPU
checks alone do not establish GPU compatibility or cluster storage visibility.

The final coverage job selects the newest artifact attempt for each requested
shard, preserves successful shards from previous attempts, and fails if any shard
is missing or failed. Its summary separates requested shapes from result rows
(one shape may run on multiple backends).

If cleanup failed, a single-shard dispatch can set `recovery_run_id` to the recent
OperatorX run from the same pool. It downloads the execution artifacts and retries
allocation/staging cleanup before allocating a new node. Recovery checks the run,
pool, and private staging parent; do not select unrelated or old Slurm executions.

`cleanup.log` records the active-job query used to confirm allocation release.
It queries the current user’s job list because querying a removed job ID directly
can return a Slurm error even after that allocation has terminated.
