# OperatorX GitHub Actions

**English** | [中文](CI_zh.md)

[OperatorX Sweep](../../.github/workflows/operatorx-sweep.yml) runs manually on
`h100-dgxc` (default), `h200-dgxc`, `b200-nscale`, `b300`, `gb200`, `gb300`,
`mi300x`, `mi325x`, or `mi355x`.
Pull requests only run the hosted planner; GPU work requires `workflow_dispatch`.

| GPU | Pool | GPUs per physical node | Image platform | Result cluster |
| --- | --- | ---: | --- | --- |
| H100 | `h100-dgxc` | 8 | `linux/amd64` | `h100_dgxc_8x` |
| H200 | `h200-dgxc` | 8 | `linux/amd64` | `h200_dgxc_8x` |
| B200 | `b200-nscale` | 8 | `linux/amd64` | `b200_nscale_8x` |
| B300 | `b300` | 8 | `linux/amd64` | `b300_dsxe_8x` |
| GB200 | `gb200` | 4 | `linux/arm64` | `gb200_nvl72_4x` |
| GB300 | `gb300` | 4 | `linux/arm64` | `gb300_nvl72_4x` |
| MI300X | `mi300x` | 8 | `linux/amd64` | `mi300x_amds_8x` |
| MI325X | `mi325x` | 8 | `linux/amd64` | `mi325x_amds_8x` |
| MI355X | `mi355x` | 8 | `linux/amd64` | `mi355x_8x` |

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
  scorer and `ci-job-*`, `ci-attempt-*`, and exactly one `nodes:1` label. All shards remain eligible; the priority/node scheduler controls physical-node
  admission. A second GitHub matrix cap can strand assigned labels on held jobs.
  Both scheduler switches must remain enabled.
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
  CPU architecture and run inside their allocation. B300 follows the inference
  launcher's compute-node import because its submit host lacks extraction space
  for this image. Enroot and GNU parallel use private temporary directories;
  Enroot uses explicit registry URLs and any pool-configured cache path. Allocation
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
uv run --no-project --python 3.12 --with pytest --with pyyaml --with torch --with numpy \
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

## AMD execution

`platforms.json` overlays the CollectiveX registry with the AMDS Slurm pools.
AMD accepts single-GPU `torch` GEMM and `torch,aiter` attention. ROCm PyTorch uses HIP events through
`torch.cuda`; FP8 selects FNUZ on gfx942 and OCP on gfx950. Unsupported formats
remain explicit. Staging lives outside `_work`, below the shared runner root
derived from `RUNNER_TEMP`. Containers never write to the checkout. MI300X/MI325X
forward `/dev/kfd` and `/dev/dri`; CPU requests follow each inference launcher.

## Attention

Select `testlists=attention_perf` for eight BF16/FP16 MHA/GQA and materialized MLA
prefill/decode cases, or `attention` for the full 2,315-case catalog. NVIDIA uses
`backends=torch`; AMD also supports `backends=torch,aiter`. Attention measures
latency in microseconds. Unsupported precision/layout combinations are recorded,
and allocation or kernel errors still fail CI. Strict CI retains unsupported
backend/operator pairs instead of dropping requested coverage.

PyTorch uses bottom-right causal masking for rectangular decode inputs. Grouped
KV expansion happens before timing. Both MLA backends time attention on
materialized Q/K/V; compressed-cache projection and RoPE are excluded. AITER calls
`flash_attn_func` directly with native grouped KV heads and bottom-right causality.
It accepts uniform BF16/FP16, contiguous KV, and head dimensions divisible by eight
up to 256; other requests remain unsupported rather than fall back to torch.
Experimental operator changes are recorded in the adjacent `perf-changelog.yaml`,
separately from the root inference-recipe changelog's config-key schema.


## Kimi K3 routed MoE benchmark profile

`testlists=kimi_k3_moe_perf`, `backends=vllm`, `world_sizes=1` runs eight BF16
routed-expert cases: 1, 16, 128 and 1024 local tokens, each with EP8 or TP8 shapes.
The generic profile follows [vLLM #50082](https://github.com/vllm-project/vllm/pull/50082):
896 experts, top-16, hidden 7168 and intermediate 3072. EP8 allocates 112 experts
with intermediate 3072; TP8 allocates 896 experts with intermediate 384.
Both execute on **one GPU**, with no live distributed groups or communication.

This is explicitly **Kimi K3 (vLLM benchmark profile)**. It measures vLLM's generic
SiLU routed expert kernel with synthetic, uniform-random local routing prepared
before timing. The timer includes the fused expert implementation's token sorting,
gate/up GEMM, SiLU-and-multiply, down GEMM and weighted reduction; it excludes
router/top-k calculation, shared experts and communication. It does not measure
the released Kimi K3 layer's SITU activation, 3584-wide latent expert path,
latent projections or shared experts. Those need a separate native-layer profile.
No model weights or Hugging Face credentials are required.

NVIDIA uses `vllm/vllm-openai:v0.19.0` (amd64/arm64); AMD uses the existing ROCm
image. Select `backends=vllm` explicitly. Unsupported precision, routing or shared
expert requests produce unsupported rows; import and kernel failures fail CI.

Useful routed matmul TFLOPS per GPU is
`6*num_tokens*top_k*hidden*(intermediate/routed_tensor_parallel_size)/(latency_us*1e6)`.
The local top-k routes all target the local expert table, matching the generic
benchmark's EP emulation. Do not divide the measured work by EP again or multiply
by the number of GPUs in the allocation. These kernel measurements exclude
activation and routing FLOPs and are not full-model throughput.

### GPU validation status

The complete eight-case BF16 profile passed on H200, MI300X and MI325X. H100 currently fails before kernel execution: its Enroot importer rejects OCI whiteout conversion for the vLLM image on both `/tmp` and `/var/tmp`. The same host limitation is recorded by CollectiveX swap-blocks. H100 requires a working image-import environment before performance can be reported; this change does not modify node configuration. B200, B300, GB200, GB300 and MI355X dispatches are awaiting shared GPU capacity. A registered pool is not runtime validation.
