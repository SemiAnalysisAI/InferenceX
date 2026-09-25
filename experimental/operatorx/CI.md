# OperatorX GitHub Actions

**English** | [中文](CI_zh.md)

[OperatorX Sweep](../../.github/workflows/operatorx-sweep.yml) runs manually on the
self-hosted runners carrying one of these labels, the labels InferenceX's benchmark
workflows use: `cluster:h100-dgxc` (default), `cluster:h200-dgxc`, `cluster:b200-nscale`,
`cluster:b300-dsxe`, `cluster:gb200-nv`, `cluster:gb300-nv`, `cluster:mi300x-amd`,
`cluster:mi325x-amds` or `cluster:mi355x-amds`.
Pull requests only run the hosted planner; GPU work requires `workflow_dispatch`.

| GPU | Runner label | GPUs per physical node | Image platform | Result cluster |
| --- | --- | ---: | --- | --- |
| H100 | `cluster:h100-dgxc` | 8 | `linux/amd64` | `h100_dgxc_8x` |
| H200 | `cluster:h200-dgxc` | 8 | `linux/amd64` | `h200_dgxc_8x` |
| B200 | `cluster:b200-nscale` | 8 | `linux/amd64` | `b200_nscale_8x` |
| B300 | `cluster:b300-dsxe` | 8 | `linux/amd64` | `b300_dsxe_8x` |
| GB200 | `cluster:gb200-nv` | 4 | `linux/arm64` | `gb200_nvl72_4x` |
| GB300 | `cluster:gb300-nv` | 4 | `linux/arm64` | `gb300_nvl72_4x` |
| MI300X | `cluster:mi300x-amd` | 8 | `linux/amd64` | `mi300x_amds_8x` |
| MI325X | `cluster:mi325x-amds` | 8 | `linux/amd64` | `mi325x_amds_8x` |
| MI355X | `cluster:mi355x-amds` | 8 | `linux/amd64` | `mi355x_8x` |

GB200/GB300 runs use one four-GPU tray, not the full NVL72 rack. Dense GEMM uses
`world_sizes=1` on every runner; reported TFLOPS remains per GPU. Hardware facts come
from CollectiveX's platform registry, and both planning and execution validate them.

## Dispatch

Once GitHub has registered the workflow, select **OperatorX Sweep → Run workflow**,
choose the source branch, and keep the initial defaults: `runner=cluster:h100-dgxc`,
`backends=vllm`, `testlists=gemm`, `world_sizes=1`, `chunk_size=500`.
This schedules the complete checked-in GEMM catalog in bounded shards (currently
5,416 cases in 11 shards). The catalog includes formats unsupported by a selected
backend and shapes that can exceed device memory. Unsupported rows remain visible;
actual kernel and allocation errors fail CI. A full catalog run is not a promise
that every case fits or is supported on H100. A newly added workflow
may need to reach the default branch before GitHub accepts manual dispatch.

```bash
gh workflow run operatorx-sweep.yml --repo SemiAnalysisAI/InferenceX \
  --ref <branch> -f runner=cluster:h100-dgxc -f backends=vllm \
  -f testlists=gemm -f world_sizes=1 -f chunk_size=500
```

`mode=timing` (default) records latency, telemetry and a profiler replay per op.
`mode=counters` instead runs each op once under Nsight Compute (NVIDIA; the host's
`/opt/nvidia/nsight-compute`) or rocprofv3 (AMD; 12 counter passes, so use a smaller
`chunk_size`), with raw counter files under `results/counters/`. Latencies from a
counters run are perturbed by the profiler.

For a quick infrastructure smoke check, explicitly select `testlists=gemm_perf`
and `chunk_size=50` (11 BF16 cases). `gemm_serving_8k1k_min` and
`gemm_serving_all_min` hold the GEMMs of InferenceX serving configurations. Unsupported operations remain visible in results. Backend
import errors, benchmark errors, and zero successful rows fail the shard.
Start with BF16 GEMM, then the quantized formats.
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
  Enroot uses explicit registry URLs and any platform-configured cache path. Allocation
  forwards account, QoS, and quarantined nodes; B300/GB nodes retain their existing
  remap-root and memory settings. B300 leaves QoS selection to its partition/account,
  matching the inference launcher; the former `batch_1_qos` override is rejected
  by the current cluster. Its former excluded node names also do not exist in
  this cluster and have been removed; Slurm still honors drained nodes. GB300 retains
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
`scripts/consolidate_results.py` is not part of CI.

## Local validation

Planning requires Python 3.11 or newer. Compute-side control code uses the existing
Python 3.10+ Slurm-host environment. CPU tests run the real planner, benchmark
orchestration and launcher with external GPU/Slurm collaborators substituted.

```bash
uv run --no-project --python 3.12 --with pytest --with pyyaml --with torch --with numpy \
  python -m pytest experimental/operatorx/tests/ -q
```

Real acceptance additionally requires a smoke run with artifacts on each selected runner, a
failed-shard rerun, and cancellation with confirmed allocation release. CPU
checks alone do not establish GPU compatibility or cluster storage visibility.

The final coverage job selects the newest artifact attempt for each requested
shard, preserves successful shards from previous attempts, and fails if any shard
is missing or failed. Its summary separates requested shapes from result rows
(one shape may run on multiple backends).

If cleanup failed, a single-shard dispatch can set `recovery_run_id` to the recent
OperatorX run from the same runner. It downloads the execution artifacts and retries
allocation/staging cleanup before allocating a new node. Recovery checks the run,
runner, and private staging parent; do not select unrelated or old Slurm executions.

`cleanup.log` records the active-job query used to confirm allocation release.
It queries the current user’s job list because querying a removed job ID directly
can return a Slurm error even after that allocation has terminated.

## AMD execution

`platforms.json` overlays the CollectiveX registry, per runner label, with the AMDS Slurm clusters.
AMD accepts single-GPU `torch`/`vllm` GEMM. ROCm PyTorch uses HIP events through
`torch.cuda`; FP8 selects FNUZ on gfx942 and OCP on gfx950. Unsupported formats
remain explicit. Staging lives outside `_work`, below the shared runner root
derived from `RUNNER_TEMP`. Containers never write to the checkout. MI300X/MI325X
forward `/dev/kfd` and `/dev/dri`; CPU requests follow each inference launcher.

## GEMM

`gemm` args describe each operand's storage and quantization. `a` is the
activation `[M, K]` and `b` the weight `[N, K]`: `{"dtype", "scale"?, "scale2"?,
"symmetric"?}`, where `scale` is `{"dtype", "static", "group": [rows, cols]}` and
`-1` spans a dimension (`[-1, -1]` per-tensor, `[1, -1]` per-token, `[-1, 1]`
per-channel, `[1, 128]` 1x128 groups, `[128, 128]` blocks). For example, FP8
block quantization is
`"a": {"dtype": "e4m3", "scale": {"dtype": "fp32", "static": false, "group": [1, 128]}}` and
`"b": {"dtype": "e4m3", "scale": {"dtype": "fp32", "static": true, "group": [128, 128]}}`;
an unquantized operand is `{"dtype": "bf16"}`. `scale.dtype` is the checkpoint's
scale format. `input` is the dtype an operand arrives in (default bf16): when it
differs from `dtype`, quantization is part of the op; when equal, the operand is
pre-quantized. The runtime format a framework converts to is reported per result.

The `vllm` backend builds a vLLM `ReplicatedLinear` under the checkpoint quant
config those descriptors imply, runs `process_weights_after_loading` and times
`layer(x)`, so vLLM picks the kernel. `metrics.backend_meta` records the chosen
kernel classes, parameter dtypes before and after loading, and the vLLM env.
Emulation-only paths are reported unsupported. AMD enables AITER, as
InferenceX's ROCm launches do.

## MoE

`moe` (`ops/moe.py`) describes one MoE layer from the router GEMM on
normed hidden states to the combined output: shape (`tokens`, `hidden`),
routed `experts` (count, top-k, intermediate size, optional biases / latent width /
zero experts, and gemm operand descriptors for `x`, `w13`, `w2`, `a2`), `router`
(gate dtype, scoring, top-k / grouped / hash selection, bias, renormalize, scale),
`activation`, optional `shared` experts, and the `routing` data distribution.
Execution (expert kernels, dispatch, shared-expert fusion or stream overlap, graphs)
is the backend's choice. The `vllm` backend (`runners/common/vllm/moe.py`) builds
vLLM's router (`GateLinear`), routed experts (`FusedMoEFactory`) and shared-expert
MLP under the quant configs the descriptors imply and times the block; vLLM picks the
expert kernels, shared-expert fusion and streams. A forced expert-load distribution
(`balanced`, `zipf`, `single_hot`) replaces the router's expert choice and keeps its
weights; zero experts are reported unsupported for now.

`testlists/moe_small.json` holds one full-size routed MoE layer per InferenceX
MoE checkpoint scheme (DeepSeek-R1, DeepSeek-V4-Pro/V4.1-Flash, Qwen3.5, Qwen3.8-Flash-Next,
GLM-5.2, Kimi-K3, MiniMax-M3 in their FP8/NVFP4/MXFP4/MXFP8 variants) at `tokens=1`,
plus `tokens=256` layers under each expert-load distribution, 28 cases. Each layer's
weights fit on one GPU.

Experimental operator changes are recorded in the adjacent `perf-changelog.yaml`,
separately from the root inference-recipe changelog's config-key schema.
