# AMD srt-slurm bring-up

This document tracks the work-in-progress integration of
[`SemiAnalysisAI/srt-slurm`](https://github.com/SemiAnalysisAI/srt-slurm) with
InferenceX AMD Slurm clusters. The project is a functional orchestration
bring-up, not a performance-tuning exercise.

Current development pin:

- repository: `SemiAnalysisAI/srt-slurm`
- branch: `agent/amd-multinode-runtime`
- commit: `7f53c35015e12ef3581f474018b5349ce454e977`

## Scope

1. Prove a single-node aggregate vLLM deployment on MI300X.
2. Prove a multi-node vLLM Router prefill/decode deployment on MI300X using
   AMD's supported MoRI-IO KV connector.
3. Exercise both paths with fixed input/output sequence lengths and lightweight
   models before introducing production-size models.
4. Validate the same paths through the upstream InferenceX GitHub Actions
   runner infrastructure.
5. Port a representative existing MI355X disaggregated configuration after the
   MI300X runtime contract is stable.

## MI300X cluster contract under validation

- eight AMD GPUs per healthy compute node;
- Slurm GPU allocation through `--gres=gpu:<count>` rather than the current
  srt-slurm `--gpus-per-node` default;
- no site-specific `--segment` directive;
- ROCm device access through `/dev/kfd` and `/dev/dri`;
- Pyxis/Enroot writable, remap-root, and mount-home behavior matching the
  established MI300X launcher;
- the shared Hugging Face cache and runner workspace remain user-owned;
- fixed-sequence validation runs InferenceX's existing
  `utils/bench_serving/benchmark_serving.py` through srt-slurm's `custom`
  benchmark hook rather than maintaining a second benchmark copy in
  srt-slurm;
- the routable inter-node network interface is selected from live cluster
  evidence rather than copied from an NVIDIA recipe.

The launcher will continue to exclude compute nodes already documented as
unsuitable. It must not resume down nodes, cancel or preempt existing jobs, or
alter unrelated shared software.

## Acceptance criteria

### Aggregate

- one srt-slurm allocation starts one aggregate vLLM service;
- every requested GPU is visible to ROCm and vLLM exactly once;
- the OpenAI-compatible health/model endpoint becomes ready;
- a fixed-sequence request completes successfully;
- srt-slurm tears down all owned processes and exits successfully.

### Disaggregated

- one allocation places distinct prefill and decode roles across multiple
  MI300X nodes;
- vLLM Router and direct vLLM workers become healthy without Dynamo, NATS,
  etcd, or bespoke per-recipe orchestration;
- role endpoints use routable node addresses and unique ports;
- KV transfer completes across AMD nodes and a fixed-sequence request succeeds;
- teardown removes only processes owned by the allocation.

### Regression safety

- NVIDIA remains the default accelerator runtime in srt-slurm;
- existing NVIDIA recipes and device binding tests remain green;
- AMD-specific mounts, Slurm directives, and environment variables live in a
  reusable cluster profile rather than duplicated recipe shell fragments.

## Current status

The srt-slurm branch now contains the first accelerator-aware runtime slice:
cluster configuration accepts `accelerator_vendor: amd`, partial-GPU workers
use Linux ROCm's `ROCR_VISIBLE_DEVICES`, and legacy NVIDIA/CUDA behavior remains
the default. It also supports `gpu_sbatch_directive: gres` without changing the
legacy NVIDIA `--gpus-per-node` default. The initial MI300X cluster profile and
small-model aggregate recipe are checked in alongside this document. The first
aggregate path uses a direct private `vllm serve` endpoint. The two-node
1-prefill/1-decode path uses the official vLLM Router and vLLM's ROCm-only
`MoRIIOConnector`: srt-slurm owns the router discovery port and generates
role-aware worker registration config from the realized Slurm topology. The
control-plane endpoints use automatic RFC1918-preferring discovery because
private NIC names vary across MI300X node generations. The earlier
Dynamo/NIXL experiment reached KV-cache initialization but failed ROCm memory
registration; that NVIDIA-oriented data plane is now explicitly out of scope
rather than patched into the AMD implementation.
