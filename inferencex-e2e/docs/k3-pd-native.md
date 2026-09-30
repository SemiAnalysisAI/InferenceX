# Native Kimi-K3 PD on MI355X

**English** | [中文](k3-pd-native_zh.md)

This configuration uses InferenceX's Python launcher and native srt-slurm orchestration for Kimi-K3 prefill/decode disaggregation. The master config and selected recipe own benchmark topology and tuning; there is no alternate Bash launcher.

## Configuration ownership

- `configs/amd-master.yaml` selects `kimik3-fp4-mi355x-vllm-disagg-agentic` and supplies matrix identity and result metadata.
- `benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml` owns roles, graph settings, concurrency, the worker and router images, and FP32 SSM state. Target weights remain MXFP4, KV is FP8, and GPU memory utilization is 0.90.
- The draft checkpoint runs through the upstream DSpark path without weight conversion. Throughput uses InferenceX's measured golden acceptance selection; eval uses real verification. The latency variant uses DSpark K7 without CPU offload; other variants use K4 with prefill SimpleCPUOffload.
- `configs/runners.yaml` owns staged models, the draft mount, fabric devices, worker network settings, memlock and image-import policy. The named srt lane enables recipe-image staging for this workload.
- `infx/launch/` owns imports, setup, submission, cancellation and result preservation. Recipe images are resolved with the job-local srt-slurm Python environment, not the launcher interpreter. Worker-image disagreement is rejected before image import, and the selected recipe's router image is staged using the existing backend.

The high-concurrency prefill configuration retains `HSA_NO_SCRATCH_RECLAIM=0`; decode is unchanged. That setting is an explicit runtime policy, not a temporary source patch. No BF16 SSM, workspace-development, READ-credit/QP or router-algorithm changes are introduced.

## Official image and temporary integration

The worker uses the official `vllm/vllm-openai-rocm:nightly-ac68c3087215e0a4f3cdfa218508c6aada57235d` image, pinned to amd64 digest `sha256:e3fdfb382f2b567718ab6de49a14f5d5695dad84efc6dfd9c38f661b1a763e19`. The master and recipe use exactly the same identity. This replaces the custom worker build without a shared-MR modification.

That nightly contains [vLLM#57700](https://github.com/vllm-project/vllm/pull/57700). Discovery templates also require [srt-slurm#508](https://github.com/NVIDIA/srt-slurm/pull/508), which is not in the pinned srt-slurm revision. Required engine backports and image/provider compatibility prerequisites belong in a separate, removable debug commit, not in the framework or configuration commits.

The temporary changes are not merge-ready engine policy. Remove them once their upstream dependencies ship:

1. Resolve an official ROCm nightly that contains the required engine fixes, update both worker-image references, and remove the corresponding temporary setup and patch assets after qualification.
2. Independently advance the srt-slurm submodule to a revision containing PR #508, then remove the temporary job-local cherry-pick and its tests. Updating the worker image does not update srt-slurm.
3. Append the performance changelog and validate the new source/image pair through the normal smoke, sweep and eval gates. Removing temporary patches makes the source stack clean; it does not replace runtime qualification or review.

## Validation scope

The Python-launcher migration is covered by behavior tests of selected images, pre-import rejection, real launch entrypoints with external scheduler/install commands stubbed, result staging, failure propagation and cancellation. Registry/source checks establish image identity and upstream inclusion, not device/provider compatibility or RDMA stability.

The earlier [c48 run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36556488243) used a different harness revision and custom image. It is historical evidence only, not qualification of this official-nightly candidate. No new GPU run, throughput, correctness or fully graceful shutdown result is claimed here.
