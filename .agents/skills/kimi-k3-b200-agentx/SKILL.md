---
name: kimi-k3-b200-agentx
description: Maintain the evolving target-hardware evidence, topology decisions, concurrency ladders, and run record for Kimi K3 AgentX tuning on 16 B200 GPUs in InferenceX PR #2475. Use with tune-agentx-submission and debug-agentx-runs.
---

# Kimi K3 B200 AgentX working record

Local-only working skill for PR #2475. Do not commit, push, publish, or include private cluster addresses. Replace stale estimates when target-hardware measurements become available.

Last updated 2026-08-03.

## Scope and constraints

- PR: https://github.com/SemiAnalysisAI/InferenceX/pull/2475
- Hardware: 2 B200 nodes, 8 GPUs per node, 16 GPUs total.
- All aggregate multinode serving must use checked-in srt-slurm recipes.
- The recipe is the serving topology source of truth. Matrix metadata must match it.
- Preserve the native 1,048,576-token model context and the existing AgentX dataset.
- Synthetic acceptance is for comparable performance measurement. Real block verification remains the correctness path.

## Blocking finding

The branch's current TP8 by PP2 DSpark recipe is not upstream-supported. The official Kimi K3 recipe excludes DSpark from pipeline parallelism, and vLLM issue #50098 records the `SupportsPP` startup failure.

Keep TP8 by PP2 only as a non-DSpark baseline until upstream support changes. Do not spend B200 time tuning a known unsupported DSpark composition.

## Official Blackwell baseline

Every candidate must verify the official settings rather than inherit the branch's omissions:

- FP8 KV cache
- prefill query quantization with `TRTLLM_RAGGED`
- prefix caching enabled
- K3 latent-MoE tail fusion
- Model Runner v2 and Rust frontend where supported
- fastsafetensors loading
- native context length
- text-only mode for AgentX
- no FlashInfer autotune.

For DSpark, official block verification uses a 7-token draft with probabilistic sampling. The checked-in performance profiles preserve K7 and the committed synthetic AL 3.84. Consider K2 only after the official K7 canaries establish a target-B200 baseline and only if one small comparison can change the selected Pareto point.

## AgentX dataset evidence

Dataset: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126

It contains 393 trajectories and 98,827 model requests.

| Statistic | Input tokens |
|---|---:|
| p50 | 142,016 |
| p75 | 310,464 |
| p90 | 549,504 |
| p95 | 682,880 |
| p99 | 863,471 |
| mean | 218,922 |

Median cached fraction is about 99.1 percent. Median uncached input is 1,664 tokens. Prefix caching is required for representative performance. Disabling it converts incremental agent turns into repeated long prefills.

## Prior B200 evidence

Run: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/30327547020

This was a non-DSpark TP8 by PP2 baseline, not proof for the candidate topologies. Startup reported:

- auto or BF16 KV
- 58.89 GiB KV cache per rank
- 4,147,581 logical KV tokens
- 3.96 maximum concurrency at 1M tokens
- prefix caching resolved false.

Observed curve:

| Conc | Total tok/s | Output tok/s | p50 TTFT | p50 TPOT | KV usage |
|---:|---:|---:|---:|---:|---:|
| 8 | 13,817 | 98.5 | 6.2 s | 45 ms | 45% |
| 16 | 11,366 | 53.2 | 12.2 s | 245 ms | 97% |
| 32 | 18,298 | 98.5 | 90.9 s | 198 ms | 100% |

Concurrency 32 increases input processing but does not improve output throughput over concurrency 8. It is not a useful operating point for that topology.

## Implemented profile matrix

| Orientation | Topology | Initial concurrency |
|---|---|---|
| Latency | TP16, EP1 | 1, 2, 4, 8 |
| Balanced | TEP16, EP16 | 8, 16, 24, 32 |
| GPU throughput | DEP16, TP1 by DP16 by EP16 | 32, 64, 96, 128, 192, 256 |
| CPU KV | DEP16 plus SimpleCPUOffloadConnector | 128, 192, 256, 384 |

This is an 18-point starting matrix. Concurrency 384 exercises nearly all 393 trajectories.

Checked-in serving recipes:

- `agg-b200-tp16-latency-dspark-agentic.yaml`, direct multi-node vLLM
- `agg-b200-tep16-balanced-dspark-agentic.yaml`, direct multi-node vLLM
- `agg-b200-dep16-throughput-dspark-agentic.yaml`, Dynamo aggregate with per-node DP ranks
- `agg-b200-dep16-throughput-vllm-simple-offload-dspark-agentic.yaml`, the same DEP topology with 220 GiB of CPU KV per rank.

The launcher pins srt-slurm renderer commit `df5baa93f4caf5169dea2a4236ad2cc742fe40e7`.

Run no more than two `agentx-fast` probes concurrently:

1. First wave, TP16 at concurrency 1 and DEP16 without offload at concurrency 32.
2. Second wave, TEP16 at concurrency 8 and DEP16 with CPU offload at concurrency 128.

Do not start the second wave until the first wave's live evidence is understood. Each probe must confirm all workers, all metrics sources, resolved FP8 KV, prefix caching, reported cache blocks, KDA headroom, routing balance, and zero deterministic request errors before expansion.

## Capacity model

K3 has 24 paged MLA layers with a 576-element latent per layer. Full-model FP8 MLA is approximately 13,824 bytes per token per rank before hybrid-cache overhead. KDA contributes fixed per-request recurrent and draft state.

With the prior 58.89 GiB HBM reservation as a contingent assumption:

| Layout | Estimated p50 request capacity | Meaning |
|---|---:|---|
| TP16 or TEP16 | about 32 | One logical cache namespace. Never multiply by 16. |
| DEP16 | about 26 per pool, 413 aggregate | Sixteen independent pools, contingent on balanced routing. |
| DEP16 plus 220 GiB CPU KV per rank | about 97 per pool, 1,544 aggregate | Cache-only upper bound, not a useful-throughput claim. |

Startup HBM, KDA retention, allocator padding, scheduler limits, prefix-cache policy, and transfer bandwidth override these estimates.

## Renderer decision

Direct multi-node vLLM correctly renders leader and headless commands for TP16 and TEP16. It does not derive node-local DP ranges.

DEP16 therefore uses srt-slurm's Dynamo aggregate path with `dp_launch_mode: per_node`. The renderer derives local size 8, start ranks 0 and 8, a shared DP address and RPC port, and hybrid load balancing. Do not move DEP to the direct frontend without a source-level renderer change and command-construction test.

## Live decision gates

Continue expanding a topology only while output throughput rises and latency remains on the intended Pareto frontier.

Stop expanding when any condition persists:

- deterministic startup or request failure
- missing or idle DP engine while peers queue
- KV usage pinned near 100 percent with growing queues
- flat or falling output throughput with sharply worse TTFT or TPOT
- CPU transfer traffic rising without useful throughput gain
- routing imbalance that invalidates aggregate capacity arithmetic
- no forward progress across repeated compute-visible metric samples.

Record the evidence before recommending cancellation. While the user is unavailable, do not cancel or mutate shared-cluster state without prior authorization.

## Access state

The private route and file-based SSH key work when the local credential agent is bypassed with `IdentityAgent=none`. Compute-visible Slurm inspection is available. Keep host addresses outside this file.

## Living run record

Keep only current evidence. After each targeted or official run, update this table and the decisions above.

| Date | Run and job | Profile and conc | Result | Measured cache and metrics | Decision |
|---|---|---|---|---|---|
| 2026-08-03 | 30327547020, prior baseline | TP8 by PP2, no DSpark, conc 1 to 32 | Completed, unofficial baseline | 58.89 GiB, 4.15M tokens, prefix false, KV cliff at 16 to 32 | Replace unsupported DSpark PP with TP16, TEP16, and DEP16 candidates |
| 2026-08-03 | 30837472733, job 91766231186, Slurm 29091 | TP16 direct vLLM, conc 1, `agentx-fast` | Canceled after deterministic startup stall | All 16 ranks reached NCCL 2.30.7. FlashInfer selected its cross-node MNNVL path, the pool reported no multicast support, GPU use stayed near 2 GiB/rank, and no weight load began for 16 minutes | Disable `VLLM_ALLREDUCE_USE_FLASHINFER` for cross-node TP16/TEP16 and rerun the same canary. The symmetric-memory warning was only a failed capability probe |
| 2026-08-03 | 30841915075, job 91780887584, Slurm 29113 | TP16 direct vLLM, conc 1, `agentx-fast` | Pending | Same profile with FlashInfer cross-node all-reduce disabled. Slurm allocation verified on the assigned runner | Monitor startup through accepted artifacts before dispatching another profile |

## Sources

- Official interactive recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3
- Official source recipe: https://github.com/vllm-project/recipes/blob/main/models/moonshotai/Kimi-K3.yaml
- Pipeline limitation: https://github.com/vllm-project/vllm/issues/50098
- Kimi K3 model: https://huggingface.co/moonshotai/Kimi-K3
- DSpark draft model: https://huggingface.co/Inferact/Kimi-K3-DSpark
- AgentX dataset: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126
