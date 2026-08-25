# Models

English | [中文](MODELS_zh.md)

This document tracks every model benchmarked by InferenceX-e2e: when it was added, which benchmark scenarios are currently active for it, and which scenarios are deprecated. Results for active scenarios are published to <https://inferencex.com/>.

## Deprecation Notice

InferenceX-e2e runs on a fixed, limited pool of GPUs and is maintained by a small team. Every scenario, precision, and recipe variant we keep alive consumes cluster hours and maintainer attention that would otherwise go to new frontier models. The deprecations below free that capacity. Where a deprecation removes one arm of an A/B pair, we keep and publish the arm that wins on the Pareto frontier.

### Monday, August 3, 2026

**Monday, August 3, 2026** is the last day for the scenarios, precisions, and recipe variants listed below. They are deprecated after that date.

**Partially enacted on 2026-08-04** in [#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493): the scenario and precision retirements in the first table were carried out. This removed 54 config keys from the active master configs and archived them under [`configs/deprecated/`](configs/deprecated/), with their benchmark scripts moved to the sibling `deprecated/` directories. The speculative-decoding A/B retirements in the second table are **not yet enacted**. See the note under that table.

Scenario and precision retirements:

| Model | Deprecated | Remains |
|---|---|---|
| MiniMax-M3 (`minimaxm3`) | Single-turn 8k1k | Agentic coding |
| Kimi-K2.5/2.6/2.7-Code (`kimik2.5`) | Agentic coding | Nothing. Single-turn 8k1k ran until August 6, 2026 and was retired on 2026-08-07 (see below) |
| Qwen3.5-397B-A17B (`qwen3.5`) | All **bf16** recipes, in every scenario, on NVIDIA and AMD | fp8 and fp4 recipes |

Speculative-decoding A/B retirements apply to each pair below. The spec-decode arm is the better Pareto frontier, so we stop running the non-spec-decode arm and publish only the spec-decode arm:

| Model | Deprecated arm | Published arm |
|---|---|---|
| DeepSeek-V4-Pro 1.6T (`dsv4`) | Agentic coding, non-MTP | Agentic coding, MTP |
| Qwen3.5-397B-A17B (`qwen3.5`) | Agentic coding, non-MTP | Agentic coding, MTP |
| MiniMax-M3 (`minimaxm3`) | Agentic coding, non-EAGLE3 | Agentic coding, EAGLE3 |
| GLM-5.2 (`glm5.2`) | Agentic coding, non-MTP | Agentic coding, MTP |
| Kimi-K3 (`kimik3`) | Agentic coding, non-DSpark (deprecated from day 0) | Agentic coding, DSpark |

**Status: not yet enacted.** Every non-spec-decode agentic arm above still runs. Removing them today would leave MiniMax-M3 and GLM-5.2 with no active config at all because their EAGLE3 and MTP agentic arms have not landed yet. It would also drop all AMD and all SGLang agentic coverage for DeepSeek-V4-Pro and Qwen3.5, neither of which has an MTP sibling on those platforms. This round runs once the replacement arms exist.

**Going forward we no longer benchmark non-spec-decode versus spec-decode as an A/B.** The non-spec-decode arm existed as a neutral baseline back when acceptance length wasn't standardized. That is now solved. [`golden_al_distribution/`](golden_al_distribution/) commits one golden acceptance-length curve per model, thinking mode, and draft length, measured on the SPEED-Bench `coding` category. AgentX pins every submission to that curve through synthetic acceptance (vLLM `synthetic_acceptance_length`, SGLang `SGLANG_SIMULATE_ACC_LEN`, TensorRT-LLM `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, etc). With a fair, engine-independent acceptance target in place, spec-decode results are directly comparable on their own and a separate non-spec-decode track is redundant. Agentic coding recipes are therefore run and published with speculative decoding enabled only, using MTP, EAGLE/EAGLE3, DSpark, or whatever draft method the model ships. The non-spec-decode arm is neither run nor published. New models are onboarded that way from day 0, as Kimi-K3 is.

### Thursday, August 6, 2026

**Thursday, August 6, 2026** is the last day for the **Single-turn 8k1k** scenario on **Kimi-K2.5/2.6/2.7-Code** (`kimik2.5`). The scenario is deprecated for these models after that date. Rationale: Kimi-K3 launched on July 27, 2026, so GPU cluster time shifts to the newer frontier model. Combined with the Agentic coding deprecation above, this leaves `kimik2.5` with no active scenario. The model is **fully retired after August 6, 2026**.

**Enacted on 2026-08-07** in [#2527](https://github.com/SemiAnalysisAI/InferenceX/pull/2527): 17 `kimik2.5` config keys were removed from the active master configs and archived under [`configs/deprecated/`](configs/deprecated/) as `nvidia-kimik2.5-8k1k-master.yaml` (10) and `amd-kimik2.5-8k1k-master.yaml` (7), and their 12 benchmark scripts were moved to the sibling `deprecated/` directories. `kimik2.5` now has **no active configuration in any master config** and is fully retired. The same PR archived `kimik2.5-int4-h100-vllm`, an agentic-coding key that #2493 left behind in `nvidia-master.yaml` after moving its script to `benchmarks/single_node/agentic/deprecated/`. It is now in `nvidia-kimik2.5-agentic-master.yaml` with its siblings. The SPEED-Bench acceptance-length script `benchmarks/single_node/speedbench/kimik2.5_fp4_b300_vllm.sh` is intentionally kept. Speedbench is driven by `speedbench-al.yml`, not the master configs, matching how #2493 treated MiniMax-M3.

### Tuesday, September 8, 2026

**Tuesday, September 8, 2026** is the last day for the **Single-turn 8k1k** scenario on **DeepSeek-V4-Pro 1.6T** (`dsv4`). The scenario is deprecated for this model after that date. **Agentic coding is unaffected and stays active for `dsv4`**, including its MTP and DSpark arms. The model is not retired: agentic coding becomes its only scenario, and it continues to run and publish.

| Model | Deprecated | Remains |
|---|---|---|
| DeepSeek-V4-Pro 1.6T (`dsv4`) | Single-turn 8k1k | Agentic coding, including the MTP and DSpark arms |

Rationale: `dsv4` carries the largest single-turn footprint in the repository. 45 active config keys use the 8k1k scenario, 32 in `configs/nvidia-master.yaml` and 13 in `configs/amd-master.yaml`, spanning H200, B200, B300, GB200, GB300, MI300X, MI325X, and MI355X across vLLM, SGLang, TensorRT-LLM, ATOM, Dynamo, and llm-d. That is a large share of every full sweep. AgentX trace replay is the scenario AI labs and the ML community ask about, and DeepSeek-V4-Pro's 19 agentic config keys are the part of `dsv4` that feeds the published North Star Pareto frontier. Retiring the fixed-sequence-length arm frees cluster hours for AgentX and for new frontier models such as Qwen3.8 2.4T without reducing what we publish for this model. Single-turn 8k1k stays active for the other models that still list it.

**Status: not yet enacted.** All 45 8k1k config keys still run. On enactment they are removed from the active master configs and archived under [`configs/deprecated/`](configs/deprecated/), with their benchmark scripts moved to the sibling `deprecated/` directories, matching how [#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493) and [#2527](https://github.com/SemiAnalysisAI/InferenceX/pull/2527) were carried out. The SPEED-Bench acceptance-length scripts for `dsv4` are intentionally kept. Speedbench is driven by `speedbench-al.yml`, not the master configs.

## Scenarios

| Scenario | ISL/OSL | Status |
|---|---|---|
| Agentic coding | Long Context, Multi Turn Realistic traffic trace replay with sub agents | Active. This is the trace-replay agentic-coding benchmark (see [`benchmarks/single_node/agentic/`](benchmarks/single_node/agentic/)). Going forward, new models will likely be onboarded with agentic coding only, and **with speculative decoding enabled only**. The non-spec-decode arm is not run or published (see [Deprecation Notice](#deprecation-notice)). |
| Single-turn 8k1k | 8192 / 1024 | Active. This is the primary fixed-sequence-length scenario. |
| Single-turn 1k1k | 1024 / 1024 | **Deprecated for all models** since 2026-07-17 ([#2263](https://github.com/SemiAnalysisAI/InferenceX/pull/2263)), to save GPU cluster time for higher-priority real-world agentic-coding benchmarks and new frontier models. Archived configs live in [`configs/deprecated/`](configs/deprecated/). |
| Single-turn 1k8k | 1024 / 8192 | **Deprecated for all models** since 2026-03-27 ([#911](https://github.com/SemiAnalysisAI/InferenceX/pull/911)), to save GPU cluster time for higher-priority real-world agentic-coding benchmarks and new frontier models. Configs were removed, not archived. |

## AgentX Guidelines

### E2E normalized interactivity and Pareto-frontier policy

E2E normalized interactivity is the primary user-side latency metric and default x-axis for AgentX trace-replay results. For every valid profiling request `i`, let `OSL_i` be its positive output sequence length and `E2EL_i` be its positive end-to-end latency, including both time to first token (TTFT) and generation time. First compute the per-request time per delivered output token:

`r_i = E2EL_i / OSL_i` (seconds per output token)

For percentile `q`, E2E normalized interactivity is:

`E2E normalized interactivity_q = 1 / percentile_q({r_i})` (output tokens/s/user)

The dashboard defaults to P90. Taking the percentile in seconds per output token before inverting preserves the slow-tail interpretation while presenting a higher-is-better rate. This is not the ratio of separately aggregated OSL and E2EL percentiles. At the request level, the metric can be understood approximately as:

`OSL / E2EL ≈ 1 / (TPOT + TTFT / OSL)`

Intuitively, it can be thought of as decode interactivity with a penalty for the queueing and prefill time during which the user receives no output tokens.

### Advantages

- It reduces the direct effect of variable AgentX output lengths on raw E2E latency: a request is not treated as slower merely because it correctly generated more output.
- It captures both TTFT and decode cadence in one user-facing delivery rate, discouraging configurations that improve decode-only interactivity while allowing TTFT or total task completion time to regress.
- It provides a higher-is-better optimization target in familiar tokens/s/user units while retaining the complete request path.

### Tradeoffs and limitations

- It is not conventional decode interactivity (`1 / TPOT`), so the absolute values are not directly comparable. E2E normalized interactivity is typically lower because it includes TTFT.
- TTFT is amortized over OSL, so short outputs receive a larger TTFT penalty than long outputs. The metric remains workload-dependent and comparisons require the same trace distribution and benchmark methodology.
- A single combined metric cannot show whether a regression came from TTFT or decode. The separate E2E latency, interactivity, and TTFT views remain diagnostic views.
- The metric requires persisted per-request traces with valid E2EL and OSL values. Runs without those traces cannot participate in the canonical frontier.

### North-star Pareto policy

For every supported y-axis metric and comparison group, the Pareto frontier computed against E2E normalized interactivity is the canonical AgentX **North Star** frontier. All other AgentX x-axis views are gated by that canonical winner set:

- The E2E normalized interactivity view displays the canonical frontier.
- The E2E latency, conventional interactivity, and TTFT views display the intersection of the canonical North Star frontier and the true Pareto frontier in the currently displayed coordinates.

Therefore, a point cannot appear on any AgentX Pareto frontier unless it is both a North Star winner and non-dominated on the selected chart. This prevents a configuration from entering a published frontier by optimizing one secondary latency metric at the expense of end-to-end user experience. Dominated points remain available in the unfiltered scatter view for diagnosis.

### Engine submission policy

Based on feedback from Tier 1 AI labs and the broader ML community about what they want to see in InferenceX AgentX, InferenceX uses an explicit model-to-framework mapping. Labs have reported that proprietary or hardware-specific engines such as TensorRT-LLM and ATOM do not always provide every feature their AgentX workloads require.

The native/upstream engines in the table below are the first-class engines for each model. If a provider supports both the native/upstream vLLM engine and native/upstream SGLang engine as first-class LLM engines, it must first submit the engine or engines assigned by this mapping before submitting additional non-vLLM/SGLang engines such as ATOM, TensorRT-LLM, or TokenSpeed. Multiple additional non-vLLM/SGLang engines may be submitted.

There are two exceptions to this ordering guideline:

1. Brand-new hardware SKUs, such as MI455X UALoE72, VR200 NVL72, Rubin NVL8, TPUv8t, and TPUv8i, may use a hardware-specific engine first for initial support. The corresponding native/upstream vLLM or SGLang submission is expected to follow shortly afterward.
2. For a new model architecture, a provider may use another engine first if it cannot support the mapped native/upstream vLLM or SGLang engine as a first-class engine and can articulate to core maintainers a fundamental, first-principles reason why the mapped framework does not yet support the hardware-model combination.

InferenceX supports the maintainers of both SGLang and vLLM and reflects feedback from AI labs and the ML community that want to see performance from both frameworks. Among models assigned to one primary framework, the mapping splits assignments evenly between vLLM and SGLang. Models mapped to both provide shared coverage. This ensures that InferenceX tests both frameworks equally without favoring one over the other.

The table also records both the agreed plan-of-record (PoR) draft-model mapping and proposals that still require partner alignment.

| Model | Primary native/upstream engines | Agreed draft model(s) (PoR) | Proposed draft model(s) pending partner alignment | Additional engines |
|---|---|---|---|---|
| DeepSeek-V4-Pro 1.6T (`dsv4`) | native/upstream vLLM engine and native/upstream SGLang engine | native MTP (Single-turn 8k1k); native DSpark heads `deepseek-ai/DeepSeek-V4-Pro-0813` (Agentic coding only, under the same synthetic-acceptance methodology as Kimi-K3's DSpark PoR) | None | Additional non-vLLM/SGLang engines under the ordering guideline and exceptions above |
| Kimi-K3 (`kimik3`) | native/upstream vLLM engine | `Inferact/Kimi-K3-DSpark` | None | Additional non-vLLM/SGLang engines under the ordering guideline and exceptions above |
| MiniMax-M3 (`minimaxm3`) | native/upstream vLLM engine | `Inferact/MiniMax-M3-EAGLE3` and/or `Inferact/MiniMax-M3-EAGLE3-GQA` | None | Additional non-vLLM/SGLang engines under the ordering guideline and exceptions above |
| GLM-5.2 (`glm5.2`) | native/upstream SGLang engine | native MTP | None | Additional non-vLLM/SGLang engines under the ordering guideline and exceptions above |
| Qwen3.5-397B-A17B (`qwen3.5`) | native/upstream SGLang engine | native MTP | None | Additional non-vLLM/SGLang engines under the ordering guideline and exceptions above |

### KV cache offloading policy

To align with the design principle of shipping quickly by limiting scope, the initial AgentX policy permits only CPU DRAM KV cache offloading, and its use is optional. Supported approaches include the vLLM Connector, LMCache, SGLang HiCache, Mooncake CPU DRAM Connector, Dynamo KVBM, CPU DRAM P2P pooling, and similar CPU-memory mechanisms. Vendors may enable or disable CPU KV cache offloading for each submission at their discretion, including disabling it when that produces better Pareto points.

The amount of CPU DDR5 used for KV cache offloading must be proportional to the fraction of a server's GPUs used by the serving configuration:

`allowed CPU DRAM = per-server baseline CPU DRAM × (GPUs used by the configuration / total GPUs in the server)`

This rule applies independently to each server.

| CPU DRAM capacity class | Example SKUs | Per-server baseline and limit |
|---|---|---|
| No standardized CPU DRAM capacity | HGX B200, HGX B300, MI355X chassis | At most 3 TB per server. A configuration using 4 of 8 GPUs may therefore use at most 1.5 TB of CPU DRAM for KV cache offloading. |
| Standardized CPU DRAM capacity | TPUv7, GB200 NVL72, GB300 NVL72 | The SKU's standard installed CPU DRAM capacity is the baseline, with no additional per-server hard cap. The proportional-GPU rule still applies. |

The 3 TB limit prevents an unrealistic memory-capacity race in which hardware vendors ask CSPs and OEMs to install the maximum number of high-capacity DIMMs, potentially reaching 6 TB per server. Total cost of ownership (TCO) per accelerator chip will be normalized by the cost of the server's total CPU DDR5 capacity.

Other offloading tiers, including NVMe KV cache offloading, are outside the initial scope and may be introduced after the InferenceX v3 release. NVMe KV cache offloading is tentatively targeted as a fast follow-up in InferenceX v3.5 or as part of InferenceX v4.

## Model support matrix

| Model architecture class | Prefix | Date added | Active scenarios | Deprecated scenarios |
|---|---|---|---|---|
| Qwen3.8 2.4T | `qwen3.8` | TBD | Agentic coding | |
| Kimi-K3 | `kimik3` | 2026-07-27 ([#2391](https://github.com/SemiAnalysisAI/InferenceX/pull/2391)) | Agentic coding (DSpark only) | Agentic coding non-DSpark arm (deprecated from day 0) |
| GLM-5.2 | `glm5.2` | 2026-07-18 ([#2268](https://github.com/SemiAnalysisAI/InferenceX/pull/2268)) | Agentic coding (the non-MTP arm still runs while the MTP-only transition remains pending, as explained in the Deprecation Notice) | |
| MiniMax-M3 | `minimaxm3` | 2026-06-12 ([#1724](https://github.com/SemiAnalysisAI/InferenceX/pull/1724)) | Agentic coding | Single-turn 1k1k, Single-turn 8k1k (removed 2026-08-04, [#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493)) |
| DeepSeek-V4-Pro | `dsv4` | 2026-04-24 ([#1130](https://github.com/SemiAnalysisAI/InferenceX/pull/1130)) | Single-turn 8k1k (last day 2026-09-08, see [Deprecation Notice](#deprecation-notice)), Agentic coding (the non-MTP arm still runs while the MTP-only transition remains pending, as explained in the Deprecation Notice) | Single-turn 1k1k |
| GLM-5 / GLM-5.1 | `glm5`, `glm5.1` | 2026-03-06 ([#762](https://github.com/SemiAnalysisAI/InferenceX/pull/762)), with GLM-5.1 added 2026-04-21 ([#1098](https://github.com/SemiAnalysisAI/InferenceX/pull/1098)) | None (retired 2026-07-18, [#2276](https://github.com/SemiAnalysisAI/InferenceX/pull/2276)) | Single-turn 1k1k, Single-turn 1k8k (GLM-5 only), Single-turn 8k1k |
| MiniMax-M2.5/2.7 | `minimaxm2.5` | 2026-02-18 ([#755](https://github.com/SemiAnalysisAI/InferenceX/pull/755)) | None (retired 2026-06-20, [#1874](https://github.com/SemiAnalysisAI/InferenceX/pull/1874)) | Single-turn 1k1k, Single-turn 1k8k, Single-turn 8k1k |
| Kimi-K2.5/2.6/2.7-Code | `kimik2.5` | 2026-02-17 ([#734](https://github.com/SemiAnalysisAI/InferenceX/pull/734)) | None (fully retired 2026-08-07, [#2527](https://github.com/SemiAnalysisAI/InferenceX/pull/2527)) | Single-turn 1k1k, Single-turn 1k8k, Agentic coding (removed 2026-08-04, [#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493)), Single-turn 8k1k (removed 2026-08-07, [#2527](https://github.com/SemiAnalysisAI/InferenceX/pull/2527)) |
| Qwen3.5-397B-A17B | `qwen3.5` | 2026-02-16 ([#704](https://github.com/SemiAnalysisAI/InferenceX/pull/704)) | Single-turn 8k1k and Agentic coding, both limited to fp8/fp4 | Single-turn 1k1k, Single-turn 1k8k, all bf16 recipes (removed 2026-08-04, [#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493)) |
| gpt-oss-120b | `gptoss` | 2025-09-09 | None (retired 2026-07-06, [#2101](https://github.com/SemiAnalysisAI/InferenceX/pull/2101)) | Single-turn 1k1k, Single-turn 1k8k, Single-turn 8k1k |
| DeepSeek-R1-0528 | `dsr1` | 2025-08-13 | Single-turn 8k1k | Single-turn 1k1k, Single-turn 1k8k |
| Llama-3.1-70B-Instruct | `llama70b` | 2025-08-12 | None (retired 2025-10-29, [#149](https://github.com/SemiAnalysisAI/InferenceX/pull/149)) | Single-turn 1k1k, Single-turn 1k8k, Single-turn 8k1k [^1] |

[^1]: `llama70b` predates the master-config system. Its configs were deleted on retirement rather than archived in `configs/deprecated/`. It first shipped as workflow templates in the initial repo import (2025-08-12).

## Notes

- The `Prefix` column is the canonical `model-prefix` used in `configs/*-master.yaml` and by `generate_sweep_configs.py --model-prefix`.
- "Retired" means the model no longer has any active scenario. Retired models' configs (except `llama70b`) are archived under [`configs/deprecated/`](configs/deprecated/).
- Deprecating a precision (e.g. Qwen3.5 bf16) or one arm of an A/B pair (e.g. non-MTP) narrows a model's recipe coverage without retiring the model. The model stays listed as active as long as one scenario still runs.
- `dsr1` began as the DeepSeek-V3 workflow templates in the initial repo import and was switched to DeepSeek-R1 benchmarking on 2025-08-13 (renamed `dsv3` → `dsr1` on 2025-08-20).
- Adding a model? Follow [Add a model + hardware recipe](docs/configuration-procedures.md#add-a-model--hardware-recipe) and add a row here (and in [`MODELS_zh.md`](MODELS_zh.md)) in the same PR.
