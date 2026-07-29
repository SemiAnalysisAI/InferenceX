# Models

English | [中文](MODELS_zh.md)

This document tracks every model benchmarked by InferenceX-e2e: when it was added, which benchmark scenarios are currently active for it, and which scenarios are deprecated. Results for active scenarios are published to <https://inferencex.com/>.

## Deprecation Notice

- **Thursday, August 6, 2026** is the last day for the **Single-turn 8k1k** scenario on **Kimi-K2.5/2.6/2.7-Code** (`kimik2.5`); the scenario is deprecated for these models after that date. Rationale: the Agentic coding (AgentX) scenario already covers them with real-world traffic, and Kimi-K3 launched on July 27, 2026, so GPU cluster time shifts to the newer frontier model.

## Scenarios

| Scenario | ISL/OSL | Status |
|---|---|---|
| Agentic coding | Long Context, Multi Turn Realistic traffic trace replay with sub agents | Active — trace-replay agentic-coding benchmark (see [`benchmarks/agentic/`](benchmarks/agentic/)). Going forward, new models will likely be onboarded with agentic coding only. |
| Single-turn 8k1k | 8192 / 1024 | Active — the primary fixed-sequence-length scenario. |
| Single-turn 1k1k | 1024 / 1024 | **Deprecated for all models** since 2026-07-17 ([#2263](https://github.com/SemiAnalysisAI/InferenceX/pull/2263)), to save GPU cluster time for higher-priority real-world agentic-coding benchmarks and new frontier models. Archived configs live in [`configs/deprecated/`](configs/deprecated/). |
| Single-turn 1k8k | 1024 / 8192 | **Deprecated for all models** since 2026-03-27 ([#911](https://github.com/SemiAnalysisAI/InferenceX/pull/911)), to save GPU cluster time for higher-priority real-world agentic-coding benchmarks and new frontier models. Configs were removed, not archived. |

## Model support matrix

| Model architecture class | Prefix | Date added | Active scenarios | Deprecated scenarios |
|---|---|---|---|---|
| Qwen3.8 2.4T | `qwen3.8` | TBD | Agentic coding | |
| Kimi-K3 | `kimik3` | 2026-07-27 ([#2391](https://github.com/SemiAnalysisAI/InferenceX/pull/2391)) | Agentic coding | |
| GLM-5.2 | `glm5.2` | 2026-07-18 ([#2268](https://github.com/SemiAnalysisAI/InferenceX/pull/2268)) | Agentic coding | |
| MiniMax-M3 | `minimaxm3` | 2026-06-12 ([#1724](https://github.com/SemiAnalysisAI/InferenceX/pull/1724)) | Single-turn 8k1k, Agentic coding | Single-turn 1k1k |
| DeepSeek-V4-Pro | `dsv4` | 2026-04-24 ([#1130](https://github.com/SemiAnalysisAI/InferenceX/pull/1130)) | Single-turn 8k1k, Agentic coding | Single-turn 1k1k |
| GLM-5 / GLM-5.1 | `glm5`, `glm5.1` | 2026-03-06 ([#762](https://github.com/SemiAnalysisAI/InferenceX/pull/762)); GLM-5.1 added 2026-04-21 ([#1098](https://github.com/SemiAnalysisAI/InferenceX/pull/1098)) | — (retired 2026-07-18, [#2276](https://github.com/SemiAnalysisAI/InferenceX/pull/2276)) | Single-turn 1k1k, Single-turn 1k8k (GLM-5 only), Single-turn 8k1k |
| MiniMax-M2.5/2.7 | `minimaxm2.5` | 2026-02-18 ([#755](https://github.com/SemiAnalysisAI/InferenceX/pull/755)) | — (retired 2026-06-20, [#1874](https://github.com/SemiAnalysisAI/InferenceX/pull/1874)) | Single-turn 1k1k, Single-turn 1k8k, Single-turn 8k1k |
| Kimi-K2.5/2.6/2.7-Code | `kimik2.5` | 2026-02-17 ([#734](https://github.com/SemiAnalysisAI/InferenceX/pull/734)) | Single-turn 8k1k, Agentic coding | Single-turn 1k1k, Single-turn 1k8k |
| Qwen3.5-397B-A17B | `qwen3.5` | 2026-02-16 ([#704](https://github.com/SemiAnalysisAI/InferenceX/pull/704)) | Single-turn 8k1k, Agentic coding | Single-turn 1k1k, Single-turn 1k8k |
| gpt-oss-120b | `gptoss` | 2025-09-09 | — (retired 2026-07-06, [#2101](https://github.com/SemiAnalysisAI/InferenceX/pull/2101)) | Single-turn 1k1k, Single-turn 1k8k, Single-turn 8k1k |
| DeepSeek-R1-0528 | `dsr1` | 2025-08-13 | Single-turn 8k1k | Single-turn 1k1k, Single-turn 1k8k |
| Llama-3.1-70B-Instruct | `llama70b` | 2025-08-12 | — (retired 2025-10-29, [#149](https://github.com/SemiAnalysisAI/InferenceX/pull/149)) | Single-turn 1k1k, Single-turn 1k8k, Single-turn 8k1k [^1] |

[^1]: `llama70b` predates the master-config system; its configs were deleted on retirement rather than archived in `configs/deprecated/`. It first shipped as workflow templates in the initial repo import (2025-08-12).

## Notes

- The `Prefix` column is the canonical `model-prefix` used in `configs/*-master.yaml` and by `generate_sweep_configs.py --model-prefix`.
- "Retired" means the model no longer has any active scenario. Retired models' configs (except `llama70b`) are archived under [`configs/deprecated/`](configs/deprecated/).
- `dsr1` began as the DeepSeek-V3 workflow templates in the initial repo import and was switched to DeepSeek-R1 benchmarking on 2025-08-13 (renamed `dsv3` → `dsr1` on 2025-08-20).
- Adding a model? Follow "Adding a benchmark configuration" in [`AGENTS.md`](AGENTS.md) and add a row here (and in [`MODELS_zh.md`](MODELS_zh.md)) in the same PR.
