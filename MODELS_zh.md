# 模型列表

[English](MODELS.md) | 中文

本文档记录 InferenceX-e2e 基准测试覆盖的所有模型：加入日期、当前启用的基准测试场景，以及已弃用的场景。启用场景的结果发布于 <https://inferencex.com/>。

## 弃用公告

- **2026 年 8 月 6 日（星期四）**为 **Kimi-K2.5/2.6/2.7-Code**（`kimik2.5`）**单轮 8k1k** 场景的最后运行日，此后该场景对这些模型弃用。原因：智能体编码（AgentX）场景已以真实流量覆盖这些模型，且 Kimi-K3 已于 2026 年 7 月 27 日发布，GPU 集群时间将转向更新的前沿模型。

## 场景

| 场景 | ISL/OSL | 状态 |
|---|---|---|
| 智能体编码（agentic coding） | 长上下文、多轮真实流量的轨迹回放，含子智能体（sub agents） | 启用 — 基于轨迹回放的智能体编码基准测试（见 [`benchmarks/agentic/`](benchmarks/agentic/)）。今后新模型预计将仅以智能体编码场景接入。 |
| 单轮 8k1k | 8192 / 1024 | 启用 — 当前主要的固定序列长度（fixed-seq-len）场景。 |
| 单轮 1k1k | 1024 / 1024 | **对所有模型均已弃用**，自 2026-07-17 起（[#2263](https://github.com/SemiAnalysisAI/InferenceX/pull/2263)），以便将 GPU 集群时间留给优先级更高的真实场景智能体编码基准测试与新的前沿模型。归档配置位于 [`configs/deprecated/`](configs/deprecated/)。 |
| 单轮 1k8k | 1024 / 8192 | **对所有模型均已弃用**，自 2026-03-27 起（[#911](https://github.com/SemiAnalysisAI/InferenceX/pull/911)），以便将 GPU 集群时间留给优先级更高的真实场景智能体编码基准测试与新的前沿模型。相关配置已删除，未归档。 |

## 模型支持矩阵

| 模型架构类别 | 前缀 | 加入日期 | 启用场景 | 已弃用场景 |
|---|---|---|---|---|
| Qwen3.8 2.4T | `qwen3.8` | 待定 | 智能体编码 | |
| Kimi-K3 | `kimik3` | 2026-07-27 ([#2391](https://github.com/SemiAnalysisAI/InferenceX/pull/2391)) | 智能体编码 | |
| GLM-5.2 | `glm5.2` | 2026-07-18（[#2268](https://github.com/SemiAnalysisAI/InferenceX/pull/2268)） | 智能体编码 | |
| MiniMax-M3 | `minimaxm3` | 2026-06-12（[#1724](https://github.com/SemiAnalysisAI/InferenceX/pull/1724)） | 单轮 8k1k、智能体编码 | 单轮 1k1k |
| DeepSeek-V4-Pro | `dsv4` | 2026-04-24（[#1130](https://github.com/SemiAnalysisAI/InferenceX/pull/1130)） | 单轮 8k1k、智能体编码 | 单轮 1k1k |
| GLM-5 / GLM-5.1 | `glm5`、`glm5.1` | 2026-03-06（[#762](https://github.com/SemiAnalysisAI/InferenceX/pull/762)）；GLM-5.1 于 2026-04-21 加入（[#1098](https://github.com/SemiAnalysisAI/InferenceX/pull/1098)） | —（2026-07-18 退役，[#2276](https://github.com/SemiAnalysisAI/InferenceX/pull/2276)） | 单轮 1k1k、单轮 1k8k（仅 GLM-5）、单轮 8k1k |
| MiniMax-M2.5/2.7 | `minimaxm2.5` | 2026-02-18（[#755](https://github.com/SemiAnalysisAI/InferenceX/pull/755)） | —（2026-06-20 退役，[#1874](https://github.com/SemiAnalysisAI/InferenceX/pull/1874)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k |
| Kimi-K2.5/2.6/2.7-Code | `kimik2.5` | 2026-02-17（[#734](https://github.com/SemiAnalysisAI/InferenceX/pull/734)） | 单轮 8k1k、智能体编码 | 单轮 1k1k、单轮 1k8k |
| Qwen3.5-397B-A17B | `qwen3.5` | 2026-02-16（[#704](https://github.com/SemiAnalysisAI/InferenceX/pull/704)） | 单轮 8k1k、智能体编码 | 单轮 1k1k、单轮 1k8k |
| gpt-oss-120b | `gptoss` | 2025-09-09 | —（2026-07-06 退役，[#2101](https://github.com/SemiAnalysisAI/InferenceX/pull/2101)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k |
| DeepSeek-R1-0528 | `dsr1` | 2025-08-13 | 单轮 8k1k | 单轮 1k1k、单轮 1k8k |
| Llama-3.1-70B-Instruct | `llama70b` | 2025-08-12 | —（2025-10-29 退役，[#149](https://github.com/SemiAnalysisAI/InferenceX/pull/149)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k [^1] |

[^1]: `llama70b` 早于 master 配置体系；退役时其配置被直接删除，未归档到 `configs/deprecated/`。该模型最初以 workflow 模板形式随仓库首次导入（2025-08-12）。

## 说明

- 「前缀」列为 `configs/*-master.yaml` 中的规范 `model-prefix`，同时用于 `generate_sweep_configs.py --model-prefix`。
- 「退役」指该模型已无任何启用场景。退役模型的配置（`llama70b` 除外）归档于 [`configs/deprecated/`](configs/deprecated/)。
- `dsr1` 最初以 DeepSeek-V3 workflow 模板的形式随仓库首次导入，2025-08-13 切换为 DeepSeek-R1 基准测试（2025-08-20 将 `dsv3` 重命名为 `dsr1`）。
- 新增模型时，请按 [`AGENTS.md`](AGENTS.md) 中「Adding a benchmark configuration」的流程操作，并在同一 PR 中同时更新本文件与 [`MODELS.md`](MODELS.md) 的表格。
