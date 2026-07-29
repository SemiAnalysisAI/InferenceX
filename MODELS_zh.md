# 模型列表

[English](MODELS.md) | 中文

本文档记录 InferenceX-e2e 基准测试覆盖的所有模型：加入日期、当前启用的基准测试场景，以及已弃用的场景。启用场景的结果发布于 <https://inferencex.com/>。

## 弃用公告

InferenceX-e2e 运行在数量固定且有限的 GPU 资源池上，并由一支小型团队维护。每保留一个场景、精度或配方变体，都会占用集群机时与维护人力，而这些资源本可投入到新的前沿模型上。以下弃用即为释放这部分产能。若某项弃用移除的是 A/B 对照中的一个分支，我们保留并发布在帕累托前沿（Pareto frontier）上更优的那个分支。

### 2026 年 8 月 3 日（星期一）

**2026 年 8 月 3 日（星期一）**为下列场景、精度与配方变体的最后运行日，此后即告弃用。

场景与精度下线：

| 模型 | 弃用内容 | 保留内容 |
|---|---|---|
| MiniMax-M3（`minimaxm3`） | 单轮 8k1k | 智能体编码 |
| Kimi-K2.5/2.6/2.7-Code（`kimik2.5`） | 智能体编码 | 单轮 8k1k，保留至 2026 年 8 月 6 日（见下文） |
| Qwen3.5-397B-A17B（`qwen3.5`） | 全部 **bf16** 配方，涵盖所有场景，NVIDIA 与 AMD 平台均在内 | fp8 与 fp4 配方 |

投机解码（speculative decoding）A/B 对照下线 —— 下列每一组对照中，启用投机解码的分支都处于更优的帕累托前沿，因此我们停止运行非投机解码分支，仅发布投机解码分支：

| 模型 | 弃用分支 | 发布分支 |
|---|---|---|
| DeepSeek-V4-Pro 1.6T（`dsv4`） | 智能体编码，非 MTP | 智能体编码，MTP |
| Qwen3.5-397B-A17B（`qwen3.5`） | 智能体编码，非 MTP | 智能体编码，MTP |
| MiniMax-M3（`minimaxm3`） | 智能体编码，非 EAGLE3 | 智能体编码，EAGLE3 |
| GLM-5.2（`glm5.2`） | 智能体编码，非 MTP | 智能体编码，MTP |
| Kimi-K3（`kimik3`） | 智能体编码，非 DSpark —— 自第 0 天（day 0）起即弃用 | 智能体编码，DSpark |

**今后我们不再以 A/B 对照的方式基准测试「非投机解码 vs 投机解码」。**当初保留非投机解码分支，是把它当作中立基线：那时接受长度（AL）完全取决于提交方草稿头（draft head）的实际水平，导致各家投机解码数据之间无法横向比较。这一问题现已解决：[`golden_al_distribution/`](golden_al_distribution/) 为每个模型、thinking 模式与草稿长度各提交了一条黄金 AL 曲线，均在 SPEED-Bench `coding` 类别上测得；AgentX 通过合成接受（synthetic acceptance）将所有提交锁定到该曲线（vLLM 用 `synthetic_acceptance_length`，SGLang 用 `SGLANG_SIMULATE_ACC_LEN`，TensorRT-LLM 用 `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`，等等）。既然已有公平且与引擎无关的接受目标，投机解码结果本身即可直接横向比较，单独保留一条非投机解码赛道已属冗余。因此，智能体编码配方一律仅在启用投机解码的条件下运行与发布 —— 具体为 MTP、EAGLE/EAGLE3、DSpark，或该模型自带的任何草稿方法 —— 非投机解码分支既不运行也不发布。新模型自第 0 天起即按此方式接入，Kimi-K3 即为一例。

### 2026 年 8 月 6 日（星期四）

**2026 年 8 月 6 日（星期四）**为 **Kimi-K2.5/2.6/2.7-Code**（`kimik2.5`）**单轮 8k1k** 场景的最后运行日，此后该场景对这些模型弃用。原因：Kimi-K3 已于 2026 年 7 月 27 日发布，GPU 集群时间将转向更新的前沿模型。叠加上文的智能体编码弃用，`kimik2.5` 将不再有任何启用场景 —— 该模型将于 **2026 年 8 月 6 日后完全退役**。

## 场景

| 场景 | ISL/OSL | 状态 |
|---|---|---|
| 智能体编码（agentic coding） | 长上下文、多轮真实流量的轨迹回放，含子智能体（sub agents） | 启用 — 基于轨迹回放的智能体编码基准测试（见 [`benchmarks/agentic/`](benchmarks/agentic/)）。今后新模型预计将仅以智能体编码场景接入，且**仅在启用投机解码的条件下**运行 —— 非投机解码分支不运行也不发布（见[弃用公告](#弃用公告)）。 |
| 单轮 8k1k | 8192 / 1024 | 启用 — 当前主要的固定序列长度（fixed-seq-len）场景。 |
| 单轮 1k1k | 1024 / 1024 | **对所有模型均已弃用**，自 2026-07-17 起（[#2263](https://github.com/SemiAnalysisAI/InferenceX/pull/2263)），以便将 GPU 集群时间留给优先级更高的真实场景智能体编码基准测试与新的前沿模型。归档配置位于 [`configs/deprecated/`](configs/deprecated/)。 |
| 单轮 1k8k | 1024 / 8192 | **对所有模型均已弃用**，自 2026-03-27 起（[#911](https://github.com/SemiAnalysisAI/InferenceX/pull/911)），以便将 GPU 集群时间留给优先级更高的真实场景智能体编码基准测试与新的前沿模型。相关配置已删除，未归档。 |

## 模型支持矩阵

| 模型架构类别 | 前缀 | 加入日期 | 启用场景 | 已弃用场景 |
|---|---|---|---|---|
| Qwen3.8 2.4T | `qwen3.8` | 待定 | 智能体编码 | |
| Kimi-K3 | `kimik3` | 2026-07-27 ([#2391](https://github.com/SemiAnalysisAI/InferenceX/pull/2391)) | 智能体编码（仅 DSpark） | 智能体编码非 DSpark 分支（自第 0 天起弃用） |
| GLM-5.2 | `glm5.2` | 2026-07-18（[#2268](https://github.com/SemiAnalysisAI/InferenceX/pull/2268)） | 智能体编码（自 2026-08-03 起仅 MTP） | |
| MiniMax-M3 | `minimaxm3` | 2026-06-12（[#1724](https://github.com/SemiAnalysisAI/InferenceX/pull/1724)） | 单轮 8k1k（至 2026-08-03）、智能体编码（自 2026-08-03 起仅 EAGLE3） | 单轮 1k1k |
| DeepSeek-V4-Pro | `dsv4` | 2026-04-24（[#1130](https://github.com/SemiAnalysisAI/InferenceX/pull/1130)） | 单轮 8k1k、智能体编码（自 2026-08-03 起仅 MTP） | 单轮 1k1k |
| GLM-5 / GLM-5.1 | `glm5`、`glm5.1` | 2026-03-06（[#762](https://github.com/SemiAnalysisAI/InferenceX/pull/762)）；GLM-5.1 于 2026-04-21 加入（[#1098](https://github.com/SemiAnalysisAI/InferenceX/pull/1098)） | —（2026-07-18 退役，[#2276](https://github.com/SemiAnalysisAI/InferenceX/pull/2276)） | 单轮 1k1k、单轮 1k8k（仅 GLM-5）、单轮 8k1k |
| MiniMax-M2.5/2.7 | `minimaxm2.5` | 2026-02-18（[#755](https://github.com/SemiAnalysisAI/InferenceX/pull/755)） | —（2026-06-20 退役，[#1874](https://github.com/SemiAnalysisAI/InferenceX/pull/1874)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k |
| Kimi-K2.5/2.6/2.7-Code | `kimik2.5` | 2026-02-17（[#734](https://github.com/SemiAnalysisAI/InferenceX/pull/734)） | 单轮 8k1k（至 2026-08-06）、智能体编码（至 2026-08-03）—— 2026-08-06 后完全退役 | 单轮 1k1k、单轮 1k8k |
| Qwen3.5-397B-A17B | `qwen3.5` | 2026-02-16（[#704](https://github.com/SemiAnalysisAI/InferenceX/pull/704)） | 单轮 8k1k、智能体编码（自 2026-08-03 起仅 MTP）；仅 fp8/fp4 —— bf16 配方于 2026-08-03 下线 | 单轮 1k1k、单轮 1k8k |
| gpt-oss-120b | `gptoss` | 2025-09-09 | —（2026-07-06 退役，[#2101](https://github.com/SemiAnalysisAI/InferenceX/pull/2101)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k |
| DeepSeek-R1-0528 | `dsr1` | 2025-08-13 | 单轮 8k1k | 单轮 1k1k、单轮 1k8k |
| Llama-3.1-70B-Instruct | `llama70b` | 2025-08-12 | —（2025-10-29 退役，[#149](https://github.com/SemiAnalysisAI/InferenceX/pull/149)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k [^1] |

[^1]: `llama70b` 早于 master 配置体系；退役时其配置被直接删除，未归档到 `configs/deprecated/`。该模型最初以 workflow 模板形式随仓库首次导入（2025-08-12）。

## 说明

- 「前缀」列为 `configs/*-master.yaml` 中的规范 `model-prefix`，同时用于 `generate_sweep_configs.py --model-prefix`。
- 「退役」指该模型已无任何启用场景。退役模型的配置（`llama70b` 除外）归档于 [`configs/deprecated/`](configs/deprecated/)。
- 弃用某一精度（如 Qwen3.5 bf16）或 A/B 对照中的某一分支（如非 MTP），只是收窄该模型的配方覆盖范围，并不等于模型退役；只要仍有一个场景在运行，该模型即继续列为启用状态。
- `dsr1` 最初以 DeepSeek-V3 workflow 模板的形式随仓库首次导入，2025-08-13 切换为 DeepSeek-R1 基准测试（2025-08-20 将 `dsv3` 重命名为 `dsr1`）。
- 新增模型时，请按 [`AGENTS.md`](AGENTS.md) 中「Adding a benchmark configuration」的流程操作，并在同一 PR 中同时更新本文件与 [`MODELS.md`](MODELS.md) 的表格。
