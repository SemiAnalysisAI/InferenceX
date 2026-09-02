# 模型列表

[English](MODELS.md) | 中文

本文档记录 InferenceX-e2e 基准测试覆盖的所有模型：加入日期、当前启用的基准测试场景，以及已弃用的场景。启用场景的结果发布于 <https://inferencex.com/>。

## 弃用公告

InferenceX-e2e 运行在数量固定且有限的 GPU 资源池上，并由一支小型团队维护。每保留一个场景、精度或配方变体，都会占用集群机时与维护人力，而这些资源本可投入到新的前沿模型上。以下弃用即为释放这部分产能。若某项弃用移除的是 A/B 对照中的一个分支，我们保留并发布在帕累托前沿（Pareto frontier）上更优的那个分支。

### 2026 年 8 月 3 日（星期一）

**2026 年 8 月 3 日（星期一）**为下列场景、精度与配方变体的最后运行日，此后即告弃用。

**已于 2026 年 8 月 4 日部分执行**（[#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493)）：第一张表中的场景与精度下线已完成。此次执行从启用的 master 配置中移除 54 个配置键并归档至 [`configs/deprecated/`](configs/deprecated/)，其基准测试脚本亦移入同级 `deprecated/` 目录。第二张表中的投机解码 A/B 下线**尚未执行**，详见该表下方说明。

场景与精度下线：

| 模型 | 弃用内容 | 保留内容 |
|---|---|---|
| MiniMax-M3（`minimaxm3`） | 单轮 8k1k | 智能体编码 |
| Kimi-K2.5/2.6/2.7-Code（`kimik2.5`） | 智能体编码 | 无。单轮 8k1k 运行至 2026 年 8 月 6 日，已于 2026-08-07 退役（见下文） |
| Qwen3.5-397B-A17B（`qwen3.5`） | 全部 **bf16** 配方，涵盖所有场景，NVIDIA 与 AMD 平台均在内 | fp8 与 fp4 配方 |

投机解码（speculative decoding）A/B 对照下线适用于以下各组。每组中，启用投机解码的分支都处于更优的帕累托前沿，因此我们停止运行非投机解码分支，仅发布投机解码分支：

| 模型 | 弃用分支 | 发布分支 |
|---|---|---|
| DeepSeek-V4-Pro 1.6T（`dsv4`） | 智能体编码，非 MTP | 智能体编码，MTP |
| Qwen3.5-397B-A17B（`qwen3.5`） | 智能体编码，非 MTP | 智能体编码，MTP |
| MiniMax-M3（`minimaxm3`） | 智能体编码，非 EAGLE3 | 智能体编码，EAGLE3 |
| GLM-5.2（`glm5.2`） | 智能体编码，非 MTP | 智能体编码，MTP |
| Kimi-K3（`kimik3`） | 智能体编码，非 DSpark（自第 0 天起即弃用） | 智能体编码，DSpark |

**状态：尚未执行。** 上表中所有非投机解码智能体分支仍在运行。若此刻移除，MiniMax-M3 与 GLM-5.2 将没有任何启用配置，因为其 EAGLE3 与 MTP 智能体分支尚未合入。这样还会使 DeepSeek-V4-Pro 与 Qwen3.5 在 AMD 及 SGLang 上的智能体覆盖全部消失，这些平台/引擎组合均无对应的 MTP 分支。待替代分支就绪后再执行本轮下线。

**今后我们不再以 A/B 对照的方式基准测试「非投机解码 vs 投机解码」。** 当初保留非投机解码分支，是把它当作中立基线。那时接受长度（AL）完全取决于提交方草稿头（draft head）的实际水平，导致各家投机解码数据之间无法横向比较。这一问题现已解决。[`golden_al_distribution/`](golden_al_distribution/) 为每个模型、thinking 模式与草稿长度各提交了一条黄金 AL 曲线，均在 SPEED-Bench `coding` 类别上测得。AgentX 通过合成接受（synthetic acceptance）将所有提交锁定到该曲线（vLLM 用 `synthetic_acceptance_length`，SGLang 用 `SGLANG_SIMULATE_ACC_LEN`，TensorRT-LLM 用 `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`，等等）。既然已有公平且与引擎无关的接受目标，投机解码结果本身即可直接横向比较，单独保留一条非投机解码赛道已属冗余。因此，智能体编码配方一律仅在启用投机解码的条件下运行与发布，具体为 MTP、EAGLE/EAGLE3、DSpark，或该模型自带的任何草稿方法。非投机解码分支既不运行也不发布。新模型自第 0 天起即按此方式接入，Kimi-K3 即为一例。

### 2026 年 8 月 6 日（星期四）

**2026 年 8 月 6 日（星期四）**为 **Kimi-K2.5/2.6/2.7-Code**（`kimik2.5`）**单轮 8k1k** 场景的最后运行日，此后该场景对这些模型弃用。原因：Kimi-K3 已于 2026 年 7 月 27 日发布，GPU 集群时间将转向更新的前沿模型。叠加上文的智能体编码弃用，`kimik2.5` 将不再有任何启用场景。该模型将于 **2026 年 8 月 6 日后完全退役**。

**已于 2026-08-07 执行**（[#2527](https://github.com/SemiAnalysisAI/InferenceX/pull/2527)）：从启用的主配置中移除 17 个 `kimik2.5` 配置项，归档至 [`configs/deprecated/`](configs/deprecated/)，分别为 `nvidia-kimik2.5-8k1k-master.yaml`（10 个）与 `amd-kimik2.5-8k1k-master.yaml`（7 个）。对应的 12 个基准测试脚本移入同级 `deprecated/` 目录。此后 `kimik2.5` 在所有主配置中**均无启用配置**，正式完全退役。同一 PR 还归档了 `kimik2.5-int4-h100-vllm`。#2493 将其脚本移入 `benchmarks/single_node/agentic/deprecated/` 时，该智能体编码配置项被遗留在 `nvidia-master.yaml` 中，现已与同类项一并归入 `nvidia-kimik2.5-agentic-master.yaml`。SPEED-Bench 接受长度脚本 `benchmarks/single_node/speedbench/kimik2.5_fp4_b300_vllm.sh` 予以保留。Speedbench 由 `speedbench-al.yml` 驱动，不经过主配置，与 #2493 处理 MiniMax-M3 的方式一致。

## 场景

| 场景 | ISL/OSL | 状态 |
|---|---|---|
| 智能体编码（agentic coding） | 长上下文、多轮真实流量的轨迹回放，含子智能体（sub agents） | 启用。此场景采用基于轨迹回放的智能体编码基准测试（见 [`benchmarks/single_node/agentic/`](benchmarks/single_node/agentic/)）。今后新模型预计将仅以智能体编码场景接入，且**仅在启用投机解码的条件下**运行。非投机解码分支不运行也不发布（见[弃用公告](#弃用公告)）。 |
| 单轮 8k1k | 8192 / 1024 | 启用。当前主要的固定序列长度（fixed-seq-len）场景。 |
| 单轮 1k1k | 1024 / 1024 | **对所有模型均已弃用**，自 2026-07-17 起（[#2263](https://github.com/SemiAnalysisAI/InferenceX/pull/2263)），以便将 GPU 集群时间留给优先级更高的真实场景智能体编码基准测试与新的前沿模型。归档配置位于 [`configs/deprecated/`](configs/deprecated/)。 |
| 单轮 1k8k | 1024 / 8192 | **对所有模型均已弃用**，自 2026-03-27 起（[#911](https://github.com/SemiAnalysisAI/InferenceX/pull/911)），以便将 GPU 集群时间留给优先级更高的真实场景智能体编码基准测试与新的前沿模型。相关配置已删除，未归档。 |

## AgentX 指南

### 端到端归一化交互性与帕累托前沿策略

端到端归一化交互性（E2E normalized interactivity）是 AgentX 轨迹回放结果的主要用户侧延迟指标，也是默认横轴。对于每个有效的性能分析请求 `i`，令 `OSL_i` 表示其正值输出序列长度，`E2EL_i` 表示其正值端到端延迟。E2EL 同时包含首 Token 延迟（TTFT）与生成时间。首先计算每个请求交付一个输出 Token 所需的时间：

`r_i = E2EL_i / OSL_i`（秒/输出 Token）

对于百分位数 `q`，端到端归一化交互性定义为：

`端到端归一化交互性_q = 1 / percentile_q({r_i})`（输出 Token/s/user）

仪表板默认使用 P90。先在“秒/输出 Token”上取百分位数，再求倒数，既保留了慢尾延迟的含义，又将结果表示为越高越好的速率。该指标并不是分别聚合 OSL 与 E2EL 的百分位数后再取两者之比。在单个请求层面，可将其近似理解为：

`OSL / E2EL ≈ 1 / (TPOT + TTFT / OSL)`

直观地说，可以将其理解为在解码交互性的基础上，对排队和预填充期间用户尚未收到任何输出 Token 的等待时间施加惩罚。

### 优势

- 它降低了 AgentX 输出长度波动对原始端到端延迟的直接影响：请求不会仅仅因为正确生成了更多输出而被视为更慢。
- 它以一个用户侧交付速率同时衡量 TTFT 与解码节奏，避免配置只优化解码交互性，却任由 TTFT 或任务总完成时间退化。
- 它使用熟悉的 Token/s/user 单位，提供越高越好的优化目标，同时覆盖完整的请求路径。

### 权衡与局限

- 它不等同于常规解码交互性（`1 / TPOT`），因此两者的绝对值不能直接比较。由于计入 TTFT，端到端归一化交互性通常更低。
- TTFT 会按 OSL 摊薄，因此短输出受到的 TTFT 惩罚大于长输出。该指标仍与工作负载有关，只有使用相同轨迹分布和基准测试方法的结果才能比较。
- 单一综合指标无法说明退化来自 TTFT 还是解码阶段。独立的端到端延迟、交互性和 TTFT 视图仍用于诊断。
- 该指标依赖持久化的逐请求轨迹，并要求 E2EL 与 OSL 均为有效值。缺少这些轨迹的运行无法参与规范的帕累托前沿。

### 北极星帕累托策略

对于每个受支持的纵轴指标和对比组，以端到端归一化交互性为横轴计算出的帕累托前沿，是 AgentX 规范的**北极星（North Star）**前沿。其他所有 AgentX 横轴视图都必须受这一规范优胜点集合约束：

- 端到端归一化交互性视图展示规范的北极星前沿。
- 端到端延迟、常规交互性与 TTFT 视图展示“规范北极星前沿”与“当前坐标系中真实帕累托前沿”的交集。

因此，只有同时属于北极星优胜点、且在所选图表中不被其他点支配的数据点，才能出现在任一 AgentX 帕累托前沿上。该规则可防止某个配置以牺牲端到端用户体验为代价，只优化某一辅助延迟指标却仍进入发布的前沿。被支配的数据点仍保留在未过滤的散点视图中，供诊断使用。

### 引擎提交策略

根据 Tier 1 AI 实验室和更广泛的机器学习社区对 InferenceX AgentX 展示内容的反馈，InferenceX 采用明确的模型到框架映射。多家实验室反馈，TensorRT-LLM、ATOM 等专有或硬件专用引擎并不总能提供其 AgentX 工作负载所需的全部功能。

下表中的原生/上游（native/upstream）引擎是各模型的一级支持引擎。如果某一提供方将原生/上游 vLLM 引擎和原生/上游 SGLang 引擎均作为一级支持的 LLM 引擎，则必须先按照本映射提交指定的一个或多个引擎，之后才能提交 ATOM、TensorRT-LLM、TokenSpeed 等其他非 vLLM/SGLang 引擎。允许提交多个其他非 vLLM/SGLang 引擎。

该提交顺序指南有两项例外：

1. 对于 MI455X UALoE72、VR200 NVL72、Rubin NVL8、TPUv8t、TPUv8i 等全新硬件 SKU，为实现初始支持，可先使用硬件专用引擎。预期相应的原生/上游 vLLM 或 SGLang 提交会在此后不久跟进。
2. 对于新的模型架构，如果提供方无法将映射指定的原生/上游 vLLM 或 SGLang 引擎作为一级支持引擎，并能向核心维护者说明该框架尚不支持相应硬件与模型组合的根本性、第一性原理原因，则可先使用其他引擎。

InferenceX 支持 SGLang 和 vLLM 双方的维护者，并响应 AI 实验室和机器学习社区希望看到两个框架性能数据的反馈。在仅指定一个主要框架的模型中，映射会将任务均衡分配给 vLLM 和 SGLang。同时指定两个框架的模型则提供共享覆盖。这确保 InferenceX 对两个框架进行同等测试，不偏向任何一方。

表中还同时记录已达成一致的草稿模型规划（PoR）以及尚待合作伙伴对齐的提案。

| 模型 | 首选原生/上游引擎 | 已达成一致的草稿模型（PoR） | 待合作伙伴对齐的草稿模型提案 | 其他引擎 |
|---|---|---|---|---|
| DeepSeek-V4-Pro 1.6T（`dsv4`） | 原生/上游 vLLM 引擎和原生/上游 SGLang 引擎 | 原生 MTP（单轮 8k1k）；`deepseek-ai/DeepSeek-V4-Pro-0813`（仅用于智能体编码，遵循与 Kimi-K3 DSpark PoR 相同的合成接受方法） | 无 | 按照上述提交顺序指南及例外处理的其他非 vLLM/SGLang 引擎 |
| Kimi-K3（`kimik3`） | 原生/上游 vLLM 引擎 | `Inferact/Kimi-K3-DSpark` | 无 | 按照上述提交顺序指南及例外处理的其他非 vLLM/SGLang 引擎 |
| MiniMax-M3（`minimaxm3`） | 原生/上游 vLLM 引擎 | `Inferact/MiniMax-M3-EAGLE3` 和/或 `Inferact/MiniMax-M3-EAGLE3-GQA` | 无 | 按照上述提交顺序指南及例外处理的其他非 vLLM/SGLang 引擎 |
| GLM-5.2（`glm5.2`） | 原生/上游 SGLang 引擎 | 原生 MTP | 无 | 按照上述提交顺序指南及例外处理的其他非 vLLM/SGLang 引擎 |
| Qwen3.5-397B-A17B（`qwen3.5`） | 原生/上游 SGLang 引擎 | 原生 MTP | 无 | 按照上述提交顺序指南及例外处理的其他非 vLLM/SGLang 引擎 |

### KV 缓存卸载策略

为遵循“通过限制范围实现快速交付”的设计原则，AgentX 初始策略仅允许 CPU DRAM KV 缓存卸载，且该功能为可选项。支持的方案包括 vLLM Connector、LMCache、SGLang HiCache、Mooncake CPU DRAM Connector、Dynamo KVBM、CPU DRAM P2P 池化以及类似的 CPU 内存机制。供应商可自行决定是否为每项提交启用 CPU KV 缓存卸载。如果禁用后能得到更优的 Pareto 点，也可选择禁用。

用于 KV 缓存卸载的 CPU DDR5 容量必须与推理配置实际使用的服务器 GPU 比例成正比：

`允许的 CPU DRAM = 每服务器 CPU DRAM 基准容量 ×（配置使用的 GPU 数 / 服务器 GPU 总数）`

该规则按每台服务器独立计算。

| CPU DRAM 容量类别 | 示例 SKU | 每服务器基准与上限 |
|---|---|---|
| CPU DRAM 容量未标准化 | HGX B200、HGX B300、MI355X 机箱 | 每服务器最多 3 TB。因此，仅使用 8 块 GPU 中 4 块的配置最多可使用 1.5 TB CPU DRAM 进行 KV 缓存卸载。 |
| CPU DRAM 容量已标准化 | TPUv7、GB200 NVL72、GB300 NVL72 | 以该 SKU 标准安装的 CPU DRAM 容量为基准，不另设每服务器硬上限，但仍须遵守 GPU 比例规则。 |

3 TB 上限旨在避免不现实的内存容量竞赛，即各硬件供应商要求云服务提供商（CSP）和原始设备制造商（OEM）安装尽可能多的高容量 DIMM，导致每服务器容量可能达到 6 TB。每颗加速器芯片的总体拥有成本（TCO）将按服务器 CPU DDR5 总容量的成本进行归一化。

其他卸载层级（包括 NVMe KV 缓存卸载）不属于初始范围，可在 InferenceX v3 发布后引入。NVMe KV 缓存卸载暂定在 InferenceX v3.5 中作为快速后续功能加入，或纳入 InferenceX v4。

## 模型支持矩阵

| 模型架构类别 | 前缀 | 加入日期 | 启用场景 | 已弃用场景 |
|---|---|---|---|---|
| Qwen3.8 2.4T | `qwen3.8` | 待定 | 智能体编码 | |
| Kimi-K3 | `kimik3` | 2026-07-27 ([#2391](https://github.com/SemiAnalysisAI/InferenceX/pull/2391)) | 智能体编码（仅 DSpark） | 智能体编码非 DSpark 分支（自第 0 天起弃用） |
| GLM-5.2 | `glm5.2` | 2026-07-18（[#2268](https://github.com/SemiAnalysisAI/InferenceX/pull/2268)） | 智能体编码（非 MTP 分支仍在运行，「仅 MTP」转换仍待执行，见弃用公告） | |
| MiniMax-M3 | `minimaxm3` | 2026-06-12（[#1724](https://github.com/SemiAnalysisAI/InferenceX/pull/1724)） | 智能体编码 | 单轮 1k1k、单轮 8k1k（2026-08-04 移除，[#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493)） |
| DeepSeek-V4-Pro | `dsv4` | 2026-04-24（[#1130](https://github.com/SemiAnalysisAI/InferenceX/pull/1130)） | 单轮 8k1k、智能体编码（非 MTP 分支仍在运行，「仅 MTP」转换仍待执行，见弃用公告） | 单轮 1k1k |
| GLM-5 / GLM-5.1 | `glm5`、`glm5.1` | 2026-03-06（[#762](https://github.com/SemiAnalysisAI/InferenceX/pull/762)），GLM-5.1 于 2026-04-21 加入（[#1098](https://github.com/SemiAnalysisAI/InferenceX/pull/1098)） | 无（2026-07-18 退役，[#2276](https://github.com/SemiAnalysisAI/InferenceX/pull/2276)） | 单轮 1k1k、单轮 1k8k（仅 GLM-5）、单轮 8k1k |
| MiniMax-M2.5/2.7 | `minimaxm2.5` | 2026-02-18（[#755](https://github.com/SemiAnalysisAI/InferenceX/pull/755)） | 无（2026-06-20 退役，[#1874](https://github.com/SemiAnalysisAI/InferenceX/pull/1874)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k |
| Kimi-K2.5/2.6/2.7-Code | `kimik2.5` | 2026-02-17（[#734](https://github.com/SemiAnalysisAI/InferenceX/pull/734)） | 无（2026-08-07 完全退役，[#2527](https://github.com/SemiAnalysisAI/InferenceX/pull/2527)） | 单轮 1k1k、单轮 1k8k、智能体编码（2026-08-04 移除，[#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493)）、单轮 8k1k（2026-08-07 移除，[#2527](https://github.com/SemiAnalysisAI/InferenceX/pull/2527)） |
| Qwen3.5-397B-A17B | `qwen3.5` | 2026-02-16（[#704](https://github.com/SemiAnalysisAI/InferenceX/pull/704)） | 单轮 8k1k 与智能体编码，二者均仅限 fp8/fp4 | 单轮 1k1k、单轮 1k8k、全部 bf16 配方（2026-08-04 移除，[#2493](https://github.com/SemiAnalysisAI/InferenceX/pull/2493)） |
| gpt-oss-120b | `gptoss` | 2025-09-09 | 无（2026-07-06 退役，[#2101](https://github.com/SemiAnalysisAI/InferenceX/pull/2101)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k |
| DeepSeek-R1-0528 | `dsr1` | 2025-08-13 | 单轮 8k1k | 单轮 1k1k、单轮 1k8k |
| Llama-3.1-70B-Instruct | `llama70b` | 2025-08-12 | 无（2025-10-29 退役，[#149](https://github.com/SemiAnalysisAI/InferenceX/pull/149)） | 单轮 1k1k、单轮 1k8k、单轮 8k1k [^1] |

[^1]: `llama70b` 早于 master 配置体系。退役时其配置被直接删除，未归档到 `configs/deprecated/`。该模型最初以 workflow 模板形式随仓库首次导入（2025-08-12）。

## 说明

- 「前缀」列为 `configs/*-master.yaml` 中的规范 `model-prefix`，同时用于 `generate_sweep_configs.py --model-prefix`。
- 「退役」指该模型已无任何启用场景。退役模型的配置（`llama70b` 除外）归档于 [`configs/deprecated/`](configs/deprecated/)。
- 弃用某一精度（如 Qwen3.5 bf16）或 A/B 对照中的某一分支（如非 MTP），只是收窄该模型的配方覆盖范围，并不等于模型退役；只要仍有一个场景在运行，该模型即继续列为启用状态。
- `dsr1` 最初以 DeepSeek-V3 workflow 模板的形式随仓库首次导入，2025-08-13 切换为 DeepSeek-R1 基准测试（2025-08-20 将 `dsv3` 重命名为 `dsr1`）。
- 新增模型时，请按[添加模型 + 硬件配方](docs/configuration-procedures_zh.md#添加模型--硬件配方)流程操作，并在同一 PR 中同时更新本文件与 [`MODELS.md`](MODELS.md) 的表格。
