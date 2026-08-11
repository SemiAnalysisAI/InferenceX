# 文档改进计划

<div align="center">

[English](./DOCUMENTATION_PLAN.md) | **中文**

</div>

## 决策

将 `docs/` 建设为 InferenceX 的长期导航层，用于说明架构、基准测试契约、运维流程与 Agent 指引。实现契约保留在代码与 Workflow 文件中，快速执行的政策规则保留在 `AGENTS.md`，而文档页面负责解释系统边界、设计原因、任务清单与故障恢复流程。不要把大型源文件复制到文档中。

本次变更建立 [`index_zh.md`](./index_zh.md)、[`agent-guide_zh.md`](./agent-guide_zh.md)、统一流程目录，以及首批架构、结果、测试与故障排查指南。本文档记录剩余整合工作的范围，以及每个页面应达到的验收标准。

## 问题

重要信息已经存在，但分散在 Agent 指令、贡献指南、Workflow 说明、配置参考、评估文档、Recipe 说明、Runner 初始化文档与故障排查文件中。在本次变更之前，Agent 必须先找到正确的文件才能理解要修改的内容，`docs/` 也只有 PR 清单，无法承担导航职责。

主要风险包括：

- 同一条权威来源规则被重复记录，其中一处变得过期。
- 配置变更只更新主 YAML，却遗漏 Recipe、变更日志、启动器或消费它的生成矩阵。
- Workflow 故障看起来相似，导致将编排、Runner、Serving、评估与收集问题混为一谈。
- 运维恢复知识被困在很长的 Agent 指令或历史调试记录中。
- 新的贡献者文档违反双语规则，或没有提供对应的中文链接。
- 现有领域参考文档没有统一从 `docs/` 发现，且许多页面没有 `_zh.md` 对应版本；迁移时必须区分贡献者文档与内部实现说明。
- AgentX 文档存在不一致：`benchmarks/single_node/agentic/README.md` 仍描述未发布的实验性 MVP，但 `MODELS.md` 与主配置已经包含有效的 `agentic-coding` 覆盖。

## 基线知识清单

下表记录分阶段整合开始前的仓库状态。

| 主题 | 本次变更前的来源 | 需要补齐的缺口 |
| --- | --- | --- |
| Agent 规则与仓库地图 | `AGENTS.md` | 快速政策很完整，但没有按任务组织导航 |
| PR 审阅与合并 | `CONTRIBUTING.md`、`docs/PR_REVIEW_CHECKLIST.md` | 知道文件位置后容易找到，但没有统一文档索引 |
| 常见 CI 与集群故障 | `KLAUD_DEBUG.md` | 恢复步骤有价值，但与历史故障背景混在一起 |
| 配置 Schema 与拓扑 | `configs/CONFIGS.md`、`utils/matrix_logic/validation.py` | Schema 参考存在，但修改流程与消费路径分散 |
| 矩阵生成与扫描复用 | `.github/workflows/README.md`、`.github/workflows/*.yml` | 运维指南位于 `.github`，不在主要文档路径下 |
| 评估与分数门禁 | `utils/evals/EVALS.md`、`utils/evals/thresholds.yaml` | 参考文档详细，但与吞吐量、结果收集的关系不直观 |
| 多节点 Recipe | `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md` | 编辑任一文件前应明确 Recipe 与主配置的耦合关系 |
| Runner 初始化 | `utils/runner_setup/RUNNER_SETUP.md`、`runners/` | 部署与运行时启动器关注点分离 |
| 模型与硬件目录 | `MODELS.md`、`configs/*-master.yaml` | 面向用户的模型列表与可运行配置列表服务于不同读者 |
| AgentX 与 Agentic Coding | `benchmarks/single_node/agentic/README.md`、`MODELS.md`、`configs/*-master.yaml` | 状态与发布说明冲突，需要一份持续维护的官方 Trace 到结果指南 |
| 产物 Schema 与 App 交接 | `utils/process_result.py`、`utils/collect_*.py`、`../InferenceX-app/.github/workflows/ingest-results.yml`、`../InferenceX-app/packages/db/src/etl/*` | 本仓库没有说明产物身份、JSON 契约以及进入 InferenceX-app 的边界 |

## 目标信息架构

每个页面回答一个问题，并链接到准确的实现来源。

| 页面 | 状态 | 主要回答的问题 |
| --- | --- | --- |
| `docs/index.md` | 本次变更完成 | 从哪里开始，当前任务应看哪份指南？ |
| `docs/procedures.md` | 本次变更完成 | 所有配置、CI、Runner、评估、恢复、AgentX 与文档日常操作的统一清单在哪里？ |
| `docs/agent-guide.md` | 本次变更完成 | 编辑或验证变更前，Agent 必须知道什么？ |
| `docs/configuration-procedures.md` | 本次变更完成 | 如何修改基准配置、Runner、镜像、Recipe 与 MTP，同时保持端到端契约？ |
| `docs/ci-procedures.md` | 本次变更完成 | 如何生成、派发、监控、重跑、暂存、复用并检查一次扫描？ |
| `docs/eval-agentx-procedures.md` | 本次变更完成 | 如何运行并校验评估或 AgentX，同时保留证据与溯源？ |
| `docs/recovery-results-procedures.md` | 本次变更完成 | 如何处理结果、验证 App 入库、恢复故障并保护 Runner Workspace？ |
| `docs/documentation-procedures.md` | 本次变更完成 | 如何新增、索引、审阅与维护双语文档？ |
| `docs/architecture.md` | 本次变更完成 | 配置如何变成基准测试结果并发布为一行数据？ |
| `docs/configuration.md` | 计划中 | 如何修改配置而不破坏 Schema、拓扑与变更日志契约？ |
| `docs/benchmark-development.md` | 计划中 | 基准测试脚本、共享 Bash 工具、启动器与运行时环境变量如何协作？ |
| `docs/agentx.md` | 计划中 | AgentX 当前状态、Trace 契约、执行路径与发布边界是什么？ |
| `docs/workflows-and-sweeps.md` | 计划中 | 如何生成、派发、监控、复用并收集一次扫描？ |
| `docs/evals.md` | 计划中 | 评估如何选择、运行、打分、校验与收集？ |
| `docs/runners-and-clusters.md` | 计划中 | Runner 标签、硬件事实、Slurm 任务、清理与日志如何管理？ |
| `docs/recipes-and-disaggregated.md` | 计划中 | srt-slurm Recipe 如何映射到主配置与分离式拓扑？ |
| `docs/results-and-ingestion.md` | 本次变更完成 | 产物如何变成聚合结果与 Dashboard 数据？ |
| `docs/glossary.md` | 计划中 | 基准测试术语、指标名称、产物字段与缩写的固定含义是什么？ |
| `docs/troubleshooting.md` | 本次变更完成 | 如何分类故障并选择安全的恢复路径？ |
| `docs/testing.md` | 本次变更完成 | 在 GPU 或 CI 执行前，哪些本地检查可以证明变更正确？ |

迁移期间保留已有的详细参考文档。只有当新页面能够保留原有契约并更新链接时，才移动内容。新页面应先链接现有来源，再逐步整合内容。

## 分阶段落地

### 第一阶段，导航与 Agent 入门

- 添加文档索引与 Agent 指南。
- 在 `AGENTS.md` 以及根目录两种语言的 README 中链接文档索引。
- 明确权威来源边界。
- 保持现有详细参考文档不变。

### 第二阶段，架构与开发流程

- 添加介绍配置到结果数据流的架构页面。
- 添加把 Schema、生成器、变更日志与校验步骤串起来的配置页面。
- 添加基准测试开发页面，说明共享 Bash 工具、环境变量传递、单节点与多节点路径，以及 MTP 的聊天模板要求。
- 只在能够澄清所有权或状态转换时使用图示。

### 第三阶段，运维与恢复

- 添加统一流程目录，覆盖标签、Canary/Fail-fast、评估修饰标签、产物复用、结果下载、暂存、重跑、恢复、AgentX 与 Runner 清理。
- 当流程目录需要更多实现细节时，再添加 Workflow 与扫描的深度页面。
- 添加 Runner 与集群页面，覆盖 AMD root 文件清理与禁止新增目录约束。
- 添加 Recipe 与分离式服务页面，提供同步编辑清单。
- 按故障层级添加故障排查页面，并链接回 `KLAUD_DEBUG.md` 中的已知案例。

### 第四阶段，结果与维护

- 添加结果处理、评估收集与入库边界说明。
- 添加专注于本地测试的页面，用命令覆盖配置、矩阵与结果契约。
- 如果仓库 CI 约定支持，为文档添加链接检查与语言配对检查。
- 新页面已建立链接并验证后，才删除重复的流程文字。

## 维护契约

- 英文版本是文档主版本。先更新英文，再在同一变更中根据英文内容翻译匹配的 `_zh.md` 页面。
- 每条运维事实都要标明其来源文件、Workflow、脚本或配置字段。
- 实现行为变化时，在同一变更中更新最近的指南。PR 清单政策变化时，同时更新校验器提示词。
- 优先使用简短任务清单与源代码链接，不要复制长段命令输出或大型生成表格。
- 历史故障细节保留在 `KLAUD_DEBUG.md` 或 Issue 记录中。可重复执行的恢复步骤放在对应运维指南，并链接回历史背景。
- 新增指南或主要权威来源文件后，检查文档索引。

## 验收标准

文档整合完成后，Agent 应当能够：

1. 不搜索整个仓库，只通过 `docs/index.md` 找到正确指南。
2. 追踪一次配置变更从主 YAML 经过矩阵生成、Workflow 执行、启动器与运行时行为，直到结果收集的完整路径。
3. 判断失败任务属于生成、编排、Runner、Serving、评估、收集还是入库问题。
4. 在派发昂贵扫描前，运行定向的本地验证。
5. 为每份贡献者指南找到中文对应版本及其实现来源。

本次变更已经完成索引、Agent 入门、统一流程目录、架构、结果入库、测试与故障排查指南，形成首轮完整的导航与运维覆盖。其余页面仍按阶段推进，确保每一页都以实际实现为依据，而不是成为另一份会逐渐偏离仓库的副本。
