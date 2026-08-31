# InferenceX 文档

<div align="center">

[English](./index.md) | **中文**

</div>

这是 InferenceX 工作的必读低上下文路由页。只选择负责当前任务的一份页面，再按需打开其中链接的源文件。仓库源文件与 Workflow 仍是行为的权威来源。

## 任务与页面索引

| 页面 | 适用场景 |
| --- | --- |
| [`index.md`](./index.md) / [`index_zh.md`](./index_zh.md) | 本任务路由页及其中文对应页面 |
| [`agent-guide.md`](./agent-guide.md) / [`agent-guide_zh.md`](./agent-guide_zh.md) | Agent 入门、安全开始、关键约束与验证 |
| [`procedures.md`](./procedures.md) / [`procedures_zh.md`](./procedures_zh.md) | 从常见任务路由到一份聚焦运维清单 |
| [`architecture.md`](./architecture.md) / [`architecture_zh.md`](./architecture_zh.md) | 配置到结果的流程、所有权边界、产物与 InferenceX-app 交接 |
| [`configuration-procedures.md`](./configuration-procedures.md) / [`configuration-procedures_zh.md`](./configuration-procedures_zh.md) | 配置、Runner、镜像、Recipe、llm-d、srt-slurm 与 MTP 变更 |
| [`ci-procedures.md`](./ci-procedures.md) / [`ci-procedures_zh.md`](./ci-procedures_zh.md) | 矩阵生成、校验、派发、PR 扫描、复用、暂存与产物下载 |
| [`eval-agentx-procedures.md`](./eval-agentx-procedures.md) / [`eval-agentx-procedures_zh.md`](./eval-agentx-procedures_zh.md) | Eval 与 AgentX 选择、执行、打分、证据与实时运行诊断 |
| [`results-and-ingestion.md`](./results-and-ingestion.md) / [`results-and-ingestion_zh.md`](./results-and-ingestion_zh.md) | 已发布结果查询、产物身份与 Schema、App 入库、去重与溯源 |
| [`recovery-results-procedures.md`](./recovery-results-procedures.md) / [`recovery-results-procedures_zh.md`](./recovery-results-procedures_zh.md) | 结果处理、入库验证与恢复、Runner 清理和故障分类 |
| [`testing.md`](./testing.md) / [`testing_zh.md`](./testing_zh.md) | 本地检查、冒烟运行、证据标准与评审门禁 |
| [`troubleshooting.md`](./troubleshooting.md) / [`troubleshooting_zh.md`](./troubleshooting_zh.md) | 故障层级诊断、已知案例、安全修复与停止条件 |
| [`documentation-procedures.md`](./documentation-procedures.md) / [`documentation-procedures_zh.md`](./documentation-procedures_zh.md) | 新增、翻译、索引、审阅与维护文档 |
| [`PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) / [`PR_REVIEW_CHECKLIST_zh.md`](./PR_REVIEW_CHECKLIST_zh.md) | CODEOWNER 审阅与精确签署要求 |
| [`DOCUMENTATION_PLAN.md`](./DOCUMENTATION_PLAN.md) / [`DOCUMENTATION_PLAN_zh.md`](./DOCUMENTATION_PLAN_zh.md) | 剩余文档缺口、目标信息架构与落地计划 |
| [`kimi-k3-mi355x-vllm-current-baseline.md`](./kimi-k3-mi355x-vllm-current-baseline.md) / [`kimi-k3-mi355x-vllm-current-baseline_zh.md`](./kimi-k3-mi355x-vllm-current-baseline_zh.md) | 复现固定的 Kimi-K3 MI355X vLLM C1/C16/C52 镜像补丁、服务配置与性能证据 |
| [`kimi-k3-mi355x-vllm-image-reproduction.md`](./kimi-k3-mi355x-vllm-image-reproduction.md) / [`kimi-k3-mi355x-vllm-image-reproduction_zh.md`](./kimi-k3-mi355x-vllm-image-reproduction_zh.md) | 重建并逐字节校验 Kimi-K3 C16 vLLM/AITER CPRR 镜像，包括精确的 PR 与自定义修复来源 |

## 权威参考

| 参考 | 负责内容 |
| --- | --- |
| [`AGENTS.md`](../AGENTS.md) | 必读的低上下文 Agent 政策与基准测试约束 |
| [`CONTRIBUTING_zh.md`](../CONTRIBUTING_zh.md) | PR 审阅、CODEOWNER 签署、扫描复用、合并与合并后职责 |
| [`.github/AGENT_OPERATIONS.md`](../.github/AGENT_OPERATIONS.md) | 翻译术语、扫描标签、派发、Eval 选择、功耗、指标与产物 |
| [`configs/CONFIGS.md`](../configs/CONFIGS.md) | 主配置 Schema、搜索空间、Runner 与拓扑字段 |
| [`.github/workflows/README.md`](../.github/workflows/README.md) | 生成器示例、Workflow 操作与复用政策 |
| [`utils/evals/EVALS.md`](../utils/evals/EVALS.md) | Eval 任务、执行、收集、校验与 SWE-bench 契约 |
| [`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md) | 分离式 Recipe 注册与主配置耦合 |
| [`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md) | Runner 部署与初始化 |
| [`MODELS_zh.md`](../MODELS_zh.md) | 支持的模型、硬件覆盖与命名 |
| [`KLAUD_DEBUG.md`](../KLAUD_DEBUG.md) | Klaud-Cold、CI、镜像、集群与 GitHub CLI 的历史故障特征 |
| [`benchmarks/single_node/agentic/README.md`](../benchmarks/single_node/agentic/README.md) | AgentX Trace 基准测试实现 |

## 上下文规则

1. 只打开当前任务所需的聚焦页面与源文件片段。
2. 如果过滤后的内容足以回答问题，不要加载大型 YAML、JSON、日志、生成矩阵或完整参考文件。
3. 源代码、Workflow YAML、Schema、启动器与收集器优先于说明文档。
4. 行为变化时，先更新最近的英文指南，再在同一变更中更新其 `_zh.md` 对应页面。
