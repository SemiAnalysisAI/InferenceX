# 运维流程

<div align="center">

[English](./procedures.md) | **中文**

</div>

本页是 InferenceX 流程知识库的低上下文入口。先读本页，再只打开与当前任务相关的一份聚焦指南。默认不要读取所有流程页面。每份聚焦指南都包含前置条件、步骤、验证、停止条件与实现来源链接。

仓库源代码、Workflow YAML、启动器脚本与结果收集器是行为的权威来源。如果指南与代码不一致，应以代码为准。英文页面是文档主版本，先更新英文，再在同一变更中根据英文内容更新对应的中文翻译。

## 任务路由

| 任务 | 首先打开 | 然后检查 |
| --- | --- | --- |
| 新增模型或 GPU 基准测试 | [配置流程](./configuration-procedures_zh.md#添加模型--硬件配方) | 最近的基准测试脚本、启动器、主 YAML、变更日志 |
| 修改已有配置 | [配置流程](./configuration-procedures_zh.md#修改主配置) | `CONFIGS.md`、校验 Schema、生成器、运行时消费者 |
| 新增 Runner | [配置流程](./configuration-procedures_zh.md#注册并设置-runner) | Runner 初始化、`configs/runners.yaml`、启动器 |
| 修改 srt-slurm 或 llm-d | [配置流程](./configuration-procedures_zh.md#注册-srt-slurm-配方) | Recipe YAML、主配置、`srtctl` 映射、启动器 |
| 修改 MTP | [配置流程](./configuration-procedures_zh.md#添加或修改-mtp) | MTP 同类实现、draft model、聊天模板路径 |
| 校验矩阵 | [CI 流程](./ci-procedures_zh.md#本地矩阵生成) | 生成器 CLI 与 Pydantic 校验 |
| 派发或监控任务 | [CI 流程](./ci-procedures_zh.md#手动端到端派发) | `e2e-tests.yml`、运行日志、产物 |
| 准备 PR 扫描 | [CI 流程](./ci-procedures_zh.md#pr-主标签与修饰标签) | `run-sweep.yml`、标签、变更日志差异 |
| 复用绿色扫描 | [CI 流程](./ci-procedures_zh.md#产物复用与-merge-with-reuse) | 复用门禁、源产物、合并工具 |
| 新增或调试评估 | [评估与 AgentX 流程](./eval-agentx-procedures_zh.md#2-添加评分-eval) | `EVALS.md`、评估模板、分数校验器 |
| 运行 AgentX | [评估与 AgentX 流程](./eval-agentx-procedures_zh.md#7-运行-agentx快速反馈与-canonical-证据) | Agentic 配置、Trace 来源、实时运行 Skill |
| 检查结果或入库 | [恢复与结果流程](./recovery-results-procedures_zh.md#结果流水线先明确应当存在什么) | 产物 Schema、收集器、App 入库 Workflow |
| 恢复失败入库 | [恢复与结果流程](./recovery-results-procedures_zh.md#失败摄取恢复) | 恢复工具、源 Run 产物、祖先关系规则 |
| 调试 Runner 或 Workspace | [恢复与结果流程](./recovery-results-procedures_zh.md#amd-root-owned-工作区的预防与恢复) | 启动器清理、`KLAUD_DEBUG.md`、集群日志 |
| 新增或更新文档 | [文档流程](./documentation-procedures_zh.md#新增面向贡献者的页面) | 最近的源文件、文档索引、`_zh.md` 配对 |

## 通用执行顺序

先完成 [Agent 指南的安全开始](./agent-guide_zh.md#安全开始)，然后：

1. 只阅读聚焦流程链接的实现章节。行为由源文件定义，不由复制的文案定义。
2. 在昂贵的 CI 或 GPU 工作前运行其中最窄的验证。
3. 流程产生外部证据时，记录 Run ID、attempt、产物名称、source SHA 与故障分类。
4. 流程或契约变化时，更新相关英文指南及其中文对应页面。

## 源 Playbook

聚焦指南会链接以下现有 Playbook，而不是复制其中完整的历史背景：

| Playbook | 范围 |
| --- | --- |
| [`add-model-hardware.md`](../.claude/commands/add-model-hardware.md) | Day-zero 模型与 SKU Recipe |
| [`recover-failed-ingest.md`](../.claude/commands/recover-failed-ingest.md) | 仅维护者可执行的产物复用恢复 |
| [`clean-amd-mi355-runner-root-files.md`](../.claude/commands/clean-amd-mi355-runner-root-files.md) | AMD Runner root 文件恢复 |
| [`debug-mi300-enroot-pyxis.md`](../.claude/commands/debug-mi300-enroot-pyxis.md) | MI300X 集群初始化调试 |
| [`merge-prs.md`](../.claude/commands/merge-prs.md) | 维护者 PR 合并协调 |
| [`fix-klaud-cron-prs.md`](../.claude/commands/fix-klaud-cron-prs.md) | 自动镜像更新 PR 修复 |
| [`nuke.md`](../.claude/commands/nuke.md) | 自动更新镜像标签，每个 Recipe 系列创建一个 PR |

聚焦页面是可发现的流程层。这些源 Playbook 保留为实现细节或特权维护路径，只有在聚焦指南明确链接时才打开。

## 上下文纪律

遵循 [`docs/index_zh.md` 的上下文规则](./index_zh.md#上下文规则)。优先读取源文件的标题或行区间，并将大型 YAML、JSON、日志、生成矩阵与产物作为定向数据源。
