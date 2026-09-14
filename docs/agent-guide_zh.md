# Agent 指南

<div align="center">

[English](./agent-guide.md) | **中文**

</div>

这是修改 InferenceX 时给 Agent 的简短入口。[`AGENTS.md`](../AGENTS.md) 是必读政策层。本页说明如何进入更大的文档集合，而不需要一次性加载全部内容。

## 安全开始

1. 确认仓库根目录、分支、Worktree 与 `git status`，保留无关修改。多文件任务使用 `.worktrees/<name>` 独立 Worktree。
2. 阅读 [`AGENTS.md`](../AGENTS.md)。涉及审阅、扫描、合并或 Runner 政策时阅读 [`CONTRIBUTING_zh.md`](../CONTRIBUTING_zh.md)。
3. 按 [`index_zh.md`](./index_zh.md) 给任务分类。对于架构、结果、测试或故障排查问题，打开负责该问题的参考页面。
4. 对于常见运维变更，使用 [`procedures_zh.md`](./procedures_zh.md) 只打开一份聚焦清单。
5. 跟进该页面的源链接，将改动字段或符号追踪到实际消费者。源文件和 Workflow YAML 优先于说明文档。
6. 派发 GPU 或 CI 任务前运行最窄的验证。产生外部证据时记录 Run ID、attempt、source SHA 与产物名称。

## 职责边界

| 需求 | 负责页面 |
| --- | --- |
| 必读 Agent 政策与基准测试约束 | [`AGENTS.md`](../AGENTS.md) |
| 任务与参考页面路由 | [`index_zh.md`](./index_zh.md) |
| 常见运维清单 | [`procedures_zh.md`](./procedures_zh.md) |
| PR 审阅、扫描复用与合并政策 | [`CONTRIBUTING_zh.md`](../CONTRIBUTING_zh.md) |
| 验证深度与证据标准 | [`testing_zh.md`](./testing_zh.md) |
| 故障分类与安全修复 | [`troubleshooting_zh.md`](./troubleshooting_zh.md) |

## 上下文纪律

只加载负责当前任务的一份页面，以及当前步骤所需的源文件章节。避免把大型 YAML、JSON、日志、生成矩阵或完整参考文件全部放入上下文。将长期有效的设计原因与常见陷阱写入聚焦指南，使后续 Agent 无需重复全仓探索即可发现。
