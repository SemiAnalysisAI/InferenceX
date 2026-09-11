# Klaud Cold 报告

<div align="center">

[English](./klaud-reporting.md) | **中文**

</div>

[`utils/klaud/reporting.py`](../utils/klaud/reporting.py) 统一管理数据结构、差值计算和渲染。agent 只提供简短观察和经过核实的证据，不手算差值。PR 正文仅包含目标和基线；评论记录尝试；生命周期完成记录证明收尾已验证。英文默认展开，简体中文放入一个默认折叠的 `<details><summary>中文</summary>` 区块。数值表格只展示一次，中文引用共用表格；生命周期评论也使用这一布局。不提及用户、不请求审查、不发布原始日志或私有遥测，也不添加 limitations 章节。

## 命令

在工作流提供候选环境的 checkout 中运行：

```bash
KLAUD=(uv run --no-project --exclude-newer PT12H --python 3.12 \
  --with 'pydantic>=2.10,<3' --with pyyaml python -m utils.klaud)

# goal.json：{"en":"Update ENGINE image from `OLD` to `NEW`.","zh":"将 ENGINE 镜像从 `OLD` 更新为 `NEW`。"}
# 先通过公开 OpenAPI 确认模型显示名称。
"${KLAUD[@]}" prepare-baseline --model 'DISPLAY MODEL NAME' \
  --goal-file "$KLAUD_EVIDENCE/goal.json" --output "$KLAUD_EVIDENCE/baseline.json"

# 创建包含实际变更的草稿 PR 后，在 GPU 工作开始前发布基线。
"${KLAUD[@]}" report --kind baseline --file "$KLAUD_EVIDENCE/baseline.json"

# 数据结构只需读取一次；后续复用本地记录更新尝试状态。
"${KLAUD[@]}" report-schema --kind attempt > "$KLAUD_EVIDENCE/attempt-schema.json"
"${KLAUD[@]}" report --kind attempt --file "$KLAUD_EVIDENCE/attempt.json"
```

`prepare-baseline` 使用候选日期、`exact=true` 查询公开 `benchmarks`，不使用 calculator view，并通过 `workflow-info` 记录生产运行 ID、head 和运行次数。只接受与所选基准提交生成的旧镜像完整 recipe fingerprint 唯一匹配的配置点。缺失指纹或来源时记为不可用，不按 SKU 和并发数近似匹配。首次发布前，按[公开 API 调查指南](./klaud_zh.md#公开-api-调查)补充已验证的历史评测和数据集来源。公开 `BenchmarkRow` 本身不含数据集身份；AgentX 数据集无法核实时，差值记为 N/A。绝不通过运行旧镜像补齐基线。

基线文件仅创建一次，重试不重新获取。首条基线评论冻结类型化记录；冲突替换会被拒绝。确需修正时，由维护者检查证据并明确记录更正，不能在修复期间静默改变基线。

创建草稿时，正文使用 `<!-- klaud-baseline -->`。辅助程序只替换此占位符一次，保留其他机器人在其外部添加的内容。如果评论已发布而正文更新中断，重试同一记录会补完正文更新。后续尝试报告不改写正文。大报告先写入以内容哈希命名的不可变分段，再更新索引，避免中断时混用不同版本。

比较键通过 `reporting.point_key()` 从规范生成的配置点计算，仅排除镜像、配置点名称、生产指纹和排队元数据。工作负载、拓扑、并发数及其他设置都必须保留。`reporting.values()` 读取 collector/API 指标，将秒换算为毫秒。不能为使结果匹配而编造别名或比较键。AgentX 还须独立匹配数据集。基线缺失或为零、配置失败、数据集或统计口径不一致时记为 N/A。吞吐量差值为 `(new / old - 1) * 100`；评测分数使用 0–1，差值以百分点展示，并匹配 suite、metric、拓扑和样本量。

每次尝试记录所属 run ID、精确 head、运行次数、类型和编号、状态、以 en/zh 句子记录的变更/下一步、分别统计的 benchmark 与 eval 应有和通过数量，以及全部配置和评测结果。目标同样使用 en/zh 句子，明确引擎及精确的新旧镜像。可选 finding 字段保存诊断证据；finding 和覆盖计数均不渲染。保留失败、取消和请求错误。初始变更编号为 0，修复为 1–5。已确认的临时基础设施问题单独记为 infrastructure-retry，不消耗 recipe 修复次数；prompt 将其限制为每次尝试最多两次。不能因为吞吐量通过就把评测失败的 smoke 写成成功。

调度后立即发布记录，再等待运行。发生实质变化或等待满 30 分钟时，更新该 run/attempt 的同一条评论；完成的尝试保留为历史。大记录拆成编号评论分段，不丢弃配置点，也不限制配置族大小。正文最多展示 12 行基线；全部数值和来源保存在基线评论。记录以 parent/candidate/run/attempt 为幂等身份，只持久化类型化公开字段，不上传整个临时目录或执行记录。

使用紧凑元数据行、精确的 `8k/1k` 简写，并将共用配置放在表格上方。只有展示的工作负载、拓扑、统计口径一致且并发数唯一时，才仅用并发数标识行；否则保留可区分的完整标签。单元格显示新值和括号内的差值。评测样本量仅在新旧计数相同时显示 `N each`，否则分别列出新旧计数。失败、请求错误和无法比较的原因以简短备注保留。不添加 Result 列、Coverage/Finding 段落、图例、存储机制说明或重复状态总结；Next 仅写下一子目标。英文及全部结果表格保持展开，只折叠中文。

## 正文格式

```markdown
**Goal:** Update ENGINE image from `OLD_IMAGE` to `NEW_IMAGE`.\
**Baseline:** DATE · `OLD_IMAGE`\
8k/1k · TP8/EP1 · Mean latency · Sources: API links

| Concurrency | Total tok/s/GPU ↑ | Output tok/s/GPU ↑ | TTFT ms ↓ | TPOT ms ↓ |
| ---: | ---: | ---: | ---: | ---: |
| C | value | value | value | value |

| Eval | Score ↑ | Samples |
| --- | ---: | ---: |
| SUITE/METRIC · cN | SCORE% | N |

<details>
<summary>中文</summary>

**目标：**将 ENGINE 镜像从 `OLD_IMAGE` 更新为 `NEW_IMAGE`。\
**基线：**DATE · `OLD_IMAGE`\
8k/1k · TP8/EP1 · 平均延迟 · 来源：API 链接；数值及异常说明见上表。

</details>
```

## 尝试评论格式

```markdown
**Repair N/5 · STATUS** · [Run ID / attempt N](RUN_URL) · UTC_TIMESTAMP\
`IMAGE` · `HEAD` · 8k/1k · TP8/EP1 · Mean latency\
**Change:** One sentence with relevant source links.

| Concurrency | Output tok/s/GPU ↑ | TTFT ms ↓ | TPOT ms ↓ |
| ---: | ---: | ---: | ---: |
| C | 110 (+10%) | 180 (-10%) | 19 (-5%) |

| Eval | Score ↑ | Samples |
| --- | ---: | ---: |
| SUITE/METRIC · cN | 97% (+0.50 pp) | 1,000 each |

**Next:** Run the final full sweep.

<details>
<summary>中文</summary>

**修复 N/5 · 状态** · [Run ID / attempt N](RUN_URL) · UTC_TIMESTAMP\
`IMAGE` · `HEAD` · 8k/1k · TP8/EP1 · 平均延迟\
**变更：**简短中文翻译；数值及异常说明见上表。\
**下一步：**运行最终完整 sweep。

</details>
```

正常结束和中断恢复均由 `finish` 使用已验证产物和冻结基线生成最终报告，并在**标记就绪之前**发布。缺失的历史基线明确记为 N/A，不编造差值，也不另跑基线。成功 sweep 可以存在性能回归；就绪表示工作和验证结束，而非每项指标都提升。Klaud 不授权 reuse，也不合并 PR。

## 最终预检与维护者重试

添加 `full-sweep-enabled` 前，验证已推送精确 head 的完整矩阵：

```bash
head_sha=$(git rev-parse HEAD)
uv run --no-project --python 3.12 --with 'pydantic>=2.10,<3' --with pyyaml \
  python utils/process_changelog.py --base-ref origin/main --head-ref "$head_sha" \
  --changelog-file perf-changelog.yaml > "$KLAUD_EVIDENCE/final-matrix.json"
"${KLAUD[@]}" check-final --matrix-file "$KLAUD_EVIDENCE/final-matrix.json"
```

验证器使用受信任 helper 代码，从精确 head 的 YAML 独立生成未过滤配置族。覆盖等价的 scenario filter 可以通过；缺失或改变的配置点、默认评测不能通过。不会执行下载的 PR 代码。旧运行遇到生成器策略变化时需要检查，不能静默降低验证标准。

已确认的阻塞原因修复后，仓库维护者可明确释放已关闭候选保留的分支：

```bash
"${KLAUD[@]}" release-candidate --parent-run-id PARENT_RUN_ID \
  --candidate-file ORIGINAL_CANDIDATE_JSON --head REVIEWED_CLOSED_PR_SHA
```

该命令拒绝 Klaud 账号及非维护者，要求父运行已结束、清理记录已验证、PR 在精确 head 上关闭且未合并、所有自有子运行均已结束。它重新检查分支，记录批准，再仅删除该保留分支。不会重新调度、抹除历史结果或接管开放 PR。普通容量或就绪性延后已由 `finish` 释放分支。
