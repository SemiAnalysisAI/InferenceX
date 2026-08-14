# CI 操作流程

<div align="center">

[English](./ci-procedures.md) | **中文**

</div>

本页用于矩阵生成、CI 派发、PR 扫描、结果预发布、产物复用与合并后发布。英文页是源版本；中文页必须在结构上与其保持同步。凡将 CI 输出作为证据，均应记录仓库、Commit SHA、Workflow Run ID、Run Attempt 与 Artifact 名称。

## 流程索引

| 任务 | 前往 |
| --- | --- |
| 确定权威实现 | [源码映射](#源码映射) |
| 避免套用错误分支的 Workflow 契约 | [源码快照警告](#源码快照警告) |
| 在本地生成并检查矩阵 | [本地矩阵生成](#本地矩阵生成) |
| 验证 YAML 与 `perf-changelog.yaml` | [YAML 与 Changelog 验证](#yaml-与-changelog-验证) |
| 启动定向 GPU 任务 | [手动端到端派发](#手动端到端派发) |
| 运维每周 overview snapshot | [每周 overview snapshot](#每周-overview-snapshot) |
| 选择 PR 扫描标签 | [PR 主标签与修饰标签](#pr-主标签与修饰标签) |
| 理解提前取消行为 | [Canary 与 Fail-fast 语义](#canary-与-fail-fast-语义) |
| 诊断或重跑 Workflow | [监控与重跑](#监控与重跑) |
| 将 PR Run 发布到预发布环境 | [暂存结果](#暂存结果) |
| 合并时不重复已批准的扫描 | [产物复用与 merge-with-reuse](#产物复用与-merge-with-reuse) |
| 恢复仅追加 Changelog 的冲突 | [Changelog 冲突恢复](#changelog-冲突恢复) |
| 不加载全部内容地检查结果 JSON | [产物下载与解析](#产物下载与解析) |
| 合并后验证发布 | [合并后预期](#合并后预期) |

## 源码映射

以下文件就是契约。应遵循目标 Ref 的源码，而不是从旧 Run 复制命令：

| 关注点 | 准确源码 |
| --- | --- |
| 生成器 CLI、过滤与 Eval 标记 | [`utils/matrix_logic/generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py) |
| 严格的 Master Config 与矩阵 Schema | [`utils/matrix_logic/validation.py`](../utils/matrix_logic/validation.py) |
| 生成器示例与复用策略 | [`.github/workflows/README.md`](../.github/workflows/README.md) |
| 手动端到端输入与矩阵扇出 | [`.github/workflows/e2e-tests.yml`](../.github/workflows/e2e-tests.yml) |
| PR/main 扫描 Gate、Canary、收集与入库派发 | [`.github/workflows/run-sweep.yml`](../.github/workflows/run-sweep.yml) |
| 单节点与多节点产物上传 | [`.github/workflows/benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml)、[`.github/workflows/benchmark-multinode-tmpl.yml`](../.github/workflows/benchmark-multinode-tmpl.yml) |
| 吞吐量与 Eval 聚合 | [`.github/workflows/collect-results.yml`](../.github/workflows/collect-results.yml)、[`.github/workflows/collect-evals.yml`](../.github/workflows/collect-evals.yml)、[`utils/collect_results.py`](../utils/collect_results.py)、[`utils/collect_eval_results.py`](../utils/collect_eval_results.py) |
| Changelog 字节、Diff 与矩阵 Gate | [`utils/validate_perf_changelog.py`](../utils/validate_perf_changelog.py)、[`utils/process_changelog.py`](../utils/process_changelog.py) |
| 复用授权与源 Run 选择 | [`utils/find_reusable_sweep_run.py`](../utils/find_reusable_sweep_run.py) |
| 受支持的复用合并与冲突准备 | [`utils/merge_with_reuse.sh`](../utils/merge_with_reuse.sh)、[`utils/prepare_perf_changelog_merge.py`](../utils/prepare_perf_changelog_merge.py) |
| 预发布请求与回调 | [`.github/workflows/stage-results.yml`](../.github/workflows/stage-results.yml)、[`.github/workflows/stage-results-callback.yml`](../.github/workflows/stage-results-callback.yml) |
| 复用 Agentic 入库的重新派发 | [`.github/workflows/recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml) |
| 合并后责任提醒 | [`.github/workflows/pr-recipe-reminder.yml`](../.github/workflows/pr-recipe-reminder.yml) |

## 源码快照警告

本页基于分支 Commit `0c28706b33d4a796b82f6f9c3594c19c46365575` 编写。当时本地 `origin/main` 为 `de493d8597035e6692833de6189b567887968460`，相关 CI 源码并不完全相同：

- 分支本地的 [`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) 要求提供 `generate-cli-command`，并将各矩阵硬编码为 `fail-fast: false`。审计过的 [`origin/main` 版本](https://github.com/SemiAnalysisAI/InferenceX/blob/de493d8597035e6692833de6189b567887968460/.github/workflows/e2e-tests.yml) 仅在特定条件下要求该命令，并新增受信任 Changelog 派发、`fail-fast` 与功耗验证输入。
- 分支本地的 [`run-sweep.yml`](../.github/workflows/run-sweep.yml) 缺少 `origin/main` 在 PR GPU Setup 之前新增的“Head 仓库必须与当前仓库相同”保护。该 `origin/main` 快照存在 [`trusted-external-sweep.yml` Workflow](https://github.com/SemiAnalysisAI/InferenceX/blob/de493d8597035e6692833de6189b567887968460/.github/workflows/trusted-external-sweep.yml)，本分支则没有。不要根据分支本地 Workflow 推断外部 Fork 的 Secret 或 GPU 行为。
- 分支本地生成器的 Agentic Eval 注释指向 SWE-bench，审计过的 `origin/main` 生成器则指向 GSM8K。在描述 `all-evals` 或 `evals-only` 选择的 Agentic 数据集之前，必须检查目标 Ref。

`workflow_dispatch` 请求使用其派发 `--ref` 中的 Workflow 定义；单独的 `inputs.ref` 控制 Job Checkout 的内容。在使用下方公共示例以外的输入前，应检查已部署定义：

```bash
gh workflow view e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref main --yaml
```

如果目标 Ref 与本源码快照不同，以其 Workflow 和脚本源码为准。不要猜测某个分支本地输入或 Fork 策略已经部署。

## 本地矩阵生成

生成器会加载所有指定的 Master Config 与 `configs/runners.yaml`，使用 Pydantic 验证输入，生成条目，验证输出结构，应用 Eval 策略，并输出一个 JSON 数组。退出码为零只能证明生成与 Schema 验证成功，不能证明容器、模型或 GPU Runner 能工作。

### 生成一个准确配置

对准确 Key 或带引号的 `*`/`?` 模式使用 `test-config`。`--conc` 必须存在于配置的并发范围/列表内，`--seq-lens` 必须匹配实际存在的 Scenario。

```bash
MATRIX=/tmp/inferencex-matrix.json
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml \
  --config-keys dsr1-fp8-h200-sglang \
  --seq-lens 8k1k \
  --conc 4 \
  --no-evals > "$MATRIX"
python3 -m json.tool "$MATRIX" >/dev/null
```

多个 Key 应逐个放在 `--config-keys` 之后。通配模式必须加引号，防止 Shell 展开：

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml \
  --config-keys '*-b200-*' \
  --conc 4 \
  --no-evals > /tmp/inferencex-b200.json
```

### 生成过滤后的扫描

`full-sweep` 不一定表示所有配置。可按模型、精度、框架、Runner、序列长度、拓扑、并发、TP/EP 或 Scenario 类型缩小范围：

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --framework sglang \
  --runner-type h200 \
  --seq-lens 8k1k \
  --min-conc 4 \
  --max-conc 4 \
  --no-evals > /tmp/inferencex-filtered.json
```

如果既未提供 `--single-node` 也未提供 `--multi-node`，两种类型都会生成。`--step-size` 必须大于 1；如果同时提供并发上下界，`--min-conc` 不能大于 `--max-conc`。

### 检查而非倾倒

检查数量与执行关键字段，不要把完整 JSON 加载进上下文：

```bash
jq 'length' "$MATRIX"
jq -r '
  .[] |
  [.["model-prefix"], .framework, .precision, .runner,
   .isl, .osl, (.conc | tostring), (.tp // "-"), (.ep // "-"),
   (.prefill.hardware // "-"), (.decode.hardware // "-"),
   (.["run-eval"] // false), (.["eval-only"] // false)] |
  @tsv
' "$MATRIX"
```

确认预期的镜像、模型、硬件/Cluster 标签、单节点或多节点拓扑、输入/输出长度、并发、TP/EP、解码模式与 Eval 标记。空输出不算成功的预检。

Eval 开关语义是明确的：

- 默认：吞吐量条目加选定的默认固定序列 Eval 子集。
- `--no-evals`：仅吞吐量；不能与 `--all-evals` 组合。
- `--evals-only`：仅选定的 Eval 子集。
- `--all-evals`：扩展到每个已生成的固定序列配置；单独使用时也等同于 Eval-only。
- `--evals-only --all-evals`：所有扩展后的 Eval，且无吞吐量。

## YAML 与 Changelog 验证

### 解析修改过的 YAML

对每个修改过的 YAML 文件执行语法解析。它能发现畸形 YAML，但不能验证 GitHub 表达式或 Workflow 依赖连线：

```bash
uv run --no-project --with pyyaml --python 3.12 \
  python -c 'import sys, yaml; [yaml.safe_load(open(path, encoding="utf-8")) for path in sys.argv[1:]]' \
  configs/nvidia-master.yaml perf-changelog.yaml .github/workflows/e2e-tests.yml
```

对 Master Config 而言，矩阵生成就是严格验证：[`validation.py`](../utils/matrix_logic/validation.py) 禁止未知字段，并同时验证 Master 条目和输出矩阵条目。运行能覆盖改动的最小准确 `test-config` 或过滤后的 `full-sweep`。

### 验证仅追加 Changelog 契约

`perf-changelog.yaml` 不是普通 YAML。除经过严格验证的 PR Link 修正外，历史字节不可修改。新条目必须追加到末尾，以恰好一个空行分隔，以一个换行结束，不得含 CR、Tab 或 NUL 字节，并且必须通过 Changelog Schema 且能生成有效矩阵。

验证器读取 Git 对象，而非未提交的 Working Tree 字节。先提交候选改动，再与真实 Base 比较：

```bash
git fetch origin main
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref HEAD
```

当 PR 将启用对应修饰标签时，加入 `--all-evals` 和/或 `--evals-only`；[`run-sweep.yml`](../.github/workflows/run-sweep.yml) 会传递相同参数。绝不使用 Formatter 重写 `perf-changelog.yaml`，也绝不能把单独通过 `yaml.safe_load` 当作充分的 Changelog 验证。

## 手动端到端派发

仅在完全相同的生成器命令已于本地成功后，才使用 [`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) 执行受限的一次性 Run。测试名称必须唯一。在通用模式中，`--ref main` 选择已部署的 Workflow 定义，输入 `ref` 则选择矩阵生成和基准 Job Checkout 的 Branch 或 SHA。

```bash
REPO=SemiAnalysisAI/InferenceX
TEST_NAME="manual-dsr1-h200-$(date -u +%Y%m%dT%H%M%SZ)"
TARGET_REF='<branch-or-full-sha>'

gh workflow run e2e-tests.yml \
  --repo "$REPO" \
  --ref main \
  -f ref="$TARGET_REF" \
  -f test-name="$TEST_NAME" \
  -f generate-cli-command='full-sweep --config-files configs/nvidia-master.yaml --single-node --model-prefix dsr1 --framework sglang --runner-type h200 --seq-lens 8k1k --min-conc 4 --max-conc 4 --no-evals' \
  -f duration-override=''
```

只使用 `gh workflow view ... --ref main --yaml` 展示的输入。不同派发 Ref 的输入可能不同，不得凭假设传入目标 Ref 特有的选项。

派发是异步的，Run 可能不会立刻出现。应按准确 Display Title 查找，而不是假定最新 Run 属于自己：

```bash
gh run list \
  --repo "$REPO" \
  --workflow e2e-tests.yml \
  --event workflow_dispatch \
  --limit 30 \
  --json databaseId,displayTitle,headBranch,headSha,status,conclusion,url

RUN_ID=$(gh run list \
  --repo "$REPO" \
  --workflow e2e-tests.yml \
  --event workflow_dispatch \
  --limit 30 \
  --json databaseId,displayTitle \
  | jq -r --arg title "e2e Test - $TEST_NAME" \
      '[.[] | select(.displayTitle == $title)][0].databaseId // empty')
```

如果 `RUN_ID` 为空，不得继续。Run Metadata 描述派发 Workflow 的 Ref，可能不等于输入 `ref`。解释 GPU 结果前，必须在 `get-jobs` 中确认唯一标题、生成器命令与 Checkout Ref。

## 每周 overview snapshot

[`weekly-overview-snapshot.yml`](../.github/workflows/weekly-overview-snapshot.yml) 每周六 06:00 UTC 运行一次精选 best-config sweep，使 `/overview` 每个 model×hardware 格子至少每周获得一个同批次数据点（#2304、#2586）。它以固定的 `test-config` key 列表调用 `e2e-tests.yml`，随后携带自身 run ID 向 InferenceX-app 派发 `ingest-results`，因此结果无需 merge 到 `main` 即可发布到生产数据库。

运维要点：

- 范围由 workflow 内默认 `config-keys` 列表决定；修改该列表，或通过 `workflow_dispatch` 输入 `config-keys` 按次覆盖。
- `workflow_dispatch` 输入 `skip-ingest: true` 只跑 sweep、不写生产数据库（冒烟测试）。
- 优先级：`schedule` 事件在 [`configs/ci-priority.yaml`](../configs/ci-priority.yaml) 中计 `-10.0`，打分器以 0 为下限，因此 snapshot job 以 `0.000` 排队——低于所有 PR 与 main-push job。
- 部分失败仍会入库：app 侧摄取会跳过失败的 benchmark 行，宁可发布半新快照也不丢掉整周。
- 恢复：sweep 已完成但 ingest 失败时走 [`recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml) 的常规恢复路径；sweep 本身失败则用相同 config keys 手动重新 dispatch 即可。
- overview 页面每个 serving series 只显示最新一次 run，因此 snapshot run 会替换其覆盖 config 的展示数字（已接受的取舍；见 #2586）。

## PR 主标签与修饰标签

[`run-sweep.yml`](../.github/workflows/run-sweep.yml) 会拒绝多个主标签。必须且只能应用一个：

| 主标签 | 矩阵范围 | Canary | 矩阵 Fail-fast |
| --- | --- | --- | --- |
| `sweep-enabled` | Changelog 矩阵裁剪为每个配置的最低并发 | 无 | 无 |
| `full-sweep-fail-fast` | 完整 Changelog 矩阵 | 有 | 有；推荐的完整扫描默认值 |
| `full-sweep-enabled` | 完整 Changelog 矩阵 | 有 | 无；需要每个矩阵点继续运行时使用 |
| `full-sweep-fail-fast-no-canary` | 完整 Changelog 矩阵 | 无 | 有 |
| `non-canary-full-sweep-enabled` | 完整 Changelog 矩阵 | 无 | 无 |

可选修饰标签不能替代主标签：

| 修饰标签 | 效果 | 合并后可复用？ |
| --- | --- | --- |
| `all-evals` | 将 Eval 选择扩展至每个已生成的固定序列配置；单独使用时是 Eval-only 简写 | 可以，前提是 Run 满足其他完整扫描复用规则 |
| `evals-only` | 禁用吞吐量，仅运行选定 Eval 条目；与 `all-evals` 组合即只运行所有 Eval | 不可以 |
| `agentx-fast` | 对 AgentX 吞吐量 Lane，在强制 Primer 后只加一次额外 Warmup Request，并使用 20 分钟 Profile；固定序列与 Eval 设置仍为规范值 | 不可以 |

修改被识别的主标签或修饰标签会共享活动扫描的 Concurrency Group，通常会取消并重启当前 Run。`skip_queue`、Patchwork、Waiver 与 Checklist 标签是 Gate/优先级输入，不是主扫描模式。Head Commit 含 `[skip-sweep]` 只会跳过 PR 基准 Setup；Changelog/复用检查仍会运行，推送到 `main` 时则忽略该标记。

## Canary 与 Fail-fast 语义

Canary 和 Fail-fast 解决不同问题：

1. 只有使用 `full-sweep-enabled` 或 `full-sweep-fail-fast` 的 PR 才创建 Canary。No-canary 标签和 `sweep-enabled` 会跳过它。
2. Canary 选择会检查单节点固定序列 `1k1k` 和 `8k1k` 条目，排除主要用途为 Eval 的条目，并选取最低并发候选。该条目随后会从单节点矩阵移除。
3. 如果没有合格候选，Canary 会被跳过。否则所有 Benchmark/Eval 矩阵都要求 Canary 成功；Canary 失败会阻止其扇出。
4. `full-sweep-fail-fast` 与 `full-sweep-fail-fast-no-canary` 会分别为每个矩阵 Job Family 设置 `strategy.fail-fast: true`。首个失败点会取消同一矩阵 Family 中排队或运行中的兄弟项；它不是跨所有独立 Family 的全局 Kill Switch。
5. 非 Fail-fast 标签会保持矩阵 Fail-fast 为 false，使其他点继续运行并保留更广泛的诊断覆盖。
6. Fail-fast Run 可能因失败后兄弟项被取消而最终显示 `cancelled`。将取消归类为基础设施事件前，必须先识别第一个真实失败。

手动 `e2e-tests.yml` 没有 Canary。它的 `fail-fast` 输入默认为 false，并传递给每个矩阵 Job Family。始终使用派发 Ref 对应的定义。

## 监控与重跑

### 监控选定 Run

```bash
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed
gh api "/repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN_ID" \
  --jq '[.id, .run_attempt, .event, .head_sha, .status, .conclusion, .html_url] | @tsv'
```

监控第一个 Canary 或矩阵失败，并在重跑前完成分类：

- **配置/运行时失败：** 可复现的 Launcher、验证、模型、镜像、就绪、OOM 或结果错误。修复源码并生成新 Run。
- **Runner/基础设施抖动：** Runner 离线、临时网络/存储/服务故障或无关取消。保留日志，确认改动本身无责后再重跑。
- **策略/Gate 失败：** 标签冲突、Changelog 无效、缺少授权、合并冲突或产物不合格。修正 Gate；重跑 GPU 无法解决。
- **已被取代的 Run：** 后续 Commit 或被识别的标签变更通过 Workflow Concurrency 将其取消。应监控替代 Run，不要复活过期证据。

### 安全重跑

不要盲目重跑仍在执行的 Run。已结束的失败 Run 可以只重跑失败 Job 及其依赖项：

```bash
gh run rerun <RUN_ID> --failed --repo SemiAnalysisAI/InferenceX
```

对已结束且为 `cancelled` 的 Fail-fast Run，应重跑整个 Attempt，从而重新创建被取消的矩阵点：

```bash
gh run rerun <RUN_ID> --repo SemiAnalysisAI/InferenceX
```

重跑沿用同一个 Workflow Run ID，但 Attempt 会增加。Artifact API 可能包含多个 Attempt 上传的产物；必须保留 `run_attempt` 并检查 Artifact 时间戳。`run-stats` 会有意统计所有 Attempt 的 Job。如果源码需要变化，不应重跑旧代码：推送修复并监控新 Run。移除再重新添加主扫描标签会强制创建新的 Labeled Run；后续 Commit 也可能使复用资格失效。

## 暂存结果

[`stage-results.yml`](../.github/workflows/stage-results.yml) 是维护者专用的 PR 结果预发布路径，不能替代合并或生产入库。

请求只有在全部满足下列条件时才可暂存：

- 评论者具有仓库 `write`、`maintain` 或 `admin` 权限。
- PR 当前具有四个完整扫描标签之一；`sweep-enabled` 不够。
- 候选是已结束的 PR `run-sweep.yml` Run，创建时完整扫描标签处于活动状态，结论为 `success`、`failure` 或 `cancelled`。
- 候选按照 Workflow 当前 Head/历史 Pin 规则与该 PR 关联。
- 存在未过期的 `changelog-metadata`，并且至少存在 `results_bmk`、`eval_results_all` 或 `bmk_agentic_*` 之一。因此失败/取消的 Run 可以暂存有用的部分数据，但空 Run 或仅有 Metadata 的 Run 不行。

授权维护者准确评论以下一条：

```text
/stage-results
/stage-results <run-id>
```

不提供 ID 时，Workflow 会选择 PR Branch 上最新的可暂存已结束 Run，并要求其 Head SHA 仍在 PR Commit 列表中。指定 ID 时允许使用明确关联的历史 Run。Workflow 会确认所选 Run，向 InferenceX-app 派发 `stage-results` 事件，并由 [`stage-results-callback.yml`](../.github/workflows/stage-results-callback.yml) 用成功图表或失败链接替换确认评论。

预发布会保留之前已暂存的 Run。再次暂存同一个 Run ID 会更新该 Run 的预发布数据。必须保留源 Run ID 与下游 App Workflow 链接；预发布成功不证明生产复用资格或合并后入库成功。

## 产物复用与 merge-with-reuse

复用可以避免已批准的完整 PR 扫描在 `main` 上再次运行；它不能绕过 Changelog 验证。

### 资格与授权

1. PR 必须保留且只保留一个完整扫描主标签：`full-sweep-enabled`、`non-canary-full-sweep-enabled`、`full-sweep-fail-fast` 或 `full-sweep-fail-fast-no-canary`。
2. `evals-only` 与 `agentx-fast` 会令 Run 不可复用。默认完整扫描以及带 `all-evals` 的完整扫描仍可复用。
3. 源 Run 必须是已结束的 PR `run-sweep.yml` Run，其 Head SHA 仍在 PR Commit 列表中，并拥有未过期的 `results_bmk`、`eval_results_all` 或 `bmk_agentic_*` 结果产物。
4. `OWNER`、`MEMBER` 或 `COLLABORATOR` 通过 `/reuse-sweep-run` 或 `/reuse-sweep-run <run_id>` 授权复用。最新的合格授权命令决定自动选择还是固定源 Run。
5. 不指定 ID 时，自动选择要求最新的合格源 Run 成功。指定 Run 是维护者的明确决定，允许结论为 `success`、`failure` 或 `cancelled`；下游入库只保留存在且有效的行，因此应将其报告为部分数据，而不是绿色 Run。

评论本身不会触发 Run。在之后的 PR `synchronize` 事件上，复用 Gate 会在 Changelog 验证后跳过另一轮 PR 扫描。在 `main` 上，映射不明确、指向无效 Run 或与标签冲突的授权会 Fail Closed。没有授权时，`main` 执行正常扫描。

### 受支持的合并路径

在具有已认证 `gh`、`git`、`jq` 和 Python 的干净 Checkout 中运行：

```bash
utils/merge_with_reuse.sh <pr-number>
```

[`merge_with_reuse.sh`](../utils/merge_with_reuse.sh) 会验证合格的成功源产物、发布授权、把 `origin/main` 合并进 PR Branch、只解决 `perf-changelog.yaml` 冲突、规范化追加条目中的 `XXX` Link、按需创建并推送 Synchronization Commit、等待 `check-changelog` 和全部 PR Check、再次确认 Head 未移动，最后执行 Admin Squash Merge。它会拒绝 Fork、脏 Working Tree、多个主标签、不兼容修饰标签、意外冲突、缺少产物、失败 Check 或移动过的 PR Head。

不要只手工复制该序列的一半。尤其是，只发表评论后直接 Squash Merge、却不执行 Synchronization/Check 阶段，可能导致 Merge Run 无法选择预期源 Run。

在 `main` Run 中，[`run-sweep.yml`](../.github/workflows/run-sweep.yml) 会向 InferenceX-app 发送两个不同 ID：

- `source-run-id`：包含 Benchmark/Eval Artifact 的 PR Run。
- `merge-run-id`：包含合并时 `changelog-metadata` 的 `main` Run。

公开数据行和链接保留源 Run 溯源。源产物覆盖范围是权威依据；后续矩阵策略变化不会凭空补出缺失点。

## Changelog 冲突恢复

每个 PR 都会在文件末尾追加，因此 `perf-changelog.yaml` 经常冲突。绝不能只接受 `ours` 或 `theirs`，也不能重新格式化或手工三方合并历史区块。

普通 PR 同步时，先记录 PR 编号，Fetch `main` 并合并：

```bash
PR=<pr-number>
git fetch origin main
git merge origin/main
```

当且仅当 `perf-changelog.yaml` 是未解决文件时，在三个冲突 Stage 仍存在的情况下使用字节保留 Helper：

```bash
python3 utils/prepare_perf_changelog_merge.py resolve-conflict \
  --changelog-file perf-changelog.yaml \
  --pr-number "$PR" \
  --repo SemiAnalysisAI/InferenceX
git add perf-changelog.yaml
git commit --no-edit
```

Helper 会从 Index Stage 1/2/3 读取 Merge Base、PR 与 Main 字节，验证 PR 侧 Delta，以当前 `main` 字节为起点，只重新追加唯一的 PR 贡献，规范化 PR Link，并验证最终原始字节契约。如果 Helper 拒绝，应停止：意外历史编辑、冲突的 PR Link 修正、贡献缺失或非 Changelog 冲突都需要维护者审查。不得猜测三方合并结果。

提交后，对 `origin/main` 运行准确 Gate：

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref origin/main \
  --head-ref HEAD
```

复用已获授权时，优先使用 [`utils/merge_with_reuse.sh`](../utils/merge_with_reuse.sh)；它会一次完成冲突准备以及所需的 Synchronization/Check 序列。

## 产物下载与解析

### 下载前列出溯源

使用 REST Endpoint，以便跨所有分页查看过期状态、大小与时间戳：

```bash
REPO=SemiAnalysisAI/InferenceX
RUN_ID=<run-id>
gh api --paginate "/repos/$REPO/actions/runs/$RUN_ID/artifacts?per_page=100" \
  --jq '.artifacts[] | [.name, .expired, .size_in_bytes, .created_at, .updated_at] | @tsv'
gh api "/repos/$REPO/actions/runs/$RUN_ID" \
  --jq '[.id, .run_attempt, .event, .head_sha, .status, .conclusion, .html_url] | @tsv'
```

只下载回答当前问题所需的命名 Artifact：

```bash
OUT="/tmp/inferencex-run-$RUN_ID"
mkdir -p "$OUT"
gh run download "$RUN_ID" --repo "$REPO" -n results_bmk -D "$OUT/results_bmk"
gh run download "$RUN_ID" --repo "$REPO" -n eval_results_all -D "$OUT/eval_results_all"
gh run download "$RUN_ID" --repo "$REPO" -n run-stats -D "$OUT/run-stats"
gh run download "$RUN_ID" --repo "$REPO" -n changelog-metadata -D "$OUT/changelog-metadata"
```

不要假设每个 Run 都有每个 Artifact。重要契约如下：

| Artifact | 典型文件 | 来源 |
| --- | --- | --- |
| `results_bmk` | `agg_bmk.json` | Result Collector 找到的 `bmk_*` 吞吐量/Agentic JSON |
| `eval_results_all` | `agg_eval_all.json` | Eval Collector 汇总的 `eval_*` Artifact |
| `run-stats` | `run_stats.json` | 跨所有 Attempt 的硬件成功计数 |
| `changelog-metadata` | `changelog_metadata.json` | 扫描 Setup 的 Search-space Metadata |
| `bmk_agentic_*` | 单 Job JSON | Agentic 入库/预发布使用的原始 AgentX Result Upload |
| `server_logs_*`、`multinode_server_logs_*`、`gpu_metrics_*`、`agentic_*` | Log、Metric 或诊断 Payload | `always()`/诊断上传；名称随 Template 与模式变化 |

### 解析有限字段

吞吐量聚合字段来自 [`utils/process_result.py`](../utils/process_result.py)：

```bash
jq -r '
  .[] |
  [.hw, .infmax_model_prefix, .framework, .precision,
   "\(.isl)/\(.osl)", .tp, .conc,
   (if .tput_per_gpu == null then "" else ((.tput_per_gpu * 100 | round) / 100) end)] |
  @tsv
' "$OUT/results_bmk/agg_bmk.json"
```

Eval 聚合字段来自 [`utils/collect_eval_results.py`](../utils/collect_eval_results.py)：

```bash
jq -r '
  .[] |
  [.hw, .model_prefix, .framework, .precision,
   .tp, .conc, .task, .score_name,
   (if .score == null then "" else ((.score * 10000 | round) / 10000) end)] |
  @tsv
' "$OUT/eval_results_all/agg_eval_all.json"
```

检查 Run Statistics 时，不要把 Skipped Job 与实际尝试的 Job 混为一谈：

```bash
jq -r 'to_entries[] | [.key, .value.n_success, .value.total] | @tsv' \
  "$OUT/run-stats/run_stats.json"
```

面对不熟悉或原始 Agentic Artifact，应从 `jq 'type, keys'` 开始，并在选择字段前阅读其生产脚本。绝不能为了回答狭窄问题而加载或粘贴数 MB 的 Artifact。报告提取值时，同时给出 Run ID、Attempt、准确 Artifact 名称与过滤条件。

## 合并后预期

在 `main` 发布路径和下游入库均得到验证前，合并在运维层面并未完成。

1. 只有 `perf-changelog.yaml` 发生变化时，推送到 `main` 才会触发 [`run-sweep.yml`](../.github/workflows/run-sweep.yml)。确认 Merge Commit 确实产生该 Run；不要假设无关合并也会触发它。
2. 没有有效复用授权时，`main` Run 会处理 Changelog Delta 并运行正常矩阵。PR Canary 逻辑不会在 `push` 上执行，PR 标签驱动的 Fail-fast 在该事件上也不可用。
3. 使用复用时，Benchmark Job 会被跳过，Run 会组合源 Artifact 与 Merge Run Changelog Metadata。应确认 Setup Output 选择了预期源 Run，不能仅凭 Job 被跳过就推断复用成功。
4. `upload-changelog-metadata` 必须产出 `changelog-metadata`。对于不含 Agentic 条目的 Search Space，`trigger-ingest` 派发 `ingest-results`；Agentic Search Space 遵循单独的 `trigger-agentic-ingest` 条件，并携带 `database-target: production` 派发 `ingest-agentic-results`。
5. 绿色 Dispatch Step 只证明 GitHub 接受了发往 InferenceX-app 的 Repository Dispatch。必须继续追踪下游 InferenceX-app Run，并验证预期数据行/链接和源 Run 溯源；其入库实现在本 Checkout 之外。
6. 检查最终 `main` Run 结论、每次重跑 Attempt、聚合 Artifact、Metadata 与发布结果。PR 作者仍负责确保全部合并后 Actions Job 通过，包括需要有依据地重跑的偶发抖动。
7. 如果有效 Artifact 已存在，不能仅因下游入库失败就重跑 GPU Benchmark。对于失败的复用 **Agentic** 入库，授权维护者可使用 [`recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml)，传入原始 Source 与 Merge ID；该 Workflow 只派发 `ingest-agentic-results`，不是通用固定序列恢复工具：

   ```bash
   gh workflow run recover-reused-ingest.yml \
     --repo SemiAnalysisAI/InferenceX \
     --ref main \
     -f source-run-id='<source-pr-run-id>' \
     -f merge-run-id='<main-merge-run-id>'
   ```

当源 Run、Merge Run、Artifact 覆盖、Changelog Metadata 或下游 Event 含糊不清时，应停止并升级处理。绝不能替换成方便的 Run ID，也不能仅凭 Actions Dispatch 就宣称发布成功。
