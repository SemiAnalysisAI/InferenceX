<div align="center">

[English](./testing.md) | **中文**

</div>

# 测试

先使用能够证伪本次变更的最小检查，仅当变更契约跨越更多层级时才扩大范围。本地检查可以快速证明语法、模式、生成与转换行为；GPU 执行则证明分配、服务、工作负载和制品行为。冒烟运行可以缩短反馈时间，但不能替代合并评审所要求的全量扫描与评测证据。

## 索引

- [事实来源](#事实来源)
- [测试层级](#测试层级)
- [测试质量](#测试质量)
- [本地检查](#本地检查)
- [冒烟、扫描与评测](#冒烟扫描与评测)
- [证据标准](#证据标准)
- [评审门禁](#评审门禁)
- [停止条件](#停止条件)

## 事实来源

- [`.github/AGENT_OPERATIONS.md`](../.github/AGENT_OPERATIONS.md#sweep-labels-and-reuse) 定义扫描标签与修饰标签；其[派发章节](../.github/AGENT_OPERATIONS.md#workflow-dispatch-and-monitoring)定义手动运行与产物检查。
- [`docs/configuration-procedures.md`](./configuration-procedures.md#validate) 是聚焦配置验证的操作流程。
- [`.github/workflows/README.md`](../.github/workflows/README.md) 记录矩阵生成、`e2e-tests.yml`、PR 扫描和复用。
- [`run-sweep.yml`](../.github/workflows/run-sweep.yml) 是可执行的 PR/push 门禁；[`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) 是手动分发的端到端路径。
- [`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) 是合并评审标准。[验证器提示词](../.github/codeowner-signoff-verify-prompt.md#check-1--a-passing-sweep--evals-ran-on-a-commit-in-this-pr) 说明如何独立核验扫描和评测证据。

当行为发生变化时，上述来源优先于本指南。先更新英文页面，再把相同结构和证据翻译到本页中文对应版本。

## 测试层级

| 层级 | 能够证明 | 不能证明 |
| --- | --- | --- |
| 解析与语法 | 编辑后的 YAML 可加载；编辑后的 Bash 可解析 | 模式有效性、运行时路由或 GPU 行为 |
| 模式与矩阵 | 配置键通过验证并发出预期矩阵字段 | 运行器可用性、服务器启动或性能 |
| 聚焦 Python 测试 | 变更后的生成器、changelog、结果、评测、收集或复用契约在覆盖输入上行为正确 | 容器、加速器、网络或 Slurm 行为 |
| 冒烟运行 | 一条严格过滤的路径可完成分配、启动服务器、运行工作负载并产生制品 | 完整并发/搜索空间或合并资格 |
| 精简 PR 扫描 | 每个选中单节点分组运行其最低并发（`sweep-enabled`） | 全量扫描所要求的中间并发点 |
| 全量扫描与评测 | 选中的未精简矩阵和评测任务在被评审提交上实际执行 | 未检查证据的正确性或无关配置 |

较后层级变绿不会弥补较早层级缺少证据。例如，绿色收集器可能只聚合了空集合，因此评审必须检查底层实际执行的任务和制品。

## 测试质量

测试应保护实际行为，而不是凑覆盖率。评审时，要明确每个测试能发现什么实际缺陷，以及它是否运行了真正交付的实现。

- 优先使用小规模输入和人工推导的预期结果，覆盖相关边界、无效输入或失败场景。不要照搬实现中的计算过程，也不要调用同一个辅助函数生成预期结果。
- 不要把当前配方数量、模型或硬件清单、镜像 pin、枚举定义或源码文本写成快照断言。新增有效配方或进行等价重构，不应迫使开发者修改无关断言。
- 保留真正的契约：数值结果、无效输入拒绝行为、稳定的产物格式，以及由不同组件独立读取的配置之间的一致性。只断言使用方真正依赖的部分。
- 必要时可以模拟外部服务或进程，但必须运行被测行为本身。测试里复制的解析器、过滤逻辑或假实现，无法发现真实实现中的回归。
- 冗余测试应直接删除，不必一一补上。只有存在实质性覆盖缺口时才扩展已有 fixture；不要为了维持测试数量而新建测试框架。

参见 [Randy Coulman 的 Tautological Tests](https://randycoulman.com/blog/2016/12/20/tautological-tests/)，了解独立预期结果与仅仅重复实现的断言之间的区别。

## 本地检查

从仓库根目录运行检查，并用实际变更路径或键替换占位符。

### 解析与语法

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('configs/runners.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
bash -n benchmarks/<path>/<script>.sh
bash -n runners/launch_<cluster>.sh
```

解析只是第一道门禁。不要把 YAML 解析结果报告为矩阵验证。

### 先精确配置，再过滤配置族

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --config-keys <exact-key>

uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --model-prefix <prefix> \
  --framework <framework> \
  --precision <precision> \
  --runner-type <runner> \
  --seq-lens 1k1k 8k1k
```

不要只检查退出码或行数，还要检查发出的值：配置键、模型、镜像、运行器、场景、并发、`max-model-len`、TP/PP/EP/DCP/PCP、prefill/decode worker、硬件、路由器、KV 传输、评测标志、`additional-settings` 和 `spec-decoding`。模式位于 [`validation.py`](../utils/matrix_logic/validation.py)，生成器是 [`generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py)。

### 按变更契约选择聚焦测试套件

| 变更 | 聚焦命令 |
| --- | --- |
| 矩阵模式或生成 | `python -m pytest utils/matrix_logic/ -v` |
| Changelog 内容或 PR 门禁 | `python -m pytest utils/test_process_changelog.py utils/changelog_gate_tests/ -v` |
| 结果处理 | `python -m pytest utils/test_process_result.py utils/test_aggregate_power.py utils/test_calc_success_rate.py -v` |
| 评测分发、批处理或补丁 | `python -m pytest utils/evals/ -v` |
| 评测收集 | `python -m pytest utils/test_collect_eval_results.py -v` |
| 扫描复用或可复用制品 | `python -m pytest utils/test_find_reusable_sweep_run.py utils/test_validate_reusable_sweep_artifacts.py -v` |

若编辑了 changelog，还要使用真实 base 和 head ref 运行 setup 所用的同一矩阵兼容性验证器：

```bash
python3 utils/validate_perf_changelog.py \
  --changelog-file perf-changelog.yaml \
  --base-ref <base-ref> \
  --head-ref <head-ref>
```

其契约实现在 [`validate_perf_changelog.py`](../utils/validate_perf_changelog.py) 中。该检查会验证生成矩阵并拒绝禁止的内容变更，但其差异读取器可能看不到仅空白的历史删除。应把精确字节差异检查作为独立证据门禁；不要改写或规范化 `perf-changelog.yaml` 历史字节。

本地矩阵不能证明 Slurm 分配或 llm-d 端点发现。多节点配方变更仍然需要上游配方检查器，并在目标集群上实际执行；详见[配置验证](./configuration-procedures.md#validate)。

## 冒烟、扫描与评测

### 冒烟

冒烟运行是手动分发的 [`e2e-tests.yml`](../.github/workflows/e2e-tests.yml) 运行，其生成器命令严格限制到变更的模型/框架/运行器/场景和最小受支持并发。先在本地生成完全相同的命令。冒烟证据用于回答“该镜像能否在该运行器上启动该服务器并产生结果制品”这类窄问题。

冒烟运行不是合并证据：它有意省略配置和并发点。同样，`agentx-fast` 是 AgentX 预检，不是规范 AgentX 结果；[派发参考](../.github/AGENT_OPERATIONS.md#workflow-dispatch-and-monitoring)定义缩短后的预热/分析行为。

### 精简与全量扫描

- `sweep-enabled` 把每个并行分组精简到最低并发，是大多数 PR 反馈的默认选择。
- `full-sweep-fail-fast` 是推荐的全量扫描标签。它使用串行单节点 canary，并在每个矩阵首次失败后停止该矩阵，同时保留已完成结果。
- 仅当 canary 已知不稳定或不具代表性时才使用无 canary 的全量扫描标签。仅当即使失败也必须让每个矩阵任务继续时，才用 `full-sweep-enabled` 代替 fail-fast。
- 必须且只能应用一个主扫描标签。只有修饰标签或存在冲突主标签都不构成有效扫描。

当前含义和资格规则由[扫描标签参考](../.github/AGENT_OPERATIONS.md#sweep-labels-and-reuse)定义，并由 [`run-sweep.yml`](../.github/workflows/run-sweep.yml) 实现。

### 评测

吞吐与评测是独立任务。默认扫描对选中的 8k1k 子集进行评测；`all-evals` 扩大评测选择，`evals-only` 抑制吞吐。根据变更范围选择修饰标签，但不要用仅评测或预检运行替代所需的全量扫描。

评测完成不能只看绿色任务。保留并检查 `meta_env.json`、`results*.json` 文件、分数验证输出、推理镜像和聚合评测制品。[`utils/evals/EVALS.md`](../utils/evals/EVALS.md) 负责任务与制品行为。[`validate_scores.py`](../utils/evals/validate_scores.py) 会拒绝缺失结果文件、低于阈值的分数和没有任何已检查指标的运行；当存在预期并发元数据时，它还会拒绝无效、不完整或失败的批次。工作流调用时没有传入 `--expected-concs`，因此评审者必须独立验证单并发制品中的 `meta_env.json`。

## 证据标准

记录足够信息，让另一位评审者无需猜测即可复现结论：

1. 精确提交 SHA，以及该提交是否仍在 PR 中。
2. 精确本地命令或工作流生成器命令，包括所有过滤器和修饰标签。
3. 工作流 URL、run ID、attempt、任务/check 名称和结论。重跑前保留失败日志。
4. 配置键，以及解析后的模型、镜像、运行器、框架、精度、场景、拓扑和并发范围。
5. 制品名称和相关结构化字段或摘要指标。不要粘贴无限量原始聚合数据。
6. 对评测记录任务、预期阈值、观测分数、完成元数据，以及被评测镜像与 PR 配置一致的证明。
7. 对失败记录首个失败层级，以及排除更早层级的证据；使用 [`troubleshooting_zh.md`](./troubleshooting_zh.md) 中的分类。

“CI 是绿色”、没有运行 URL 的截图、没有实际执行任务的收集器成功，或来自已被 rebase 移出 PR 的提交制品，都不是充分证据。

## 评审门禁

1. **开始 GPU 工作前：**解析、精确键生成、相关聚焦测试套件和发出字段检查均为绿色。精确的分发生成器命令已在本地运行。
2. **扩大范围前：**冒烟或 canary 已证明变更后的运行时路径。如果失败，先诊断该层级，再花费全量扫描资源。
3. **CODEOWNER 签署前：**遵循 [`PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md)，包括适用的代码质量、架构、镜像来源、上游配方、补丁/豁免、聊天模板和 AgentX 要求。
4. **扫描/评测验收：**当前仍在 PR 中的至少一个提交拥有成功、未跳过且实际执行的 `single-node */` 与 `eval /` 检查。仅 `collect-evals` 成功不够。下载对应评测制品，确认其非空、准确率达标且使用同一推理镜像。这些可执行规则位于[验证器检查 1 和 2](../.github/codeowner-signoff-verify-prompt.md#check-1--a-passing-sweep--evals-ran-on-a-commit-in-this-pr)。
5. **合并时复用：**获授权的 `OWNER`、`MEMBER` 或 `COLLABORATOR` 必须在受支持的合并路径前发布独占一行的 `/reuse-sweep-run` 命令（可附带合格来源 run ID）。验证器会把命令缺失或发布者未授权视为失败；参见[验证器检查 4](../.github/codeowner-signoff-verify-prompt.md#check-4--reuse-sweep-command-explicitly-posted)和[复用流程](../.github/workflows/README.md#reusing-an-approved-pr-full-sweep)。
6. **合并时：**CODEOWNER 的精确签署须由 [`codeowner-signoff-verify.yml`](../.github/workflows/codeowner-signoff-verify.yml) 独立接受。如果 PR head 变化，重新评估并签署新提交的证据。
7. **合并后：**作者按照 [`CONTRIBUTING.md`](../CONTRIBUTING.md#after-merging) 的要求确认 main 分支任务通过。

## 停止条件

遇到以下情况时停止并修复或升级处理，不要扩大运行或批准：

- 语法、模式、精确键生成、changelog 验证或聚焦测试套件失败；
- 即使命令退出为零，生成字段仍与预期配置不同；
- 多节点变更无法在目标运行器/集群上执行必要前置验证；
- 冒烟或 canary 失败且失败层级仍未知；
- 预期任务被跳过、取消、缺失，或仅关联到已不在 PR 中的提交；
- 评测制品缺失、为空、不完整、低于阈值，或使用不同镜像；
- 必需证据、上游配方、豁免或由评审者负责确认的清单事实仍未知。

不要把未知项转成通过，不要在未保存原始失败证据的情况下反复重跑直至偶然通过，也不要用更广扫描掩盖更窄失败。
