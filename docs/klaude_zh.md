# Klaud Cold 自动扫描

<div align="center">

[English](./klaude.md) | **中文**

</div>

[`klaude-plan.yml`](../.github/workflows/klaude-plan.yml) 使用 Python 准备候选，通过简短的只读 Claude 检查排除与开放 PR 重复的工作，再调用 [`klaude-candidate.yml`](../.github/workflows/klaude-candidate.yml)。每个所选候选由一个自主运行的 Klaud Cold 会话负责分支、草稿 PR、benchmark、修复和报告。只读 Stop hook 在同一会话内检查未结束的运行。后续步骤仅保存脱敏诊断；没有 attempt 工作流、Python 执行控制器或后续报告 agent。

PR 检查使用 `claude-opus-5`（Opus 5），关闭 fast mode（`fastMode: false`），最多运行 200 轮。候选执行使用 `claude-fable-5-1`（Fable 5.1），关闭 fast mode。Agent 的显示名称为 **Klaud Cold**；现有 `klaude` 工作流文件名、CLI、产物、分支和 secret 标识保持兼容。

## 候选选择

[`api.py`](../utils/klaude/api.py) 通过同一 HTTP 实现处理公开和私有读取。[统一 CLI](../utils/klaude/__main__.py) 为 `python -m utils.klaude`：`plan --directory DIR` 准备候选和开放 PR；`select --max-candidates-per-run N --directory DIR` 校验 Claude 的结构化 `KLAUDE_PR_REVIEW` 输出并应用工作流中的候选数量上限；`check-capacity --cluster ID` 在调度前重新检查容量。在 `plan` 前传入 `--root PATH`，可指定用于获取候选基准 SHA 的 checkout；不再加载配置文件。

- 公开读取使用 `/api/v1/latest-images` 和 `/api/v1/framework-releases`。候选身份和版本键直接来自响应。版本不匹配和不稳定镜像仅作为检查线索，兼容的新镜像由 Klaud Cold 核实。
- 私有读取仅使用 `/api/status/clusters`。要求 schema-v6 响应有效且新鲜、API 对集群判定为 `stale: false`、观测与接收时间戳有效且顺序正确、状态为 operational 或 degraded，并且至少有一个空闲节点。集群数据的过期阈值由 API 配置决定；Klaud 的 120 秒限制只用于响应获取/生成时间，不用于集群观测时间。排队任务、调度器预留和优先级覆盖不会改变低于 20% 的利用率规则。
- 公开 API/CDN 负责 HTTP 缓存的新鲜度。Klaud 不再因刚获取响应的 `Age` 超过两小时而拒绝它：镜像接口允许 24 小时 CDN 缓存，版本接口允许在重新验证期间返回旧缓存。本地获取时间检查仍阻止复用过期的已获取快照，响应内容校验仍然有效。读取失败会报告接口和固定错误码（例如 `images:http-503`）；校验失败会列出具体问题码。准备阶段以非零状态退出，在 Actions 错误和摘要中说明原因，不输出 traceback 或原始响应。私有容量的新鲜度检查独立且不变。
- `plan` 将公开硬件类别与符合容量条件的遥测集群求交集，按公开候选身份去重，跳过已知 Klaud 分支/PR，再将不同候选随机打乱一次。每个准备好的候选都有机会，无需重复抽取。它通过分页获取匹配的分支和全部开放 PR，包括草稿和任意名称的分支。它一次性预取变更文件路径供检查使用。读取失败或达到 GitHub 的 3,000 个文件上限时，列表保持不完整，交由 Claude 进一步调查；重命名文件同时包含原路径。仓库身份来自 `GITHUB_REPOSITORY`。
- Claude 根据 runner 和现有配置，对照私有 `capacity.json` 中的路由线索，解析当前配置族及其**全部实际目标集群**。每个目标都必须符合条件；同类硬件的健康兄弟集群不能替代另一个集群。无法证实映射或目标不可用时标记为 `uncertain`，并按随机顺序继续检查，直到补足名额。路由解释仍由 agent 负责，不新增 Python recipe/别名目录。Claude 还会检查开放 PR 的变更文件，再按需阅读正文和 diff。已有镜像更新或重叠的兼容性修改会阻止候选，即使目标镜像 tag 不同；仅涉及相同硬件的无关工作不算重复。结构化决策包含 `candidate-id`、`decision`（`proceed`、`duplicate` 或 `uncertain`）、`family`、精确的 `telemetry-clusters`、重叠 PR 编号列表 `pull-requests` 和不含私有资格数据的简短 `reason`。
- `select` 仅接受通过校验的 `proceed` 决策，刷新私有容量数据，要求检查结果中的每个目标仍符合条件，按解析后的配置族去重，再按随机顺序应用总数量上限。`capacity-deferred-candidates` 记录最终容量检查未通过的候选。某个配置族被标记为重复或不确定时，即使另一条观测允许继续，也会排除整个配置族。缺失或未检查的候选延后。检查 action 失败、输出格式错误、未知/重复 ID 或容量 API 不可用会让本次选择延后，绝不绕过重叠检查。候选失去容量资格后，可由后续已检查且仍符合条件的配置族补位。

两个 agent 都明确获得证据目录的访问权限。检查 agent 使用 Read/Glob/Grep 检查本地内容，每次 Bash 调用只执行一个允许的只读 gh/git 命令，避免 shell 包装和管道。两个 Klaud 工作流均不设置作业/步骤超时或 Bash 超时覆盖，使用 GitHub Actions 默认限制。预取也不设置整体截止时间。`selection.json` 记录延后原因，作业摘要报告所选/延后数量。`review-diagnostics.json` 仅保留时长、轮数、费用等数值指标、按工具汇总的拒绝次数及固定的 Bash 分类（例如 shell 包装或文件过滤）；不包含原始命令、路径、消息、结果或凭据。历史日志只提供拒绝次数，无法还原之前被拒绝的具体命令；新增分类和访问指令仍需实际运行验证。诊断文件缺失不阻止收尾。检查步骤之外的基础设施故障或整个作业被取消仍可能导致无法完成。

私有容量门槛是 **节点利用率严格低于 20%**：`(summary.allocatedNodes + summary.mixedNodes) * 5 < summary.totalNodes`，不做舍入。完全分配和部分使用的节点均计入已使用节点；恰好 20% 时不放行。不扣除预留节点。要求至少有一个空闲节点，避免整个集群不可用时仍以 0% 利用率通过检查。缺失、无效、不一致、过期或不可用的数据均拒绝。硬件匹配只用于初筛；检查阶段必须解析每个实际目标，选择阶段重新检查这些精确 ID。完整物理节点需求的准入仍由调度器负责。

`klaude-plan` 产物仅显式包含 `candidates.json`、`open-prs.json`、`selection.json`、`review-diagnostics.json` 及每个所选候选的 `candidate.json`。本地 `capacity.json` 为检查阶段提供遥测 ID 和资格线索，**绝不上传**；任意临时文件也不会上传。每份交接文件包含公开观测、版本线索、检查原因、分支、基准 SHA、公开 API 发现 URL 和通过校验的 `pr-review`，不包含私有节点计数或原始遥测。所选候选并行运行，各自获得独立 Klaud Cold 会话。一个候选失败不会取消其他候选。不再复制模型/runner 目录，也不保留 `recipes.py`；agent 使用现有 InferenceX 配置和工具理解实际 recipe。

## Klaud Cold 负责执行

完成 checkout 和上下文准备后，candidate 工作流将控制权交给 Klaud Cold，并提供 `CLAUDE_PAT`、`ANTHROPIC_API_KEY` 和私有 API 只读密钥。Klaud Cold 将公开观测解析到一个活动主配置族，检查当前镜像和已有 PR，并在**编辑或创建分支/PR 之前**核实检查结果中的目标 ID 和实时容量。随后在使用 GPU 前认领分支，产生实际修改并创建草稿 PR。所有生成的 PR 标题必须以 `[Klaud Cold] ` 开头，后接英文 / 简体中文描述。不得在 GitHub 上 @提及用户/团队，也不得请求 review/re-review；这些操作由自动流程处理。它在认领分支前立即重新检查开放 PR，因为检查只是快照，不能锁住后来创建的人工 PR。有歧义、已退役或已经更新的候选直接停止，不运行扫描。提交、推送、调度、监控、诊断、修复及双语 PR 更新都由同一会话完成。

提示词要求 Klaud Cold 通过 `main` 上现有的 `e2e-tests.yml` 仅测量更新后的镜像及其修复尝试，将实际测量提交的 SHA 传入 `inputs.ref`，将完整配置族的 `test-config` 命令传入 `generate-cli-command`。它读取当前 `configs/*-master.yaml`、`configs/runners.yaml` 并使用现有矩阵生成器 CLI，不再维护另一份 recipe 目录。保留默认 eval、所有配置测试点、物理 `nodes:N` 标签、MTP chat template 和产物约定。每次调度前，使用 `check-capacity --cluster ID` 检查精确目标；通过重复 `--cluster` 指定每个可能的目标。退出状态 0 要求全部目标均通过新鲜度、可用性和低于 20% 利用率检查。其他结果表示报告延后并停止，不等待恢复，也不承诺自动继续。已有 PR 保持草稿，不创建占位 PR。命令不打印容量详情。

Klaud Cold 读取运行产物和日志，计算匹配性能差值并在 PR 中发布证据。更新后的镜像能够正常工作、通过所选 benchmark 和默认 eval，即视为成功；性能优化尽力而为，性能回退如实报告，不按百分比阈值拒绝更新。PR 和最终报告必须包含双语 Markdown 汇总表，分别记录带发布日期的已有基线及每次更新或修复尝试：尝试编号、镜像/SHA、运行 URL、benchmark/eval 结果、相对基线的匹配吞吐量和延迟变化，以及诊断结论。明确标注各测试点的差值；对于缺失或不可比较的测量，包括失败或取消的尝试，填写 `N/A` 并说明原因。无效、重复或未匹配的测试点不得用于声称性能提升，并应披露相关限制。所选 benchmark 成功、性能证据完整和全局 PR 审批是不同结论。下方的公开 API 调查指南说明如何按需读取补充信息。公开 dashboard 中已有的数据作为基线；只有新镜像尝试消耗 GPU 时间。空汇总文件和成功的收集作业不能证明 benchmark/eval 成功。原始 benchmark 产物保留在对应 e2e 运行中；结束后仅上传脱敏候选诊断，排除任意临时文件和执行记录。

无论测试点数量多少，都运行完整的所选配置族。单次 benchmark 运行可能长达三小时；Klaud Cold 在候选作业的总时限内监控其进展。Klaud 不再为单次 benchmark 运行设置额外超时。

两个设置直接保留在工作流中：[`klaude-plan.yml`](../.github/workflows/klaude-plan.yml) 的 `MAX_CANDIDATES_PER_RUN` 控制 planner 的候选数量上限；[`klaude-candidate.yml`](../.github/workflows/klaude-candidate.yml) 的 `MAX_REPAIRS` 将修复次数上限直接传入 agent 提示词。

工作流并发和选择规则由代码强制执行。Klaud Cold 最多运行 200 轮，必须在 GitHub Actions 默认作业时限内完成报告和清理。**修复判断和报告内容仍由 agent 负责；Stop hook 检查所属运行的状态。** Klaud Cold 必须在成功、修复预算耗尽、重复失败无进展、容量丢失或写操作结果不确定时停止，并取消未完成子运行、确认结束。作业超时或 runner/agent 突然终止仍可能留下未结束运行或未完成报告，因为没有后续作业接管。

Klaud Cold 调度 `e2e-tests.yml` 时显式设置布尔输入 `klaud-run: true`，并使用 `klaud-` 测试名称方便识别。手动和复用调用中的该输入均默认为 false，并传递至所有 benchmark/eval 模板。只有这个标志会为作业名称添加 `klaud | ` 前缀，供 InferenceX Dash 优先级调度器识别；普通测试名称不会改变优先级。配套调度器改动使 Klaud 始终排在普通人工任务之后，不受任务到达时间或 recipe 分数影响；Klaud 不获得等待时长加分或节点预留，skip-queue 请求也不生效。Klaud 使用剩余容量，不抢占已经运行的任务。显式管理员优先级覆盖保留原有优先顺序，Klaud 不得主动请求。启用 auto-sweep 前需部署配套 dashboard 调度器改动。

### 已发布基线与会话完成

基线来自 **`https://inferencex.semianalysis.com` 的公开 dashboard API**。将 `candidate.source.date` 传给 `workflow-info` 和 `benchmarks`；从 OpenAPI 解析展示模型名称，设置 `date` 和 `exact=true`，不使用计算器 `view`。核实旧镜像以及完整的模型、硬件、框架、精度、推测解码和工作负载身份，再逐点匹配拓扑、并发量及数据集。记录 API 查询、发布日期和每个测试点的来源 `run_url`/SHA，区分逻辑曲线快照与实际数据来源。按需读取已发布 eval，所有尝试共用这份固定基线。缺失或不可比较的数据填写 `N/A` 并说明原因。绝不调度或重跑旧镜像基线。

调度运行或创建草稿不代表任务完成。使用 `gh run watch --interval 60` 留在同一会话中等待，工具超时后继续等待，并检查作业级状态，因为 queued 工作流可能包含正在运行的作业。benchmark 矩阵失败后，eval 作业仍可能继续。定位首个服务端错误而非清理阶段症状；在原有范围、预算和容量规则内修复。工具调用被拒绝时改用允许的工具或命令，不得提前报告成功。先将所有尝试的最终结果写入 PR 表格，再报告停止原因、修复次数、已确认的子运行结束状态和 PR URL。不得承诺稍后继续监控，也不得仅为结束会话而取消正常运行。

[Claude Code Stop hook](https://code.claude.com/docs/en/hooks#stop) 运行 `python -m utils.klaude check-stop`，读取候选准备时间之后 `e2e-tests.yml` 手动调度的全部分页，并匹配 `e2e Test - $KLAUDE_TEST_NAME`。每次尝试必须使用固定标识 `klaud-<parent-run-id>-<candidate-id>`。匹配到未结束的运行，或列表不可用/不完整时，阻止正常停止并要求同一 agent 继续。其他候选的运行不受影响；没有所属运行或全部结束时允许停止。hook 不调度、不取消、不修复，也不调用模型。GitHub 作业时限、Claude 内置 Stop-hook 循环上限、API 错误、中断或异常终止仍可能导致任务未完成；它不是外部监督服务。不增加自定义超时或继续执行预算。

action 结束后，`diagnostics` 复用 planner 的脱敏逻辑，仅将 `candidate-diagnostics.json` 上传至 `klaude-candidate-<candidate-id>`。保留 action 结果、允许列表中的终止类型、布尔错误状态、数值型耗时/轮数/成本及固定权限拒绝类别。未知终止类型记为 `unknown`；执行文件缺失或不可读时记为不可用。排除原始消息、结果、命令、路径、错误文本、凭据和私有遥测。runner 仍可用时这些步骤通过 `always()` 执行，不更新 PR，也不恢复 Claude。planner 诊断同步增加相同终止字段。

## 公开 API 调查

从当前 [API 参考文档](https://inferencex.semianalysis.com/api) 和 [OpenAPI 文档](https://inferencex.semianalysis.com/api/openapi.json) 获取参数、响应结构和限制。planner 只需要现有的两个公开数据源；Klaud Cold 根据候选情况选择额外读取。API schema 和模型/runner 清单仍以各自的现有来源为准。

下表所有路径均相对于 `/api/v1/`：

| 调查内容 | 可用读取接口 |
| --- | --- |
| 已发布覆盖范围与 recipe 溯源 | `availability` 提供实际场景和日期；`workflow-info` 提供运行尝试次数、SHA、变更日志中的配置键及各次运行覆盖范围；`submissions` 按日期汇总拓扑和测试点数量。 |
| 镜像与性能历史 | `benchmarks` 和 `benchmarks/history` 提供原始指标、镜像、拓扑、recipe fingerprint 和产出运行 URL。省略 calculator 视图以保留全部可用指标。 |
| 同组测试点 | `benchmark-siblings` 提供相关测试点及其并发/拓扑、来源 GitHub 运行和数据集 slug。比较前仍需筛选目标工作负载和拓扑。 |
| 已发布失败与质量信息 | `evaluations` 提供任务分数和运行来源；`reliability` 提供硬件/日期维度的成功计数，不能归因到具体镜像。 |
| 运行时诊断 | `log-availability` 检查日志是否保留；`server-log-files` 列出文件名；`server-log-search` 搜索全部保留文件并返回数量受限的片段；`server-log` 分块读取指定文件。通过 `nextOffset` 继续读取，并检查搜索结果是否被截断。 |
| AgentX 缓存、延迟与工作负载诊断 | `trace-availability`、`agentic-aggregates`、`derived-agentic-metrics`、`trace-histograms`、`trace-server-metrics` 和 `request-timeline` 提供已保留的 trace、百分位、token 样本、缓存/队列/吞吐量时间序列，以及请求阶段和取消信息。先读汇总，仅在需要时获取详细 trace。 |
| AgentX 数据集背景 | `datasets`、`datasets/{slug}`、`datasets/{slug}/conversations` 和 `datasets/{slug}/conversations/{convId}` 提供数据集元信息、分布与会话结构。使用该次运行的数据集 slug，不预设数据集。 |
| 通信背景 | `collectivex/latest`、`collectivex/runs` 和 `collectivex/runs/{runId}` 提供带版本的通信测量，可用于相关的多节点故障调查。必须明确指定受支持版本；接口可能返回已存储的回退数据。 |

解读响应时应注意以下实现细节：

- `latest-images` 和 `availability` 返回原始模型键；benchmark 查询要求使用当前 OpenAPI 枚举中的展示名称。核对返回的模型及完整候选身份。`latest-images` 记录的是观测结果，不能代替当前仓库配置族目录，也不提供兼容替换镜像清单。版本数据源可能覆盖不全或返回 null。
- 向 `workflow-info` 传入观测日期：虽然参考文档说省略日期会查询全部日期，当前查询实现会将该参数转换为 SQL 日期。变更日志中的配置键和运行覆盖范围可以缩小查找范围，但不能证明配置族唯一。
- 对于 append-only 运行，`exactRun=true` 可能包含同一镜像的前序运行链。保留各测试点产出运行的 `run_url`，并与逻辑 `curve_*` 快照元数据区分。benchmark 行的 `workflow_run_id` 是数据库 ID；`runId` 参数和 `workflow-info` 中的运行标识是 GitHub ID。诊断接口要求传入数据库基准测试结果的正整数 `id`。
- `submissions` 按配置/日期聚合测试点，未排序便选取一个非 null 镜像，因此不能用于确定精确镜像基线。公开缓存和数据导入可能滞后于新运行。诊断数据缺失表示证据不可用，不代表数值为零或测试通过。
- 原始 benchmark 时间指标使用秒；请求时间线使用纳秒偏移，延迟字段名称明确标注毫秒。服务器时间序列偏移使用秒。物理芯片数与逻辑 TP 独立；只有分离式部署才应将 prefill/decode 芯片数相加。

应用还有一些有用的**未发布 UI 读取接口**：`/api/unofficial-run` 将尚未导入的运行产物归一化，`/api/gpu-metrics` 读取 GPU 指标产物，`/api/v1/eval-samples` 和 `/api/v1/eval-samples-live` 可下钻失败样本，`/api/v1/trace-server-metric-source` 获取所选 worker/来源的时间序列。按需使用前先检查当前 handler；这些契约由页面内部使用。非官方 benchmark 行使用合成的 `id: 0`，不能用于查询已存储的诊断数据。新旧镜像比较的证据仍须对应原始运行产物。

`tco-feed` 提供经过插值的图表/TCO 背景，不用于精确 recipe 比较。`overview`、`request-chart-data` 和 `resident-sequence-lengths` 是 UI 投影，没有此处候选选择所需的额外数据。反馈与管理路由不参与候选调查。

## 修改范围与 PR 策略

Klaud Cold 只应修改所选主配置族的镜像，以及它已经引用且未被其他配置族共享的 srt-slurm recipe YAML 镜像/后端兼容性设置。模型、精度、拓扑、推测解码、工作负载、命令、资源和 recipe 引用保持不变。`model.container` 及存在时的 `identity.container.image` 必须与主配置镜像一致。共享脚本、launcher、库、工作流/控制文件和无关配置族保持不变。Klaud Cold 使用 **uv** 运行针对性检查；agent 之后没有独立补丁校验器。

明确的任务规则要求 **`perf-changelog.yaml` 逐字节保持不变**。这与 [`CONTRIBUTING.md`](../CONTRIBUTING.md) 和 [`claude-pr-review.yml`](../.github/workflows/claude-pr-review.yml) 对此类修改要求变更日志条目的规定冲突。Klaud Cold 必须说明冲突并让所有 PR 保持草稿。现有检查和人工审核仍然有效；没有授权自动合并或绕过规则。尚未检查分支保护要求的检查项。

## 工作流操作与凭据

入口工作流仅支持手动触发（`workflow_dispatch`）。每六小时的 cron（`0 */6 * * *`，UTC）仍以注释形式禁用。条件 `github.ref == 'refs/heads/main' && github.run_attempt == 1` 会跳过功能分支、tag 和重新运行。只有 candidate 暴露 `workflow_call`。`klaude-auto-sweep` 并发组不取消已有运行，并防止多次 auto-sweep 调用重叠；同一次调用中的候选并行运行，工作流不额外限制候选并行度。

planner 的 Python 准备和最终容量检查步骤使用 dashboard key；准备步骤还使用只读工作流 token。限轮数的 Claude PR 检查使用 `ANTHROPIC_API_KEY` 和具有 `pull-requests: read` 权限的只读工作流 token。它接收私有资格线索，但不接收 `CLAUDE_PAT` 或 dashboard key，也不执行 GitHub 写操作。candidate 获得用于分支/PR 写入及 e2e 调度/取消的 `CLAUDE_PAT`、用于 Klaud Cold 的 `ANTHROPIC_API_KEY`，以及覆盖 clusters 的限期 `status:read` `KLAUDE_DASHBOARD_API_KEY`。Klaud Cold 不得发布凭据或私有 API 响应。共享 HTTP 读取器固定来源、限制 GET 响应大小并拒绝重定向。非有限 JSON 数值（包括 `1e400` 这样的指数溢出）会在校验或哈希计算前被拒绝。不需要新增控制器、数据库或 environment 配置。

所有外部 action 均固定完整提交 SHA，已于 2026-09-04 核对上游 release/tag 元数据和实现测试。内部调用使用 `./.github/workflows/klaude-candidate.yml` 解析调用者的精确提交，并显式传递三个必需 secret。

| Action | 版本 | 提交 |
| --- | --- | --- |
| `anthropics/claude-code-action` | `v1.0.216` | [`d75b94d5ad42`](https://github.com/anthropics/claude-code-action/commit/d75b94d5ad426cb8546e6628b6f5f19b84e5cce1) |
| `actions/checkout` | `v7.0.1` | [`3d3c42e5aac5`](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1) |
| `actions/upload-artifact` | `v7.0.1` | [`043fb46d1a93`](https://github.com/actions/upload-artifact/commit/043fb46d1a93c77aae656e7c1c64a875d1fc6a0a) |
| `actions/download-artifact` | `v8.0.1` | [`3e5f45b2cfb9`](https://github.com/actions/download-artifact/commit/3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c) |
| `astral-sh/setup-uv` | `v10.0.1` | [`20cfd1bf945f`](https://github.com/astral-sh/setup-uv/commit/20cfd1bf945f4377ade1205e4dbc17946fc9a30d) |

## 本地验证

```bash
uv run --no-project --python 3.12 --with "pydantic>=2.10,<3" python -m utils.klaude --help
uvx zizmor==1.30.0 --offline --no-config --no-ignores .github/workflows/klaude-plan.yml .github/workflows/klaude-candidate.yml
```

CLI 和工作流检查不能证明 GPU 实际可运行。Klaud Cold 使用现有 InferenceX 校验和 e2e 工作流验证候选修改。本地验证不调用真实模型、不调度 benchmark、不创建 PR、不部署。

常规 zizmor 扫描报告一项低严重性的 `self-repository` 建议：倾向使用 `$/`，而非仓库现有的 `./` 工作流调用形式。auditor 模式还会提示权限缺少说明，以及直接使用仓库 dashboard secret 而未使用专用 GitHub environment；最终容量刷新步骤新增一项 auditor 模式的 `secrets-outside-env` 提示，涉及现有 dashboard key；该 key 仍仅在对应 Python 步骤中提供。未添加忽略规则。`actionlint` 1.7.12 可直接检查两个 Klaud 工作流，无需在临时副本中改写语法。
