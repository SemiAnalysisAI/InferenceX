# Klaud Cold 自动扫描

<div align="center">

[English](./klaud.md) | **中文**

</div>

[`klaud-plan.yml`](../.github/workflows/klaud-plan.yml) 先收尾有记录的中断会话，再用 Python 准备候选，经只读 Claude 检查排除重叠 PR 后调用 [`klaud-candidate.yml`](../.github/workflows/klaud-candidate.yml)。每个候选仍由一个自主 Klaud Cold 会话负责修改、诊断和修复。`finish` 命令验证结果或执行清理，并发布完成记录；只读 Stop hook 和诊断步骤对照 GitHub 验证该记录。恢复工作放在下一次现有 autosweep 中，不增加第二个 agent 或工作流。

PR 检查使用 `claude-opus-5`（Opus 5），关闭 fast mode（`fastMode: false`），最多运行 200 轮。候选执行使用 `claude-fable-5-1`（Fable 5.1），关闭 fast mode。Agent 的显示名称为 **Klaud Cold**；工作流文件名、CLI、产物、分支及环境变量/secret 标识统一使用 `klaud` / `KLAUD`。旧拼写的候选分支仍会阻止重复选择。调度重命名后的工作流前，须配置 `KLAUD_DASHBOARD_API_KEY`。

## 候选选择

[`api.py`](../utils/klaud/api.py) 通过同一 HTTP 实现处理公开和私有读取。[统一 CLI](../utils/klaud/__main__.py) 为 `python -m utils.klaud`：`plan --directory DIR` 准备候选和开放 PR；`select --max-candidates-per-run N --directory DIR` 校验 Claude 的结构化 `KLAUD_PR_REVIEW` 输出并应用工作流中的候选数量上限；`check-capacity --cluster ID` 在调度前重新检查容量。在 `plan` 前传入 `--root PATH`，可指定用于获取候选基准 SHA 的 checkout；不加载单独的 Klaud 策略配置文件。

- 公开读取使用 `/api/v1/latest-images` 和 `/api/v1/framework-releases`。公开观测和版本键直接来自响应；候选配置族身份来自当前主配置。版本不匹配和不稳定镜像仅作为检查线索，兼容的新镜像由 Klaud Cold 核实。
- 私有读取仅使用 `/api/status/clusters`。要求响应新鲜且实际读取的字段有效、API 对集群判定为 `stale: false`、观测与接收时间戳有效且顺序正确、状态为 operational 或 degraded，并且至少有一个空闲节点。集群数据的过期阈值由 API 配置决定；Klaud 的 120 秒限制只用于响应获取/生成时间，不用于集群观测时间。排队任务、调度器预留和优先级覆盖不会改变低于 80% 的利用率规则。
- 公开 API/CDN 负责 HTTP 缓存的新鲜度；Klaud 校验响应内容和本地获取时间。准备失败时，Actions 日志和摘要会报告具体错误码。
- `plan` 使用现有矩阵生成器和 runner 元数据，从当前根目录的 `configs/*-master.yaml` 生成配置身份。按模型前缀、硬件、框架、精度、投机解码、分离部署、场景、ISL/OSL 和当前镜像与公开观测求交集，已归档工作负载不会挤掉有效匹配。无法生成矩阵的配置族单独告警并排除。每个有效配置族选取最新匹配观测作为基线，再对不同配置族随机打乱一次。分支身份包含精确主配置文件/键及源镜像/发布元数据；旧的粗粒度身份和拼写对应的占用仍用于去重。关闭的 PR 不参与去重，但保留的分支仍占用候选。分页读取全部打开的 PR 及修改文件，包括草稿和重命名；失败或不完整的读取留给检查阶段处理。
- Claude 根据 runner 和现有配置，对照私有 `capacity.json` 中的路由线索，验证已提供的精确当前配置族及其**全部实际目标集群**。每个目标都必须符合条件；同类硬件的健康兄弟集群不能替代另一个集群。无法证实映射或目标不可用时标记为 `uncertain`，并按随机顺序继续检查，直到补足名额。路由解释仍由 agent 负责，不新增 Python recipe/别名目录。Claude 还会检查开放 PR 的变更文件，再按需阅读正文和 diff。已有镜像更新、重叠的修改或共享依赖会阻止候选，即使目标镜像 tag 不同；不同配置族仅模型或镜像相同不足以判为重复。不得替换为兄弟配置或归档工作负载。结构化决策包含 `candidate-id`、`decision`（`proceed`、`duplicate` 或 `uncertain`）、`family`、精确的 `telemetry-clusters`、重叠 PR 编号列表 `pull-requests` 和不含私有资格数据的简短 `reason`。
- `select` 仅接受通过校验且保留已提供配置族的 `proceed` 决策，刷新私有容量数据，要求检查结果中的每个目标仍符合条件，按解析后的配置族去重，再按随机顺序应用总数量上限。`capacity-deferred-candidates` 记录最终容量检查未通过的候选。某个配置族被标记为重复或不确定时，即使另一条观测允许继续，也会排除整个配置族。缺失或未检查的候选延后。检查 action 失败、输出格式错误、未知/重复 ID 或容量 API 不可用会让本次选择延后，绝不绕过重叠检查。候选失去容量资格后，可由后续已检查且仍符合条件的配置族补位。

两个 agent 都明确获得证据目录的访问权限。检查 agent 使用 Read/Glob/Grep 检查本地内容，每次 Bash 调用只执行一个允许的只读 gh/git 命令，避免 shell 包装和管道。两个 Klaud 工作流均不设置作业/步骤超时或 Bash 超时覆盖，使用 GitHub Actions 默认限制。预取也不设置整体截止时间。`selection.json` 记录延后原因，作业摘要报告所选/延后数量。`review-diagnostics.json` 仅保留时长、轮数、费用等数值指标、按工具汇总的拒绝次数及固定的 Bash 分类（例如 shell 包装或文件过滤）；不包含原始命令、路径、消息、结果或凭据。历史日志只提供拒绝次数，无法还原之前被拒绝的具体命令；新增分类和访问指令仍需实际运行验证。检查阶段诊断文件缺失不阻止选择收尾。检查步骤之外的基础设施故障或整个作业被取消仍可能导致无法完成。

私有容量门槛是 **节点利用率严格低于 80%**：`(summary.allocatedNodes + summary.mixedNodes) * 5 < summary.totalNodes * 4`，不做舍入。完全分配和部分使用的节点均计入已使用节点；恰好 80% 时不放行。不扣除预留节点。要求至少有一个空闲节点，避免整个集群不可用时仍以 0% 利用率通过检查。缺失、无效、不一致、过期或不可用的数据均拒绝。硬件匹配只用于初筛；检查阶段必须解析每个实际目标，选择阶段重新检查这些精确 ID。符合条件的作业可以先排队，由调度器等待完整物理节点需求能够满足后再启动。兼容性按实际读取的字段判断，不依赖 `schemaVersion`；新增字段或版本变化不会排除其他方面均有效的集群。

`klaud-plan` 产物仅显式包含 `candidates.json`、`open-prs.json`、`selection.json`、`review-diagnostics.json` 及每个所选候选的 `candidate.json`。本地 `capacity.json` 为检查阶段提供遥测 ID 和资格线索，**绝不上传**；任意临时文件也不会上传。每份交接文件包含公开观测、版本线索、检查原因、分支、基准 SHA、公开 API 发现 URL 和通过校验的 `pr-review`，不包含私有节点计数或原始遥测。所选候选并行运行，各自获得独立 Klaud Cold 会话。一个候选失败不会取消其他候选。不再复制模型/runner 目录，也不保留 `recipes.py`；agent 使用现有 InferenceX 配置和工具理解实际 recipe。

## Klaud Cold 负责执行

维护者可用 `klaud-handoff` 标签接管打开的候选 PR（首次使用前先创建仓库标签）。Klaud 在修改 PR/分支或取消运行前检查该标签；发现接管后保留 PR、分支、标签和运行中的作业，返回 `handoff`，Stop hook 同时释放监控责任。Klaud 不得自行添加或移除该标签。旧草稿仍占用候选，必须由维护者明确恢复，或关闭并释放分支；更新提示词不会自动迁移旧会话。

除维护者明确接管外，会话结束时仅在最终验证成功后保留开放且 ready 的 PR。否则先报告失败或延后原因，取消所属未完成运行并确认所有作业结束，补全各次尝试评论中的结果，移除 sweep 标签、改回草稿并关闭 PR。对于已确认的基础设施或容量阻塞（包括缺少预置权重或挂载目录），删除远程候选分支以允许重试。对于已确认的镜像不兼容、镜像修复预算耗尽或原因不明，保留分支并解释原因；原因不明应交由人工检查，不得直接声称不兼容。开放 PR 阻止整个配置族；保留的分支仅阻止精确候选，因此源镜像或 release 元数据变化后仍可选中。不得关闭其他所有者的 PR 或删除其分支。没有自己创建的 PR 时，仅报告停止原因，不创建占位 PR，也不移除已有候选认领。

PR 正文只包含一到两句话说明目标，以及**基线**章节：发布日期、旧镜像、工作负载/拓扑、来源链接和精简的 benchmark/eval 数值。正文保持稳定；当前状态、下一步和所有尝试只放在评论中。不要添加限制章节、检查清单或套话。若基线存在事实上的不匹配，就在对应数据旁简短说明。正文和每条评论均先写英文，再用 Markdown 分隔线（`---`）分隔，最后写自然的简体中文。私有数据、凭据、@提及和请求审阅的现有限制仍然适用。

完成 checkout 和上下文准备后，candidate 工作流将控制权交给 Klaud Cold，并提供 `CLAUDE_PAT`、`ANTHROPIC_API_KEY` 和私有 API 只读密钥。Klaud Cold 将公开观测解析到一个活动主配置族，检查当前镜像和已有 PR，并在**编辑或创建分支/PR 之前**核实检查结果中的目标 ID 和实时容量。随后在使用 GPU 前认领分支，产生实际修改并创建草稿 PR。所有生成的 PR 标题必须以 `[Klaud Cold] ` 开头，后接英文 / 简体中文描述。不得在 GitHub 上 @提及用户/团队，也不得请求 review/re-review；这些操作由自动流程处理。它在认领分支前立即重新检查开放 PR，因为检查只是快照，不能锁住后来创建的人工 PR。有歧义、已退役或已经更新的候选直接停止，不运行扫描。提交、推送、调度、监控、诊断、修复及双语 PR 更新都由同一会话完成。

定向尝试只测试更新后的镜像：从 `main` 调度 `e2e-tests.yml`，将 `inputs.ref` 设为实际测量 SHA，并使用 `generate-cli-command="test-config --config-files FILE --config-keys FAMILY --trim-conc"`。现有矩阵生成器为每种部署形态保留最低并发，保持原有工作负载及 smoke eval 选择规则。这只能证明启动和兼容性，不能代表完整曲线。smoke 通过后，追加精确配置族的 changelog 条目，不使用场景、eval 选择或 append-only 修饰项；将 PR 标记为 ready 并添加 `full-sweep-enabled`。最终 sweep 保留全部测试点和默认 eval。最终失败后，先移除 sweep 标签并退回草稿，再推送修复。Klaud 不自行 staging、授权复用、请求 review 或合并。

在编辑或创建分支之前、每次定向调度之前，以及最终 sweep 的 ready/标签转换之前，使用 `check-capacity --cluster ID` 检查精确目标；通过重复 `--cluster` 指定每个可能的目标。退出状态 0 要求全部目标均通过新鲜度、可用性和低于 80% 利用率检查。如果该检查在定向调度、最终 sweep 转换或恢复调度之前失败，Klaud 会先在已有 PR 的评论中记录可公开的容量延后原因和当前尝试状态。随后取消并确认全部所属运行已经结束，再用终态或已取消行及确认后的状态更新尝试评论。最后移除所有 sweep 标签、将 PR 改回草稿、关闭 PR，并删除远程 Klaud 分支，使后续扫描可以重试该候选。如果尚无 PR，则在最终响应中记录延后结果，不创建占位 PR。调度后利用率上升不会导致健康运行被取消。Klaud 不会等待恢复或承诺自动继续。命令不打印容量详情。

Klaud Cold 读取运行产物/日志，在评论中维护完整尝试记录。为**首次尝试**、每次实际使用的**修复 N/5**和**最终完整扫描**各创建一条简短的纵向评论；持续更新镜像/SHA、运行链接、修改内容、benchmark/eval 结果、匹配的吞吐量/延迟差值、诊断及下一步，包括最终或已取消的结果。首次尝试不占用五次修复预算。出现实质诊断、修复、最终 sweep 或停止状态变化时，追加简短里程碑评论，链接对应尝试而不重复全部结果。合并相邻里程碑；等待时仅在有实质变化或距离上次更新已过 30 分钟时更新。评论只描述可观察的工作，不包含内部推理或原始日志。紧凑表格只用于数值比较。缺失或不可比较的数值标为 `N/A` 并就地简短说明；无效、重复或未匹配的测试点不得用于声称性能提升，不另设限制章节。定向 benchmark/default eval 通过、精确 head 的最终验证通过和全局 PR 审批是不同结论。性能提升尽力而为，回退如实报告，不设置百分比拒绝阈值。公开 dashboard 数据作为基线，只有新镜像尝试消耗 GPU 时间。空汇总或成功的收集作业不能证明通过。原始产物保留在对应运行中；结束后仅上传脱敏诊断和结构化终态。

最终 sweep 无论测试点数量多少，都运行完整的所选配置族。单次 benchmark 运行可能长达三小时；Klaud Cold 在候选作业的总时限内监控其进展。Klaud 不再为单次 benchmark 运行设置额外超时。

两个设置直接保留在工作流中：[`klaud-plan.yml`](../.github/workflows/klaud-plan.yml) 的 `MAX_CANDIDATES_PER_RUN` 控制 planner 的候选数量上限；[`klaud-candidate.yml`](../.github/workflows/klaud-candidate.yml) 的 `MAX_REPAIRS` 将修复次数上限直接传入 agent 提示词。

Klaud Cold 最多运行 500 轮，使用 GitHub Actions 默认作业时限。诊断、修复选择和尝试评论仍由 agent 负责；`finish` 通过代码验证结果覆盖、子运行结束、最终报告和分支处理。作业时限、API 错误或 runner 被终止仍可能中断会话；下一次 autosweep 会在选择新候选前收尾有归属记录的会话。不增加自定义工作流超时。

Klaud Cold 调度 `e2e-tests.yml` 时显式设置布尔输入 `klaud-run: true`，并使用 `klaud-` 测试名称方便识别。手动和复用调用中的该输入均默认为 false，并传递至所有 benchmark/eval 模板。只有这个标志会为作业名称添加 `klaud | ` 前缀，供 InferenceX Dash 优先级调度器识别；普通测试名称不会改变优先级。配套调度器改动使 Klaud 始终排在普通人工任务之后，不受任务到达时间或 recipe 分数影响；Klaud 不获得等待时长加分或节点预留，skip-queue 请求也不生效。Klaud 使用剩余容量，不抢占已经运行的任务。显式管理员优先级覆盖保留原有优先顺序，Klaud 不得主动请求。启用 auto-sweep 前需部署配套 dashboard 调度器改动。

### 已发布基线与会话完成

基线来自 **`https://inferencex.semianalysis.com` 的公开 dashboard API**。将 `candidate.source.date` 传给 `workflow-info` 和 `benchmarks`；从 OpenAPI 解析展示模型名称，设置 `date` 和 `exact=true`，不使用计算器 `view`。核实旧镜像以及完整的模型、硬件、框架、精度、推测解码和工作负载身份，再逐点匹配拓扑、并发量及数据集。记录 API 查询、发布日期和每个测试点的来源 `run_url`/SHA，区分逻辑曲线快照与实际数据来源。按需读取已发布 eval，所有尝试共用这份固定基线。缺失或不可比较的数据填写 `N/A` 并说明原因。绝不调度或重跑旧镜像基线。

调度运行或创建草稿不代表任务完成。使用 `gh run watch --interval 60` 留在同一会话中等待，工具超时后继续等待，并检查作业级状态，因为 queued 工作流可能包含正在运行的作业。benchmark 矩阵失败后，eval 作业仍可能继续。定位首个服务端错误而非清理阶段症状；在原有范围、预算和容量规则内修复。工具调用被拒绝时改用允许的工具或命令，不得提前报告成功。先将所有尝试的最终结果写入 PR 尝试评论，再报告停止原因、修复次数、已确认的子运行结束状态和 PR URL。不得承诺稍后继续监控，也不得仅为结束会话而取消正常运行。

[Stop hook](https://code.claude.com/docs/en/hooks#stop) 通过 `check-stop` 检查 `$KLAUD_EVIDENCE/outcome.json`。结束前，将请求的 `CandidateOutcome` JSON 写入单独文件，运行 `uv run --no-project --python 3.12 --with "pydantic>=2.10,<3" python -m utils.klaud finish --outcome-file "$KLAUD_EVIDENCE/requested-outcome.json"`，再原样返回已验证的 `outcome.json`。命令从父运行原始创建时间起发现所有自有定向和最终运行，包括旧 head、已关闭或移除标签的 PR。先发布失败或延后报告，再取消运行；全部结束后才移除 sweep 标签、退回草稿并关闭 PR。`capacity-deferred`/`readiness-blocked` 删除所属 PR 的分支；其他失败结果保留分支。完成记录包含实际运行 ID 和分支策略。取消或标签事件作业仍在进行时，等待后重试 `finish`。维护者接管优先于清理；不覆盖其他所有者、fork、已移动的 head、已合并 PR 或不明确的状态。没有所属 PR 时保留已有分支占用。hook 仅做验证，不修改状态，也不能突破 Claude 内置的停止循环上限。

执行结束后，`diagnostics` 对照 GitHub 和完成记录验证结构化终态，并与脱敏运行指标一起写入 `candidate-diagnostics.json`。终态仅允许固定类别（`validated`、`capacity-deferred`、`readiness-blocked`、`incompatible`、`duplicate`、`retired`、`already-updated`、`uncertain`、`failed`、`handoff`、`unexpected-error`）、阶段、数字 PR/运行 ID 和修复次数；不接受自由文本或遥测。作业摘要通过紧凑双语表格链接 PR 和运行，包括未创建 PR 的会话。输出缺失、格式错误或 action 失败时记录 `unexpected-error`，阶段和修复次数为未知，并让报告步骤失败。正常延后表示会话完成，不表示基准验证成功。不得上传原始执行消息、命令、凭据或私有响应。

`run-sweep.yml` 为 Klaud 单独上传 `klaud-sweep-manifest`，记录实际调度矩阵、精确 head、运行 ID/attempt 和完整 sweep 状态。验证器按完整 recipe fingerprint、并发量和镜像比较全部 benchmark 点，并复用现有 eval 身份、原始结果及汇总一致性验证器检查所需 eval；缺失、重复或不匹配时拒绝完成。旧运行没有该 manifest 时不能自动认定为已验证。现有 `changelog-metadata` 产物保持原格式。

### 中断恢复

选择步骤在 agent 启动前上传小型 `klaud-ownership` 产物，保存父运行及所选候选的 ID、配置族和基准 SHA，最长保留 90 天。下一次 autosweep 的 `recover` 只读取受信任 `main` 工作流产生的记录，并要求原父运行已结束。它尊重接管或已合并状态，验证已完成结果；遇到中断时，先等待已经调度的精确 head 最终 sweep，保留成功且覆盖完整的结果，否则取消并确认所属作业、发布双语清理报告、关闭未完成 PR，保留原因不明候选的分支供人工审查。待完成清理记录将原因和修复次数绑定到原 PR head；容量清理中途崩溃后，仍会释放分支以便重试。head 已变化时要求人工检查。未知修复次数写为 `null`，不猜测为零。PR 创建时间限定在所属会话内，旧关闭 PR 不会妨碍同名分支的新尝试；恢复前后均检查 head、标签和父运行状态。单个候选状态不明不会阻止检查其他候选，但会暂停新的调度并在摘要中提示检查。父运行的全部候选收尾后，只删除这个待处理归属产物，保留计划、诊断和 PR 报告，避免反复扫描已完成历史。旧版会话或已过期记录不自动迁移；禁用 schedule 或 GitHub 不可用会推迟恢复。恢复不新建 PR、不修复镜像、不合并，也不取消其他人的新运行。

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

Klaud Cold 只应修改所选主配置族的镜像，以及它已经引用且未被其他配置族共享的 srt-slurm recipe YAML 镜像/后端兼容性设置。模型、精度、拓扑、推测解码、工作负载、命令、资源和 recipe 引用保持不变。`model.container` 及存在时的 `identity.container.image` 必须与主配置镜像一致。共享脚本、launcher、库、工作流/控制文件和无关配置族保持不变。固定镜像必须原样运行：禁止对推理引擎或 serving 技术栈打补丁、改写源码或容器文件、就地修改 site-packages、monkey-patch，以及覆盖安装 fork 或重新构建的 wheel。候选镜像或现有启动路径若依赖其中任一操作，Klaud 会将其判定为不兼容。Klaud 的运行时补丁数量必须为零，没有豁免例外；应使用未经修改的受支持镜像。Klaud Cold 使用 **uv** 运行针对性检查；agent 之后没有独立补丁校验器。

定向验证通过后，Klaud 会追加必要的 changelog 条目，同时保持历史字节不变，重新检查容量，再将候选标记为 ready 并添加 `full-sweep-enabled`；`run-sweep.yml` 会跳过草稿 PR。最终 sweep 失败时，先移除标签并将 PR 改回草稿，再开始修复。对于容量相关失败，Klaud 会执行必要的恢复前容量检查，并仅在该检查失败时执行完整的关闭与释放清理。现有检查和人工审核仍然有效；没有授权自动 staging、复用、请求 review、合并或绕过规则。

## 工作流操作与凭据

入口工作流每六小时运行一次（`0 */6 * * *`，UTC），并支持手动触发（`workflow_dispatch`）。每次调用最多选择五个候选并行运行。条件 `github.ref == 'refs/heads/main' && github.run_attempt == 1` 会跳过功能分支、tag 和重新运行。candidate 作业也跳过重跑；应重新调度 autosweep，先执行恢复。只有 candidate 暴露 `workflow_call`。`klaud-auto-sweep` 并发组不取消已有运行，并防止多次 auto-sweep 调用重叠；同一次调用中的候选并行运行，工作流不额外限制候选并行度。

恢复步骤单独使用 `CLAUDE_PAT` 执行已确认归属的清理；只读重叠检查 agent 不获得该凭据。planner 的 Python 准备和最终容量检查步骤使用 dashboard key；准备步骤还使用只读工作流 token。限轮数的 Claude PR 检查使用 `ANTHROPIC_API_KEY` 和具有 `pull-requests: read` 权限的只读工作流 token。它接收私有资格线索，但不接收 `CLAUDE_PAT` 或 dashboard key，也不执行 GitHub 写操作。candidate 获得用于分支/PR 写入及 e2e 调度/取消的 `CLAUDE_PAT`、用于 Klaud Cold 的 `ANTHROPIC_API_KEY`，以及覆盖 clusters 的限期 `status:read` `KLAUD_DASHBOARD_API_KEY`。Klaud Cold 不得发布凭据或私有 API 响应。共享 HTTP 读取器固定来源、限制 GET 响应大小并拒绝重定向。非有限 JSON 数值（包括 `1e400` 这样的指数溢出）会在校验或哈希计算前被拒绝。不需要部署额外服务、数据库或新增 environment 配置。

所有外部 action 均固定完整提交 SHA；下表与当前工作流中的固定版本一致。内部调用使用 `./.github/workflows/klaud-candidate.yml` 解析调用者的精确提交，并显式传递三个必需 secret。

| Action | 版本 | 提交 |
| --- | --- | --- |
| `anthropics/claude-code-action` | `v1.0.218` | [`0d0e0876d3ea`](https://github.com/anthropics/claude-code-action/commit/0d0e0876d3eaa933f45dc692f7a4312c83caf36f) |
| `actions/checkout` | `v7.0.1` | [`3d3c42e5aac5`](https://github.com/actions/checkout/commit/3d3c42e5aac5ba805825da76410c181273ba90b1) |
| `actions/upload-artifact` | `v7.0.1` | [`043fb46d1a93`](https://github.com/actions/upload-artifact/commit/043fb46d1a93c77aae656e7c1c64a875d1fc6a0a) |
| `actions/download-artifact` | `v8.0.1` | [`3e5f45b2cfb9`](https://github.com/actions/download-artifact/commit/3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c) |
| `astral-sh/setup-uv` | `v10.0.1` | [`20cfd1bf945f`](https://github.com/astral-sh/setup-uv/commit/20cfd1bf945f4377ade1205e4dbc17946fc9a30d) |

## 本地验证

```bash
uv run --no-project --python 3.12 --with "pydantic>=2.10,<3" python -m utils.klaud --help
uvx zizmor==1.30.0 --offline --no-config --no-ignores .github/workflows/klaud-plan.yml .github/workflows/klaud-candidate.yml
```

CLI 和工作流检查不能证明 GPU 实际可运行。Klaud Cold 使用现有 InferenceX 校验和 e2e 工作流验证候选修改。本地验证不调用真实模型、不调度 benchmark、不创建 PR、不部署。

常规 zizmor 扫描报告一项低严重性的 `self-repository` 建议：倾向使用 `$/`，而非仓库现有的 `./` 工作流调用形式。auditor 模式还会提示权限缺少说明，以及直接使用仓库 dashboard secret 而未使用专用 GitHub environment；最终容量刷新步骤新增一项 auditor 模式的 `secrets-outside-env` 提示，涉及现有 dashboard key；该 key 仍仅在对应 Python 步骤中提供。未添加忽略规则。`actionlint` 1.7.12 可直接检查两个 Klaud 工作流，无需在临时副本中改写语法。

### 结果文件命名与最终 sweep 调度

定向和最终工作流使用同一个有长度上限的结果文件前缀。长前缀对完整身份（包括完整 recipe fingerprint）做哈希，JSON 中保留原始 fingerprint。对不含文件命名辅助程序的旧提交运行基准测试时，工作流会回退为对完整身份做哈希。SRT 测试点文件保留数字并发/GPU 后缀，只压缩过长的配置名；长度预算包含 `power_validation_`、`.json` 和原子写入的 `.tmp` 后缀。较短的测试点文件名保持不变。

最终 sweep 的每个 benchmark/eval 调用（包括 canary）都会为同仓库、由 `Klaud-Cold` 创建且分支为 `klaud/auto-*` 或旧拼写的 PR 传递后台优先级标志。人工 PR 不会因标题或标签而获得 Klaud 优先级；最终运行与定向运行使用相同的 `klaud |` 调度器标志。
