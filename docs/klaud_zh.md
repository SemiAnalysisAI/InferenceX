# Klaud Cold 自动扫描

<div align="center">

[English](./klaud.md) | **中文**

</div>

[`klaud-plan.yml`](../.github/workflows/klaud-plan.yml) 先收尾有记录的中断会话，再用 Python 准备候选，经只读 Claude 检查排除重叠 PR 后调用 [`klaud-candidate.yml`](../.github/workflows/klaud-candidate.yml)。每个候选仍由一个自主 Klaud Cold 会话负责修改、诊断和修复。`finish` 命令验证结果或执行清理，并发布完成记录；只读 Stop hook 和诊断步骤对照 GitHub 验证该记录。恢复工作放在下一次现有 autosweep 中，不增加第二个 agent 或工作流。

PR 检查使用 `claude-opus-5`（Opus 5），关闭 fast mode（`fastMode: false`），最多运行 500 轮，为有界候选批次解析重复工作、目标集群及公开基线模型名。候选执行使用 `claude-fable-5-1`（Fable 5.1），关闭 fast mode，并负责上游镜像调查。Agent 的显示名称为 **Klaud Cold**；工作流文件名、CLI、产物、分支及运行时环境变量统一使用 `klaud` / `KLAUD`。旧拼写的候选分支仍会阻止重复选择。调度前须配置 `DASH_API_KEY`；工作流仍将其传给现有的 `KLAUD_DASHBOARD_API_KEY` 运行时变量。

## 候选选择

[`api.py`](../infx/klaud/api.py) 通过同一 HTTP 实现处理公开和私有读取。[统一 CLI](../infx/klaud/__main__.py) 为 `python -m infx.klaud`：`plan --directory DIR --review-batch-size N --cooldown-hours H` 准备候选和开放 PR；`select --max-candidates-per-run N --directory DIR` 校验 Claude 的结构化 `KLAUD_PR_REVIEW` 输出并应用工作流中的候选数量上限；`check-capacity --cluster ID` 在调度前重新检查容量。在 `plan` 前传入 `--root PATH`，可指定用于获取候选基准 SHA 的 checkout；不加载单独的 Klaud 策略配置文件。

- 规划阶段仅将 `/api/v1/latest-images` 用作已发布 benchmark 基线索引。候选配置族身份和当前镜像来自现有主配置。该数据源及 `/api/v1/framework-releases` 均不负责判断镜像是否最新或兼容；候选 agent 会独立检查官方上游源码、版本和可用镜像。
- 私有读取仅使用 `/api/status/clusters`。要求响应新鲜且实际读取的字段有效、API 对集群判定为 `stale: false`、观测与接收时间戳有效且顺序正确、状态为 operational 或 degraded，并且至少有一个空闲节点。集群数据的过期阈值由 API 配置决定；Klaud 的 120 秒限制只用于响应获取/生成时间，不用于集群观测时间。排队任务、调度器预留和优先级覆盖不会改变低于 80% 的利用率规则。
- 公开 API/CDN 负责 HTTP 缓存的新鲜度；Klaud 校验响应内容和本地获取时间。网络故障及 HTTP 408、429、5xx 响应会进行两次短暂且有界的重试；无效 JSON、schema 错误及其他语义错误立即失败。准备失败时，Actions 日志和摘要会报告具体错误码。
- `plan` 使用现有矩阵生成器和 runner 元数据，从当前根目录的 `configs/*-master.yaml` 生成配置身份。按模型前缀、硬件、框架、精度、投机解码、分离部署、场景、ISL/OSL 和当前镜像与已发布基线求交集；比较镜像时将 registry 的 `/` 与 enroot 的 `#` 写法视为相同。已归档工作负载不会挤掉有效匹配。无法生成矩阵的配置族单独告警并排除。每个有效配置族选取最新匹配观测作为 benchmark 证据，再对不同配置族随机打乱一次。分支身份包含精确主配置文件/键及规范化当前镜像。现有配置族认领和开放 PR 仍用于去重；失败会话完成清理后删除分支，使配置族回到候选池。分页读取全部打开的 PR 及修改文件，包括草稿和重命名；失败或不完整的读取留给检查阶段处理。工作流每次最多向检查步骤发送 64 个候选。同一基准 SHA 的候选产物形成 24 小时软冷却：候选仍在池中，但排在近期未获得 agent 的候选之后。
- Claude 根据 runner 和现有配置，对照私有 `capacity.json` 路由线索验证**全部实际目标集群**，检查与开放 PR 的语义重叠，并解析公开 API 基线查询所需的精确展示模型名。每个目标都必须符合条件；同类硬件的健康兄弟集群不能替代另一个集群。映射、重叠或容量无法证实时标记为 `uncertain`。重复或不确定的候选不占调度名额。结构化 schema 要求每个候选恰有一个决策，使后续符合条件的配置族能够补位容量变化、基线缺失或并发认领导致的空缺。Claude 检查开放 PR 的变更文件，再按需阅读正文和 diff。已有镜像更新、重叠修改或共享依赖会阻止候选，即使目标镜像 tag 不同；不同配置族仅模型或镜像相同不足以判为重复。结构化决策包含 `candidate-id`、`decision`、`family`、`baseline-model`、精确的 `telemetry-clusters`、`pull-requests` 和不含私有资格数据的简短 `reason`。上游源码、版本和镜像兼容性调查由候选 agent 负责。
- `select` 仅接受通过校验且保留已提供配置族的 `proceed` 决策，刷新私有容量数据，要求检查结果中的每个目标仍符合条件，并在创建认领或启动 agent 前重建完整公开基线。基线不完整或有歧义的候选记录在 `baseline-deferred-candidates` 中，后续已检查配置族可以补位。随后按配置族去重，并按随机顺序应用总数量上限。`capacity-deferred-candidates` 记录最终容量检查未通过的候选。某个配置族被标记为重复或不确定时，即使另一条观测允许继续，也会排除整个配置族。检查不完整时延后本次选择，不将其报告为候选池已耗尽。检查 action 失败、输出格式错误、未知/重复 ID 或共享基线/容量状态不可用会停止后续选择；重叠和基线检查绝不绕过。

两个 agent 都明确获得证据目录的访问权限。检查 agent 使用 Read/Glob/Grep/WebFetch，并在每次 Bash 调用中只执行一个允许的只读 gh/git 命令，避免 shell 包装和管道。两个 Klaud 工作流均不设置作业/步骤超时或 Bash 超时覆盖，使用 GitHub Actions 默认限制。预取也不设置整体截止时间。`selection.json` 记录容量与基线延后原因，作业摘要报告所选/延后数量。`review-diagnostics.json` 仅保留时长、轮数、费用等数值指标、按工具汇总的拒绝次数及固定的 Bash 分类（例如 shell 包装或文件过滤）；不包含原始命令、路径、消息、结果或凭据。检查阶段诊断文件缺失不阻止选择收尾。检查步骤之外的基础设施故障或整个作业被取消仍可能导致无法完成。

私有容量门槛是 **节点利用率严格低于 80%**：`(summary.allocatedNodes + summary.mixedNodes) * 5 < summary.totalNodes * 4`，不做舍入。完全分配和部分使用的节点均计入已使用节点；恰好 80% 时不放行。不扣除预留节点。要求至少有一个空闲节点，避免整个集群不可用时仍以 0% 利用率通过检查。缺失、无效、不一致、过期或不可用的数据均拒绝。硬件匹配只用于初筛；检查阶段必须解析每个实际目标，选择阶段重新检查这些精确 ID。符合条件的作业可以先排队，由调度器等待完整物理节点需求能够满足后再启动。兼容性按实际读取的字段判断，不依赖 `schemaVersion`；新增字段或版本变化不会排除其他方面均有效的集群。

`klaud-plan` 产物仅显式包含 `candidates.json`、`open-prs.json`、`selection.json`、`review-diagnostics.json` 及每个所选候选的 `candidate.json`。本地 `capacity.json` 为检查阶段提供遥测 ID 和资格线索，**绝不上传**；任意临时文件也不会上传。每份交接文件包含已发布基线观测、已验证的 `baseline-model`、分支、基准 SHA、公开 benchmark 查询 URL 和通过校验的 `pr-review`，不包含私有节点计数或原始遥测。所选候选并行运行，各自获得独立 Klaud Cold 会话。一个候选失败不会取消其他候选。不再复制模型/runner 目录，也不保留 `recipes.py`；agent 使用现有 InferenceX 配置和工具理解实际 recipe 及上游镜像。

## Klaud Cold 负责执行

维护者可用 `klaud-handoff` 标签接管打开的候选 PR（首次使用前先创建仓库标签）。Klaud 在修改 PR/分支或取消运行前检查该标签；发现接管后保留 PR、分支、标签和运行中的作业，返回 `handoff`，Stop hook 同时释放监控责任。Klaud 不得自行添加或移除该标签。仍在活动的旧草稿继续占用候选，直到维护者恢复或恢复流程完成清理；已验证的历史完成记录会迁移到始终删除分支的策略。

除维护者明确接管外，会话结束时仅在最终验证成功后保留开放且 ready 的 PR。否则先报告失败或延后原因，取消所属未完成运行并确认所有作业结束，补全各次尝试评论中的结果，移除 sweep 标签、改回草稿、关闭 PR、删除未移动的精确 head 候选分支，并释放配置族认领。容量或就绪性阻塞、镜像不兼容、修复次数耗尽、原因不明及意外失败均采用同一清理方式，使配置族回到候选池；原因不明仍不得直接声称不兼容。不得关闭其他所有者的 PR，也不得删除已移动或已接管的分支。没有自己创建的 PR 时，仅报告停止原因，不创建占位 PR，也不删除无法验证归属的分支。

[报告指南](./klaud-reporting_zh.md)统一定义正文、评论模板及类型化报告命令。正文仅包含目标和冻结的公开基线；尝试评论包含计数/状态/运行、紧凑元数据与 Change、benchmark/eval 表格，以及仅说明下一子目标的 Next。英文展开，简体中文折叠在 中文 区块，数值表格共用一次。单元格显示新值和差值，表头使用箭头，长度使用精确的 8k/1k 简写。不添加 Result 列、可见的 Coverage/Finding 段落、图例、存储说明或限制章节；失败及无法比较的原因保留为简短备注。目标/变更/下一步使用 en/zh 记录。

完成 checkout 和上下文准备后，candidate 工作流将控制权交给 Klaud Cold，并提供 `AGENT_PAT`、`ANTHROPIC_API_KEY` 和私有 API 只读密钥。Klaud Cold 将公开观测解析到一个活动主配置族，检查当前镜像和已有 PR，并在**编辑或创建分支/PR 之前**核实检查结果中的目标 ID 和实时容量。随后在使用 GPU 前认领分支，产生实际修改并创建草稿 PR。所有生成的 PR 标题必须以 `[Klaud Cold] ` 开头，后接英文 / 简体中文描述。不得在 GitHub 上 @提及用户/团队，也不得请求 review/re-review；这些操作由自动流程处理。它在认领分支前立即重新检查开放 PR，因为检查只是快照，不能锁住后来创建的人工 PR。有歧义、已退役或已经更新的候选直接停止，不运行扫描。提交、推送、调度、监控、诊断、修复及使用上述布局的 PR 正文/评论更新都由同一会话完成。

选择或修复镜像前，比较新旧镜像实际内置版本对应的 vLLM、SGLang、ATOM 或 TensorRT-LLM 源码标签/提交，以及关联的 serving 依赖。检查参数和配置解析器及执行路径，确认默认值或语义变化、改名/移除的选项和相关新增选项。仅看 release notes 不够；核实镜像与源码的对应关系，无法确认时如实说明，不能假定最新 `main` 就是镜像内容。将源码链接、相关变化和范围内的决策写入尝试评论。这不扩大允许修改的范围，也不允许运行时补丁。

定向尝试只测试更新后的镜像：从 `main` 调度 `e2e-tests.yml`，将 `inputs.ref` 设为实际测量 SHA，并使用 `generate-cli-command="test-config --config-files FILE --config-keys FAMILY --smoke"`。生成器将最低并发的吞吐量检查与规范并发上的代表性 eval 分开，不再把长评测降到吞吐量最低并发。这只能证明启动和兼容性，不能代表完整曲线。smoke 通过后，追加精确配置族的 changelog 条目，不使用场景、eval 选择或 append-only 修饰项。描述必须是最多 120 个字符的一句简洁英文，只写引擎版本变化，并仅在必要时附带一项关键兼容性调整。证据、结果、版本摘要、理由和限制写入尝试评论。保持 PR 为草稿并添加 `full-sweep-fail-fast`。最终 sweep 保留全部测试点和默认 eval。任何同仓库 PR 均可通过 sweep 标签授权草稿运行；fork PR 仍使用受信任调度路径。最终失败后，先移除 sweep 标签、保持草稿，再推送修复。补全尝试评论后调用 `finish`；它先验证全部必需结果，再将 PR 标记为 ready，触发自动审阅。标记为 ready 不会启动 sweep。确认结果为 `validated` 后，Klaud 仅发布一次 `/use <verified-final-run-id>`，保留已完成运行供合并时复用。不要求性能差值为正；回退应如实报告。Klaud 不自行 staging、请求 review 或合并。

在编辑或创建分支之前、每次定向调度之前，以及最终 sweep 的标签转换之前，使用 `check-capacity --cluster ID` 检查精确目标；通过重复 `--cluster` 指定每个可能的目标。退出状态 0 要求全部目标均通过新鲜度、可用性和低于 80% 利用率检查。如果该检查在定向调度、最终 sweep 转换或恢复调度之前失败，Klaud 会先在已有 PR 的评论中记录可公开的容量延后原因和当前尝试状态。随后取消并确认全部所属运行已经结束，再用终态或已取消行及确认后的状态更新尝试评论。最后移除所有 sweep 标签、将 PR 改回草稿、关闭 PR，并删除远程 Klaud 分支，使后续扫描可以重试该候选。如果尚无 PR，则在最终响应中记录延后结果，不创建占位 PR。调度后利用率上升不会导致健康运行被取消。Klaud 不会等待恢复或承诺自动继续。命令不打印容量详情。

Fail-fast 只会取消失败矩阵内排队或运行中的其他任务；其他矩阵仍可能继续运行。先诊断首个失败，并等待所有自有任务结束后再重试。被取消的测试点仍须取得成功结果；若基础设施故障导致整个 run 以 cancelled 结束，应在同一 head 上重跑整个 attempt。保留现有重试预算，不得将确定性的镜像失败归类为基础设施问题。只有在已记录基础设施例外、需要让健康任务在同矩阵其他任务失败后继续运行时，才使用 `full-sweep-enabled`；必须等自有任务结束后才能切换标签。已有的非 fail-fast 运行仍可用于验证。两种模式都要求唯一的 sweep 标签、精确 head 上成功的最终运行，以及全部基线测试点和默认 eval。

使用 `report` 分别记录初次尝试、修复 N/5、基础设施重试和最终完整 sweep。在等待前持久化所属 run/head/attempt。发生实质变化或等待满 30 分钟时更新同一条评论，完成的历史保留。渲染器统一处理匹配、单位、差值及大记录拆分。已确认的临时基础设施重试与 recipe 修复分开计算，每次尝试最多两次。`finish` 从已验证产物生成最终报告，发布后才标记就绪。缺失基线记为 N/A，回归如实报告，不设置拒绝阈值。

最终 sweep 无论测试点数量多少，都运行完整的所选配置族。单次 benchmark 运行可能长达三小时；Klaud Cold 在候选作业的总时限内监控其进展。Klaud 不再为单次 benchmark 运行设置额外超时。

工作流设置与其执行位置放在一起：[`klaud-plan.yml`](../.github/workflows/klaud-plan.yml) 中的 `MAX_CANDIDATES_PER_RUN`、`REVIEW_BATCH_SIZE` 和 `CANDIDATE_COOLDOWN_HOURS` 分别控制并行调度、检查批次大小和软重复排序；[`klaud-candidate.yml`](../.github/workflows/klaud-candidate.yml) 的 `MAX_REPAIRS` 将修复次数上限直接传入 agent 提示词。

Klaud Cold 最多运行 500 轮，使用 GitHub Actions 默认作业时限。诊断、修复选择和尝试评论仍由 agent 负责；`finish` 通过代码验证结果覆盖、子运行结束、最终报告和分支处理。作业时限、API 错误或 runner 被终止仍可能中断会话；下一次 autosweep 会在选择新候选前收尾有归属记录的会话。不增加自定义工作流超时。

Klaud Cold 调度 `e2e-tests.yml` 时显式设置布尔输入 `klaud-run: true`，并使用 `klaud-` 测试名称方便识别。手动和复用调用中的该输入均默认为 false，并传递至所有 benchmark/eval 模板。只有这个标志会为作业名称添加 `klaud | ` 前缀，供 InferenceX Dash 优先级调度器识别；普通测试名称不会改变优先级。配套调度器改动使 Klaud 始终排在普通人工任务之后，不受任务到达时间或 recipe 分数影响；Klaud 不获得等待时长加分或节点预留，skip-queue 请求也不生效。Klaud 使用剩余容量，不抢占已经运行的任务。显式管理员优先级覆盖保留原有优先顺序，Klaud 不得主动请求。启用 auto-sweep 前需部署配套 dashboard 调度器改动。

### 已发布基线与会话完成

基线来自 **`https://inferencex.semianalysis.com` 的公开 dashboard API**。规划阶段先解析 OpenAPI 展示模型名，并在启动 agent 前预检完整测试点清单。候选使用 `candidate.json` 中这一精确值，将 `candidate.source.date` 传给 `workflow-info` 和 `benchmarks`，设置 `date` 和 `exact=true`，不使用计算器 `view`。预检只判断资格；候选解析精确的新旧镜像目标后，仍须冻结并发布自己的基线。核实旧镜像以及完整的模型、硬件、框架、精度、推测解码和工作负载身份，再逐点匹配拓扑、并发量及数据集。记录 API 查询、发布日期和每个测试点的来源 `run_url`/SHA，区分逻辑曲线快照与实际数据来源。按需读取已发布 eval，所有尝试共用这份固定基线。缺失或不可比较的数据填写 `N/A` 并说明原因。绝不调度或重跑旧镜像基线。

调度运行或创建草稿不代表任务完成。使用 `gh run watch --interval 60` 留在同一会话中等待，工具超时后继续等待，并检查作业级状态，因为 queued 工作流可能包含正在运行的作业。benchmark 矩阵失败后，eval 作业仍可能继续。定位首个服务端错误而非清理阶段症状；在原有范围、预算和容量规则内修复。工具调用被拒绝时改用允许的工具或命令，不得提前报告成功。先将所有尝试的最终结果写入 PR 尝试评论，再报告停止原因、修复次数、已确认的子运行结束状态和 PR URL。不得承诺稍后继续监控，也不得仅为结束会话而取消正常运行。

[Stop hook](https://code.claude.com/docs/en/hooks#stop) 通过 `check-stop` 检查 `$KLAUD_EVIDENCE/outcome.json`。结束前，将请求的 `CandidateOutcome` JSON 写入单独文件，运行 `uv run --no-project --exclude-newer PT12H --python 3.12 --with "pydantic>=2.10,<3" --with pyyaml python -m infx.klaud finish --outcome-file "$KLAUD_EVIDENCE/requested-outcome.json"`，然后仅调用一次 `StructuredOutput`，传入已验证的 `outcome.json` 对象，不附加说明文字，也不将其编码为字符串。诊断优先使用该验证文件；仅在文件不可用或无效时，才回退到 action 的重复结构化输出。命令从父运行原始创建时间起发现所有自有定向和最终运行，包括旧 head、已关闭或移除标签的 PR。先发布失败或延后报告，再取消运行；全部结束后才移除 sweep 标签、退回草稿、关闭 PR，并为所有失败结果删除未移动的精确 head 分支。完成记录包含实际运行 ID 和清理状态。取消或 PR 状态转换事件的作业仍在进行时，等待后重试 `finish`。维护者接管优先于清理；不覆盖其他所有者、fork、已移动的 head、已合并 PR 或不明确的状态。没有所属 PR 时不删除无法验证归属的分支。hook 仅做验证，不修改状态，也不能突破 Claude 内置的停止循环上限。

每次 agent 步骤结束后，无论 action 结果如何，`recover-current` 都执行一次受信任的非阻塞收尾。没有 PR/run 的会话会立即释放；健康子运行保留给后续恢复；已结束工作则完成清理或验证。随后诊断优先使用经 GitHub 验证的生命周期记录。固定错误码区分会话状态不可用、记录无效、结构化输出无效和生命周期验证失败。无法验证时，先上传脱敏产物，再让候选作业失败。仅记录固定结果类别、数字 ID/指标及 head/attempt，不包含原始执行消息、命令、凭据或私有响应。

`run-sweep.yml` 上传包含完整矩阵、精确 head 和 run ID/attempt 的 `klaud-sweep-manifest`。`check-final` 与最终验证均使用受信任代码，从精确 head 的 YAML 独立生成未过滤配置族。覆盖等价的 scenario filter 可以通过，缺失或改变的配置点与默认 eval 不能通过。验证器选择当前 attempt 的 manifest，以及同一 run/head 下每个名称最新的产物。仅当对应配置生产作业没有重跑时，才保留之前的 aggregate；全部覆盖及原始结果/汇总一致性仍须通过。不同 archive 不叠加解压。缺失、过期产物及生成器策略变化需要检查；无 manifest 的旧运行不能自动认证。

### 中断恢复

选择步骤先原子创建每个配置族的 `klaud/claim-*` ref，保存经验证的归属，同时上传兼容旧流程的 `klaud-ownership`。ref 覆盖上传产物前的中断，并阻止同配置族其他 release 重复启动。恢复先验证完整清单，仅处理受信任 main 的已结束父运行，使用每会话 `klaud/recovery-*` 租约。旧租约所属工作流结束后，才通过非强制 fast-forward 的比较交换接管；没有全局工作流锁。健康子运行继续执行，下一轮再检查，不阻塞新候选选择。未解决配置族继续排除，其余配置族可以推进。全局归属或清单无效时仍停止调度。

恢复会先发布并认证成功的完整最终结果，再标记就绪。已结束但无法认证的结果以需检查的终态关闭并删除未移动的精确 head 分支，不能据此推断镜像不兼容。中断清理从绑定 head 的记录继续，不把未知修复次数改为零；不修改已移动 head、已合并、已接管或其他所有者的工作。恢复还会根据已验证的历史完成记录，删除过去保留的精确 head 分支。取消后未结束的子作业留到后续检查，恢复不阻塞等待。仅退休全部已解决的归属产物，公开报告保留。正常收尾或独占恢复释放配置族认领。恢复不会启动另一 agent、创建 PR 或调度替代 sweep。

## 公开 API 调查

从当前 [API 参考文档](https://inferencex.semianalysis.com/api) 和 [OpenAPI 文档](https://inferencex.semianalysis.com/api/openapi.json) 获取参数、响应结构和限制。planner 只用 `latest-images` 定位已发布 benchmark 基线；Klaud Cold 根据候选情况选择额外读取，并独立检查上游源码与镜像。API schema 和模型/runner 清单仍以各自的现有来源为准。

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

- `latest-images` 和 `availability` 返回原始模型键；benchmark 查询要求使用当前 OpenAPI 枚举中的展示名称。核对返回的模型及完整候选身份。`latest-images` 仅用作 benchmark 基线观测，不能代替当前仓库配置族、当前版本或兼容替换镜像清单；Klaud 会独立检查上游源码和镜像。
- 向 `workflow-info` 传入观测日期：虽然参考文档说省略日期会查询全部日期，当前查询实现会将该参数转换为 SQL 日期。变更日志中的配置键和运行覆盖范围可以缩小查找范围，但不能证明配置族唯一。
- 对于 append-only 运行，`exactRun=true` 可能包含同一镜像的前序运行链。保留各测试点产出运行的 `run_url`，并与逻辑 `curve_*` 快照元数据区分。benchmark 行的 `workflow_run_id` 是数据库 ID；`runId` 参数和 `workflow-info` 中的运行标识是 GitHub ID。诊断接口要求传入数据库基准测试结果的正整数 `id`。
- `submissions` 按配置/日期聚合测试点，未排序便选取一个非 null 镜像，因此不能用于确定精确镜像基线。公开缓存和数据导入可能滞后于新运行。诊断数据缺失表示证据不可用，不代表数值为零或测试通过。
- 原始 benchmark 时间指标使用秒；请求时间线使用纳秒偏移，延迟字段名称明确标注毫秒。服务器时间序列偏移使用秒。物理芯片数与逻辑 TP 独立；只有分离式部署才应将 prefill/decode 芯片数相加。

应用还有一些有用的**未发布 UI 读取接口**：`/api/unofficial-run` 将尚未导入的运行产物归一化，`/api/gpu-metrics` 读取 GPU 指标产物，`/api/v1/eval-samples` 和 `/api/v1/eval-samples-live` 可下钻失败样本，`/api/v1/trace-server-metric-source` 获取所选 worker/来源的时间序列。按需使用前先检查当前 handler；这些契约由页面内部使用。非官方 benchmark 行使用合成的 `id: 0`，不能用于查询已存储的诊断数据。新旧镜像比较的证据仍须对应原始运行产物。

`tco-feed` 提供经过插值的图表/TCO 背景，不用于精确 recipe 比较。`overview`、`request-chart-data` 和 `resident-sequence-lengths` 是 UI 投影，没有此处候选选择所需的额外数据。反馈与管理路由不参与候选调查。

## 修改范围与 PR 策略

权威[候选指令](../.github/klaud-candidate-prompt.md)仅允许修改所选 master 镜像，以及其已引用且未共享的单节点 recipe 或 srt-slurm YAML 中有源码依据的兼容性参数/环境变量。模型、精度、拓扑、推测解码、工作负载/数据集、时长、资源、全部配置点和 eval 保持不变。`model.container` 和 `identity.container.image` 必须一致。Klaud 不进行广泛调优，也不修改共享脚本、launcher、库或工作流。运行时引擎/serving 技术栈补丁必须为零，包括选定路径已有的补丁：禁止改写源码、容器、site-packages，禁止 overlay、monkey patch 或重建/fork 的 wheel。选择可原样运行的镜像，否则报告不兼容。

smoke benchmark 和代表性 eval 都通过后，在 changelog 物理末尾追加条目并保留历史字节，推送精确 head，生成/检查完整矩阵并重新检查容量。保持草稿，只添加 `full-sweep-fail-fast`。最终失败后，先移除 sweep 标签并保持草稿，再修复。仅 `finish` 在结果验证和最终报告发布后标记就绪；就绪触发审查，不启动新 sweep。随后 Klaud 用 `/use` 记录已验证的最终运行；审查、staging 和合并仍由维护者决定。

## 工作流操作与凭据

入口工作流每六小时运行一次（`0 */6 * * *`，UTC），并支持手动触发（`workflow_dispatch`）。每次调用最多选择五个候选并行运行。条件 `github.ref == 'refs/heads/main' && github.run_attempt == 1` 会跳过功能分支、tag 和重新运行。candidate 作业也跳过重跑；应重新调度 autosweep，先执行恢复。只有 candidate 暴露 `workflow_call`。两个 Klaud 工作流均不设置并发组：新调用可以与已有调用重叠，且不会取消已有运行。五个候选的上限按每次调用计算，因此多轮重叠时，活跃候选总数可以超过五个。开放 PR 检查和 agent 对精确分支的原子认领仍用于防止重复工作；审查只是快照，agent 必须在认领前再次检查。恢复先于选择执行，但不阻塞等待活动子运行。配置族认领和每会话恢复租约保护重叠调用，未解决配置族仍被排除。

恢复和最终选择步骤使用 `AGENT_PAT` 执行已确认归属的清理及配置族认领；只读重叠检查 agent 不获得该凭据。planner 的 Python 准备和最终容量检查步骤使用 dashboard key；准备步骤还使用只读工作流 token。限轮数的 Claude PR 检查使用 `ANTHROPIC_API_KEY` 和具有 `pull-requests: read` 权限的只读工作流 token。它接收私有资格线索，但不接收 `AGENT_PAT` 或 dashboard key，也不执行 GitHub 写操作。candidate 获得用于分支/PR 写入及 e2e 调度/取消的 `AGENT_PAT`、用于 Klaud Cold 的 `ANTHROPIC_API_KEY`，以及覆盖 clusters 的限期 `status:read` `DASH_API_KEY`。Klaud Cold 不得发布凭据或私有 API 响应。共享 HTTP 读取器固定来源，同时限制压缩和解码后的 GET 响应大小，支持明确的 gzip/identity JSON 响应，并拒绝重定向或不支持的编码。非有限 JSON 数值（包括 `1e400` 这样的指数溢出）会在校验或哈希计算前被拒绝。不需要部署额外服务、数据库或新增 environment 配置。

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
uv run --no-project --exclude-newer PT12H --python 3.12 --with "pydantic>=2.10,<3" python -m infx.klaud --help
uvx --exclude-newer PT12H zizmor@latest --offline --no-config --no-ignores .github/workflows/klaud-plan.yml .github/workflows/klaud-candidate.yml
```

CLI 和工作流检查不能证明 GPU 实际可运行。Klaud Cold 使用现有 InferenceX 校验和 e2e 工作流验证候选修改。本地验证不调用真实模型、不调度 benchmark、不创建 PR、不部署。

严格 zizmor 扫描没有未忽略的发现。使用 `--no-ignores` 检查时，会报告允许多轮重叠的并发例外，以及五项现有仓库 secret environment 提示。`actionlint` 1.7.12 尚不识别 runner 的 `$/` 同仓库工作流调用语法，因此会报告该现有调用，但 GitHub Actions 可以接受它。

### 结果文件命名与最终 sweep 调度

定向和最终工作流使用同一个有长度上限的结果文件前缀。长前缀对完整身份（包括完整 recipe fingerprint）做哈希，JSON 中保留原始 fingerprint。对不含文件命名辅助程序的旧提交运行基准测试时，工作流会回退为对完整身份做哈希。SRT 测试点文件保留数字并发/GPU 后缀，只压缩过长的配置名；长度预算包含 `power_validation_`、`.json` 和原子写入的 `.tmp` 后缀。较短的测试点文件名保持不变。

最终 sweep 的每个 benchmark/eval 调用（包括 canary）都会为同仓库、由 `Klaud-Cold` 创建且分支为 `klaud/auto-*` 或旧拼写的 PR 传递后台优先级标志。人工 PR 不会因标题或标签而获得 Klaud 优先级；最终运行与定向运行使用相同的 `klaud |` 调度器标志。
