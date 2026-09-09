# 文档维护流程

<div align="center">

[English](./documentation-procedures.md) | **中文**

</div>

新增或更新面向贡献者的文档、准备配套 GitHub 文本或审阅文档改动时，请使用本指南。仓库源文件和工作流始终是行为的权威来源；文档用于解释这些来源的契约、设计理由和安全操作流程。

> **硬性要求：** 英文页面是主要来源版本。先编写或更新英文页面，再在同一个改动中将其翻译为对应的简体中文 `_zh.md` 页面。两个页面的标题、结构、链接、命令、表格和代码块必须同步，并在顶部提供 `English | 中文` 语言切换器。

## 流程索引

| 流程 | 适用场景 |
| --- | --- |
| [确定文档范围](#确定文档范围) | 判断应更新现有页面、新建聚焦页面，还是修改与源码相邻的参考文档 |
| [新增面向贡献者的页面](#新增面向贡献者的页面) | 创建新的双语指南 |
| [更新现有双语页面对](#更新现有双语页面对) | 修改契约、命令、链接、图片或说明 |
| [更新文档索引](#更新文档索引) | 让新页面可被发现，或调整页面用途 |
| [维护权威来源链接](#维护权威来源链接) | 在不制造第二套权威规则的前提下说明实现行为 |
| [编写双语 GitHub 内容](#编写双语-github-内容) | 准备 PR、issue、评论、审阅或 commit 文本 |
| [完成 CODEOWNER 签核](#完成-codeowner-签核) | 按仓库审阅策略批准 PR |
| [审阅文档改动](#审阅文档改动) | 检查正确性、双语一致性、导航和可读性 |

## 规则权威来源

编辑前先阅读与改动相关的来源：

| 来源 | 管理内容 |
| --- | --- |
| [`AGENTS.md`](../AGENTS.md) | 双语文档与 GitHub 文本规则、commit 格式和 Agent 政策 |
| [`.github/AGENT_OPERATIONS.md`](../.github/AGENT_OPERATIONS.md#translation-terminology) | 首选翻译术语 |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | PR 审阅顺序、CODEOWNER 批准、扫描结果复用和合并后责任 |
| [`docs/index.md`](./index.md) 与 [`docs/index_zh.md`](./index_zh.md) | 维护中的文档地图与导航措辞 |
| [`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) | 当前 CODEOWNER 签核模板与合并标准 |
| [`.github/CODEOWNERS`](../.github/CODEOWNERS) | PR 所改路径对应的所有者 |
| [`.github/codeowner-signoff-verify-prompt.md`](../.github/codeowner-signoff-verify-prompt.md) | 签核验证器编码的独立检查项 |
| [`.github/workflows/codeowner-signoff-verify.yml`](../.github/workflows/codeowner-signoff-verify.yml) | 签核触发事件、精确短语检测与状态发布 |
| [`.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`](../.github/PULL_REQUEST_TEMPLATE/pull_request_template.md) | PR 字段和作者检查清单 |
| [`.github/ISSUE_TEMPLATE/`](../.github/ISSUE_TEMPLATE/) | Bug 与功能需求 issue 模板 |

本文采用 [`InferenceX-app` 文档中心](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/index.md)中有效的信息架构模式：中心页面链接到聚焦的专题页面；[架构页面](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/architecture.md)解释决策与取舍；[测试页面](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/testing.md)明确强制门禁和审阅标准；[新增实体页面](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/adding-entities.md)使用有序工作流、前置条件、检查清单和明确验证步骤。

## 确定文档范围

1. **明确读者和任务。** 一个页面应解决贡献者或运维人员会反复遇到的一个问题。如果读者必须加载多个无关子系统才能使用该页面，应将其拆分。
2. **先搜索文档地图。** 如果最接近的现有页面已经负责该主题，就更新该页面。不得创建第二套约定或重复流程。
3. **起草前先找到实现来源。** 针对每项操作性陈述，阅读其背后的准确配置 schema、工作流、脚本或策略文件。相邻指南只能提供上下文，不能证明当前行为。
4. **选择页面类型：**
   - **架构/设计理由：** 所有权、数据流、不变量、备选方案，以及当前设计存在的原因。
   - **操作流程：** 前置条件、有序步骤、验证、失败/停止条件和权威链接。
   - **参考：** 稳定的 schema、术语、支持值或导航地图。
   - **故障排查：** 可观察症状、需收集的证据、根因分支、安全修复方法和升级边界。
5. **源码相邻的参考文档应继续靠近源码。** 详细 schema 或命令文档可能更适合放在代码旁，例如 [`configs/CONFIGS.md`](../configs/CONFIGS.md)、[`.github/workflows/README.md`](../.github/workflows/README.md) 或 [`utils/evals/EVALS.md`](../utils/evals/EVALS.md)。应从 `docs/` 链接它们，而不是复制内容。
6. **编辑前定义完成条件。** 明确双语页面对、索引位置、实现链接和所需审阅证据。

## 新增面向贡献者的页面

### 1. 命名页面对

在 `docs/` 下使用语义明确的 kebab-case 文件名：

```text
docs/<topic>.md
docs/<topic>_zh.md
```

简体中文页面必须使用 `_zh.md` 后缀。仅面向 Agent 的说明文件（`AGENTS.md`、`CLAUDE.md`、`KLAUD_DEBUG.md`）、`.github/` 或 `utils/` 下的内部参考资料，以及 `configs/CONFIGS.md` 和 `experimental/README.md` 等贴近实现的参考资料，是已规定的仅英文例外。不要自行增加例外。

### 2. 添加双向语言切换器

将切换器紧跟在各页面标题之后。

英文页面：

```html
<div align="center">

**English** | [中文](./<topic>_zh.md)

</div>
```

中文页面：

```html
<div align="center">

[English](./<topic>.md) | **中文**

</div>
```

使用相对链接，确保页面对可在 fork、分支和本地渲染器中正常工作。

### 3. 构建聚焦且有索引的页面

仅使用主题实际需要的章节，通常按以下顺序组织：

1. 用一段话说明用途和权威边界。
2. 如果遗漏某项要求会导致静默数据丢失、算力浪费、审阅失效或不安全操作，应突出关键前置条件或不变量。
3. 页面包含多个工作流时，提供流程索引。
4. 列出规则来源或架构上下文。
5. 编写有序流程，前置条件必须在编辑步骤之前。
6. 明确验证和停止条件。
7. 提供最终审阅检查清单。

优先使用简短决策表、有序流程、检查清单和精确链接。对不直观的约束说明其**原因**。不要把页面写成文件清单，也不要粘贴会随实现漂移的大段代码。

### 4. 先完成英文，再翻译并审阅

- 将英文页面视为主要来源版本：先完成其中的陈述、结构、标题、链接、命令、表格和代码块，再翻译对应的 `_zh.md` 页面。
- 两个页面的标题层级、流程顺序、表格、链接、徽章、图片、警告和代码块必须一致。
- 翻译意图，不翻译语法。命令、路径、环境变量、flag、模型名称、硬件 SKU、框架名称、日志和堆栈跟踪保持不变。
- 使用自然的技术简体中文，不做逐字翻译。遵循[首选术语](../.github/AGENT_OPERATIONS.md#translation-terminology)。
- 对应标题的语义应保持一致，从而保留有意义的锚点。翻译后测试跨页面链接和页内链接。
- 如果某项陈述无法在一种语言中准确表达，应先修正原始陈述；不得让两个页面产生分歧。

### 5. 添加导航并验证

在同一个改动中更新两个文档索引，然后执行[审阅文档改动](#审阅文档改动)中的检查。仅创建页面文件不代表工作完成。

## 更新现有双语页面对

1. 编辑前阅读两个页面。确认当前是否一致；将已有不一致与本次请求的改动分开记录。
2. 阅读当前权威实现或策略。仓库级文档规则从 [`AGENTS.md`](../AGENTS.md) 开始；审阅策略从 [`CONTRIBUTING.md`](../CONTRIBUTING.md) 和 `main` 上的最新 checklist 开始。
3. 先更新作为主要来源的英文页面。在翻译前，完成其中每一项已改动的陈述、标题、链接、警告、表格行、徽章、图片和代码块。
4. 在同一个 PR 中，将最终确定的英文改动翻译到对应的 `_zh.md` 页面。代码和证据保持原文；翻译外围说明，而不是 CLI flag、路径名、日志或堆栈跟踪。
5. 删除已被新契约废止的陈述。除非实现仍支持旧路径且读者确实需要迁移规则，否则不要遗留已弃用路径、别名或相互矛盾的流程。
6. 如果页面标题、范围或推荐入口发生变化，更新双语文档索引。
7. 按源码重新核对每项操作性陈述。仅包含文档的 diff 也可能在行为上是错误的。

### 特殊情况：PR 审阅清单

[`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) 是合并标准的权威来源。其模板在两个语言版本的 checklist 中都必须保持完全一致的英文原文，因为验证器依赖精确英文措辞进行检测。

如果新增、删除或实质性改写 checklist 项，应同步更新 [`.github/codeowner-signoff-verify-prompt.md`](../.github/codeowner-signoff-verify-prompt.md)，使独立检查编码相同策略，最好放在同一个 PR 中。格式调整、拼写修正和 `_zh` 翻译同步不需要修改验证逻辑。然后执行[验证 verifier 变更](#验证-verifier-变更)。

### 验证 verifier 变更

本地 Markdown 或 YAML 检查不能证明 verifier 变更有效。变更合并后：

1. 创建一个临时 `[DO NOT MERGE]` 测试 PR。
2. 发布包含精确短语 `As a PR reviewer and CODEOWNER` 的 CODEOWNER 签署评论，否则 Workflow 不会触发。
3. 阅读 verifier 发布的结论，确认其执行了预期检查。
4. 关闭测试 PR。

## 更新文档索引

每个新的面向贡献者页面都必须能从维护中的中心页面被发现。

1. 先在作为主要来源的英文 [`docs/index.md`](./index.md) 中添加并最终确定条目。链接到 `<topic>.md`，并写出简洁的英文“适用场景”描述。
2. 将最终确定的条目翻译到 [`docs/index_zh.md`](./index_zh.md) 的对应位置，并链接到 `<topic>_zh.md`。
3. 描述页面解决的问题，而不是仅重复标题。
4. 将条目放入现有文档家族或导航表，不要为同一类别新建第二个索引章节。
5. 如果新页面替代旧入口，在同一改动中更新所有索引链接并删除旧条目。不得保留两个“规范”页面。
6. 两个索引的结构必须一致：条目顺序、目标页面集合、指南家族和系统边界相同。
7. 从渲染后的两个索引分别进入对应语言页面，再通过切换器返回另一语言页面，确认整个导航链路。

## 维护权威来源链接

文档是源码的入口，不是源码的替代品。

### 将陈述链接到负责该行为的来源

| 陈述 | 首选来源 |
| --- | --- |
| 仓库贡献规则或双语规则 | [`AGENTS.md`](../AGENTS.md) 或 [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| CODEOWNER 身份 | [`.github/CODEOWNERS`](../.github/CODEOWNERS) |
| 审阅 checklist 措辞 | `main` 上的 [`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) |
| CI 触发器、权限或 job 行为 | [`.github/workflows/`](../.github/workflows/) 下的准确文件 |
| 配置字段或允许值 | 主 YAML、[`configs/CONFIGS.md`](../configs/CONFIGS.md) 以及验证器/生成器代码 |
| 运行时 flag 或环境变量 | 消费该值的基准测试脚本、启动器或共享 helper |
| 产物 schema 或评估门禁 | 收集器/验证器代码及其源码相邻参考文档 |
| 外部框架行为 | 带版本的上游文档、recipe、release 或源码链接 |

### 链接编写规则

- InferenceX 仓库内的文件优先使用仓库相对链接，以便在 fork 和分支预览中继续工作。
- 链接到准确文件或聚焦的源码相邻指南，而不是仅链接仓库根目录或宽泛目录。
- 使用稳定的上游 URL。如果陈述依赖特定版本，应固定到 commit 或 release；如果有意引用动态策略，则链接上游默认分支。
- 存在两种表达形式时，明确哪一个来源优先。例如，工作流 YAML 定义 CI 行为，而对应 README 解释调用方式。
- 只复制执行任务所需的短小语法示例。默认值、支持值、分支逻辑和失败行为应链接到实现。
- 源码移动或契约变化时，在同一 PR 中更新所有指向它的文档链接。搜索旧路径和旧术语，不要只修复当前打开的页面。
- 除非命令确实在所声明环境中执行过，否则不得写成“已验证”。必须区分静态语法检查与运行时或 CI 结果。

## 编写双语 GitHub 内容

双语规则适用于每个 PR、每个 issue 和人工编写的 PR 评论。机器人生成的评论遵循其工作流模板。未经授权，不得发布外部评论、review、label 或 merge 命令。

### PR 与 issue

标题使用以下格式：

```text
<English title> / <中文标题>
```

正文要求：

1. 完整填写现有 PR 或 issue 模板；不要删除必填字段或检查清单。
2. 用英文写明摘要、动机、改动、验证、风险和来源链接。
3. 添加 `## 中文说明` 章节，镜像英文中的实质内容。
4. 代码块、命令、日志、堆栈跟踪、路径、标识符和 URL 保持不变。在原始块前后用中文解释其含义。
5. 证据必须对称。英文部分链接了工作流、产物、issue 或源码时，中文部分必须保留相同链接。
6. 只有实际观察到对应操作完成后，才能勾选 checklist 项。

最小结构：

```markdown
## Summary
- Explain what changed and why.

## Verification
- Link or state the exact check that ran.

## 中文说明
### 摘要
- 说明改动内容及原因。

### 验证
- 保留相同的检查结果或链接。
```

### 评论与审阅总结

- 简短评论：单行使用 `<English> / <中文>`。
- 较长评论：先写英文段落，再添加 `中文：` 段落或标题明确的中文章节。
- 行内 review comment、对话评论和 review summary 都必须提供中文翻译。
- 代码摘录和证据保持不变；翻译诊断、影响和所需修复。
- CODEOWNER checklist 签核是例外：必须复制精确英文原文，不要在签核内容中追加翻译后的 checklist。

### Commit

使用 conventional commit 格式的英文 subject，并在 commit body 中加入 subject 和关键正文要点的简体中文翻译：

```text
docs: add documentation procedures

Explain the source-of-truth and review impact in English when a body is needed.

中文：新增文档维护流程，并说明权威来源与审阅影响。
```

Squash merge commit 会继承双语 PR 标题，因此自动满足 subject 要求。不要利用文档措辞绕过扫描或合并策略。

## 完成 CODEOWNER 签核

请求或发布签核前，先遵循 [`CONTRIBUTING.md`](../CONTRIBUTING.md)。

### 作者准备工作

1. 完成双语文档页面对，以及所有必需的索引/源码更新。
2. 完整填写 PR 模板，并链接准确的验证证据。
3. 通过要求的 PR 验证路径。根据当前策略，CODEOWNER 签核依赖 PR 中某个 commit 上包含评估的全量绿色扫描；不得使用陈旧或无关的运行结果替代。
4. 根据 [`.github/CODEOWNERS`](../.github/CODEOWNERS) 向改动路径对应的 owner 请求审阅。
5. CODEOWNER 发布有效 checklist 签核之前，不要请求核心维护者进行最终批准。

### CODEOWNER 操作流程

1. 签核前立即从 `main` 打开最新版 [`docs/PR_REVIEW_CHECKLIST.md`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/docs/PR_REVIEW_CHECKLIST.md)。不要复用保存过的副本。
2. 确认自己是改动路径适用的 CODEOWNER。阅读当前 [`.github/CODEOWNERS`](../.github/CODEOWNERS)；所有权由当前路径规则决定，不能只根据公司归属判断。
3. 复制完整的当前模板，不要翻译或改写。以下开头必须完全一致：

   > As a PR reviewer and CODEOWNER, I have reviewed this and have:

4. 勾选前独立验证每一项。审阅 diff、源码行为、全量扫描与评估证据、上游 recipe 状态、镜像来源、架构限制、patch/waiver 状态、chat-template 要求，以及适用时的 AgentX acceptance 证据。
5. 在 additional-detail section 中填写准确的验证和评估工作流链接、已合并的上游 vLLM recipe/SGLang cookbook PR 或已发布 recipe 链接，并对每个例外或不适用项给出明确理由。
6. 在 `Signed:` 中填写真实 GitHub 用户名。不得代替其他审阅者签名。
7. 将精确英文模板作为对话评论、review summary 或 inline review comment 发布。验证器支持这三类事件。
8. 确认 [`.github/workflows/codeowner-signoff-verify.yml`](../.github/workflows/codeowner-signoff-verify.yml) 已触发，并阅读 verdict。验证器会重新推导合并门禁陈述；不会直接信任勾选结果。
9. 如果签核后 PR head 前进，应重新审阅新 diff 并发布新的签核。旧证据只对应之前审阅的 commit。
10. 只有获得授权的维护者才能记录 `/reuse-sweep-run` 并使用受支持的合并路径。CODEOWNER 批准本身不会授予该权限。

缺少必需的来源、工作流链接、recipe、例外理由或验证结果时，应停止而不是签核。保留未勾选项并提出具体后续要求；绝不能把未知状态写成批准声明。

## 审阅文档改动

应把文档当作可执行接口来审阅：读者会照着其中的命令和策略说明操作。

### 正确性与权威性

- [ ] 每项操作性陈述都与当前负责该行为的源码一致。
- [ ] 命令、路径、flag、配置 key、工作流名称、label 和 URL 完全准确。
- [ ] 页面明确指出权威来源，且没有创建相互竞争的契约。
- [ ] 前置条件位于操作步骤之前；验证和停止条件清晰明确。
- [ ] 示例真实，不声称执行过实际未运行的检查或任务。
- [ ] 策略变更更新了所有受影响的策略表面，包括必要时同步 checklist/verifier。

### 双语一致性

- [ ] 两个文件均存在，并使用要求的 `<name>.md` / `<name>_zh.md` 命名。
- [ ] 两个切换器均位于顶部、相互指向、使用相对链接且目标文件存在。
- [ ] 标题层级、流程顺序、警告、表格、链接、徽章、图片和代码块在语义上对应。
- [ ] 中文是自然的技术简体中文，并遵循仓库术语。
- [ ] 标识符、命令、日志、模型名称、硬件 SKU 和框架名称保持不变。
- [ ] CODEOWNER 英文模板在任何出现位置都保持精确原文。

### 导航与可用性

- [ ] 两个文档索引在正确家族中包含对应条目。
- [ ] 索引描述清楚说明何时使用该页面。
- [ ] 两种语言页面中的每个相对链接和标题锚点都能解析。
- [ ] 页面主题聚焦；重复实现细节已替换为准确源码链接。
- [ ] 决策表和检查清单清楚展示前置条件、所有权和完成标准，无需读者加载无关的大型手册。

### 审阅证据

1. 检查两种语言中完整的改动章节，而不是只看孤立的 diff 行。
2. 渲染或预览两个页面，检查标题、列表、表格、引用块、代码围栏和切换器。
3. 打开每个新增或修改的链接。对于外部链接，确认目标是预期版本或动态策略页面。
4. 运行仓库支持的最小文档检查。至少使用以下命令检查空白错误：

   ```bash
   git diff --check -- docs/
   ```

5. 如果文档描述了已变更的命令或运行时契约，应运行该源码模块对应的最小验证。Markdown 渲染不能证明运行时行为。
6. 在 PR 中用双语总结审阅结果。将已验证事实与推测分开，并明确列出任何未能运行的检查。

不能仅因为 Markdown 可以渲染就批准改动。导航损坏、命令过期、双语不一致和错误的合并策略都属于阻塞性文档缺陷。
