# 为 InferenceX 做贡献

<div align="center">

[English](./CONTRIBUTING.md) | **中文**

</div>

感谢你的贡献！我们欢迎 PR。本页介绍每个 PR 在合并前需要经过的审阅流程。

## PR 审阅流程

每个 PR 描述都必须包含 **AI model disclosure（AI 模型使用说明）** 部分，列出准备该 PR 时实际使用的完整模型名称/版本及各自的工作内容，包括委派给其他 agent 的工作。不能只写 Claude Code、Cursor 或 Perplexity Computer 等工具名。模型标识应以运行环境提供的信息为准，不得猜测；如果运行环境未提供确切模型，须明确说明无法确认。完全未使用 AI 的 PR 须填写 `No AI used`。后续修改使用其他模型时，须同步更新说明。

1. 打开你的 PR 并通过 PR 验证。添加 `full-sweep-fail-fast` 标签，强烈推荐使用此标签，因为变更有问题时每个矩阵最多浪费一个任务，而不是整个扇出。仅当需要任务在失败后继续运行时才使用 `full-sweep-enabled`。让基准测试 sweep 运行，并在 PR 的某个 commit 上获得全绿的完整 sweep，包括 evals。
2. 若修改的文件归属于仓库管理员及 `@SemiAnalysisAI/core` 之外的 CODEOWNER，请联系一位有资格的 [CODEOWNER](.github/CODEOWNERS) 审阅，并在批准评论中填写 **PR Review Checklist** 签署（见下文）。
3. 在 Slack 上联系核心维护者进行最终批准；若要求清单签署，请先完成签署。
4. 由授权维护者发布 `/use <run_id>`（见下文），然后通过 reuse 路径合并 PR。

**性能变更日志要求：** 凡是可能影响基准测试性能的变更，以及任何配方（recipe）的新增或修改，都**必须**在 `perf-changelog.yaml` 文件的物理末尾追加一个新条目。历史条目**严禁**编辑。

## Pareto 覆盖度

性能提交合并前，**强烈建议每条受影响的吞吐量与 E2EL Pareto 前沿至少有
5 个实测点**。这不是不可豁免的最低门槛：点数不足时，可以在进一步审阅后，
由管理员明确记录绕过决定再合并。缺少证据或无法核实覆盖度时也适用同一流程。
警告不能视为普通签核通过。

CODEOWNER 必须在清单的 Additional detail section 中记录每条受影响曲线的
点数和证据。点数不足会显示 `⚠️` 警告，并提醒 @functionstackx（Oren）、
@cquil11（Cam）、@Oseltamivir（Bryan）和 @adibarra（Alec）。管理员需在 PR 中写明理由、受影响
曲线及已评估 SHA；验证器需核实评论作者的仓库 `permission: admin` 和
`role_name: admin`。普通批准、`/use`、Core 团队成员身份或机器人评论都不等于
本项例外授权。即使已获授权，裁定仍保留 WARN 并链接该决定。

这是审阅政策，不会新增 GitHub 必需状态检查。现有签核工作流仍只更新参考评论，
不能阻止 GitHub 合并，也不会授予绕过权限。分支规则、sweep 和 eval 要求均不变。
代理不得自行使用管理员绕过权限。

### 统计仪表板前沿，而非 sweep 矩阵大小

- 使用每芯片总 token 吞吐量（`tput_per_gpu`，图表 `y_tpPerGpu`）与以秒计的
  E2EL。固定序列使用 `median_e2el`；AgentX 使用本次审阅的百分位，默认为 P90。
  若提交 P75，须单独报告。不得替换为 TTFT、交互性、仅输出吞吐量或集群吞吐量。
- 按模型、场景、硬件、框架、精度及已评估镜像和 run/date 分别统计。
  固定序列的不同投机解码方法属于独立曲线；AgentX 同一曲线内可包含不同拓扑、
  投机方法和 KV offload 配置。不得跨曲线或历史结果凑足点数。
- 统计实测前沿顶点，不统计配置的并发数、成功任务总数、重复坐标、插值线段或
  失败测量。E2EL 与吞吐量必须为有限正值；数据无效或缺失时应警告。
- append-only 提交统计最终整条曲线，包括有可复用 artifact 证据、配方未变且
  使用同一镜像的已有点。不要求本次新增 5 个点，也不得混入不兼容旧镜像的点。
- 提供 sweep run/attempt、原始 artifact、源 SHA、镜像、场景、百分位、指标及
  逐曲线点数。不影响基准测试曲线的文档或工具改动可记录 N/A，并说明原因。

实现依据为 InferenceX-app 提交
[`d507f3689274c82972709abb531bdedcb2e77946`](https://github.com/SemiAnalysisAI/InferenceX-app/commit/d507f3689274c82972709abb531bdedcb2e77946)：

- [`metric-registry.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/metric-registry.ts)
  为吞吐量与 E2EL 图选择 `upper_right`。
- [`paretoFrontUpperRight`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/lib/chart-utils.ts)
  按延迟升序、同延迟时吞吐量降序排序，保留吞吐量递增点，以及延迟不同但吞吐量
  相等的平台点。完全相同的坐标只计一次。平台点规则是当前应用行为，不等同于
  严格的数学非支配关系，不应在计数时悄悄更改。
- [`chartFrontier`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/utils/powerCurves.ts)
  仅在存在 canonical 标记时调用
  [`canonicalParetoIntersection`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/utils/canonicalFrontier.ts)。
  应先对所有有效点计算 E2EL 前沿，再与已核实的标记取交集；提前过滤可能使原本
  被支配的点进入前沿。当前
  [`ChartDisplay`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/ui/ChartDisplay.tsx)
  不会给普通官方 E2EL 点添加这些标记。不能仅凭未被调用的 canonical-ID helper，
  就认定所有 AgentX 曲线都必须取交集。

可信审阅工具接收已按曲线分组的实测坐标：

```bash
node .github/scripts/pareto-coverage.cjs < /tmp/pareto-curves.json
```

输入示例（只有一个实测点，因此返回 WARN）：

```json
[{"key":"model/scenario/hwKey/precision/run/percentile/image","points":[{"x":1.0,"y":100.0}]}]
```

须包含所有受影响曲线，包括空曲线。若本次评估的应用路径提供
`isOnNormalizedInteractivityFrontier` 标记，须保留已核实的标记。工具本身不会
下载 artifact、验证来源、发现遗漏曲线或批准例外，这些工作仍由验证器独立完成。
若应用前沿语义已偏离上述锁定版本，应报告 WARN 并解决差异，不能声称点数等价。

## Draft 模型精度

投机解码提交必须使用原始、未量化的 draft 权重，并保留其原生精度。此规则适用于内嵌的 MTP/NextN/EAGLE draft head 及独立 draft 模型（包括 DSpark），覆盖所有硬件厂商和框架。相对于参考实现，不得对 draft 精度进行任何更改，包括量化、降精度、升精度、同位宽 dtype 转换（如 BF16 转 FP16）或混合精度覆盖设置。此要求涵盖 draft 权重、激活、计算及 draft KV cache，无论更改发生在离线、加载时还是服务运行时；同样不得替换为经过精度转换的 draft checkpoint。

审阅者必须核实 draft 的实际运行精度，不能只看启动参数。检查 checkpoint 元数据、量化排除项、环境变量、锁定镜像中的框架默认行为，以及从 target 模型继承的量化设置。Target/verifier 模型仍可在满足现有 eval 要求的前提下量化，但其量化不能同时作用于 draft 组件。不得仅凭 target checkpoint 的名称或精度标签推断 draft 精度。

此规则涵盖 `--speculative-draft-model-quantization quark_mxfp4`、`SGLANG_GLM_NEXTN_MOE_PTPC=1` 等会量化 draft 计算的设置。上游配方、通过 evals、声称接受长度（AL）未变，或重新测得的 AL 曲线，都不能作为豁免理由。AgentX 的合成接受也不能证明 draft 精度保持不变。

涉及投机解码的改动，CODEOWNER 必须在 Additional detail section 中注明 draft checkpoint 及其 revision（或内嵌 head）、原生精度和实际运行精度，并提供用于确认 draft 权重和精度均未改变的元数据或锁定版本实现。无法核实精度时，该条目不满足要求。参见[审阅清单](docs/PR_REVIEW_CHECKLIST_zh.md)及[验证器检查 13](.github/codeowner-signoff-verify-prompt.md#check-13--draft-weights-and-precision-are-unchanged)。

此要求与 [MLPerf Inference Rules 附录 C：Speculative Decoding](https://github.com/mlcommons/inference_policies/blob/ff7edba545fded369e7e7e3d5a2f0bab4a95eece/inference_rules.adoc#appendix-c-speculative-decoding) 保持参考 head 精度的原则相似。该规则要求参考 MTP head 使用提供时的相同精度（"at the same precision as provided"），并禁止参考 head 权重量化及其他人为操纵接受率的行为。所引用的 MLPerf 版本对特定量化边缘工作负载设有例外，InferenceX 不采用该例外。此处仅比较保留 draft 权重和精度的原则，不引入 MLPerf 的允许模型列表、投机解码配置或接受率测试方法。

## PR Review Checklist（CODEOWNER 签署）

CODEOWNER 自动验证目前仅供审阅参考。工作流会核验提交的清单并更新同一条裁定评论，不再发布提交状态。GitHub 单独设置的 Core 团队和 CODEOWNER 批准要求仍然有效，除非有权限的维护者使用绕过权限。

仅当修改的文件存在仓库管理员及 `@SemiAnalysisAI/core` 之外的 CODEOWNER 时，才要求签核。归属以 PR 目标分支当前最新提交中的 CODEOWNERS 为准：先解析该分支的 SHA，再使用同一 SHA 校验并读取 CODEOWNERS，最后匹配的规则生效；重命名同时检查旧路径和新路径。归属规则不从 PR 的 Head 或其记录中可能过期的基础提交读取。同一文件有 core 团队作为 owner，不会豁免其他 owner。个人管理员必须同时拥有仓库 `permission: admin` 和 `role_name: admin`；其他团队和邮箱 owner 均要求签核。归属信息缺失或权限查询失败不能授予豁免。不涉及此类 owner 的改动会跳过验证。

由一名符合条件的 CODEOWNER 审阅者在批准评论中填写最新的 [PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md)（[中文说明](docs/PR_REVIEW_CHECKLIST_zh.md)）模板。

**每个 PR 只需一名符合条件的 CODEOWNER 审阅者发布清单。** 发布前先检查是否已有清单；其他审阅者无需重复发布。需要更正条目、补充证据或重试验证时，原审阅者必须**编辑自己已有的清单评论**，不要另发一条。只有原评论被删除时才创建替代评论。

友情提醒。请**正确**遵循最新的清单模板：

- 务必从 `main` 分支上**当前**的 [docs/PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) 复制模板。清单会不断演进，使用过期副本的签署会被标记为缺项。
- 保持模板的开头语句原样不变（必须保留英文原文）：

  > As a PR reviewer and CODEOWNER, I have reviewed this and have:

  我们的 CI 验证工作流 [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml) 正是通过这句话触发的。**如果批准评论缺少这句话，工作流就不会核验该清单。**
- 签署可以以普通会话评论、review 总结或行内 review 评论的形式发布。这三种方式都会触发验证。
- 请在 PR 处于打开且非草稿状态时提交或编辑清单。推送、重新打开或退出草稿状态不会触发验证。如果合并冲突期间遗漏了 Review 事件，请在解决冲突后编辑已有清单或手动分发工作流来重试。
- 启动 Claude 要求触发者为具有合格仓库写权限的人类用户。
- 请在 "Additional detail section" 中填写清单要求的链接（验证/评测工作流运行、对应的 [vLLM recipe](https://github.com/vllm-project/recipes) / [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new) PR，以及任何例外理由）。

签署发布后，CI 会独立复核审阅清单中的各项声明，包括 CODEOWNER 身份、PR 内 commit 上的全绿 sweep 与 evals、所链接的 recipe、复用命令、是否使用最新清单模板、上游 [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) 镜像、没有更改模型架构的基准测试 hack、投机解码是否使用 chat template、draft 模型和 draft head 的权重与精度是否保持不变，以及 Pareto 覆盖度。随后，CI 会为整个 PR 创建或更新同一条裁定评论，并注明实际评估的 SHA。未通过的条目及 Pareto 警告直接显示；已通过和不适用（N/A）的条目统一放入折叠区域。旧版按提交生成的裁定评论会被复用；如果评论已删除，下次验证会创建替代评论。勾选项不会被无条件信任，请只勾选你确实核实过的条目。

裁定只记录实际评估的提交，不会将批准延续到后续提交。需要重新评估时，由原审阅者编辑已有清单，或由有权限的协作者传入 `pr-number` 及其 `comment_url`（两者必须指向同一 PR）手动分发 `codeowner-signoff-verify.yml`。流程会更新同一条裁定评论。

## 使用 `/use` 在合并时复用 PR 的全绿 sweep

完整基准测试 sweep 花费昂贵的 GPU 时间，且 runner 由所有打开的 PR 共享。如果不复用，一个已批准 PR 的 sweep 将运行**两次**，一次用于 PR 验证，另一次在合并后于 `main` 上运行。reuse 路径避免了重复运行：

- 当你的 PR 拥有符合条件的全绿完整 sweep 后，授权维护者（`OWNER`/`MEMBER`/`COLLABORATOR`）在 PR 上评论 `/use <run_id>` 来指定该 Run。命令和 Run ID 必须放在同一行。
- `/reuse-sweep-run <run_id>` 仍受支持，行为完全相同。不带 ID 的 `/reuse-sweep-run` 会自动选择源 Run；不带 ID 的 `/use` 会被拒绝。
- 合并到 `main` 的运行随后会验证并摄取该 PR sweep 的 artifacts，而不是在 `main` 上重新运行整个 sweep。
- **这为每个人减少了 CI 排队时间。** 每次复用合并都会为其他 PR 释放数小时的 GPU runner 时间，因此请优先选择 reuse 路径，而不是不带它直接合并。仅有全绿 sweep 还不够。复用命令必须在评论记录中（签署验证会检查这一点），否则 `main` 会静默地重新运行完整 sweep。
- 复用不要求保留 sweep 标签。机器人会在命令被接受时添加 👍，拒绝时添加 👎，详情见 Actions 运行摘要；合并时仍会重新验证源产物。
- `utils/merge_with_reuse.sh <pr-number>` 是受支持的合并路径。它会发布命令、将分支与 `main` 同步、等待检查并 squash 合并。资格详情见 [workflows README](.github/workflows/README.md#reusing-an-approved-pr-full-sweep)。

## AMD 集群：严禁在 runner 工作区留下 root 所属文件

AMD MI355X TW 集群上的多节点基准测试通过 Slurm 提交容器化任务，这些容器通常以 **root** 身份运行。如果容器将文件（通常是 `benchmark_logs/logs/slurm_job-*`）写入 GitHub Actions runner 工作区，而任务在 teardown 执行前被**取消**，root 所属目录就会被遗留。runner 用户无法删除这些文件，导致 `actions/checkout` 失败：

```
Error: File was unable to be removed
Error: EACCES: permission denied, rmdir '.../benchmark_logs/logs/slurm_job-<id>'
```

**这会阻塞该 runner 上的所有后续任务**，直到拥有 `sudo` 权限的人在共享 `/it-share` 存储上手动删除这些文件。由于所有 AMD MI355X 扫描共享同一个 runner 池，一个遗留的 root 所属目录就会阻塞整个队列，影响所有人。

**基准测试脚本和 Slurm 容器规则：**

1. **严禁以 root 身份写入 runner 工作区。** 如果容器必须以 root 运行，请将输出写入 `_work/` 之外的临时目录（例如 `/tmp` 或专用暂存路径）。
2. **如果 root 写入不可避免**，请添加清理 trap 或 teardown 步骤，在任务退出前（包括取消时，使用 `trap cleanup EXIT`）`chown` 或 `rm` 工作区下所有 root 所属文件。
3. **测试你的 teardown 路径。** 在运行中途取消基准测试，验证工作区中不会残留 root 所属文件。

如果发现遗留的 root 所属文件阻塞了 runner，恢复流程参见 [`.claude/commands/clean-amd-mi355-runner-root-files.md`](.claude/commands/clean-amd-mi355-runner-root-files.md)：SSH 到中转主机，使用 `sudo` 扫描 `_work` 目录并删除问题文件。

## 合并之后

**PR 作者有责任确保合并后所有 GitHub Action 任务完全通过。** 很多时候失败只是偶发抖动（flake），重新运行失败的任务即可解决。[参见 GitHub 关于重新运行失败任务的文档](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs#re-running-failed-jobs-in-a-workflow)。
