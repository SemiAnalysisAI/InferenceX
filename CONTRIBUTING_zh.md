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

## Draft 模型精度

规则很简单：**按发布时的默认方式运行 draft。** 投机解码提交必须使用随所服务 checkpoint 一同发布的 draft head 或 draft 模型，保持其发布时的精度，并采用锁定上游框架对该 checkpoint 的默认加载处理。此规则适用于内嵌的 MTP/NextN/EAGLE draft head 及独立 draft 模型（包括 DSpark），覆盖所有硬件厂商和框架。

"按发布时的默认方式"既不是指 BF16，也不是指"未量化的原始发布版本"。它指的是默认值：checkpoint 作者为该 checkpoint 发布的内容，以及锁定上游镜像开箱即用时对它的处理方式。基线是：同一 checkpoint 由同一锁定镜像加载，且不带任何来自提交方的 draft 相关设置。提交的 draft 实际运行精度与该基线一致即为合规；使 draft 比该基线更"便宜"即为违规。

允许（这是基线本身，不是例外）：

- 内嵌于所服务 checkpoint 或随其一同发布的 draft head / draft 权重，保持其存储精度。若所服务的 FP8 checkpoint 内嵌 FP8 MTP head，则 FP8 head 就是正确的 draft。示例：Qwen FP8 checkpoint 以 FP8 存储其 MTP 权重，因此内嵌的 FP8 head 是合规的；换用 BF16 发布版本的 head，或强制添加 unquantized-draft 覆盖设置，才是违规，而不是"修复"。
- 锁定上游框架对该 checkpoint 默认执行的加载时处理，包括 dtype 转换。示例：锁定的 SGLang 镜像将 DeepSeek V4.1 DSpark 以 FP8 存储的 `wo_a` 投影加载为 BF16，与 DeepSeek 参考实现一致。该转换就是发布路径；用本地补丁将其保持为 FP8 会改变 draft 计算，属于违规。
- 框架对该 checkpoint 的默认 draft KV cache dtype，以及对 target 和 draft 一致应用的上游支持的 KV cache dtype（例如从 FP8 target 继承的 FP8 draft KV）。
- 锁定镜像中自带、且被在意准确性的客户在生产中实际使用的真实上游优化：上游为该模型默认启用的融合或低精度 kernel、调度优化，以及任何在不降低 draft 权重或激活精度的前提下让同样的 draft 计算跑得更快的手段。

禁止（提交方对 draft 精度的 hack）：

- 今后明确禁止启用 `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE`（`=1` 或锁定实现认可的其他启用值），包括从环境、launcher、容器或镜像默认值继承的启用设置。上述基线允许项不能豁免此 flag。审阅者必须核实它在实际配方中已禁用；只有锁定实现确认未启用时，未设置或 `=0` 才可接受。历史运行不能作为当前待审提交的先例或例外，此要求也适用于仅更新镜像或重新启用配方的提交。本规则不追溯否定生效前的运行。
- 通过 flag、环境变量、配置文件或转换步骤，将 draft 权重、激活或计算量化到发布精度以下，无论在线还是离线。包括对 BF16 MTP head 使用 `--speculative-draft-model-quantization quark_mxfp4`、`SGLANG_GLM_NEXTN_MOE_PTPC=1`，以及 `exclude_layer` 模式未覆盖整个 draft head 的 ATOM `--online_quant_config`。
- 针对 draft 的 dtype 或 KV cache dtype 覆盖设置，使其精度低于框架对该 checkpoint 的默认值。
- 替换为经过精度转换或不同量化方式的 draft checkpoint，或使用与所服务 target 不同发布版本的 draft head。
- 给锁定镜像打补丁，使其以不同于默认的精度加载或计算 draft，无论升高还是降低。引擎补丁规则本已禁止此行为；waiver 不能豁免 draft 精度要求。
- 以 checkpoint 作者未发布的方式减少 draft FLOPs，例如裁剪 draft 层或 expert。

Target/verifier 模型仍可在满足现有 eval 要求的前提下量化。只要 target 的量化不会同时把 draft 量化到发布精度以下即可；请核查继承的量化设置和 `exclude_layer` 覆盖范围，而不是假定。

匹配上游配方、通过 evals、声称接受长度（AL）未变，或重新测得的 AL 曲线，都不能豁免降低 draft 精度的提交。AgentX 的合成接受在任何方向上都不构成证据。draft 精度的改变会把准确性损失转移到接受率上，而 target 模型的 evals 无法测量这一点。

审阅者必须对照发布基线核实 draft 的实际运行精度，不能只看启动参数。检查 checkpoint 元数据与量化排除项、环境变量、锁定镜像中的框架默认行为，以及从 target 模型继承的量化设置。不得仅凭 target checkpoint 的名称或精度标签推断 draft 精度。

涉及投机解码的改动，CODEOWNER 必须在 Additional detail section 中注明 draft checkpoint 及其 revision（或内嵌 head）、其发布精度、锁定上游镜像对其的默认处理方式，以及实际运行精度，以便审阅者确认后两者一致。无法核实时，该条目不满足要求。参见[审阅清单](docs/PR_REVIEW_CHECKLIST_zh.md)及[验证器检查 13](.github/codeowner-signoff-verify-prompt.md#check-13--draft-runs-as-shipped)。

此要求与 [MLPerf Inference Rules 附录 C：Speculative Decoding](https://github.com/mlcommons/inference_policies/blob/ff7edba545fded369e7e7e3d5a2f0bab4a95eece/inference_rules.adoc#appendix-c-speculative-decoding) 的原则一致：参考 MTP head 使用提供时的相同精度（"at the same precision as provided"），并禁止参考 head 权重量化及其他人为操纵接受率的行为。InferenceX 不采用该版本针对特定量化边缘工作负载的例外、其允许模型列表，或其投机解码配置与接受率测试方法。

## PR Review Checklist（CODEOWNER 签署）

CODEOWNER 自动验证目前仅供审阅参考。工作流会核验新提交及已编辑的清单，并为每个签署资源关联一条裁定评论，不再发布提交状态。GitHub 单独设置的 Core 团队和 CODEOWNER 批准要求仍然有效，除非有权限的维护者使用绕过权限。

仅当修改的文件存在仓库管理员及 `@SemiAnalysisAI/core` 之外的 CODEOWNER 时，才要求签核。归属以 PR 目标分支当前最新提交中的 CODEOWNERS 为准：先解析该分支的 SHA，再使用同一 SHA 校验并读取 CODEOWNERS，最后匹配的规则生效；重命名同时检查旧路径和新路径。归属规则不从 PR 的 Head 或其记录中可能过期的基础提交读取。同一文件有 core 团队作为 owner，不会豁免其他 owner。个人管理员必须同时拥有仓库 `permission: admin` 和 `role_name: admin`；其他团队和邮箱 owner 均要求签核。归属信息缺失或权限查询失败不能授予豁免。不涉及此类 owner 的改动会跳过验证。

由一名符合条件的 CODEOWNER 审阅者在批准评论中填写最新的 [PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md)（[中文说明](docs/PR_REVIEW_CHECKLIST_zh.md)）模板。

**每个 PR 只需一名符合条件的 CODEOWNER 审阅者发布清单。** 发布前先检查是否已有清单；其他审阅者无需重复发布。需要更正条目或补充证据时，原审阅者必须**编辑自己已有的清单评论**，不要另发一条。只有原评论被删除时才创建替代评论。

友情提醒。请**正确**遵循最新的清单模板：

- 务必从 `main` 分支上**当前**的 [docs/PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) 复制模板。清单会不断演进，使用过期副本的签署会被标记为缺项。
- 保持模板的开头语句原样不变（必须保留英文原文）：

  > As a PR reviewer and CODEOWNER, I have reviewed this and have:

  我们的 CI 验证工作流 [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml) 正是通过这句话触发的。**如果批准评论缺少这句话，工作流就不会核验该清单。**
- 签署可以以普通会话评论、review 总结或行内 review 评论的形式发布。这三种方式都会触发验证。
- 请在 PR 处于打开且非草稿状态时提交新清单。编辑该清单会再次触发验证；推送、重新打开或退出草稿状态不会触发验证。如果合并冲突期间遗漏了 Review 事件，请在解决冲突后手动分发工作流来重试。
- 启动 Claude 要求触发者为具有合格仓库写权限的人类用户。
- 请在 "Additional detail section" 中填写清单要求的链接（验证/评测工作流运行、对应的 [vLLM recipe](https://github.com/vllm-project/recipes) / [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new) PR，以及任何例外理由）。

签署发布后，CI 会独立复核审阅清单中的各项声明，包括 CODEOWNER 身份、PR 内 commit 上的全绿 sweep 与 evals、所链接的 recipe、复用命令、是否使用最新清单模板、上游 [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) 镜像、没有更改模型架构的基准测试 hack、投机解码是否使用 chat template，以及 draft 模型和 draft head 的权重与精度是否保持不变。CI 会为该签署资源创建一条裁定评论，并注明实际评估的 SHA。编辑同一清单时只更新与其关联的裁定。替代清单或新增清单会获得独立裁定；与旧签署关联的裁定保持不变。未通过的条目直接显示；已通过和不适用（N/A）的条目统一放入折叠区域。勾选项不会被无条件信任，请只勾选你确实核实过的条目。

裁定只记录实际评估的提交，不会将批准延续到后续提交。需要重新评估时，先按需更正已有清单，再由有权限的协作者传入 `pr-number` 及其 `comment_url`（两者必须指向同一 PR）手动分发 `codeowner-signoff-verify.yml`。手动重新评估会更新与该签署资源关联的裁定。

## 使用 `/use` 在合并时复用 PR 的全绿 sweep

完整基准测试 sweep 花费昂贵的 GPU 时间，且 runner 由所有打开的 PR 共享。如果不复用，一个已批准 PR 的 sweep 将运行**两次**，一次用于 PR 验证，另一次在合并后于 `main` 上运行。reuse 路径避免了重复运行：

- 当你的 PR 拥有符合条件的全绿完整 sweep 后，授权维护者（`OWNER`/`MEMBER`/`COLLABORATOR`）在 PR 上评论 `/use <run_id>` 来指定该 Run。命令和 Run ID 必须放在同一行。
- `/reuse-sweep-run <run_id>` 仍受支持，行为完全相同。不带 ID 的 `/reuse-sweep-run` 会自动选择源 Run；不带 ID 的 `/use` 会被拒绝。
- 合并到 `main` 的运行随后会验证并摄取该 PR sweep 的 artifacts，而不是在 `main` 上重新运行整个 sweep。
- **这为每个人减少了 CI 排队时间。** 每次复用合并都会为其他 PR 释放数小时的 GPU runner 时间，因此请优先选择 reuse 路径，而不是不带它直接合并。仅有全绿 sweep 还不够。复用命令必须在评论记录中（签署验证会检查这一点），否则 `main` 会静默地重新运行完整 sweep。
- 复用不要求保留 sweep 标签。机器人会在命令被接受时添加 👍，拒绝时添加 👎，详情见 Actions 运行摘要；合并时仍会重新验证源产物。
- 缺少授权维护者发布的复用命令时，Check 4 会给出 **WARN**，不会因此拒绝签署。警告会在签署裁定中保持展开；要实际复用产物，仍需先发布有效的授权命令。
- `uv run --extra workflows python -m infx.workflows.merge_with_reuse <pr-number>` 是受支持的合并路径。它会发布命令、将分支与 `main` 同步、等待检查并 squash 合并。资格详情见 [workflows README](.github/workflows/README.md#reusing-an-approved-pr-full-sweep)。

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
