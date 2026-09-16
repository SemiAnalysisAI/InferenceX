# 为 InferenceX 做贡献

<div align="center">

[English](./CONTRIBUTING.md) | **中文**

</div>

感谢你的贡献！我们欢迎 PR。本页介绍每个 PR 在合并前需要经过的审阅流程。

## PR 审阅流程

1. 打开你的 PR 并通过 PR 验证。添加 `full-sweep-fail-fast` 标签，强烈推荐使用此标签，因为变更有问题时每个矩阵最多浪费一个任务，而不是整个扇出。仅当需要任务在失败后继续运行时才使用 `full-sweep-enabled`。让基准测试 sweep 运行，并在 PR 的某个 commit 上获得全绿的完整 sweep，包括 evals。
2. 若修改的文件归属于仓库管理员及 `@SemiAnalysisAI/core` 之外的 CODEOWNER，请联系一位有资格的 [CODEOWNER](.github/CODEOWNERS) 审阅，并在批准评论中填写 **PR Review Checklist** 签署（见下文）。
3. 在 Slack 上联系核心维护者进行最终批准；若要求清单签署，请先完成签署。
4. 由授权维护者发布 `/reuse-sweep-run`（见下文），然后通过 reuse 路径合并 PR。

**性能变更日志要求：** 凡是可能影响基准测试性能的变更，以及任何配方（recipe）的新增或修改，都**必须**在 `perf-changelog.yaml` 文件的物理末尾追加一个新条目。历史条目**严禁**编辑。

## PR Review Checklist（CODEOWNER 签署）

仅当修改的文件存在仓库管理员及 `@SemiAnalysisAI/core` 之外的 CODEOWNER 时，才要求签核。归属以 PR 目标分支当前最新提交中的 CODEOWNERS 为准：先解析该分支的 SHA，再使用同一 SHA 校验并读取 CODEOWNERS，最后匹配的规则生效；重命名同时检查旧路径和新路径。归属规则不从 PR 的 Head 或其记录中可能过期的基础提交读取。同一文件有 core 团队作为 owner，不会豁免其他 owner。个人管理员必须同时拥有仓库 `permission: admin` 和 `role_name: admin`；其他团队和邮箱 owner 均要求签核。归属信息缺失或权限查询失败不能授予豁免。不涉及此类 owner 的改动会获得成功的“不适用”状态，无需启动验证器。

由一名符合条件的 CODEOWNER 审阅者在批准评论中填写最新的 [PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md)（[中文说明](docs/PR_REVIEW_CHECKLIST_zh.md)）模板。

**每个 PR 只需一名符合条件的 CODEOWNER 审阅者发布清单。** 发布前先检查是否已有清单；其他审阅者无需重复发布。需要更正条目、补充证据或重试验证时，原审阅者必须**编辑自己已有的清单评论**，不要另发一条。只有原评论被删除时才创建替代评论。

友情提醒。请**正确**遵循最新的清单模板：

- 务必从 `main` 分支上**当前**的 [docs/PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) 复制模板。清单会不断演进，使用过期副本的签署会被标记为缺项。
- 保持模板的开头语句原样不变（必须保留英文原文）：

  > As a PR reviewer and CODEOWNER, I have reviewed this and have:

  我们的 CI 验证工作流 [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml) 正是通过这句话触发的。**如果你的批准评论没有遵循清单模板，包括这句话，签署验证 CI 将完全不会触发**，你的签署也不会计入合并要求。
- 签署可以以普通会话评论、review 总结或行内 review 评论的形式发布。这三种方式都会触发验证。
- 首次 PASS 前，更新 Head、重新打开 PR 或退出草稿状态会在当前 Head 上补查最新的合格签署，找回合并冲突期间遗漏的 Review，无需重复发布清单。
- 启动 Claude 仍要求触发者具备合格的仓库写权限。如果更新来自无写权限用户或未获允许的机器人，具有写权限的协作者可发起首次验证。延续已有 PASS 无需重新验证。
- 请在 "Additional detail section" 中填写清单要求的链接（验证/评测工作流运行、对应的 [vLLM recipe](https://github.com/vllm-project/recipes) / [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new) PR，以及任何例外理由）。

签署发布后，CI 会独立复核决定合并的各项声明，包括 CODEOWNER 身份、PR 内 commit 上的全绿 sweep 与 evals、所链接的 recipe、`/reuse-sweep-run` 命令、是否使用最新清单模板、上游 [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) 镜像、没有更改模型架构的基准测试 hack，以及投机解码是否使用 chat template。随后，CI 会为整个 PR 创建或更新同一条裁定评论，并注明实际评估的 SHA。未通过的条目直接显示；已通过和不适用（N/A）的条目统一放入折叠区域。旧版按提交生成的裁定评论会被复用；如果评论已删除，下次验证会创建替代评论。勾选项不会被无条件信任，请只勾选你确实核实过的条目。

**管理员更新不会使签署失效，非管理员更新则会。** 可信工作流使用实际推送更新的 GitHub 认证用户，并要求仓库权限同时为 `permission: admin` 和 `role_name: admin`。提交中的作者姓名和邮箱不能获得豁免。若管理员更新从已覆盖的 head 开始，自动化会扩展覆盖范围，无需调用 Claude。非管理员更新（包括合并 `main`）需要重新验证签署；随后由管理员推送也不能消除这一要求。

裁定评论记录实际验证的提交，以及后续管理员更新所覆盖的 head。旧的 `codeowner-signoff-verified` 终身有效标签会被移除，不再作为接受依据。可信的旧版裁定必须包含已验证 SHA；贡献者撰写的评论、缺失的来源信息或无法查询的权限都不能扩展覆盖范围。若无法将某次更新连接到已记录的覆盖 head，就需要验证。删除裁定评论也会移除该证明。验证新的非管理员改动时，应由原审阅者编辑已有检查清单，或由有权限的协作者传入 `pr-number` 及其 `comment_url`（两者必须指向同一 PR）手动分发 `codeowner-signoff-verify.yml`。流程更新同一条裁定评论，重新评估被拒绝时会撤销接受状态。

## 使用 `/reuse-sweep-run` 在合并时复用 PR 的全绿 sweep

完整基准测试 sweep 花费昂贵的 GPU 时间，且 runner 由所有打开的 PR 共享。如果不复用，一个已批准 PR 的 sweep 将运行**两次**，一次用于 PR 验证，另一次在合并后于 `main` 上运行。reuse 路径避免了重复运行：

- 当你的 PR 拥有符合条件的全绿完整 sweep 后，授权维护者（`OWNER`/`MEMBER`/`COLLABORATOR`）在 PR 上评论 `/reuse-sweep-run`（也可固定某次运行：`/reuse-sweep-run <run_id>`）。
- 合并到 `main` 的运行随后会验证并摄取该 PR sweep 的 artifacts，而不是在 `main` 上重新运行整个 sweep。
- **这为每个人减少了 CI 排队时间。** 每次复用合并都会为其他 PR 释放数小时的 GPU runner 时间，因此请优先选择 reuse 路径，而不是不带它直接合并。仅有全绿 sweep 还不够。`/reuse-sweep-run` 评论必须在记录中（签署验证会检查这一点），否则 `main` 会静默地重新运行完整 sweep。
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
