# 为 InferenceX 做贡献

<div align="center">

[English](./CONTRIBUTING.md) | **中文**

</div>

感谢你的贡献！我们欢迎 PR。本页介绍每个 PR 在合并前需要经过的审阅流程。

## PR 审阅流程

1. 打开你的 PR 并通过 PR 验证：添加 `full-sweep-fail-fast` 标签（强烈推荐 — 变更有问题时每个矩阵最多浪费一个任务，而不是整个扇出；仅当需要任务在失败后继续运行时才使用 `full-sweep-enabled`）以运行基准测试 sweep，并在 PR 的某个 commit 上获得全绿的完整 sweep（包括 evals）。
2. 向你所在公司的 [CODEOWNER](.github/CODEOWNERS) 请求审阅。
3. CODEOWNER 审阅后在批准评论中填写 **PR Review Checklist** 签署（见下文）。
4. 只有在清单签署发布之后，才应在 Slack 上联系核心维护者进行最终批准。
5. 由授权维护者发布 `/reuse-sweep-run`（见下文），然后通过 reuse 路径合并 PR。

## PR Review Checklist（CODEOWNER 签署）

CODEOWNER 批准 PR 时，必须在批准评论中填写最新的 [PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md)（[中文说明](docs/PR_REVIEW_CHECKLIST_zh.md)）模板。

友情提醒 — 请**正确**遵循最新的清单模板：

- 务必从 `main` 分支上**当前**的 [docs/PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) 复制模板。清单会不断演进；使用过期副本的签署会被标记为缺项。
- 保持模板的开头语句原样不变（必须保留英文原文）：

  > As a PR reviewer and CODEOWNER, I have reviewed this and have:

  我们的 CI 验证工作流 [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml) 正是通过这句话触发的。**如果你的批准评论没有遵循清单模板 — 包括这句话 — 签署验证 CI 将完全不会触发**，你的签署也不会计入合并要求。
- 签署可以以普通会话评论、review 总结或行内 review 评论的形式发布 — 三种方式都会触发验证。
- 请在 "Additional detail section" 中填写清单要求的链接（验证/评测工作流运行、对应的 [vLLM recipe](https://github.com/vllm-project/recipes) / [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new) PR，以及任何例外理由）。

签署发布后，CI 会立即在当前 PR head 上创建必需的 `codeowner-signoff-verify` Check Run，并独立复核决定合并的各项声明 — CODEOWNER 身份、PR 内 commit 上的全绿 sweep + evals、所链接的 recipe、`/reuse-sweep-run` 命令、是否使用最新清单模板、上游 [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) 镜像、没有更改模型架构的基准测试 hack，以及投机解码是否使用 chat template。详细结果会显示在 Check Run 中，不再自动发布 PASS/FAIL 评论。如果 PR head 发生变化，仅当 PR 的实际 diff 未改变时，CI 才会自动复核已有签署；若内容发生变化，则必须由 CODEOWNER 重新签署。勾选项不会被无条件信任，请只勾选你确实核实过的条目。

## `/reuse-sweep-run` — 在合并时复用 PR 的全绿 sweep

完整基准测试 sweep 花费昂贵的 GPU 时间，且 runner 由所有打开的 PR 共享。如果不复用，一个已批准 PR 的 sweep 将运行**两次** — PR 验证一次，合并后在 `main` 上再一次。reuse 路径避免了这一点：

- 当你的 PR 拥有符合条件的全绿完整 sweep 后，授权维护者（`OWNER`/`MEMBER`/`COLLABORATOR`）在 PR 上评论 `/reuse-sweep-run`（也可固定某次运行：`/reuse-sweep-run <run_id>`）。
- 合并到 `main` 的运行随后会验证并摄取该 PR sweep 的 artifacts，而不是在 `main` 上重新运行整个 sweep。
- **这为每个人减少了 CI 排队时间** — 每次复用合并都会为其他 PR 释放数小时的 GPU runner 时间，因此请优先选择 reuse 路径，而不是不带它直接合并。仅有全绿 sweep 还不够：`/reuse-sweep-run` 评论必须在记录中（签署验证会检查这一点），否则 `main` 会静默地重新运行完整 sweep。
- `utils/merge_with_reuse.sh <pr-number>` 是受支持的合并路径；它会发布命令、将分支与 `main` 同步、等待检查并 squash 合并。资格详情见 [workflows README](.github/workflows/README.md#reusing-an-approved-pr-full-sweep)。

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
