# 为 InferenceX 做贡献

<div align="center">

[English](./CONTRIBUTING.md) | **中文**

</div>

感谢你的贡献！我们欢迎 PR。本页介绍每个 PR 在合并前需要经过的审阅流程。

## PR 审阅流程

1. 打开你的 PR 并通过 PR 验证：添加 `full-sweep-enabled`（或 `full-sweep-fail-fast`）标签以运行基准测试 sweep，并在 PR 的某个 commit 上获得全绿的完整 sweep（包括 evals）。
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

签署发布后，CI 会独立复核决定合并的各项声明 — CODEOWNER 身份、PR 内 commit 上的全绿 sweep + evals、所链接的 recipe、`/reuse-sweep-run` 命令、是否使用最新清单模板、上游 [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) 镜像、没有更改模型架构的基准测试 hack，以及投机解码是否使用 chat template — 并在 PR 上发布裁定评论。勾选项不会被无条件信任，请只勾选你确实核实过的条目。

## `/reuse-sweep-run` — 在合并时复用 PR 的全绿 sweep

完整基准测试 sweep 花费昂贵的 GPU 时间，且 runner 由所有打开的 PR 共享。如果不复用，一个已批准 PR 的 sweep 将运行**两次** — PR 验证一次，合并后在 `main` 上再一次。reuse 路径避免了这一点：

- 当你的 PR 拥有符合条件的全绿完整 sweep 后，授权维护者（`OWNER`/`MEMBER`/`COLLABORATOR`）在 PR 上评论 `/reuse-sweep-run`（也可固定某次运行：`/reuse-sweep-run <run_id>`）。
- 合并到 `main` 的运行随后会验证并摄取该 PR sweep 的 artifacts，而不是在 `main` 上重新运行整个 sweep。
- **这为每个人减少了 CI 排队时间** — 每次复用合并都会为其他 PR 释放数小时的 GPU runner 时间，因此请优先选择 reuse 路径，而不是不带它直接合并。仅有全绿 sweep 还不够：`/reuse-sweep-run` 评论必须在记录中（签署验证会检查这一点），否则 `main` 会静默地重新运行完整 sweep。
- `utils/merge_with_reuse.sh <pr-number>` 是受支持的合并路径；它会发布命令、将分支与 `main` 同步、等待检查并 squash 合并。资格详情见 [workflows README](.github/workflows/README.md#reusing-an-approved-pr-full-sweep)。

## 合并之后

**PR 作者有责任确保合并后所有 GitHub Action 任务完全通过。** 很多时候失败只是偶发抖动（flake），重新运行失败的任务即可解决。[参见 GitHub 关于重新运行失败任务的文档](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs#re-running-failed-jobs-in-a-workflow)。
