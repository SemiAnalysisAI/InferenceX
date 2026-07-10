# 推理引擎补丁豁免（Waiver）

<div align="center">

[English](./README.md) | **中文**

</div>

InferenceX 的基准测试对象必须是社区能够实际拉取并运行的软件。**PR 不得对推理引擎或 serving 技术栈打补丁。** 锁定的镜像必须原样运行 —— 禁止任何形式的修改，包括（但不限于）：

- `.patch` 文件、`git apply` 或 `patch` 调用
- **内嵌在基准测试脚本中的行内补丁** —— 例如 `benchmarks/**` 下的 `.sh` 脚本中，通过 `python3 - <<EOF` / `sed` / `cat > file` 等 heredoc 在启动 `vllm serve` / SGLang 之前改写已安装的引擎源码
- 使用 `sed`/`awk` 等就地编辑已安装的引擎源码（例如 `site-packages` 下的任何内容）
- 通过环境变量钩子、sitecustomize 或拷贝文件注入的 Python monkey-patch
- 覆盖或遮蔽容器镜像内的文件
- 在锁定镜像之上 `pip install` fork 或重新构建的引擎 wheel
- 从修改过的源码树重新构建引擎镜像

安装基准测试工具链及其客户端依赖（如 aiperf、评测工具）没有问题 —— 本规则针对的是产出性能数据的 **serving** 技术栈。

**唯一的例外：该补丁已由本文件夹中一份填写完整的 `WAIVER.md` 豁免覆盖。**

## 什么情况可以获得豁免？

豁免是临时的过渡桥梁，不是漏洞。合理的场景例如：

- 上游修复已合并但尚未包含在任何已发布镜像中，且没有该修复基准测试完全无法运行
- 新硬件或新模型架构的 bring-up，且上游支持正在积极推进落地

以下情况**不**可豁免：上游尚未接受的性能优化补丁 —— 如果一个补丁让基准测试更快，社区就无法用已发布的镜像复现该数据，这违背了 InferenceX 的初衷。

## 如何提交豁免

1. 在引入补丁的**同一个 PR** 中，创建 `docs/waiver/<slug>/WAIVER.md`，其中 `<slug>` 标识该补丁（例如 `2026-07-glm5-gb200-sglang-moe-kernel`）。
2. 复制下方模板并**逐项**填写全部字段（使用英文）。
3. 在 CODEOWNER 签署的 "Additional detail section" 中给出该豁免的链接 —— 签署验证器（[codeowner-signoff-verify.yml](../../.github/workflows/codeowner-signoff-verify.yml)）会独立核查 PR 中的任何补丁行为是否已被豁免覆盖。
4. 核心维护者必须在 PR 评论中明确批准该豁免；并在豁免文件中链接该评论。
5. 上游修复发布后，在同一个 PR 中移除补丁并删除对应的豁免文件夹。

## 模板（请保持英文原文）

```markdown
# WAIVER: <one-line description of the patch>

- **PR:** <link to the InferenceX PR that introduces the patch>
- **Date filed:** <YYYY-MM-DD>
- **Filed by:** @<github-username>
- **Configs affected:** <model / precision / SKU / framework, and the master-config entries touched>
- **What is patched:** <exact files and where the patch is applied (script line, Dockerfile step); include the patch contents or a link to it>
- **Why the patch is required:** <why the unmodified upstream image cannot run this benchmark>
- **Upstream status:** <link to the upstream PR/issue that removes the need for this patch>
- **Removal plan:** <which upstream release or condition lets us drop the patch, and the expected date>
- **Performance impact:** <does the patch change performance vs. the unpatched upstream image? link the evals run>
- **Core maintainer approval:** @<maintainer> — <link to the approving PR comment>
```

## 当前生效的豁免

无。
