# PR 审阅清单

<div align="center">

[English](./PR_REVIEW_CHECKLIST.md) | **中文**

</div>

当相应硬件 AI 芯片公司的 [CODEOWNER](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/CODEOWNERS) 审阅并批准其相关 PR 时，请在批准评论中填写以下表单，然后再联系核心维护者进行最终批准。

我们欢迎 InferenceX 合作伙伴与社区提交 PR，对本清单进行符合 InferenceX 原则的合理增补或删减 — 总体原则是：删除一条准则的流程应当与新增一条准则同样容易。

我们同样欢迎 InferenceX 合作伙伴与机器学习社区改进 [codeowner-signoff-verify.yml](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml)（独立复核这些签署的 CI 机器人），使其更加严谨。

> **重要：模板请保持英文原文，原样复制粘贴，不要翻译。** CI 签署验证工作流 [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml) 通过开头语句 "As a PR reviewer and CODEOWNER, I have reviewed this and have" 触发；模板被改写或翻译后，签署验证 CI 将不会触发。

## 模板（请复制英文原文）
```
As a PR reviewer and CODEOWNER, I have reviewed this and have:
- [ ] Verified that as of the moment of typing this, this is the latest version of [PR_REVIEW_CHECKLIST.md](https://github.com/SemiAnalysisAI/InferenceX/edit/main/docs/PR_REVIEW_CHECKLIST.md)
- [ ] Verified that the general code quality meets the InferenceX standard and does not make the code quality any worse.
- [ ] Verified that this PR has passed PR validation. Please link to GitHub Action workflow that shows this.
- [ ] Verified that this PR passes evals.  Please link to GitHub Action workflow that shows this.
- [ ] Verified that speculative decoding PRs uses chat templates to align the AL distribution to real world
- [ ] For agentic workloads: verified that speculative-decoding configs (EAGLE / MTP / draft models) run with simulated synthetic acceptance, with the acceptance-length value taken from the committed golden AL curve in [golden_al_distribution/](https://github.com/SemiAnalysisAI/InferenceX/tree/main/golden_al_distribution) for that model, thinking mode, and draft length. A submission may choose any supported draft length, but it may not substitute a different acceptance target.
- [ ] Verified that the model architecture isn't changed with benchmark hacks like using --hf-overrides to skipping indexer for every x layers on models that don't natively support this. As a general rule, we won't accept optimizations that reduces the number of model architecture FLOPs. Anything that makes that same computation run faster is fair game; FLOPs at lower precisions is fine, given that the config passes private evals. As an general north star princple, we should only use optimizations which is used in production by customers that care about accuracy
- [ ] If an company claims that they support vLLM/SGLang as first class LLM inference engines on their hardware, I have verified that the respective vLLM submission made using upstream https://hub.docker.com/u/vllm docker repo, upstream SGLang https://hub.docker.com/u/lmsysorg docker repo. The only exceptions are for new hardware, such as MI455X UALoE72, Vera Rubin NVL72, Rubin NVL8, etc., and for new model architectures where there is an actual reason why vLLM/SGLang does not fundamentally support them yet as supported by vLLM/SGLang community maintainers
- [ ] If an company claims that they support vLLM/SGLang as first class upstream in-tree LLM inference engines on their hardware, I have have verified that the respective vLLM/SGLang submission has been made before additional frameworks (TRT-LLM, ATOM, etc.). The only exceptions are for new hardware, such as MI455X UALoE72, Vera Rubin NVL72, Rubin NVL8, etc., and for new model architectures where there is an actual reason why vLLM/SGLang does not fundamentally support them yet.
- [ ] Verified that the single-node recipes are similar to the official [vLLM recipes](https://recipes.vllm.ai/) and/or the[SGLang cookbook](https://docs.sglang.io/cookbook/intro):
  - [ ] If they are not, I have verified that a PR has been opened in [vLLM recipe repo](https://github.com/vllm-project/recipes) or [SGLang repo](https://github.com/sgl-project/sglang/tree/main/docs_new) and linked it below in the additional detail section:
- [ ] Verified that this PR does not patch the inference engine or serving stack — the pinned image must run as shipped. This covers .patch files / git apply / patch, inline patches embedded in benchmark scripts (e.g. a python3/sed heredoc that rewrites installed engine sources before serving), in-place edits of site-packages, monkey-patching, overwriting container files, and installing forked/rebuilt engine wheels on top of the pinned image. The only exception is a patch covered by a filled-out waiver at [docs/waiver/](https://github.com/SemiAnalysisAI/InferenceX/tree/main/docs/waiver)`<PR_NUMBER>.md` — named after the PR that introduces the patch and filed in that same PR, stating what is patched, why the unmodified upstream image cannot run this benchmark, the upstream PR/issue link, and the removal plan — which I have linked below in the additional detail section.
- [ ] If any of the above criteria cannot reasonably be satisfied, I have provided additional reasoning below.

### Additional detail section:
- insert any additional info here

Signed: `FILL_IN_GITHUB_USERNAME`
```
## 各条目中文对照说明

1. 已确认在填写此清单时，使用的是 [PR_REVIEW_CHECKLIST.md](https://github.com/SemiAnalysisAI/InferenceX/blob/main/docs/PR_REVIEW_CHECKLIST.md) 的最新版本。
2. 已确认整体代码质量达到 InferenceX 标准，且不会使代码质量变差。
3. 已确认该 PR 通过了 PR 验证，并附上能证明这一点的 GitHub Action 工作流链接。
4. 已确认该 PR 通过了 evals（准确性评测），并附上能证明这一点的 GitHub Action 工作流链接。
5. 已确认投机解码（speculative decoding）PR 使用 chat template，使接受长度（AL）分布与真实场景对齐。
6. 对 agentic 工作负载：已确认投机解码配置（EAGLE / MTP / draft 模型）启用了模拟合成接受（simulated synthetic acceptance），且接受长度（AL）取值来自 [golden_al_distribution/](https://github.com/SemiAnalysisAI/InferenceX/tree/main/golden_al_distribution) 中该模型、thinking 模式与 draft 长度对应的已提交黄金 AL 曲线。提交可选择任意受支持的 draft 长度，但不得替换为其他接受目标。
7. 已确认模型架构未被基准测试 hack 更改 — 例如在不原生支持的模型上使用 `--hf-overrides` 每 x 层跳过 indexer。一般规则：不接受减少模型架构 FLOPs 的优化；让同样的计算跑得更快没有问题；更低精度的 FLOPs 也可以，前提是该配置通过私有 evals。北极星原则：只使用在意准确性的客户在生产中实际使用的优化。
8. 如果公司声称在其硬件上将 vLLM/SGLang 作为一等 LLM 推理引擎支持，已确认相应 vLLM 提交使用上游 [vLLM docker 仓库](https://hub.docker.com/u/vllm)、SGLang 提交使用上游 [lmsysorg docker 仓库](https://hub.docker.com/u/lmsysorg)。唯一例外：新硬件（如 MI455X UALoE72、Vera Rubin NVL72、Rubin NVL8 等），以及经 vLLM/SGLang 社区维护者确认上游尚未从根本上支持的新模型架构。
9. 如果公司声称在其硬件上将 vLLM/SGLang 作为一等上游 in-tree LLM 推理引擎支持，已确认相应 vLLM/SGLang 提交先于其他框架（TRT-LLM、ATOM 等）完成。例外情形同上。
10. 已确认单节点 recipe 与官方 [vLLM recipes](https://recipes.vllm.ai/) 和/或 [SGLang cookbook](https://docs.sglang.io/cookbook/intro) 相似；如果不相似，已确认在 [vLLM recipe 仓库](https://github.com/vllm-project/recipes)或 [SGLang 仓库](https://github.com/sgl-project/sglang/tree/main/docs_new)开了 PR，并在下方 Additional detail section 中给出链接。
11. 已确认该 PR 未对推理引擎或 serving 技术栈打补丁 —— 锁定的镜像必须原样运行。涵盖：.patch 文件 / `git apply` / `patch`、内嵌在基准测试脚本中的行内补丁（例如在启动服务前用 python3/sed heredoc 改写已安装的引擎源码）、就地编辑 site-packages、monkey-patch、覆盖容器文件、以及在锁定镜像之上安装 fork 或重新构建的引擎 wheel。唯一例外：该补丁已由 [docs/waiver/](https://github.com/SemiAnalysisAI/InferenceX/tree/main/docs/waiver)`<PR_NUMBER>.md`（以引入补丁的 PR 编号命名，并在同一 PR 中提交）中填写完整的豁免覆盖 —— 写明补丁内容、为何未修改的上游镜像无法运行该基准测试、上游 PR/issue 链接及移除计划 —— 并已在下方 Additional detail section 中给出链接。
12. 如果上述任何条目无法合理满足，已在下方提供额外说明。

## 示例

<img width="667" height="701" alt="image" src="https://github.com/user-attachments/assets/0c832d48-c81b-4bdb-bb53-43f39ff18b9b" />


<img width="569" height="632" alt="image" src="https://github.com/user-attachments/assets/491d9763-ab09-4734-b0f1-39eefe1ab5c4" />
