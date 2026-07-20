# PR Review Checklist

<div align="center">

**English** | [中文](./PR_REVIEW_CHECKLIST_zh.md)

</div>

When [CODEOWNER](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/CODEOWNERS) from the respective hardware AI chip company is reviewing & approving their respective PRs, please fill in the following form in your approval comment before pinging an core maintainer for final approval

We welcome InferenceX partners & the community to submit PRs for reasonable additions to this checklist — or deletions — that follow the principles of InferenceX, and the general principle that deleting a guideline should be as easy a process as adding a new one.

We also welcome InferenceX partners & the ML community to improve [codeowner-signoff-verify.yml](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml) — the CI bot that independently verifies these sign-offs — and make it more rigorous too.

## Template
```
As a PR reviewer and CODEOWNER, I have reviewed this and have:
- [ ] Verified that as of the moment of typing this, this is the latest version of [PR_REVIEW_CHECKLIST.md](https://github.com/SemiAnalysisAI/InferenceX/blob/main/docs/PR_REVIEW_CHECKLIST.md)
- [ ] Verified that the general code quality meets the InferenceX standard and does not make the code quality any worse.
- [ ] Verified that this PR has passed PR validation. Please link to GitHub Action workflow that shows this.
- [ ] Verified that this PR passes evals.  Please link to GitHub Action workflow that shows this.
- [ ] Verified that speculative decoding PRs uses chat templates to align the AL distribution to real world
- [ ] For agentic workloads: verified that speculative-decoding configs (EAGLE / MTP / draft models) run with simulated synthetic acceptance, with the acceptance-length value taken from the committed golden AL curve in [golden_al_distribution/](https://github.com/SemiAnalysisAI/InferenceX/tree/main/golden_al_distribution) for that model, thinking mode, and draft length. A submission may choose any supported draft length, but it may not substitute a different acceptance target.
- [ ] Verified that the model architecture isn't changed with benchmark hacks like using --hf-overrides to skipping indexer for every x layers on models that don't natively support this. As a general rule, we won't accept optimizations that reduces the number of model architecture FLOPs. Anything that makes that same computation run faster is fair game; FLOPs at lower precisions is fine, given that the config passes private evals. As an general north star princple, we should only use optimizations which is used in production by customers that care about accuracy
- [ ] If an company claims that they support vLLM/SGLang as first class LLM inference engines on their hardware, I have verified that the respective vLLM submission made using upstream https://hub.docker.com/u/vllm docker repo, upstream SGLang https://hub.docker.com/u/lmsysorg docker repo. The only exceptions are for new hardware, such as MI455X UALoE72, Vera Rubin NVL72, Rubin NVL8, etc., and for new model architectures where there is an actual reason why vLLM/SGLang does not fundamentally support them yet as supported by vLLM/SGLang community maintainers
- [ ] If an company claims that they support vLLM/SGLang as first class upstream in-tree LLM inference engines on their hardware, I have have verified that the respective vLLM/SGLang submission has been made before additional frameworks (TRT-LLM, ATOM, etc.). The only exceptions are for new hardware, such as MI455X UALoE72, Vera Rubin NVL72, Rubin NVL8, etc., and for new model architectures where there is an actual reason why vLLM/SGLang does not fundamentally support them yet.
- [ ] Verified that every single-node vLLM/SGLang recipe in this PR is documented in the official [vLLM recipes](https://recipes.vllm.ai/) and/or the [SGLang cookbook](https://docs.sglang.io/cookbook/intro):
  - [ ] I linked the corresponding upstream PR in the [vLLM recipe repo](https://github.com/vllm-project/recipes) or [SGLang repo](https://github.com/sgl-project/sglang/tree/main/docs_new) and verified that it is **MERGED** before this InferenceX PR merges. An opened, draft, or closed-without-merge upstream PR does not satisfy this requirement. If the matching recipe was already published, I linked the published recipe/cookbook page in the additional detail section below.
- [ ] Verified that this PR does not patch the inference engine or serving stack — the pinned image must run as shipped. This covers .patch files / git apply / patch, inline patches embedded in benchmark scripts (e.g. a python3/sed heredoc that rewrites installed engine sources before serving), in-place edits of site-packages, monkey-patching, overwriting container files, and installing forked/rebuilt engine wheels on top of the pinned image. The only exception is a patch covered by a filled-out waiver at [docs/waiver/](https://github.com/SemiAnalysisAI/InferenceX/tree/main/docs/waiver)`<PR_NUMBER>.md` — named after the PR that introduces the patch and filed in that same PR, stating what is patched, why the unmodified upstream image cannot run this benchmark, the upstream PR/issue link, and the removal plan — which I have linked below in the additional detail section.
- [ ] If any of the above criteria cannot reasonably be satisfied, I have provided additional reasoning below.

### Additional detail section:
- insert any additional info here

Signed: `FILL_IN_GITHUB_USERNAME`
```

## Example

<img width="667" height="701" alt="image" src="https://github.com/user-attachments/assets/0c832d48-c81b-4bdb-bb53-43f39ff18b9b" />


<img width="569" height="632" alt="image" src="https://github.com/user-attachments/assets/491d9763-ab09-4734-b0f1-39eefe1ab5c4" />

