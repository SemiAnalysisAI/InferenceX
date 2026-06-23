# PR Review Checklist

When [CODEOWNER](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/CODEOWNERS) from the respective hardware AI chip company is reviewing & approving their respective PRs, please fill in the following form in your approval comment before pinging an core maintainer for final approval

## Template

As a PR reviewer and CODEOWNER, I have reviewed this and have:

- [ ] Verified that the general code quality meets the InferenceX standard and does not make the code quality any worse.
- [ ] Verified that this PR has passed PR validation.
- [ ] Verified that this PR passes evals.
- [ ] Verified that speculative decoding PRs uses chat templates to align the AL distribution to real world 
- [ ] If an company claims that they support vLLM/SGLang as first class LLM inference engines on their hardware, I have have verified that the respective vLLM/SGLang submission has been made before additional frameworks (TRT-LLM, ATOM, etc.). The only exceptions are for new hardware, such as MI455X UALoE72, Vera Rubin NVL72, Rubin NVL8, etc., and for new model architectures where there is an actual reason why vLLM/SGLang does not fundamentally support them yet.
- [ ] Verified that the single-node recipes are similar to the official [vLLM recipes](https://recipes.vllm.ai/) and/or the[SGLang cookbook](https://docs.sglang.io/cookbook/intro):
  - If they are not, I have verified that a PR has been opened in [vLLM recipe repo](https://github.com/vllm-project/recipes) or [SGLang repo](https://github.com/sgl-project/sglang/tree/main/docs_new) and linked it below in the additional detail section:
- [ ] If any of the above criteria cannot reasonably be satisfied, I have provided additional reasoning below.

### Additional detail section:
- insert any additional info here

Signed: `FILL_IN_GITHUB_USERNAME`

## Example

<img width="667" height="701" alt="image" src="https://github.com/user-attachments/assets/0c832d48-c81b-4bdb-bb53-43f39ff18b9b" />


<img width="569" height="632" alt="image" src="https://github.com/user-attachments/assets/491d9763-ab09-4734-b0f1-39eefe1ab5c4" />

