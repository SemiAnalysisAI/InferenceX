# PR Review Checklist

When reviewing & approving an PR, please fill in the following form in your approval comment before pinging an core maintainer for final approval

As a PR reviewer and CODEOWNER, I have reviewed this and have:

- [ ] Verified that the general code quality meets the InferenceX standard and does not make the code quality any worse.
- [ ] Verified that this PR has passed PR validation.
- [ ] Verified that this PR passes evals.
- [ ] Verified that the respective vLLM/SGLang submission has been made before additional frameworks (TRT-LLM, ATOM, etc.). The only exceptions are for new hardware, such as MI455X UALoE72, Vera Rubin NVL72, Rubin NVL8, etc., and for new model architectures where there is an actual reason why vLLM/SGLang does not fundamentally support them yet.
- [ ] Verified that the single-node recipes are similar to the official vLLM recipes and/or the SGLang cookbook:
  - [SGLang cookbook](https://docs.sglang.io/cookbook/intro)
  - [vLLM recipes](https://recipes.vllm.ai/)
  - If they are not, I have verified that a PR has been opened and linked it below to the SGLang/vLLM recipe repositories:
    - [vLLM recipe repository](https://github.com/vllm-project/recipes)
    - [SGLang documentation repository](https://github.com/sgl-project/sglang/tree/main/docs_new)
- [ ] If any of the above criteria cannot reasonably be satisfied, I have provided additional reasoning below.

Signed: `FILL_IN_GITHUB_USERNAME`
