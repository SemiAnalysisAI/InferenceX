# srt-slurm patches

As shown in the [CODEOWNERS](../../../../.github/CODEOWNERS) file, InferenceX core maintainers control the patches here, so there is ZERO dependency on upstream srt-slurm NVIDIA maintainers for any srt-slurm patch, ensuring that InferenceX is vendor neutral. srt-slurm allows for declarative YAML launching instead of the previous unmaintainable, low-quality pile of 1000+ bash scripts.

The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) applies every `*.patch` here to the job's srt-slurm clone after checking out the pinned submodule.

Each patch is a temporary fix for an open upstream PR. When the PR merges and the submodule pin includes it, delete the patch and its row.

| Patch | Upstream PR | Fix |
|-------|-------------|-----|
| `548-atom-served-model-name.patch` | [NVIDIA/srt-slurm#548](https://github.com/NVIDIA/srt-slurm/pull/548) | ATOM serves evals the role's `served-model-name` instead of the literal `--model` path, so AToMesh accepts eval requests |
| `smg-frontend.patch` | [SemiAnalysisAI/srt-slurm#38](https://github.com/SemiAnalysisAI/srt-slurm/pull/38) (with #41 and #42) | [DNM] `frontend.type: smg` (Shepherd Model Gateway) |
| `tokenspeed.patch` | [SemiAnalysisAI/srt-slurm#37](https://github.com/SemiAnalysisAI/srt-slurm/pull/37) and its SMG follow-up | [DNM] `engine: tokenspeed`: `dynamo.tokenspeed` workers behind Dynamo, or TokenSpeed gRPC engines behind SMG, aggregated or prefill/decode |
