# Native Kimi-K3 PD on MI355X

**English** | [中文](k3-pd-native_zh.md)

This draft integrates Kimi-K3 prefill/decode disaggregation through InferenceX's shared Python launcher and native srt-slurm lifecycle. There is no alternate launcher or router algorithm.

## Configuration ownership

- `configs/amd-master.yaml` selects `kimik3-fp4-mi355x-vllm-disagg-agentic` and supplies matrix identity and result metadata.
- `benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml` owns the topology, worker/router images and serving settings: MXFP4 target weights, FP32 SSM, FP8 KV and GMU 0.90.
- The existing variants remain 1P1D c1/c10/c48 and 1P2D c24/c48. The latency variant uses DSpark K7 without CPU offload; the others use K4 with prefill SimpleCPUOffload. Draft weights and precision are unchanged. Throughput uses automatic golden acceptance selection; eval uses real verification.
- `configs/runners.yaml` declares staged models, draft/fabric mounts, worker network settings, memlock and image-import policy. `infx/launch/` uses the job-local srt-slurm Python to resolve recipe images, rejects worker-image disagreement before import, stages the recipe-owned router image, and retains shared submission, cancellation and result collection.

High-concurrency prefill retains `HSA_NO_SCRATCH_RECLAIM=0`. Existing graph modes, transport settings and 1% request-error gates are unchanged. No BF16 SSM, workspace-development, shared-MR, QP/credit implementation or client cancellation patch is added.

## Pinned runtime and removable debug layer

The worker image is `vllm/vllm-openai-rocm:nightly-ac68c3087215e0a4f3cdfa218508c6aada57235d@sha256:e3fdfb382f2b567718ab6de49a14f5d5695dad84efc6dfd9c38f661b1a763e19`. Master and recipe use the same immutable identity. This image already includes vLLM #57700; no backport of that PR remains.

The temporary engine setup applies two checked-in Python runtime patches: synchronous READ zeroing and synthetic-only draft gathering. It verifies SHA256 and `git apply --check` before applying each, without replacing compiled extensions or downloading a moving PR head. The parser is the unmodified nightly implementation; no EOF backport is applied. A standalone router skips engine setup; a worker with missing vLLM or an incompatible patch fails startup.

| Dependency | Pinned source | Purpose |
| --- | --- | --- |
| [vLLM #59164](https://github.com/vllm-project/vllm/pull/59164) | `c5b1350f1f2bf10a128127a9b85e93d5f9f18e62` | Exclude synchronous READ destinations from newly allocated KV-page zeroing. |
| Synthetic draft-gathering experiment | `k3-synthetic-unproposed-drafts.patch`, against the pinned nightly | Restore pre-#58784 input gathering only when synthetic acceptance is active. Normal/block verification still rejects never-proposed slots. This is benchmark compatibility, not a general correctness fix. |
| [srt-slurm #508](https://github.com/NVIDIA/srt-slurm/pull/508) | tested revision `51cee8904a0b402a834887a26008adb79b8cd26b` | Bind discovery topology inside connector templates in the matching job's disposable checkout. |

READ-zeroing patch hash: `3da3746e85d53e4a3b17062b4113475d31a86cc07418e87ef5b4bb0c906cad20`. The setup does not include EOF, heartbeat, FULL-context, draft-fence or transfer-ownership patches from #58968.

Synthetic experiment hash: `33a6f792b13d39705a50562ca037a1d3c49dc054a8bdd8539fd8a154667f39df`. The automatic golden acceptance curve, draft model/count, parser and error gates are unchanged. This restores historical treatment of bootstrap placeholders, which can change the first visible token and subsequent generation. Its results require separate interpretation from ordinary model-quality evaluation and are not yet qualified as performance-equivalent.

The debug layer also retains the cluster-declared, read-only single-file `ionic-provider` mount. This is image/kernel ABI compatibility, not shared-MR. Keep the image's libibverbs core and unrelated providers unchanged. The srt-slurm dependency URL, submodule pin and AIPerf source are not replaced by this integration.

## Removal conditions

1. Once a qualified official worker image contains #59164, update master and recipe together and remove that patch/application step. Remove the separate synthetic experiment once its compatibility question is resolved; it is not an upstream fix awaiting inclusion. Delete the worker setup script, recipe reference and its tests once no runtime patch remains.
2. Independently update the shared srt-slurm pin to a revision containing #508, then remove the job-local cherry-pick and its focused tests. A worker-image update does not upgrade srt-slurm.
3. Remove the provider mount only after the replacement image opens all expected RDMA devices without it on the target fleet. Device enumeration alone does not qualify RDMA traffic.
4. Append the performance changelog and complete the applicable smoke, sweep and eval gates before merge. These temporary engine patches remain a draft debug integration, not an exception to upstream-image policy.

## Evidence and remaining qualification

The current PR sweep temporarily retains all five one-hour throughput points and only the generated 1P2D c48 full GSM8K evaluation. A separate debug commit filters the PR #3582 plan after native generation; it does not change evaluator inputs, real block rejection, sample limits, score thresholds or benchmark gates. It rejects an unexpected scope rather than silently reducing it. Other PRs and manual workflows are unaffected. Remove this selection commit before claiming the repository's full evaluation coverage; the omitted vendor checks remain unqualified.

EOF is excluded from the current candidate. Earlier EOF regression and throughput results remain historical evidence for a different parser stack, not qualification of this candidate.

The matched [1P1D c48 one-hour run](https://github.com/billishyahao/InferenceMINI/actions/runs/36847087330) used the same immutable worker image, #59164 and byte-identical EOF runtime. Warmup: 531 valid / 0 empty; profile: 4040 valid / 4 empty (0.0989%). Submission, result export and native cleanup completed. This is supporting runtime evidence from a separately adapted harness, not a sweep on this PR's current commit or an accuracy certification.

The earlier [InferenceX full-feature control](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36800732704) used a broader stack. The later #59164-only [InferenceX run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36814088271) was cancelled while queued, without a GPU runner. Neither qualifies the consolidated candidate. The minimal combination still needs current-PR 1P2D, sweep/eval and loaded-service teardown qualification; omission of other independently identified fixes is not proof that their failure modes are impossible.
