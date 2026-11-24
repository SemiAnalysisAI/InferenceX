# VideoGenX Standard v0.1

Draft. Scope: MiniMax-H3 ([`specs/h3-v2.yaml`](specs/h3-v2.yaml) set `release-v1`)
and Wan2.2-T2V-A14B ([`specs/wan22.yaml`](specs/wan22.yaml) set `release-v1`),
lossless tier only, one node with 1, 2, 4 or 8 GPUs, concurrency 1, on SGLang and
vLLM-Omni. Not every deployment runs all four GPU counts; the
[deployment list](../inferencex-e2e/docs/video-generation.md) has the current set.
Scope owner: `@edwingao28`.

Changing a rule bumps the version. Adding a model under unchanged rules does not:
the PR adds the model's spec and its released plan hash and goes through CODEOWNER
review. Review of every video change follows
[CONTRIBUTING.md](../CONTRIBUTING.md#the-pr-review-checklist-codeowner-sign-off).

## Rules for a result

A result counts toward VideoGenX v0.1 only when it meets every rule below. Rule 12
defines the latency metric; every other rule blocks a result that breaks it. One
exemption: Wan2.2 is exempt from rule 5 until it moves to spec schema 2 in v0.2.

| # | Rule | Why | Owner |
| --- | --- | --- | --- |
| 1 | Every measured deployment runs its model's spec and set from [`specs/`](specs/) unchanged. | Deployments with the same plan hash ran the same work. | `@functionstackx` |
| 2 | The image is pinned by SHA256 digest, the engine by its 40-character commit (`VIDEO_RUNTIME_REVISION`) and the model by revision. | Anyone can rerun the exact point. | `@edwingao28` |
| 3 | One node per deployment. Each GPU count is its own deployment and names a `video/agg-<layout>.yaml` recipe, where `<layout>` states the GPU split. | Each layout qualifies on its own sweep. | `@edwingao28` |
| 4 | Concurrency 1. | Rule 11's zero failure tolerance and fixed sample count assume one request at a time. | `@edwingao28` |
| 5 | Every engine encodes with libx264 `ultrafast`, CRF 18 and yuv420p; H3 adds AAC audio at 32 kHz stereo. | Encoding time is inside the measured latency. | `@functionstackx` |
| 6 | No work is reused across requests: the prompt-embedding (conditioning) cache is off. | Every request does the full work. | `@functionstackx` |
| 7 | Only the [lossless tier](#lossless-tier) is published. The rule is the same for every engine; [Enforcement on main](#enforcement-on-main) states the evidence behind it for each one. | Lossy speedups change the output and need paired quality scores, which v0.1 does not have. | `@functionstackx` |
| 8 | An optimization outside the [registry](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/app/src/components/video-benchmark/optimizations.ts) counts as lossy; the registry, not the submitter, decides. | A new technique cannot enter the lossless ranking unreviewed. | `@SemiAnalysisAI/core` |
| 9 | All configuration lives in the recipe and the master config row: no launcher or client change, no cache patching, no launch flags outside the recipe, no secrets in settings. | The recipe fingerprint describes the whole deployment. | `@edwingao28` |
| 10 | Every clip passes the technical checks: it decodes fully; video is H.264 yuv420p at a stable resolution; frame count matches and fps is within 0.1% with even frame spacing; duration is within one frame; the picture is not blank; frames change where the prompt needs motion. H3 audio is AAC at 32 kHz stereo with continuous timestamps, above −80 dBFS where the prompt needs sound. | A clip that fails cannot count toward speed. | `@functionstackx` |
| 11 | A deployment qualifies only when its warmup clip is valid, all 20 measured clips are valid and its native power is valid. | The median is taken over the complete population. | `@functionstackx` |
| 12 | Latency per clip is `submit_to_media_seconds`: the time from submit until the finished MP4 is downloaded, including encoding and polling. Charts use the median over the valid measured clips; the warmup is not counted. | One definition for every engine. | `@functionstackx` |
| 13 | The qualifying run is a full regular sweep whose video rows carry the same recipe fingerprints as the merged commit's video rows, and whose `validate-video-results` job passed. A maintainer selects it with `/use <run_id>`; it may come from any PR. Merging publishes that run; nothing reruns on `main`. | The published number is the reviewed number. | `@SemiAnalysisAI/core` |

## Lossless tier

- Allowed, because they change how the work is scheduled, not what is computed:
  tensor, Ulysses, ring and CFG parallelism; text-encoder tensor parallelism; VAE
  tiling and patch parallelism; a choice among attention backends and kernels that
  compute in BF16.
- Also allowed: changing where the weights live. CPU offload, layerwise offload
  and FSDP sharding move weights without changing the result, and the transfer
  time counts toward latency like any other work. SGLang's `--performance-mode`
  values `auto` (its default) and `memory` choose these settings from free GPU
  memory, and `speed` can turn on FSDP at 2 or more GPUs, so the recipe sets the
  mode explicitly.
- Not allowed: changing any workload or encoding setting; weights or activations in
  any precision other than BF16, including FP8, FP4 and other quantization;
  quantized attention, such as INT8 query-key products; work reused across
  requests; step caching such as TeaCache or FirstBlockCache; sparse attention;
  LoRA; launch flags or patches outside the recipe.

## Recorded, not gating

These values are recorded with each result and never block it.

- Frozen-frame fraction. No calibrated threshold exists yet.
- Reference PSNR, as [Reference PSNR](README.md#reference-psnr) describes.
  `quality_status` stays `unjudged`, and Wan2.2 has no reference, so its clips stay
  unscored. Lossless engines and layouts already spread widely in PSNR against one
  reference, so the score cannot gate.
- TF32 and other settings that change how FP32 matrix multiplies compute or
  accumulate. They are not weight or activation precisions, which the lossless
  tier fixes at BF16.

## v0.1 deliverables

These do not gate a single result. They are what v0.1 delivers as a project.

| Deliverable | What it requires | Why | Owner |
| --- | --- | --- | --- |
| VLM judge calibration | A VLM judge is calibrated offline against human-rated clips; v0.1 requires only the calibration report. The judge never blocks publication, and the dashboard does not change in v0.1. | The technical checks cannot tell whether a clip shows its prompt. | `@edwingao28` |
| Engine coverage | SGLang, vLLM-Omni and Sol-Engine all run under these rules. | Every engine in scope is measured the same way. | `@edwingao28` |
| Workload size | Each workload states its size per clip: latent tokens, DiT forward passes (denoising steps × guidance branches) and estimated DiT FLOPs. The size describes the workload and never ranks results; results from different workloads are never merged into one curve. | Video seconds do not measure work: H3 (5.17 s) and Wan2.2 (5.06 s) are nearly the same length, yet H3 needs about 1.29× the DiT compute per clip. | `@edwingao28` |

## Enforcement on main

What the code on `main` checks today. This section changes when a check is added
or removed; the rules above do not. Status is Enforced, Partial (the code checks
part of the rule) or Not checked.

| Rule | Status | Checked by | Not checked |
| --- | --- | --- | --- |
| 1 | Partial | The server binds the plan hash ([`srt_selected.py`](evaluator/srt_selected.py)); `validate_video_artifacts` ([`video_artifacts.py`](../inferencex-e2e/infx/results/video_artifacts.py)) rechecks it before publication. | That the plan hash is the released one for its spec and set. A sweep with an edited spec passes. |
| 2 | Partial | The server checks the image digest ([`srt_selected.py`](evaluator/srt_selected.py)); the binder checks the engine commit and model revision format ([`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py)). | The engine commit is declared by the recipe. Nothing checks it against the image. |
| 3 | Enforced | Master config validation ([`validation.py`](../inferencex-e2e/infx/matrix/validation.py)) allows video only as a single-node scenario and checks the recipe name. | |
| 4 | Enforced | Config and matrix validation ([`validation.py`](../inferencex-e2e/infx/matrix/validation.py)) accept only `conc-list: [1]` for a video row. | |
| 5 | Enforced for H3 | [`h3-v2.yaml`](specs/h3-v2.yaml) puts the encoding in the plan hash; [`media.py`](evaluator/media.py) reads CRF and preset from the x264 SEI header. | Wan2.2 (exempt): `wan22.yaml` is schema 1, so the runner neither requests nor checks an encoding. |
| 6 | Partial | The binder ([`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py)) requires each engine's off switch for a schema 2 spec (H3); the SGLang attestation requires `disable_conditioning_cache`. | Wan2.2 on vLLM-Omni. |
| 7 | Partial | SGLang: the attestation ([`attestation.py`](evaluator/attestation.py)) checks from the engine's startup arguments and log that no cache-dit config, quantization or LoRA is set, that the conditioning cache is off, and that every request ran the declared shape, seed and steps. [`srt_bundle.py`](evaluator/srt_bundle.py) marks a failed cell unverified. | Publication accepts a bundle with no attestation. The attestation does not fail on an optimization turned on outside the startup arguments, such as cache-dit through `SGLANG_CACHE_DIT_ENABLED`. Nothing checks that a recipe sets `--performance-mode`. vLLM-Omni has no attestation and no recipe check. |
| 8 | Not checked | | No InferenceX code records a run's optimizations, so every published run reads as lossless. |
| 9 | Partial | The binder ([`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py)) rejects launcher-owned and credential-like environment names. | Who published the image. Code on the Python path (the shared [`video.yaml`](../inferencex-e2e/configs/srt-recipes/video.yaml) puts the repository checkout on the client's path, and each recipe does the same for its server) and a service `preamble` get human review only. |
| 10 | Enforced | The runner ([`runner.py`](evaluator/runner.py)) builds the checks from the plan; [`media.py`](evaluator/media.py) marks a clip valid only when every check passes. | |
| 11 | Enforced | The runner stops before measuring after a failed warmup; `run_complete` ([`bench/video.py`](../inferencex-e2e/infx/bench/video.py)) requires every clip; `validate_video_artifacts` checks power and the clip population at publication. | |
| 12 | Partial | The runner writes `submit_to_media_seconds` per clip; [`serving.py`](evaluator/serving.py) takes the median; the App reads the per-clip values ([`tradeoff.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/app/src/components/video-benchmark/tradeoff.ts)). | When a run has no per-clip value, the App falls back to `summary.latency_median_seconds`, the median of a different field, `latency_seconds`. |
| 13 | Partial | The planner ([`plan.py`](../inferencex-e2e/infx/matrix/plan.py)) rejects append-only video; `validate_video_artifacts` requires a full video sweep; `/use` may name a run from any PR ([`reuse.py`](../inferencex-e2e/infx/workflows/reuse.py)); Merge Ingest ([`merge-ingest.yml`](../.github/workflows/merge-ingest.yml)) revalidates the reused run's bundles against that run's own sweep manifest. | That the merged commit's recipe fingerprints equal the reused run's. Publication approves every artifact of the run, not only the validated ones, and counts an entry whose receipt names no merge run. `validate-video-results` is not a required check on `main`. |

| Deliverable | Status | Today |
| --- | --- | --- |
| VLM judge calibration | Not started | No judge and no calibration report exist in this repository. |
| Engine coverage | Partial | SGLang and vLLM-Omni run. Sol-Engine is not supported: the binder accepts only `sglang` and `vllm-omni`, and there is no attestation for it. |
| Workload size | Not started | No spec or artifact field records it, and the dashboard does not show it. |
