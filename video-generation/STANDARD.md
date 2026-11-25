# VideoGenX Standard v0.1

Draft. Scope: MiniMax-H3 ([`specs/h3-v2.yaml`](specs/h3-v2.yaml) set `release-v1`)
and Wan2.2-T2V-A14B ([`specs/wan22.yaml`](specs/wan22.yaml) set `release-v1`),
lossless tier only, one node with 1, 2, 4 or 8 GPUs, concurrency 1, on SGLang and
vLLM-Omni. Not every deployment runs all four GPU counts; the
[deployment list](../inferencex-e2e/docs/video-generation.md) has the current set.
Scope owner: `@edwingao28`.

Changing a rule bumps the version. Adding a model under unchanged rules does not:
the PR adds the model's spec with its `release-v1` set.
[Trust boundary](#trust-boundary) states what a merge requires.

## Rules for a result

A result counts toward VideoGenX v0.1 only when it meets every rule below. Rule 12
defines the latency metric; every other rule blocks a result that breaks it. One
exemption: Wan2.2 is exempt from rule 5 until it moves to spec schema 2 in v0.2.

| # | Rule | Why | Owner |
| --- | --- | --- | --- |
| 1 | Every measured deployment runs a released set, `release-v<N>`, of a spec for its model in [`specs/`](specs/). A released set never changes. A change to prompts, seeds or run counts adds the next `release-v<N>` to the same spec file. A change to the file's `model`, `generation` or `fairness` block, such as the denoising steps or the resolution, adds a new spec file whose first set is `release-v1`, because every set in a file shares those blocks. | Deployments with the same plan hash ran the same work. | `@functionstackx` |
| 2 | The image is pinned by SHA256 digest, the engine by its 40-character commit (`VIDEO_RUNTIME_REVISION`) and the model by revision. | Anyone can rerun the exact point. | `@edwingao28` |
| 3 | One node per deployment. Each GPU count is its own deployment and names a `video/agg-<layout>.yaml` recipe, where `<layout>` states the GPU split. | Each layout qualifies on its own sweep. | `@edwingao28` |
| 4 | Concurrency 1. | Rule 11's zero failure tolerance and fixed sample count assume one request at a time. | `@edwingao28` |
| 5 | Every engine encodes with libx264 `ultrafast`, CRF 18 and yuv420p; H3 adds AAC audio at 32 kHz stereo. | Encoding time is inside the measured latency. | `@functionstackx` |
| 6 | No work is reused across requests: the prompt-embedding (conditioning) cache is off. | Every request does the full work. | `@functionstackx` |
| 7 | Only the [lossless tier](#lossless-tier) is published. The rule is the same for every engine; [Enforcement on main](#enforcement-on-main) states the evidence behind it for each one. | Lossy speedups change the output and need paired quality scores, which v0.1 does not have. | `@functionstackx` |
| 8 | A row may use only engine settings and optimizations that a registry lists as reviewed. The registry, not the submitter, decides; anything outside it counts as lossy. | A new technique cannot enter the lossless ranking unreviewed. | `@SemiAnalysisAI/core` |
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
  memory, so SGLang rows set `--performance-mode speed`, which can turn on FSDP at
  2 or more GPUs. #5220 (open) enforces `speed` at bind time.
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

## Trust boundary

The code that checks a result runs at three trust levels:

- **(a) PR code.** Run Sweep's GPU jobs and `validate-video-results` run the PR merge
  ref's code, including the validator and the workflow YAML. Its collect and compare
  jobs run `main`'s tooling. On the fork route, the fork's code runs the GPU job,
  packaging and attestation.
- **(b) InferenceX `main`'s code.** Merge Ingest and Video Quality.
- **(c) InferenceX-app `main`'s code.** `h3-media.yml` and the publish endpoint.

`main`'s ruleset requires no status checks and no code-owner review; one approval
from `@SemiAnalysisAI/core` merges. So checks at (a) are advisory. Merge Ingest is the
only InferenceX stage that can refuse a result, and only on the pushes that run it:
pushes that touch `perf-changelog.yaml`.

Known v0.1 limit: for a fork run, `main` re-checks bookkeeping only, which is
checksums, identities, fingerprints and the clip population. It reads the power and
attestation verdicts without recomputing them, does not decode the media again and
does not see which image actually ran.

Each fact is checked once per trust level, at the earliest stage that has the
information.

## Enforcement on main

What the code on `main` checks today. This section changes when a check is added
or removed; the rules above do not. Status is Enforced, Partial (the code checks
part of the rule) or Not checked. The Open column lists PRs and branches that are
not on `main` yet.

| Rule | Status | Checked on `main` | Open | Not checked |
| --- | --- | --- | --- | --- |
| 1 | Partial | The binder ([`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py)) writes the selected set's plan hash into the bound recipe, so the recipe fingerprint covers it. Serve and client refuse a plan that does not hash to it (`selected_plan`, [`srt_selected.py`](evaluator/srt_selected.py)). In Merge Ingest, `require_merged_rows` ([`video_artifacts.py`](../inferencex-e2e/infx/results/video_artifacts.py)) refuses a merge whose video rows differ in fingerprint from the reused run's (#5213). | #5218: config time ([`validation.py`](../inferencex-e2e/infx/matrix/validation.py)) accepts only a `release-v<N>` set. `released_sets.py` reads every commit of `main` since `283fb864d` that touches the specs and requires each released set to keep the plan hash of the first commit that held it. It runs on a pull request into `main` (advisory) and in Merge Ingest before a video publication (refuses). | That a released set is unchanged: today a PR that edits one and sweeps with it publishes. Limit after #5218: Merge Ingest runs the check only on a push that publishes video, so an edit in any other push can land on `main`. The next video publication reads that commit too, so it refuses and keeps refusing until the edit is reverted. |
| 2 | Partial | The binder ([`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py)) checks the engine commit and model revision format, and forwards the row's image as `VIDEO_CONTAINER_IMAGE`. The serve step records that declared digest as `image_digest` ([`srt_selected.py`](evaluator/srt_selected.py)). | #5218: config time ([`validation.py`](../inferencex-e2e/infx/matrix/validation.py)) requires a video row's image to be pinned by digest, and is the only owner of that rule. | Known v0.1 gap: no code observes the image that actually ran. The engine commit is declared by the recipe; nothing checks it against the image. |
| 3 | Enforced | Master config validation ([`validation.py`](../inferencex-e2e/infx/matrix/validation.py)) allows video only as a single-node scenario and checks the recipe name. | | |
| 4 | Enforced | Config and matrix validation ([`validation.py`](../inferencex-e2e/infx/matrix/validation.py)) accept only `conc-list: [1]` for a video row (#5216). | | |
| 5 | Enforced for H3 | [`h3-v2.yaml`](specs/h3-v2.yaml) puts the encoding in the plan hash; [`media.py`](evaluator/media.py) reads CRF and preset from the x264 SEI header. | | Wan2.2 (exempt): `wan22.yaml` is schema 1, so the runner neither requests nor checks an encoding. |
| 6 | Enforced | The binder ([`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py)) requires each engine's off switch for every spec (#5219). The SGLang attestation requires `disable_conditioning_cache`. | | |
| 7 | Partial | SGLang: the attestation ([`attestation.py`](evaluator/attestation.py)) checks, from the engine's startup arguments and log, that no cache-dit config, quantization, precision override or LoRA is set, that every logged attention backend is dense, that the DiT weights load in BF16, and that every request ran the declared shape, seed and steps. Publication requires a valid attestation for every SGLang cell (`validate_video_artifacts`, #5217). | #5220: the binder refuses engine settings outside the registry of rule 8 and requires SGLang `--performance-mode speed`. #5223: the attestation fails when the server log shows cache-dit. #5181: the attestation records SGLang's resolved weight placement. | Known v0.1 gap: vLLM-Omni has no runtime evidence, so its rows are checked only as declared. On `main`, a log that shows cache-dit turned on outside the startup arguments passes. |
| 8 | Not checked | | #5220: `_REVIEWED` and its per-model table in [`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py) decide which engine flags, environment variables and preambles a row may set. The binder refuses anything else. | No InferenceX code records a run's optimizations, so the App's registry ([`optimizations.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/app/src/components/video-benchmark/optimizations.ts)) has nothing to classify and every published run reads as lossless. |
| 9 | Partial | The binder ([`srt_slurm/video.py`](../inferencex-e2e/infx/srt_slurm/video.py)) rejects launcher-owned recipe keys and launcher-owned or credential-like environment names. | #5214: the `video-server` service moves into the shared [`video.yaml`](../inferencex-e2e/configs/srt-recipes/video.yaml). #5222, after #5214: one allowlist of the keys a recipe may set; the binder refuses every other key. | Who published the image. Code on the Python path (the shared `video.yaml` puts the repository checkout on the client's path, and each recipe does the same for its server) and a service `preamble` get human review only. |
| 10 | Enforced | The runner ([`runner.py`](evaluator/runner.py)) builds the checks from the plan; [`media.py`](evaluator/media.py) marks a clip valid only when every check passes. | | |
| 11 | Enforced | The runner stops before measuring after a failed warmup; `run_complete` ([`bench/video.py`](../inferencex-e2e/infx/bench/video.py)) requires every clip; `validate_video_artifacts` refuses invalid power and an incomplete clip population at publication. Power bundles keep the measured sensor and scope (#5215). | Branch `fix/vgx-video-power-required`: the GPU job fails on invalid power. Today that surfaces only at publication. InferenceX-app branch `fix/vgx-one-qualification-rule`: `/vgx`, History and the API share one qualification rule and one power rule. | |
| 12 | Partial | The runner writes `submit_to_media_seconds` per clip; [`serving.py`](evaluator/serving.py) takes the median; the App reads the per-clip values ([`tradeoff.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/app/src/components/video-benchmark/tradeoff.ts)). | | When a run has no per-clip value, the App falls back to `summary.latency_median_seconds`, the median of a different field, `latency_seconds`. |
| 13 | Partial | The planner ([`plan.py`](../inferencex-e2e/infx/matrix/plan.py)) rejects append-only video. `/use` may name a run from any PR ([`reuse.py`](../inferencex-e2e/infx/workflows/reuse.py)). `validate_video_artifacts` requires a bundle for every swept row, and `require_merged_rows` requires the merged commit's video-row fingerprints to equal the reused run's (#5213). Both run in `video-reuse-check` on each push after `/use` (#5237, advisory) and in Merge Ingest ([`merge-ingest.yml`](../.github/workflows/merge-ingest.yml)), which refuses. Merge Ingest writes a selection of the validated bundles (#5212), and the App approves only bundles sent with a merge run and that selection (InferenceX-app #1855). | Branch `chore/vgx-check-each-fact-once`: Merge Ingest validates video before it dispatches any publication. InferenceX-app branch `fix/video-publish-provenance`: the publish endpoint verifies that the merge run is a Merge Ingest push run on InferenceX `main`, and takes bundle names and digests from that run's selection. | The publish endpoint trusts the merge run and selection its caller sends. |

| Deliverable | Status | Today |
| --- | --- | --- |
| VLM judge calibration | Not started | No judge and no calibration report exist in this repository. |
| Engine coverage | Partial | SGLang and vLLM-Omni run. Sol-Engine is not supported: the binder accepts only `sglang` and `vllm-omni`, and there is no attestation for it. |
| Workload size | Not started | No spec or artifact field records it, and the dashboard does not show it. #5224 (open) computes it at launch and carries it in the bundle. |
