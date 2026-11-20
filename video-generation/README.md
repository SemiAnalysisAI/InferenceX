# Video-generation harness [experimental]

**English** | [中文](README_zh.md)

This package is the client and server side of the `video-generation` benchmark
scenario: MiniMax-H3 served by one diffusion engine on one node, the selected
frozen YAML workload sent at every configured concurrency, each generated clip
decoded and validated, and every cell's
measured window published to srt-slurm's native GPU power telemetry.

InferenceX owns scheduling and results. `inferencex-e2e/configs/nvidia-master.yaml`
and `amd-master.yaml` declare the deployments (`h3-bf16-<sku>-<engine>-video`), `infx.matrix` plans
them from `perf-changelog.yaml`, `run-sweep.yml` runs the `video` bucket through
`benchmark-tmpl.yml`, and `python -m infx.launch run` takes the `srt-video` path:
the SKU recipe under `benchmarks/single_node/srt-slurm-recipes/h3/` starts the
engine through `python3 -m infx.bench video serve` as a generic service and runs
`python3 -m infx.bench video client` as `benchmark.type: custom`. This directory
is mounted into the job as `/video-package`. Each `*-video` entry in the master
configs names its spec and set. See
[native video delivery](../inferencex-e2e/docs/video-generation.md) for
artifact identity, reuse validation and post-merge publication.

## VideoGenX Standard v0.1

Draft. Scope: MiniMax-H3 ([`specs/h3-v2.yaml`](specs/h3-v2.yaml) set `release-v1`)
and Wan2.2-T2V-A14B ([`specs/wan22.yaml`](specs/wan22.yaml) set `release-v1`),
lossless tier only, one node with 1, 2, 4 or 8 GPUs, concurrency 1, on SGLang,
vLLM-Omni and Sol-Engine. Scope owner: `@edwingao28`.

A result counts toward VideoGenX v0.1 only when it meets every rule below. A rule
change lands in this table and bumps the version. "Enforced by" links the code
that blocks or computes the rule at commit `89ad295`; "gap" means no code checks
it yet.

| # | Rule | Why | Enforced by | Effect | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Every measured deployment runs its model's spec and set unchanged. H3: 20 prompts with one fixed seed each, 49 denoising updates, 124 frames at 24 fps and 1344×768, and a separate warmup prompt. Wan2.2: four prompts by five seeds, 40 steps, 81 frames at 16 fps and 832×480, and one warmup. | Deployments with the same plan hash ran the same work. | [`srt_selected.py:190`][plan-bind] binds the plan hash; [`video_artifacts.py:163`][plan-pub] rechecks it before publication | Blocking | `@functionstackx` | In force |
| 2 | The image is pinned by SHA256 digest, the engine by its 40-character commit (`VIDEO_RUNTIME_REVISION`) and the model by revision. | Anyone can rerun the exact point. | [`srt_selected.py:266`][image], [`srt_slurm/video.py:336`][engine], [`srt_slurm/video.py:149`][model] | Blocking | recipe CODEOWNERS | In force |
| 3 | One node per deployment. Each GPU count is its own deployment with its own `video/agg-<layout>.yaml` recipe. | Each layout qualifies on its own sweep. | [`validation.py:918`][single-node] (video is a single-node scenario only); [`validation.py:452`][naming] | Blocking | `@edwingao28` | In force |
| 4 | Concurrency 1. | The failure tolerance and the sample population in rule 12 assume one request at a time. | Gap: [`validation.py:488`][conc] accepts any unique positive concurrency; every current row lists `[1]` | Blocking | `@edwingao28` | In force |
| 5 | Every engine encodes with libx264 `ultrafast`, CRF 18 and yuv420p; H3 adds AAC audio at 32 kHz stereo. | Encoding time is inside the measured latency. | H3: [`h3-v2.yaml:157`][encoding] puts the encoding in the plan hash and [`media.py:581`][sei] reads CRF and preset from the x264 SEI header. Gap for Wan2.2: `wan22.yaml` is schema 1, so [`runner.py:183`][encoding-v2] requests and checks no encoding | Blocking | `@functionstackx` | In force |
| 6 | No work is reused across requests: the prompt-embedding (conditioning) cache is off. | Every request does the full work. | H3: [`srt_slurm/video.py:329`][cache] requires each engine's off switch. Gap for Wan2.2: that check applies to schema 2 specs only | Blocking | `@functionstackx` | In force |
| 7 | Lossless tier only; see [the lists below](#lossless-tier). | Lossy speedups change the output and need paired quality scores, which v0.1 does not have. | SGLang: engine-log attestation [`attestation.py:36`][policy], gated in [`srt_bundle.py:378`][attest-gate]. Gap for vLLM-Omni and Sol-Engine: [`attestation.py:66`][policies] has an SGLang policy only, and [`video_artifacts.py:146`][attest-presence] passes a bundle with no attestation | Blocking | `@functionstackx` | In force |
| 8 | An optimization outside the registry counts as lossy; the registry, not the submitter, decides. | A new technique cannot enter the lossless ranking unreviewed. | App [`optimizations.ts:26`][registry]. Gap: no InferenceX code records a run's optimizations, so every published run reads as lossless | Blocking | `@SemiAnalysisAI/core` | In force |
| 9 | All configuration lives in the recipe and the master config row: no launcher or client change, no cache patching, no launch flags outside the recipe, no secrets in settings. | The recipe fingerprint describes the whole deployment. | [`srt_slurm/video.py:102`][env] rejects launcher-owned and credential-like environment names. Gap: a patched image is not detected | Blocking | recipe CODEOWNERS | In force |
| 10 | Every clip passes the technical checks: it decodes fully; video is H.264 yuv420p at a stable resolution; frame count matches and fps is within 0.1% with even frame spacing; duration is within one frame; the picture is not blank; frames change where the prompt needs motion. H3 audio is AAC at 32 kHz stereo with continuous timestamps, above −80 dBFS where the prompt needs sound. | A clip that fails cannot count toward speed. | [`runner.py:946`][expected] builds the checks from the plan; [`media.py:806`][clip-valid] marks a clip valid only when every check passes | Blocking | `@functionstackx` | In force |
| 11 | The frozen-frame fraction is recorded, not checked. | No calibrated threshold exists yet. | [`media.py:753`][frozen] checks it only when a plan sets `max_frozen_fraction`; neither spec does | Advisory | `@functionstackx` | In force |
| 12 | A deployment qualifies only when its warmup clip is valid, all 20 measured clips are valid (no failure tolerance) and its native power is valid. | The median is taken over the complete population. | [`runner.py:1124`][warmup] stops before measuring after a failed warmup; [`bench/video.py:28`][complete]; [`video_artifacts.py:140`][power] and [`video_artifacts.py:183`][population] at publication | Blocking | `@functionstackx` | In force |
| 13 | Latency per clip runs from submit until the finished MP4 is downloaded, including encoding and polling. Charts use the median over the valid measured clips; the warmup is not counted. | One definition for every engine. | [`runner.py:1073`][latency]; [`serving.py:91`][median] | Definition | `@functionstackx` | In force |
| 14 | The qualifying run is a full regular sweep of the final changelog matrix on a PR commit, and its `validate-video-results` job passes. After merge the same run is published; nothing reruns on `main`. | The published number is the reviewed number. | [`plan.py:178`][append-only] rejects append-only; [`video_artifacts.py:76`][full-sweep] requires a full video sweep; [`run-sweep.yml:910`][validate-job] | Blocking | `@SemiAnalysisAI/core` | In force |
| 15 | One eligible CODEOWNER posts the PR Review Checklist and a core maintainer approves. | Every rule change and result gets a second reviewer. | The `main` ruleset requires one approval from `@SemiAnalysisAI/core`. Gap: code-owner review is not required, and an admin merge bypasses the ruleset | Blocking | `@SemiAnalysisAI/core` | In force |
| 16 | Reference PSNR is reported and never gates reuse or merge; `quality_status` stays `unjudged`. Wan2.2 has no reference, so its clips stay unscored. | Lossless engines and layouts already spread widely in PSNR against one reference. | [`quality.py:192`][unjudged]; no workflow reads the score as a gate | Advisory | `@functionstackx` | In force |
| 17 | A VLM judge scores each measured clip, and the score is reported next to the clip. It does not block publication. | The technical checks cannot tell whether a clip shows its prompt. | Gap: no judge exists in this repository | Advisory | `@edwingao28` | In force |
| 18 | Every engine, SGLang, vLLM-Omni and Sol-Engine alike, meets the same rules: a PR rebased on `main`, a public image pinned by digest, and a passing engine-log attestation. | One evidence bar for every engine. | [`srt_selected.py:266`][image] for the digest; [`srt_bundle.py:378`][attest-gate] for the attestation. Gap: [`srt_slurm/video.py:253`][runtime] accepts only `sglang` and `vllm-omni`, and the attestation policy is SGLang only (rule 7) | Blocking | `@edwingao28` | In force |

### Lossless tier

- Allowed, because they change how the work is scheduled, not what is computed:
  tensor, Ulysses, ring and CFG parallelism; text-encoder tensor parallelism; VAE
  tiling and patch parallelism; attention backend and kernel choice.
- Not allowed: changing any workload or encoding setting; any precision other
  than BF16, including FP8, FP4 and other quantization; work reused across
  requests; step caching such as TeaCache or FirstBlockCache; sparse attention;
  LoRA; launch flags or patches outside the recipe.

[plan-bind]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/srt_selected.py#L190-L191
[plan-pub]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L163-L168
[image]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/srt_selected.py#L266-L268
[engine]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L336-L338
[model]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L149-L150
[single-node]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/validation.py#L918-L920
[naming]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/validation.py#L452-L467
[conc]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/validation.py#L488
[encoding]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/specs/h3-v2.yaml#L157-L164
[sei]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/media.py#L581-L595
[encoding-v2]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L182-L183
[cache]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L329-L335
[policy]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/attestation.py#L36-L65
[attest-gate]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/srt_bundle.py#L378-L389
[policies]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/attestation.py#L66
[attest-presence]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L146-L150
[registry]: https://github.com/SemiAnalysisAI/InferenceX-app/blob/d4f50964689aaf4beb97c40e0f7476f6b83e0a86/packages/app/src/components/video-benchmark/optimizations.ts#L26-L27
[env]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L102-L108
[expected]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L946-L955
[clip-valid]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/media.py#L806
[frozen]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/media.py#L753-L763
[warmup]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L1124-L1128
[complete]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/bench/video.py#L28-L34
[power]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L140-L145
[population]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L183-L191
[latency]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/runner.py#L1073
[median]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/serving.py#L91
[append-only]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/matrix/plan.py#L178
[full-sweep]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/results/video_artifacts.py#L75-L76
[validate-job]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/.github/workflows/run-sweep.yml#L910
[unjudged]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/video-generation/evaluator/quality.py#L192-L193
[runtime]: https://github.com/SemiAnalysisAI/InferenceX/blob/89ad29552dc2d8ea76a6b6ec723c99df59b7016a/inferencex-e2e/infx/srt_slurm/video.py#L246-L253

## Entry points

The native entry `python3 -m infx.bench video` calls `evaluator.srt_selected`
directly. Both actions first load the selected spec and set from `specs/` and
check its `plan_sha256` against the bound value.

- `serve` records the sealed server spec and launch, then execs the engine
  (`sglang` or `vllm-omni`) on the job's GPUs.
- `client` runs the same plan at every `CONC_LIST` concurrency against the
  service endpoint, one cell per concurrency under `gpu/c<N>/`, and publishes
  each cell's measured window.

After the job, the launcher runs `python -m evaluator.cli srt-finalize`. It joins
the cells, their power validation sidecars and the server record into the
`h3-results-<run>-<attempt>-<recipe-fingerprint>` bundle the dashboard ingests.

`specs/video_spec.py` loads the YAML workload, validates it and hashes the
selected plan; `infx.srt_slurm.video` imports it when binding a point.

For offline Diffusers reference media, use the [pinned environment and generation
command](reference/README.md). Reference generation is separate from serving
latency and leaves cross-engine quality unjudged.

[`specs/h3-v2.yaml`](specs/h3-v2.yaml) supplies a separate 20-prompt,
5-second H3 contract under the `release-v1` selector. It requests 49 denoising
updates, expects 124 delivered frames, and uses a disjoint warmup. The runner
sends each engine's explicit H.264 options and records the encoding policy in
the plan hash. Media decoding verifies observable stream properties, plus CRF
and the preset's fixed options from x264's SEI header. A plan may request only
a preset whose options the check can verify. Exact encoder implementation and
cross-engine quality still require separate evidence. Comparisons leave cross-engine quality unjudged, while
same-implementation numerical checks require matching sealed image, launch,
environment, GPU and algorithm identity.

## Reference PSNR

[Video Quality](../.github/workflows/video-quality.yml) scores every measured clip
of a finished Run Sweep against the Diffusers references. It runs main's code on a
GitHub-hosted runner and reads the sweep's `h3-results-*` artifacts only as data.
[`reference/h3-v2.json`](reference/h3-v2.json) names the reference release, the
generation settings and implementation identity, and each case's prompt, seed,
MP4 SHA256 and size. The job downloads the release assets, rejects any asset whose
hash differs from the descriptor, and runs:

```bash
python -m evaluator.cli score-quality <artifact-id>=<bundle.zip|dir> ... \
  --descriptor reference/h3-v2.json --references <release-mp4-dir> \
  --run-id <run> --source-revision <git-sha> --output <new-dir>
```

A clip is scored only when its recorded prompt, seed, model and generation
settings equal the descriptor; `compare_media` then compares every decoded frame.
Other clips are listed under `unscored` with a reason: `invalid_candidate`,
`checksum_mismatch`, `no_reference`, or `settings_mismatch` with the differing
fields. A bundle that cannot be read is listed under `unreadable` and the others are still
scored. Each deployment contributes its newest bundle up to the scored attempt. The
`h3-quality-<run>-<attempt>` artifact holds `quality.json`, a copy of
the descriptor and `SHA256SUMS`. PSNR measures technical fidelity to one reference
implementation, so `quality_status` stays `unjudged`. To score an earlier run,
dispatch the workflow with its `run_id` and `run_attempt`.

The dashboard's [quality guide](https://inferencex.semianalysis.com/vgx/quality)
explains how these scores judge a lossy speedup and plots one published run's
per-clip PSNR for each GPU.

## Local checks

```bash
cd video-generation
PYTHONPATH=../inferencex-e2e uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pyyaml>=6,<7' --with 'pytest==9.2.0' python -m pytest -q
```

[Test Video Harness](../.github/workflows/test-video-harness.yml) runs these
CPU checks on relevant changes. They make no model, scheduler or GPU calls;
sweep artifacts are the acceptance evidence.
