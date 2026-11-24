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
lossless tier only, one node with 1, 2, 4 or 8 GPUs, concurrency 1, on SGLang and
vLLM-Omni. SGLang results are attested from the engine log; vLLM-Omni results are
checked at recipe level (rule 7, G2). Sol-Engine is in scope but not supported yet
(G8). Not every deployment runs all four GPU counts ([deployments][gpu-counts]).
Scope owner: `@edwingao28`.

A result counts toward VideoGenX v0.1 only when it meets every rule below. One
exemption: Wan2.2 is exempt from rule 5 until it moves to spec schema 2 in v0.2.
Changing a rule lands in this table and bumps the version. Adding a model under
unchanged rules does not: it adds a spec and its released plan hash under
CODEOWNER review. "Enforced by" links the code that blocks or computes the rule
at commit `8dd3577`; "Gap" means no code checks it yet, and a branch or PR named
as pending is not merged. Status is `Enforced`, `Partial` when code checks only
part of the rule, `Gap (G#)` with the ID from [Open gaps](#open-gaps), or
`Advisory` for a rule that never gates.

| # | Rule | Why | Enforced by | Effect | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Every measured deployment runs its model's spec and set unchanged. H3: 20 prompts with one fixed seed each, 49 denoising updates, 124 frames at 24 fps and 1344×768, and a separate warmup prompt. Wan2.2: four prompts by five seeds, 40 steps, 81 frames at 16 fps and 832×480, and one warmup. | Deployments with the same plan hash ran the same work. | [`srt_selected.py:190`][plan-bind] binds the plan hash; [`video_artifacts.py:163`][plan-pub] rechecks it before publication. Not checked yet: that a released plan hash stays unchanged (pending `fix/vgx-release-plan-pins`) | Blocking | `@functionstackx` | Partial |
| 2 | The image is pinned by SHA256 digest, the engine by its 40-character commit (`VIDEO_RUNTIME_REVISION`) and the model by revision. The engine commit is declared by the recipe; nothing verifies it against the image. | Anyone can rerun the exact point. | [`srt_selected.py:266`][image], [`srt_slurm/video.py:336`][engine], [`srt_slurm/video.py:149`][model]. [`runner.py:797`][engine-declared] accepts any 40-character commit as operator-declared | Blocking | `@edwingao28` | Partial |
| 3 | One node per deployment. Each GPU count is its own deployment and names a `video/agg-<layout>.yaml` recipe, where `<layout>` states the GPU split. | Each layout qualifies on its own sweep. | [`validation.py:918`][single-node] (video is a single-node scenario only); [`validation.py:453`][naming] | Blocking | `@edwingao28` | Enforced |
| 4 | Concurrency 1. | The failure tolerance and the sample population in rule 12 assume one request at a time. | Gap: [`validation.py:488`][conc] accepts any unique positive concurrency; every current row lists `[1]` | Blocking | `@edwingao28` | Gap (G1) |
| 5 | Every engine encodes with libx264 `ultrafast`, CRF 18 and yuv420p; H3 adds AAC audio at 32 kHz stereo. | Encoding time is inside the measured latency. | H3: [`h3-v2.yaml:157`][encoding] puts the encoding in the plan hash and [`media.py:581`][sei] reads CRF and preset from the x264 SEI header. Gap for Wan2.2: `wan22.yaml` is schema 1, so [`runner.py:183`][encoding-v2] requests and checks no encoding; Wan2.2 moves to schema 2 in v0.2, which changes its plan hash and needs a new sweep | Blocking | `@functionstackx` | Enforced; Wan2.2 exempt until v0.2 |
| 6 | No work is reused across requests: the prompt-embedding (conditioning) cache is off. | Every request does the full work. | H3: [`srt_slurm/video.py:329`][cache] requires each engine's off switch. Wan2.2 on SGLang: the attestation requires `disable_conditioning_cache` ([`attestation.py:45`][sglang-no-reuse]). Not checked yet: Wan2.2 on vLLM-Omni (pending `fix/vgx-wan22-no-reuse`) | Blocking | `@functionstackx` | Partial |
| 7 | Lossless tier only; see [the lists below](#lossless-tier). v0.1 publishes no lossy result. | Lossy speedups change the output and need paired quality scores, which v0.1 does not have. | SGLang: engine-log attestation [`attestation.py:36`][policy], gated in [`srt_bundle.py:378`][attest-gate]. Three parts are open: [`video_artifacts.py:146`][attest-presence] passes a bundle with no attestation (G3); the attestation does not fail on every optimization it can detect (G4); nothing checks that the recipe sets `--performance-mode` until [#5181](https://github.com/SemiAnalysisAI/InferenceX/pull/5181) merges, and the attestation records the mode but not the offload settings it resolves. vLLM-Omni: checked at recipe level, not attested from logs. [`attestation.py:66`][policies] has an SGLang policy only (G2); a bind-time check that refuses lossy switches is pending on `feat/vgx-vllm-omni-flag-check`. TF32 and reduced-precision settings are recorded, not gated, in v0.1 | Blocking | `@functionstackx` | Gap (G2, G3, G4; #5181) |
| 8 | An optimization outside the registry counts as lossy; the registry, not the submitter, decides. | A new technique cannot enter the lossless ranking unreviewed. | App [`optimizations.ts:26`][registry]. Gap: no InferenceX code records a run's optimizations, so every published run reads as lossless | Blocking | `@SemiAnalysisAI/core` | Gap (G4) |
| 9 | All configuration lives in the recipe and the master config row: no launcher or client change, no cache patching, no launch flags outside the recipe, no secrets in settings. | The recipe fingerprint describes the whole deployment. | [`srt_slurm/video.py:102`][env] rejects launcher-owned and credential-like environment names. Gap (G5): nothing checks who published the image. G5 is a provenance check on the image repository, not patch detection. Review only: code on the Python path (the shared block puts the repository checkout on the client's path at [`video.yaml:8`][python-path], and each recipe does the same for its server) and a service `preamble` | Blocking | `@edwingao28` | Gap (G5) |
| 10 | Every clip passes the technical checks: it decodes fully; video is H.264 yuv420p at a stable resolution; frame count matches and fps is within 0.1% with even frame spacing; duration is within one frame; the picture is not blank; frames change where the prompt needs motion. H3 audio is AAC at 32 kHz stereo with continuous timestamps, above −80 dBFS where the prompt needs sound. | A clip that fails cannot count toward speed. | [`runner.py:946`][expected] builds the checks from the plan; [`media.py:806`][clip-valid] marks a clip valid only when every check passes | Blocking | `@functionstackx` | Enforced |
| 11 | The frozen-frame fraction is recorded, not checked. | No calibrated threshold exists yet. | [`media.py:753`][frozen] checks it only when a plan sets `max_frozen_fraction`; neither spec does | Advisory | `@functionstackx` | Advisory |
| 12 | A deployment qualifies only when its warmup clip is valid, all 20 measured clips are valid (no failure tolerance) and its native power is valid. | The median is taken over the complete population. | [`runner.py:1124`][warmup] stops before measuring after a failed warmup; [`bench/video.py:28`][complete]; [`video_artifacts.py:140`][power] and [`video_artifacts.py:183`][population] at publication | Blocking | `@functionstackx` | Enforced |
| 13 | Latency per clip is `submit_to_media_seconds`: the time from submit until the finished MP4 is downloaded, including encoding and polling. Charts use the median over the valid measured clips; the warmup is not counted. | One definition for every engine. | Written at [`runner.py:1073`][latency]; median taken at [`serving.py:105`][median]; read per clip by the App at [`tradeoff.ts:93`][app-latency]. `summary.latency_median_seconds` ([`runner.py:750`][latency-summary]) is the median of a second field, `latency_seconds`, which the App uses only when a run has no per-clip value ([`tradeoff.ts:120`][app-fallback]) | Definition | `@functionstackx` | Enforced |
| 14 | The qualifying run is a full regular sweep of the final changelog matrix on a PR commit, and its `validate-video-results` job passes. After merge the same run is published; nothing reruns on `main`. | The published number is the reviewed number. | [`plan.py:178`][append-only] rejects append-only; [`video_artifacts.py:76`][full-sweep] requires a full video sweep. The gate at merge is [`merge-ingest.yml:131`][merge-gate], which validates the reused run's bundles before publishing them. Not complete: `validate-video-results` ([`run-sweep.yml:910`][validate-job]) is not a required check on `main`, and `fix/vgx-merge-matrix-check` is pending | Blocking | `@SemiAnalysisAI/core` | Partial |
| 15 | One eligible CODEOWNER posts the PR Review Checklist and a core maintainer approves. | Every rule change and result gets a second reviewer. | The `main` ruleset requires one approval from `@SemiAnalysisAI/core`. Gap: code-owner review is not required, and an admin merge bypasses the ruleset. G6 names the video owners in CODEOWNERS; closing the gap needs an admin ruleset change | Blocking | `@SemiAnalysisAI/core` | Gap (G6) |
| 16 | Reference PSNR is reported and never gates reuse or merge; `quality_status` stays `unjudged`. Wan2.2 has no reference, so its clips stay unscored. | Lossless engines and layouts already spread widely in PSNR against one reference. | [`quality.py:192`][unjudged]; no workflow reads the score as a gate | Advisory | `@functionstackx` | Advisory |
| 17 | A VLM judge is calibrated offline against human-rated clips; v0.1 requires only that calibration report. The judge never blocks publication, and the dashboard does not change in v0.1. | The technical checks cannot tell whether a clip shows its prompt. | Gap: no judge and no calibration report exist in this repository | Advisory | `@edwingao28` | Gap (G7) |
| 18 | Every engine meets the same rules. What differs in v0.1 is the evidence behind rule 7: SGLang is attested from its engine log, vLLM-Omni is checked at recipe level, and Sol-Engine is not supported yet. | A reader can tell how strongly each engine's lossless claim is backed. | SGLang: [`attestation.py:66`][policies]. vLLM-Omni: Gap (G2). Sol-Engine: Gap, [`srt_slurm/video.py:253`][runtime] accepts only `sglang` and `vllm-omni` | Blocking | `@edwingao28` | Gap (G8) |
| 19 | Each workload states its size per clip: latent tokens, DiT forward passes (denoising steps × guidance branches) and estimated DiT FLOPs. [`specs/README.md`](specs/README.md#workload-size) gives the derivation and each workload's numbers. They describe the workload and never rank results; results from different workloads are never merged into one curve. | Video seconds do not measure work: H3 (5.17 s) and Wan2.2 (5.06 s) are nearly the same length, yet H3 needs about 1.29× the DiT compute per clip. | Gap: no spec or artifact field records them, and the dashboard does not show them | Definition | `@edwingao28` | Gap (G9) |

### Lossless tier

- Allowed, because they change how the work is scheduled, not what is computed:
  tensor, Ulysses, ring and CFG parallelism; text-encoder tensor parallelism; VAE
  tiling and patch parallelism; attention backend and kernel choice.
- Also allowed: changing where the weights live. CPU offload, layerwise offload
  and FSDP sharding move weights without changing the result, and the transfer
  time counts toward latency like any other work. SGLang's `--performance-mode`
  values `auto` (its default) and `memory` choose these settings from free GPU
  memory, and `speed` can turn on FSDP at 2 or more GPUs, so the recipe sets the
  mode explicitly (rule 7 says what checks it).
- Not allowed: changing any workload or encoding setting; any precision other
  than BF16, including FP8, FP4 and other quantization; work reused across
  requests; step caching such as TeaCache or FirstBlockCache; sparse attention;
  LoRA; launch flags or patches outside the recipe.

### Open gaps

- G0: the [Test Video Harness][harness-paths] path filter omits files the harness
  imports or this table cites, such as `inferencex-e2e/infx/bench/env.py` and
  `inferencex-e2e/infx/srt_slurm/video.py`.
- G1 (rule 4): a concurrency other than 1 is accepted.
- G2 (rules 7 and 18): vLLM-Omni has no log attestation; its bind-time flag
  check is pending on `feat/vgx-vllm-omni-flag-check`.
- G3 (rule 7): publication passes a bundle that has no attestation.
- G4 (rules 7 and 8): no run records its optimizations, and the attestation does
  not fail on every one it can detect.
- G5 (rule 9): nothing checks who published the image.
- G6 (rule 15): code-owner review is not required, and an admin merge bypasses
  the ruleset.
- G7 (rule 17): no VLM judge and no calibration report exist.
- G8 (rule 18): Sol-Engine is not supported: no binder support and no attestation
  policy.
- G9 (rule 19): no spec or artifact field records workload size, and the
  dashboard does not show it.

Pending without a gap ID: `fix/vgx-release-plan-pins` (rule 1),
`fix/vgx-wan22-no-reuse` (rule 6), `fix/vgx-merge-matrix-check` (rule 14) and
[#5181](https://github.com/SemiAnalysisAI/InferenceX/pull/5181) (rule 7).
Wan2.2's exemption from rule 5 ends with its move to schema 2 in v0.2.

[plan-bind]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/srt_selected.py#L190-L191
[plan-pub]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/results/video_artifacts.py#L163-L168
[image]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/srt_selected.py#L266-L268
[engine]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/srt_slurm/video.py#L336-L338
[model]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/srt_slurm/video.py#L149-L150
[engine-declared]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/runner.py#L797-L799
[single-node]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/matrix/validation.py#L918-L920
[naming]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/matrix/validation.py#L453-L467
[conc]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/matrix/validation.py#L488
[encoding]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/specs/h3-v2.yaml#L157-L164
[sei]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/media.py#L581-L595
[encoding-v2]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/runner.py#L182-L183
[cache]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/srt_slurm/video.py#L329-L335
[sglang-no-reuse]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/attestation.py#L45
[policy]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/attestation.py#L36-L65
[attest-gate]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/srt_bundle.py#L378-L389
[policies]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/attestation.py#L66
[attest-presence]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/results/video_artifacts.py#L146-L150
[registry]: https://github.com/SemiAnalysisAI/InferenceX-app/blob/dbcd81e6dae24c83eb045db6a8e45a91ea428dc4/packages/app/src/components/video-benchmark/optimizations.ts#L26-L27
[env]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/srt_slurm/video.py#L102-L108
[python-path]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/configs/srt-recipes/video.yaml#L8
[expected]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/runner.py#L946-L955
[clip-valid]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/media.py#L806
[frozen]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/media.py#L753-L763
[warmup]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/runner.py#L1124-L1128
[complete]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/bench/video.py#L28-L34
[power]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/results/video_artifacts.py#L140-L145
[population]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/results/video_artifacts.py#L183-L191
[latency]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/runner.py#L1073
[median]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/serving.py#L105-L107
[append-only]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/matrix/plan.py#L178
[full-sweep]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/results/video_artifacts.py#L75-L76
[merge-gate]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/.github/workflows/merge-ingest.yml#L131-L160
[validate-job]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/.github/workflows/run-sweep.yml#L910
[unjudged]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/quality.py#L192-L193
[runtime]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/infx/srt_slurm/video.py#L246-L253
[gpu-counts]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/inferencex-e2e/docs/video-generation.md?plain=1#L3-L15
[app-latency]: https://github.com/SemiAnalysisAI/InferenceX-app/blob/dbcd81e6dae24c83eb045db6a8e45a91ea428dc4/packages/app/src/components/video-benchmark/tradeoff.ts#L93
[latency-summary]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/video-generation/evaluator/runner.py#L723-L750
[app-fallback]: https://github.com/SemiAnalysisAI/InferenceX-app/blob/dbcd81e6dae24c83eb045db6a8e45a91ea428dc4/packages/app/src/components/video-benchmark/tradeoff.ts#L120-L133
[harness-paths]: https://github.com/SemiAnalysisAI/InferenceX/blob/8dd3577fae706986d6786f5b8b174f44594eb202/.github/workflows/test-video-harness.yml#L5-L10

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
