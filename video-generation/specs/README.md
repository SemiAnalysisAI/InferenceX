# Video workload specs

`h3.yaml` defines the MiniMax-H3 text-to-video-and-audio workload that SGLang and vLLM-Omni both run.
`video_spec.py` loads and validates it, then builds the plan object that `evaluator/runner.py` (#4658) executes and hashes.
`h3-vllm-omni-serving.md` records the pinned vLLM-Omni image, server command, parallelism and attention backend that served this workload on 2x H100.

`h3-v2.yaml` is a separate proposed 5-second H3 workload. Its `release-v1` set contains 20 distinct measured prompts with fixed seeds and a disjoint warmup prompt. The expected delivered clip is 124 frames at 24 fps (124/24 seconds), 1344×768, with 32 kHz stereo audio. The request uses 49 denoising updates; the frozen schedule identity records 50 sigma points and video/audio shifts 12/3.

`h3-i2va.yaml` is the image-to-video (first frame) workload. It keeps every `h3-v2.yaml` generation and fairness value and adds one conditioning image per case: 20 measured images and a disjoint warmup image, each a 1344×768 PNG that every engine already receives at its output size, so no engine resizes it.

`h3.yaml` uses schema version 1, `h3-v2.yaml` schema version 2 and `h3-i2va.yaml` schema version 3.
Version 2 adds a `warmup_case` to every set and a `fairness` block that seals the schedule, the requested libx264 ultrafast/CRF 18/yuv420p plus AAC encoding, the disabled cross-request conditioning cache, and the official Diffusers source, model and RNG candidate.
The loader enforces the version 2 policy once: 20 measured cases with distinct prompt text, one warmup run on a disjoint prompt, a schedule that matches the request, and encoding audio that matches generation.
Those pins identify a proposed reference producer; they do not assert that a reference was generated, that encoder settings were observed, or that cross-engine quality passed.

Version 3 adds an `images` registry and names one image per case. The registry records each PNG's SHA256, byte size, dimensions and media type, plus its source (URL, Commons file page and SHA-1, license, author and crop box). Its `fairness` block replaces `reference_identity` with `conditioning`, which fixes the `fl2va` task, one keyframe at frame 0, and how the images were prepared. No official i2va reference producer exists yet.
The loader also requires every image to have the generation width and height, no repeated SHA256, a distinct image per measured case, and a warmup image no measured case uses, so a warmup cannot prefill a cross-request conditioning cache.

A version 2 plan carries `schema_version: 2` and `fairness`; a version 1 plan omits `schema_version`, so its recorded hash still holds.
Code that needs the version calls `schema_version(plan)` instead of testing which keys a plan has.

## What the YAML records

- `model` names the checkpoint and its 40-character revision.
- `generation` holds the request and clip values.
  SGLang rejects caller-set fps and frame counts and derives them from its `target`; only vLLM-Omni receives fps, frame count, width, and height.
  The runner checks the clip's duration against `frame_count / fps` (107 / 24 = 4.458 s), not `duration_seconds`.
- `images` (version 3) registers each conditioning image by its PNG SHA256. The SHA256 is the image's identity; the source fields are provenance.
- `prompts` defines each prompt once with its motion and sound requirements.
  Each entry in `sets` lists cases that pair a prompt with a seed; no two cases repeat both prompt text and seed.
- `reference` (version 1 only) records the SGLang run on #2916 whose plan hash `frozen-c124` reproduces.

## Hash contract

`canonical_json_bytes` serializes with sorted keys, compact separators, UTF-8, and no NaN.
Both hashes below and the runner's recorded `plan_sha256` are SHA-256 over those bytes.

- `plan_sha256(plan_for(spec, set_id))` equals the `plan_sha256` that the runner records for the same plan.
  The `frozen-c124` set reproduces the `h3-serving-2gpu-c124-v1` plan that the reference run recorded, whose hash is `8d51a472caa7db0f5d4c33adec1e9c9f852e7144cf25356085802edaf7d80d68`; `test_video_spec.py` asserts it.
- `spec_sha256(spec)` covers the model, the generation values, and each set's resolved cases and counts.
- For `h3-v2.yaml`, both hashes also cover `warmup_case` and `fairness`, and the plan hash covers `schema_version`. `test_video_spec.py` pins both, because sealed text-to-video results carry them.
- For `h3-i2va.yaml`, each resolved case carries `condition: {task, role, frame_index, sha256, width, height, media_type}`, so both hashes change when any image's bytes change. The registry's source fields are outside both hashes.
- `reference`, YAML key order, comments, and formatting are outside both hashes.
  Neither hash includes GPU UUIDs, job IDs, ports, or hosts.
- `plan_for(spec, set_id, repetitions=..., warmup_runs=...)` overrides the counts.
  If either count differs from the set's own, the plan ID gains a `-r<repetitions>-w<warmup_runs>` suffix, so a shortened run never shares an ID or a hash with the full one.

## Run the tests and print the hashes

From the repository root:

```bash
uv run --no-project --python 3.12 --with 'pyyaml>=6,<7' --with 'pytest>=8,<9' python -m pytest video-generation/tests/test_video_spec.py -q
uv run --no-project --python 3.12 --with 'pyyaml>=6,<7' python video-generation/specs/video_spec.py
```

The second command prints `spec_sha256` and each set's `plan_id` and `plan_sha256` as JSON.
To hash another spec file, pass its path as the argument.

## Rebuild and check the image-to-video images

The PNGs live outside the repository; the spec carries their hashes. `prepare_images.py` rebuilds each one from its source original and checks it against the spec. The sources directory holds each original as `<commons_sha1>.<ext>`, downloaded from its `source.url`. The script writes `<sha256>.png`, the layout `vgbench run --image-dir` reads, and exits 1 if any image differs from the spec.

```bash
uv run --no-project --python 3.12 --with pillow==11.3.0 --with 'pyyaml>=6,<7' \
  python video-generation/specs/prepare_images.py video-generation/specs/h3-i2va.yaml SOURCES OUTPUT
```

It pins Pillow 11.3.0 because LANCZOS and zlib output can change between Pillow versions.
