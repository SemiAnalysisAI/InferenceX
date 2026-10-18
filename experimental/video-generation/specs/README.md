# Video workload specs

**English** | [中文](README_zh.md)

## MiniMax-H3

`h3.yaml` defines the MiniMax-H3 text-to-video-and-audio workload that SGLang and vLLM-Omni both run.
`h3_spec.py` loads and validates it, then builds the plan object that the runner from PR #2916 executes and hashes.
The H3 spec is an explicit dependency of the shared loader's default command.
`h3-vllm-omni-serving.md` records the pinned vLLM-Omni H3 image and two-GPU server command.
The `release-v1` set crosses the four existing prompts with five recorded seeds for 20 measured clips.

## What the YAML records

- `model` and every `generation` field hold a `value` and a `source` citation of the form `<ref>@<commit>:<path>:<line>`.
- `sent.sglang` and `sent.vllm-omni` say whether that engine's request carries the field.
  SGLang rejects caller-set fps and frame counts, runs at a fixed 24 fps, and derives the frame count and dimensions from its `target`.
  Only vLLM-Omni receives fps, frame count, width, and height.
- `checked_on_clip` says whether the delivered clip is compared to the value.
  The runner checks the clip's duration against `frame_count / fps` (107 / 24 = 4.458 s), so `duration_seconds` itself is not checked.
- `prompts` defines each prompt once with its motion and sound requirements.
  Each entry in `sets` lists cases that pair a prompt with a seed, and its `seed_sources` cites every seed.
- `reference` records the SGLang reference run. It is informational, and no code reads or hashes it.
  The run used workflow commit f7c7f58, whose spec, `mvp_runner.py`, and `mvp_media.py` match 0cb4c847d byte for byte.

## Hash contract

`canonical_json_bytes` is a byte-for-byte copy of the function in #2916's `evaluator/mvp_runner.py`.
It serializes with sorted keys, compact separators, UTF-8, and no NaN.

- `plan_sha256(plan_for(spec, set_id))` equals the `plan_sha256` that #2916's runner records for the same plan.
  The `frozen-c124` set reproduces #2916's `h3-serving-2gpu-c124-v1` plan, whose hash is `8d51a472caa7db0f5d4c33adec1e9c9f852e7144cf25356085802edaf7d80d68`.
- `spec_sha256(spec)` covers the model, the generation values, and each set's resolved cases and counts.
- Citations, `sent`, `checked_on_clip`, and `reference` are outside both hashes, as are YAML key order, comments, and formatting.
  Neither hash includes GPU UUIDs, job IDs, ports, or hosts.
- `plan_for(spec, set_id, repetitions=..., warmup_runs=...)` overrides the counts.
  If either count differs from the set's own, the plan ID gains a `-r<repetitions>-w<warmup_runs>` suffix, so a shortened run never shares an ID or a hash with the full one.

## Run the tests and print the hashes

From the repository root:

```bash
uv run --no-project --python 3.12 --with 'pyyaml>=6,<7' --with 'pytest>=8,<9' python -m pytest experimental/video-generation/tests/test_h3_spec.py experimental/video-generation/tests/test_wan_spec.py -q
uv run --no-project --python 3.12 --with 'pyyaml>=6,<7' python experimental/video-generation/specs/h3_spec.py
```

The second command prints `spec_sha256` and each set's `plan_id` and `plan_sha256` as JSON.
To hash another spec file, pass its path as the argument.

## Wan2.2-T2V-A14B

`wan.yaml` defines the Wan2.2-T2V-A14B text-to-video workload in the same schema and is loaded by the same `h3_spec.py`.
Its `frozen-3733` set reproduces PR #3733's `wan22-t2v-a14b-serving-smoke-v1` plan, whose hash is `1f1a093b391bfd94f64d6e492948989850508ac4dad472eeaa98785fd26327d6`; its `shared-v1` set runs every #3733 prompt with its #3733 seed and a second seed 100 higher (the second seeds are new, so each cites the #3733 line of the seed it derives from).
The `release-v1` set crosses the four recorded prompts with five cited seeds for 20 measured clips and one warmup on each engine.
Both engines receive every generation field (`frame_count` travels as `num_frames`); width, height, frame count and fps are checked on the clip, and the video-only model contract adds the audio-absence check.
`wan-vllm-omni-serving.md` records the pinned vLLM-Omni image, server command and parallelism that served this workload on 8x H100, and `../tests/fixtures/wan-vllm-omni-a8576ccb/` holds the request, server record and ffprobe from that run.
The Wan tests live in `tests/test_wan_spec.py` and `tests/test_wan_runner.py`; `.github/workflows/test-h3-video.yml` runs the whole `experimental/video-generation` suite on draft and ready pull requests.
