# H3 workload spec

`h3.yaml` defines the MiniMax-H3 text-to-video-and-audio workload that SGLang and vLLM-Omni both run.
`h3_spec.py` loads and validates it, then builds the plan object that the runner from PR #2916 executes and hashes.
`h3-vllm-omni-serving.md` records the pinned vLLM-Omni image, server command, parallelism and attention backend that served this workload on 2x H100.

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
uv run --no-project --python 3.12 --with 'pyyaml>=6,<7' --with 'pytest>=8,<9' python -m pytest experimental/video-generation/tests/test_h3_spec.py -q
uv run --no-project --python 3.12 --with 'pyyaml>=6,<7' python experimental/video-generation/specs/h3_spec.py
```

The second command prints `spec_sha256` and each set's `plan_id` and `plan_sha256` as JSON.
To hash another spec file, pass its path as the argument.
