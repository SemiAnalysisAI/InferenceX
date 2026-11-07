# Video-generation harness

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
is mounted into the job as `/video-package`. H100, H200, B200 and MI355X select
`h3-v2.yaml/release-v1`: 20 distinct measured prompts at C1. See
[native video delivery](../inferencex-e2e/docs/video-generation.md) for
artifact identity, reuse validation and post-merge publication.

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
the plan hash. Media decoding verifies observable stream properties; preset,
CRF, exact encoder implementation, and cross-engine quality still require
separate evidence. Comparisons leave cross-engine quality unjudged, while
same-implementation numerical checks require matching sealed image, launch,
environment, GPU and algorithm identity.

[`specs/h3-i2va.yaml`](specs/h3-i2va.yaml) is the image-to-video version of that
contract: the same generation and encoding, plus one first-frame image per case.
The runner reads each image from `--image-dir` as `<sha256>.png`, refuses bytes
whose hash differs from the plan, and sends them inline: an SGLang `data:` URI
condition or a vLLM-Omni multipart `input_reference`. Stored request payloads
keep only the image hash. The runner also refuses to start unless the recorded
server launch turns off cross-request conditioning caches. Each record keeps the
request size and the time to send it. The media check records frame-0 PSNR and
SSIM against the input image without affecting validity. The native srt path
does not pass an image directory yet.

## Local checks

```bash
cd video-generation
PYTHONPATH=../inferencex-e2e uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pyyaml>=6,<7' --with 'pytest>=8,<9' python -m pytest -q
```

[Test Video Harness](../.github/workflows/test-video-harness.yml) runs these
CPU checks on relevant changes. They make no model, scheduler or GPU calls;
sweep artifacts are the acceptance evidence.
