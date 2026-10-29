# H3 video-generation harness

**English** | [中文](README_zh.md)

This package is the client and server side of the `video-generation` benchmark
scenario: MiniMax-H3 served by one diffusion engine on one node, the selected
frozen YAML workload sent at every configured concurrency, each generated clip
decoded and validated, and every cell's
measured window published to srt-slurm's native GPU power telemetry.

InferenceX owns scheduling and results. `inferencex-e2e/configs/nvidia-master.yaml`
declares the deployments (`h3-bf16-<sku>-<engine>-video`), `infx.matrix` plans
them from `perf-changelog.yaml`, `run-sweep.yml` runs the `video` bucket through
`benchmark-tmpl.yml`, and `python -m infx.launch run` takes the `srt-video` path:
the SKU recipe under `benchmarks/single_node/srt-slurm-recipes/h3/` starts the
engine through `python3 -m infx.bench video serve` as a generic service and runs
`python3 -m infx.bench video client` as `benchmark.type: custom`. This directory
is mounted into the job as `/h3-package`. H100, H200 and B200 select
`h3-v0.1.yaml/release-v1`: 20 distinct measured prompts at C1. GB200 retains
`h3.yaml/release-v1` with its separate 107-frame contract. See
[native video delivery](../../inferencex-e2e/docs/video-generation.md) for
artifact identity, reuse validation and post-merge publication.

## Entry points

`python -m evaluator.cli` (`vgbench`) exposes the commands the native entry calls:

- `srt-yaml-serve`: load the selected spec and set from `specs/`, check its
  `plan_sha256` against the bound value, record the sealed server spec and
  launch, and exec the engine (`sglang` or `vllm-omni`) on the job's GPUs.
- `srt-yaml-client`: run the same plan at every `CONC_LIST` concurrency against
  the service endpoint, one cell per concurrency under `gpu/c<N>/`, and publish
  each cell's measured window.
- `srt-finalize`: join the cells, their power validation sidecars and the
  server record into the `h3-results-<run>-<attempt>-<recipe-fingerprint>`
  bundle the dashboard ingests.

`specs/h3_spec.py` loads the YAML workload, validates it and hashes the
selected plan; `infx.srt_slurm.video` imports it when binding a point.

[`specs/h3-v0.1.yaml`](specs/h3-v0.1.yaml) supplies a separate 20-prompt,
5-second H3 contract under the `release-v1` selector. It requests 49 denoising
updates, expects 124 delivered frames, and uses a disjoint warmup. The runner
sends each engine's explicit H.264 options and records the encoding policy in
the plan hash. Media decoding verifies observable stream properties; preset,
CRF, exact encoder implementation, and cross-engine quality still require
separate evidence. Comparisons leave cross-engine quality unjudged, while
same-implementation numerical checks require matching sealed image, launch,
environment, GPU and algorithm identity.

## Local checks

```bash
cd experimental/video-generation
PYTHONPATH=../../inferencex-e2e uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pyyaml>=6,<7' --with 'pytest>=8,<9' python -m pytest -q
```

[Test H3 Video Harness](../../.github/workflows/test-h3-spec.yml) runs these
CPU checks on relevant changes. They make no model, scheduler or GPU calls;
sweep artifacts are the acceptance evidence.
