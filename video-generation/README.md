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

## Paired lossless quality

A deployment that lists lossy optimizations in `run.json`
`configuration.optimizations`, such as `cache-dit`, is scored against its lossless
twin instead of the Diffusers reference. Dispatch Video Quality with
`baseline_run_id` set to the run that holds the lossless deployment. It may be the
same run, but a deployment never pairs with itself. The job downloads the newest
`h3-results-*` bundles of both runs and runs:

```bash
python -m evaluator.cli score-quality <artifact-id>=<bundle.zip|dir> ... \
  --baseline <artifact-id>=<bundle.zip|dir> ... --baseline-run-id <run> \
  --run-id <run> --source-revision <git-sha> --output <new-dir>
```

Only baseline deployments without optimizations are baselines. A deployment pairs
with the one that has the same hardware, engine, image digest, runtime revision, GPU
count, precision, concurrency and `spec.json` `server` flags. Only the flags its
declared optimizations set may differ: `OPTIMIZATION_SERVER_KEYS` in
[`evaluator/quality.py`](evaluator/quality.py) lets `subblock-sparse-attn` change
`attention_backend`. Each clip pairs with the same case and repetition and is scored
only when the prompt, seed, generation settings and sealed request hash are equal.
Other clips are listed under `unscored` as `no_baseline`, `ambiguous_baseline`,
`invalid_baseline`, `settings_mismatch` with the differing fields,
`invalid_candidate` or `checksum_mismatch`.

SSIM and LPIPS take about 55 s per clip on a GitHub-hosted runner, and
byte-identical frames skip both. Before scoring, the job refuses a dispatch whose
differing clips would need more than 100 of its 120 minutes (`--budget-minutes`,
about 109 clips), so it fails with a message instead of timing out with no artifact.

Each pair gets per-frame PSNR, SSIM on BT.601 luma with a Gaussian window of
σ 1.5, and LPIPS (AlexNet, v0.1, weight hashes pinned), with the clip mean and
the worst frame of each. Each deployment gets the median and worst clip. An
identical clip has `video_identical` true, PSNR `null`, SSIM 1 and LPIPS 0. The
`h3-paired-quality-<run>-<attempt>-<baseline-run>-<baseline-attempt>` artifact holds `quality.json`
with `mode: paired_lossless`, and `quality_status` stays `unjudged`. Reference
scoring stays PSNR only, and its artifact name is unchanged.

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
