# Native video delivery

The `video-generation` scenario schedules one server job per deployment. SGLang
selects C1 deployments on 1, 2, 4 and 8 B200, H200 or MI355X GPUs (Ulysses
across all GPUs) and on 2, 4 and 8 H100 or A100 GPUs (TP2, since the model does
not fit one 80 GB GPU), and vLLM-Omni the same B200 and A100 deployments, all using
`h3-v2.yaml`: 20 distinct measured prompts, a separate warmup, 49 denoising
updates, and 124 frames at 24 fps and 1344×768. A five-second request produces
124/24 seconds of video. Config presence is not GPU or quality qualification.
Each deployment's image pin is its `*-video` entry in
[`configs/nvidia-master.yaml`](../configs/nvidia-master.yaml) or
[`configs/amd-master.yaml`](../configs/amd-master.yaml).
Actual step execution, media,
and power must be checked in the original run. The YAML workload has no `model.path`
or Python field: the binder derives the pinned model snapshot path from `MODEL`,
`HF_HUB_CACHE`, and the model revision. The service recipe selects the engine executable.
Native power is accepted only when its sampled GPU UUIDs match the server's
`spec.json` and `launch.json` assignment. On ROCm the server names each GPU by
its board serial, which is how the AMD power exporter identifies it.
The H100 runner provides an HF hub cache mount, but the pinned H3 snapshot must
be verified there before execution. Historical artifacts remain evidence of their
original workloads.

Each video service uses `command: [python3, -m, infx.bench, video, serve]`.
Its `args` select `--model-view snapshot` or `--model-view h3-fl2va`, followed
by `--` and the complete engine executable and flags. TP, Ulysses, offload,
compilation, host, and port are explicit recipe settings. `services[].env`
owns engine environment values, including offline mode and vLLM storage and
worker settings. The binder substitutes `{model_path}`, `{model_id}`,
`{model_revision}`, and `{gpus_per_node}` in service args/env, and rejects
model, revision, port, or GPU-count mismatches before submission. Other native
srt-slurm service placeholders retain their existing meaning. For a schema version 2
fairness plan, the binder also requires SGLang's `--disable-conditioning-cache`
or vLLM-Omni's `OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0` in the selected recipe.
It forwards every point's image as `VIDEO_CONTAINER_IMAGE`; the server refuses an
image without a SHA256 digest and records the digest in its launch receipt.

The launch receipt records recipe-wide `environment` and server `env` values
after binding, with server values taking precedence. Use these fields only for
nonsecret settings: credential-like names and launcher-owned workload, device,
service-discovery and power variables are rejected before submission. Inherited
credentials are not copied into the receipt. The rendered recipe fingerprint
includes the declared settings.

The Python entrypoint validates the workload and GPU identity, prepares an
explicitly selected model view, records the final command, then starts it.
It adds no engine flags or performance environment defaults. The SGLang child
checks the assigned UUID order before importing its engine. The previous B200
`frozen-c124` four-clip JSON and YAML artifacts remain readable as historical evidence.
The App shows their four-sample cells in History, below its ten-valid-sample chart
floor.
Each recipe's `readiness.timeout_seconds` sets how long the server may load and
warm up.

The shared [`configs/srt-recipes/video.yaml`](../configs/srt-recipes/video.yaml)
block runs `python3 -m infx.bench video client` as the custom benchmark for every
recipe. It resolves the native service IP and measurement-window directory at
runtime, preserving per-concurrency outputs and native power windows. Its request
timeout and environment are supplied by that block.

`perf-changelog.yaml` → `infx.matrix.plan` → `run-sweep.yml` →
`benchmark-tmpl.yml` → `infx.launch` → native srt-slurm service/client → result bundle.
A video canary runs one complete deployment and is removed from the remaining
matrix. `full-sweep-fail-fast` retains the repository's matrix-scoped fail-fast.
Video's scheduler allocation and workflow timeout, including preparation and
cleanup, are set in
[`benchmark-tmpl.yml`](../../.github/workflows/benchmark-tmpl.yml); the caller's
video limit overrides cluster profile defaults. Each request retains its explicit
timeout. No automatic retry is added.

Artifact names are `h3-results-<run>-<attempt>-<recipe-fingerprint>`; the full
fingerprint separates hardware, engines and deployments. `manifest.artifact_name`
and `ci.artifact_name` preserve the exact name. `manifest.git_commit` is the executed
checkout; `manifest.ci.head_sha` is the PR head and can differ from that checkout.
The selected `plan_sha256` and `engine.name/revision` come from per-cell run records.
Every job also retains its bound plan and engine identity before startup; failed
startup bundles retain that declaration and cannot pass qualification.
Power producer/exporter provenance is copied before the bundle is hashed.

The `video-sweep-manifest` artifact retains the complete selected matrix for the
current attempt and is replaced on rerun; earlier result bundles remain immutable.
Only full video sweeps upload this reuse marker. PR
validation and merge ingestion require every selected deployment and concurrency
from each deployment's newest bundle up to the trusted run attempt, consistent CI identity, valid native power, the
frozen plan’s full measured request population and matching media/file hashes. This checks retained evidence; it does not
rerun media decoding or establish perceptual quality, comparison sample sufficiency,
or production acceptance. Old unsuffixed bundles remain readable by the App but
do not qualify this new native reuse path. Video append-only sweeps are unsupported;
use the regular sweep so one server's complete workload stays together.

After authorized reuse and merge, Merge Ingest validates source artifacts and
sends `publish-video-results` with the original `source_run_id` to the App's media
publisher. Mixed runs separately retain their existing LLM/AgentX ingest event.
Verify the downstream publication and playback before claiming delivery. The
trusted main-branch `/use` handler must support video artifacts before accepting
the first native video reuse request; a PR-local checker cannot bypass that gate.

## Contribute a video deployment

To add or change a deployment, follow the
[H3 submission guide](../../video-generation/SUBMITTING.md).

### Render a deployment locally

Render locally from `inferencex-e2e/`, using the repository's pinned srt-slurm
submodule and locked dependencies. This writes files only and submits no job:

```bash
recipe_output=$(mktemp -d)
uv run --locked --extra recipes infx generate \
  --config-key h3-bf16-h100-sglang-video --output-dir "$recipe_output"
uv run --locked --all-extras --group test python -m infx.matrix.changed \
  --base origin/main --json
```

Inspect `manifest.json` and its generated YAML for the selected config, image,
model, GPU count, full command, environment and client concurrency. The recipe
fingerprint binds the matrix row and rendered recipe; a successful render
proves schema/binding, not model execution. Run the focused binder, entrypoint,
matrix and result tests for changed behavior.
