# Native video delivery

The `video-generation` scenario schedules one server job per deployment. H100,
B200 and MI355X SGLang select four-GPU C1 deployments using
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
A recipe that enables a lossy optimization, such as Cache-DiT or sparse attention,
declares it in a top-level `optimizations:` list of App registry keys (`cache-dit`,
`subblock-sparse-attn`). The binder rejects a key the runtime's work attestation
cannot prove, so a misspelled key fails before submission. It passes the list to
the client, which records it as `configuration.optimizations` in each cell's
`run.json`, so the App draws the run apart from the lossless deployment. A lossy
deployment keeps the lossless key's image, layout and workload under its own key,
so its clips can be scored against the lossless clips.

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

For an existing video runtime/API, change its native recipe and master config rather
than adding a launcher or modifying the Python client. The configured deployments
are the `*-video` keys in [`configs/nvidia-master.yaml`](../configs/nvidia-master.yaml)
and [`configs/amd-master.yaml`](../configs/amd-master.yaml);
each names its recipe directory under
[`benchmarks/single_node/srt-slurm-recipes/h3/`](../benchmarks/single_node/srt-slurm-recipes/h3/).
Their presence does not establish final-head GPU qualification.

1. Choose an existing combination and keep its frozen workload, model revision,
   concurrency population and measurement boundary. Put the full engine command
   after `services[].args`' `--`; put nonsecret process settings in `services[].env`
   or recipe-wide `environment`. Keep the native serve/client entrypoints and
   telemetry binding. A new model, engine API, topology or model-view format
   needs maintainer adapter validation before it becomes a recipe-only contribution.
2. Change the existing row in its master config, or copy it with a
   distinct key and recipe for a deployment that must remain separately selectable.
   Pin the image by digest and use its real interpreter and architecture. The
   H100/B200 SGLang rows use the public CUDA amd64 image and the MI355X row its
   ROCm MI35x build. A shared image does
   not qualify a different topology or SKU: validate
   the selected deployments through their sweep. Maintainers stage model snapshots in the runner's
   existing HF cache; contributors do not patch the cache or SSH extra launch flags.
3. Render locally from `inferencex-e2e/`, using the repository's pinned srt-slurm
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
4. Append the PR's single `perf-changelog.yaml` entry with every affected config
   key and a `scenario-type` that lists `video-generation`. The planner rejects an
   entry whose named video config would otherwise plan no jobs. Follow [the contribution workflow](../../CONTRIBUTING.md) for review,
   CODEOWNER sign-off, the primary sweep label and trusted reuse. A maintainer
   controls fork scheduling. Use the final PR-selected matrix and preserve failed
   attempts; an ad-hoc diagnostic cannot replace its qualifying sweep.
5. Inspect the original run artifacts. `video-sweep-manifest` identifies the
   selected attempt and deployments. Each `h3-results-*` bundle retains
   `manifest.json`, `ci.json`, `SHA256SUMS` and `gpu/c*/` request, media and power
   evidence. Each cell's `attestation.json` checks, from the engine's own startup
   arguments and log, that every request ran the declared shape, seed and denoising
   steps without quantization or LoRA, and that the log shows exactly the run's
   declared optimizations running, so undeclared step caching or sparse attention
   fails; a failed attestation leaves the cell unverified. A valid `release-v1`
   deployment needs 20 valid measured C1 clips plus its separate warmup.
   Four-sample B200 cells remain History evidence below the App's ten-valid-sample
   chart floor.

During execution, native `recipe.lock.yaml` and the scheduler job record identify
the submitted configuration. Inspect `logs/video/server/launch.json` for the actual
argv/environment and assigned GPU UUIDs; `spec.json` retains the selected model
and workload. Service startup failures are in `logs/service_video-server.out` and
`logs/sweep_<job>.log`. Client failures remain in each cell's `baseline/run.json`;
native power failures retain `power-validation.json`. Keep run ID, attempt, source
head, executed checkout, recipe fingerprint and plan hash distinct when reporting
a problem. Never hand-edit a bundle to make its result eligible.

After authorized merge and publication, use the App's published History and
read-only video API to retrieve the original source. An unpublished source is not
an official result, and opening a comparison must not publish it. Follow the
publication handoff above rather than copying media by hand.
