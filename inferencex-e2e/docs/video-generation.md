# Native video delivery

**English** | [中文](video-generation_zh.md)

The `video-generation` scenario schedules one server job per deployment. H3 SGLang
has active H200 and B200 configs. Each serves C1/C2/C4 with four measured requests
and one warmup per cell: 12 measured requests and three warmups per SKU. The frozen
workload is 107 frames at 24 fps, 1344×768. Config presence is not GPU qualification.
H100 is excluded pending checkpoint staging/write-access validation; MI355X is
excluded because the prepared CUDA image cannot run on AMD. Historical artifacts
remain evidence of their original workloads.

`perf-changelog.yaml` → `infx.matrix.plan` → `run-sweep.yml` →
`benchmark-tmpl.yml` → `infx.launch` → native srt-slurm service/client → result bundle.
A video canary runs one complete deployment and is removed from the remaining
matrix. `full-sweep-fail-fast` retains the repository's matrix-scoped fail-fast.
Video uses a 75-minute scheduler allocation and a 90-minute workflow timeout,
including preparation and cleanup; the caller's video limit overrides cluster
profile defaults. `job_seconds` in the legacy GPU spec does not bound the native
matrix client. Each request retains its explicit timeout. No automatic retry is added.

Artifact names are `h3-results-<run>-<attempt>-<recipe-fingerprint>`; the full
fingerprint separates hardware, engines and deployments. `manifest.artifact_name`
and `ci.artifact_name` preserve the exact name. `manifest.git_commit` is the executed
checkout; `manifest.ci.head_sha` is the PR head and can differ from that checkout.
The selected `plan_sha256` and `engine.name/revision` come from per-cell run records.
Selected YAML workloads also retain their bound plan and engine identity before
startup; failed startup bundles retain that declaration and cannot pass qualification.
Power producer/exporter provenance is copied before the bundle is hashed.

The `video-sweep-manifest` artifact retains the complete selected matrix for the
current attempt and is replaced on rerun; earlier result bundles remain immutable.
Only full video sweeps upload this reuse marker. PR
validation and merge ingestion require every selected deployment and concurrency
from the trusted run attempt, consistent CI identity, valid native power, the
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
