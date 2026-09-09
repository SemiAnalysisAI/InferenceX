# H3 video CI smoke

**English** | [中文](README_zh.md)

This experimental lane runs the existing H3 supervisor inside InferenceX CI on
SemiAnalysis H200 resources. Its first target is a bounded same-build smoke:
original generated MP4s, full video/audio validation, measured requests, and
verified cleanup. It does not publish a native InferenceX database/UI result.
Successful executions also publish a [versioned frontend result contract](RESULTS.md)
with validated GPU power/energy, original measurements and a portable power report.

The runner supports two frozen 16:9 cells at 1344×768 and 24 FPS: a 4-second
request resolves to 107 frames, while an 8-second request resolves to 192 frames.
These counts follow the pinned H3 runtime's temporal rounding. Freeze a new plan
for a changed prompt or duration; retain earlier runs as their original cells.

A successful smoke job means the configured measurements and evidence completed.
Its uncalibrated regression decision remains inconclusive and
`ci_accepted: false`. No successful H3 run is established by adding these files
or passing CPU tests.

## Prepare the existing runtime

Set repository variable `H3_SITE_CONFIG` to the absolute path of a reviewed JSON
file on the `cluster:h200-dgxc` submission runner. Start from
[site.example.json](site.example.json); its placeholders are not executable.
The configuration binds the persistent workspace, existing rootfs/readiness
record, entry-only script and SHA256, container Python, frozen supervisor spec
and SHA256, resource limits, task identity, and optional prior allocation receipts.
The spec must record actual compute/model-use approval. Selecting the manual H3
route requests only that configured, reviewed workload; dispatch accepts no shell
command, model path, arbitrary config contents, or alternate provider.

The runner requires Python 3.11+, Git, the Slurm tools, and access to the declared
shared paths. PyAV/NumPy and the pinned H3 runtime/model must already be prepared
inside the existing runtime. A rootfs-created marker alone is not proof of model
compatibility. The adapter checks saved preparation and input identity before
allocation; it does not install packages, import an image, or create a rootfs.

Adapt [runtime-entry.example.sh](runtime-entry.example.sh) from the saved working
entry command, then pin its digest. It must enter the existing Enroot rootfs,
map `workspace.host` to `/work`, preserve the Slurm step's GPU/CPU binding and
metadata, forward its command, and propagate its exit code. It translates
`SLURM_STEP_GPUS` global IDs to physical UUIDs, exports
`H3_ASSIGNED_GPU_UUIDS`, and retains the original mask in
`H3_ORIGINAL_CUDA_VISIBLE_DEVICES`. In-container admission checks the actual
driver UUIDs against that assignment. Do not invoke an old allocating launcher
as the entry script.

The pinned SGLang runtime expects numeric device IDs. Each H3 child receives the
selected devices' observed NVML indices, then verifies their ordered CUDA driver
UUIDs before importing SGLang. An enumeration mismatch fails startup; ownership
locks and telemetry continue to use the assigned UUIDs.

The adapter recovers task-owned allocation receipts before allocating. Imported
receipts must match task identity, Unix ownership, and the scheduler's exact
allocation identity; ambiguous intent blocks another submission. The fixed site
is `main` / `sa-shared`. A new exclusive allocation reserves eight GPUs;
the example step selects four GPUs, 32 CPUs, and 1 TiB of host memory. The pinned
four-rank loader exceeded 256 GiB during CPU weight staging; 1 TiB is a tested
working allowance, not a measured minimum. Charge reserved capacity.
`resources.minutes` is the total allocation cap, at most 90 minutes. The step
reserves five minutes for outer cleanup, and the supervisor plus ten minutes
must fit the allocation. For example: 90-minute allocation, 85-minute step,
75-minute supervisor. Reused allocations need enough remaining time. Preserve
the prepared rootfs; clean only owned processes/steps and release only allocations
this execution owns.

## Dispatch through InferenceX

The first run belongs in CI. For a feature branch, use the already registered
End-to-End Tests workflow, which calls the branch's reusable H3 workflow:

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX \
  --ref feat/h3-video-ci -f h3-video=true -f test-name=h3-first-smoke
```

Review the branch before dispatch. The original actor and rerun actor must have
write, maintain, or admin permission. The H3 route checks out `github.sha`:
`--ref` selects its workflow and source; the ordinary LLM `ref` input is not
used. It skips LLM matrix generation, all dependent LLM sweeps/collectors, and
their success-rate calculation. External PR events cannot launch this route.
After the standalone workflow is registered on the default branch,
`h3-video.yml` can also be manually dispatched.

When priority scheduling is enabled, node-slot scheduling must also be enabled.
The queued job requests exactly `nodes:1` plus its native
`ci-job-<priority>-<token>` and `ci-attempt-<attempt>` labels on
`cluster:h200-dgxc`. If priority scheduling is disabled, it follows the
repository's native cluster-label route. GitHub admission and Slurm resource
verification remain distinct. This lane uses native workflow permission and
scheduler admission; it does not add an OIDC service or claim independent
hardware attestation.

To export already accepted H3 evidence without repeating generation, supply one
or two source run IDs through the same trusted route:

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX \
  --ref feat/h3-video-ci -f h3-video=true \
  -f h3-reuse-run-ids=34291306687,34293342829 -f test-name=h3-power-export
```

The sources must be successful manual executions in this repository. CI verifies
their commit, original artifacts and persistent Slurm receipts. It inventories
the same node and GPU UUIDs using the existing runtime, with a fixed ten-minute
allocation cap (at most 1.3333 reserved GPU-hours), then releases its allocation.
This records current hardware identity and power limits without loading H3.
Later limits cannot establish historical generation settings. Existing task
allocations are checked for reuse before requesting a new one.

## Results and local checks

Every attempt uploads `h3-video-<run-id>-<attempt>` for 14 days, with compression
disabled for media. The upload runs even after failure and includes the complete
adapter evidence: receipts, original MP4s, telemetry, checksums, and the portable
report when available. Compiler cache subtrees stay on persistent storage and
are excluded from the upload. Missing reports or media remain missing; fixtures never
replace them. Persistent source evidence remains at the configured workspace.
Retain/download the complete artifact before GitHub retention expires.

The hosted export job publishes `h3-results-<run-id>-<attempt>` containing
`index.json`, the JSON schema, bilingual metric definitions, and one source
subdirectory per original execution. Each source contains `result.json`, original
media/logs/report, per-GPU power series, phase integration/coverage and
`power-report.html`. Reprocessing also publishes `h3-hardware-<run-id>-<attempt>`;
the verified raw inventory is copied into each result. Original CI identities and
checksum seals are preserved separately from exporter identities and new seals.
Missing or invalid telemetry withholds power; export failures retain error logs
and return an unsuccessful status. Original workload failures still upload their
raw evidence even when export cannot start.

In smoke mode, exit 0 means both roles completed every planned warmup and
measurement with verified timing, fresh valid media, and clean teardown. Exit 1
means a completed workload contains an invalid outcome; exit 2 means execution
or evidence verification failed. Latency/fidelity thresholds remain separate:
the report can show a failed or inconclusive comparison after a successful smoke.
Regression mode additionally requires the existing calibrated acceptance gate.

```bash
cd experimental/video-generation
PYTHONPATH=../.. uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pytest>=8,<9' --with 'jsonschema>=4,<5' python -m pytest -q
bash -n runtime-entry.example.sh
```

[Test H3 Video](../../.github/workflows/test-h3-video.yml) runs these CPU checks
and workflow linting on relevant changes. They make no model, scheduler, or GPU
calls. Real CI execution and artifact inspection are separate acceptance evidence.
