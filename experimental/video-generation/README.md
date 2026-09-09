# H3 video CI smoke

**English** | [中文](README_zh.md)

This experimental lane runs the existing H3 supervisor inside InferenceX CI on
prepared SemiAnalysis NVIDIA resources. Its first target is a bounded same-build smoke:
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
command, model path, arbitrary config contents, or alternate provider. The
optional `h3-site-config` dispatch input selects an existing reviewed JSON file;
`h3-cluster` must match its declared site before any allocation.

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
allocation identity; ambiguous intent blocks another submission. The default H200 site
is `main` / `sa-shared`. An explicit `site` records the cluster, partition, account,
and expected GPU model. Currently admitted clusters are `h200-dgxc`, `h100-dgxc`,
and `b200-nscale`; admission is implementation support, not a completed hardware run.
`resources.allocated_gpus` records the full allocation separately from participating
`resources.gpus`; set it to eight on whole-node H100. A paired allocation reserves eight GPUs;
the example step selects four GPUs, 32 CPUs, and 1 TiB of host memory. The pinned
four-rank loader exceeded 256 GiB during CPU weight staging; 1 TiB is a tested
working allowance, not a measured minimum. Charge reserved capacity.
`resources.minutes` is the total allocation cap, at most 240 minutes. The step
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

To export accepted H3 evidence using a retained hardware inventory, supply one
or two H3 source run IDs and the inventory run ID:

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX \
  --ref feat/h3-video-ci -f h3-video=true \
  -f h3-reuse-run-ids=34291306687,34293342829 \
  -f h3-inventory-run-id=34297499754 -f test-name=h3-power-export
```

This path uses hosted CPU export and skips the native H200 job. The original
actor and rerun actor pass the same authorization checks. Hosted export
independently verifies the accepted H3 executions and the completed inventory
job, including artifact seals, Git/CI/Slurm identities, and the same physical GPU
UUIDs. It makes no new GPU queries or model requests. Keep the source artifacts
within GitHub's retention period.

Omitting `h3-inventory-run-id` records a new inventory through the native Slurm
route. CI verifies the source commits, original artifacts and persistent Slurm
receipts, then inventories the same node and GPU UUIDs using the existing runtime.
The allocation has a fixed ten-minute cap (at most 1.3333 reserved GPU-hours);
existing task allocations are checked for reuse first. The inventory loads no H3
model and releases its owned allocation after cleanup. Later power limits cannot
establish historical generation settings.

The retained [inventory run 34297499754](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34297499754)
([raw inventory artifact](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34297499754/artifacts/10083702100))
completed Slurm **82290.0** on `worker-10`, observing the original four UUIDs at
`2026-09-09T01:02:44Z`. Recorded NVIDIA H200 PCI device/subsystem IDs
`233510DE` / `18BE10DE` identify H200 SXM, with a manufacturer maximum configurable
TDP of **700 W per GPU**. At this later observation, configured, enforced, default
and maximum limits were all 700 W on all four devices. Historical generation
limits remain unknown. The original inventory profile's unknown classification
is preserved; the exporter classifies its original XML with the new producer
commit, accepting PCI IDs with or without `0x`, without querying hardware again.
See [the retained A/A measurements and their limits](RESULTS.md#observed-aa-evidence).

## Optional serving load

Add `"serving": {"concurrency": 2, "delivery_deadline_seconds": 300}` to the
reviewed supervisor spec and update its pinned SHA256 in the site configuration.
This example deadline is operator-selected, not a calibrated acceptance gate.
Concurrency accepts 1–32; omitting `serving` preserves serial regression behavior.
The direct client exposes the same options as `--serving-concurrency` and
`--delivery-deadline-seconds`; without `--execute`, it still only previews.

Each job measures one concurrency against one supervised endpoint. Workers
submit another request after downloading the previous output; media validation
runs separately. Warmup stays serial and separate. To compare loads, repeat the
same frozen prompt/seed/generation plan and runtime on separately recorded jobs,
including an explicit serving-concurrency-1 control. The default mode does not launch a load sweep. Reuse the existing CI allocation/runtime route and retain every
request outcome; uncertain remote completion stops new submissions.

This measures closed-loop delivery throughput, not a fixed arrival rate or
sustainable serving capacity. Server queue/execution timestamps, actual batch
sizes, multi-replica layouts and full deployment cost remain unavailable.
Serving runs require an uncalibrated policy. CPU fixtures test the harness;
they do not establish H3 concurrency support or hardware performance.

Set the reviewed site configuration to `"mode": "serving-smoke"` for the bounded
C1/C2/C4 matrix. Its plan contains 4–200 measured requests per cell, plus
explicit warmups. It boots the baseline runtime once per cell in one allocation
and stops after a failed cell. Allocation GPU count defaults to the participating
count; `resources.allocated_gpus` declares a larger required allocation explicitly.
Ordinary paired smoke keeps its existing allocation behavior. `serving-smoke.json`, `gpu/cN/` and `report/index.html` retain
the matrix, original request/media/telemetry evidence and playable report. An
interrupted attempt is counted separately from an unstarted request. This mode
skips the paired frontend export and cannot claim regression acceptance.

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
`power-report.html`. A new inventory job publishes `h3-hardware-<run-id>-<attempt>`;
CPU-only replay downloads the retained inventory instead. The verified raw
inventory is copied into each result. Original CI identities and
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

## Cross-hardware serving matrices

`h3-preflight-only=true` records the selected CI runner's identity, public SSH
host keys, and command availability without reserving or querying GPUs. Its
separate `h3-site-preflight` artifact is not benchmark or runtime qualification
evidence. It also accepts `mi355x-amds`; actual AMD generation remains unsupported.
Preflight cannot be combined with historical result reuse. Hardware sites have
independent workflow concurrency groups; each retains its existing Slurm checks.

The existing `serving-smoke` mode accepts 4–200 measured requests per concurrency
from `plan.cases × plan.repetitions`; concurrency remains 1, 2, and 4. Twenty per
cell produces sixty measured requests plus three separate warmups when
`warmup_runs: 1`. Failures and unstarted requests remain in the declared denominator.
Freeze identical model files, prompts/seeds, video settings, and quality requirements
across sites. Record different runtime builds and deployment topology explicitly.
Small-sample percentiles are preliminary; this closed-loop sweep does not establish
sustainable open-loop arrival capacity. AMD runtime/device admission is not yet
implemented. Missing sites and measurements must not be represented by fixture data.
