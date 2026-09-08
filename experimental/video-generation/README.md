# H3 video CI smoke

**English** | [中文](README_zh.md)

This experimental lane runs the existing H3 supervisor inside InferenceX CI on
SemiAnalysis H200 resources. Its first target is a bounded same-build smoke:
original generated MP4s, full video/audio validation, measured requests, and
verified cleanup. It does not publish a native InferenceX database/UI result.

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

The adapter recovers task-owned allocation receipts before allocating. Imported
receipts must match task identity, Unix ownership, and the scheduler's exact
allocation identity; ambiguous intent blocks another submission. The fixed site
is `main` / `sa-shared`. A new exclusive allocation reserves eight GPUs;
the example step selects four GPUs and 32 CPUs. Charge reserved capacity.
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

## Results and local checks

Every attempt uploads `h3-video-<run-id>-<attempt>` for 14 days, with compression
disabled for media. The upload runs even after failure and includes the complete
adapter evidence: receipts, original MP4s, telemetry, checksums, and the portable
report when available. Compiler cache subtrees stay on persistent storage and
are excluded from the upload. Missing reports or media remain missing; fixtures never
replace them. Persistent source evidence remains at the configured workspace.
Retain/download the complete artifact before GitHub retention expires.

The CLI returns 0 only for a completed, verified smoke; 1 for a detected regression in verified evidence;
and 2 for inconclusive infrastructure or an unmet regression gate. A green smoke
does not turn an uncalibrated inner gate into an accepted regression. Keep
measurement completion, regression decision, and workflow provenance separate.

```bash
cd experimental/video-generation
uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pytest>=8,<9' python -m pytest -q
bash -n runtime-entry.example.sh
```

[Test H3 Video](../../.github/workflows/test-h3-video.yml) runs these CPU checks
and workflow linting on relevant changes. They make no model, scheduler, or GPU
calls. Real CI execution and artifact inspection are separate acceptance evidence.
