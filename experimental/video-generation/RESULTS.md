# H3 backend result contract

**English** | [中文](./RESULTS_zh.md)

`result.json` is the frontend entry point for a downloaded H3 CI artifact. Its
`schema_version` is **1.0.0**; [result.schema.json](./result.schema.json) describes
its external shape. Reject unknown versions. This contract does not ingest a
result into the InferenceX database or qualify a release.

The bundle retains original media, raw request records and telemetry, runtime
and client logs, model/runtime identities, Slurm receipts, and the existing
portable `report/index.html`. The exporter adds `result.json`, timestamped
`power/baseline.json` and `power/candidate.json`, and `power-report.html`.
Original files are not rewritten. The publisher refreshes `SHA256SUMS` after
export and preserves the source checksum file separately when reprocessing.
All media/report references are relative to the artifact root and include
SHA256. Unzip the whole artifact before opening either report.

## Reading a result

1. Check `status` and `invalid_reasons`. An invalid artifact writes a failed
   result and raises an error so CI can preserve logs and fail the export.
2. Read `workload_status` separately from `regression_status`. Valid A/A clips
   with an uncalibrated policy can finish successfully while regression is
   `inconclusive`. `release_qualified` is always false for this MVP.
3. Read each `roles.<role>.metrics.status` and each power phase's `valid` flag.
   A complete export can contain unavailable power. Invalid power/energy is
   null, never zero. A phase is not made valid by another phase passing.
4. Use `execution.ci` for the **original GPU execution**, including its commit,
   run and attempt. `producer` identifies the exporter commit/current CI and
   source hashes. Reprocessing an artifact does not create new GPU measurements.
5. `hardware.selected_gpu_count` is the measured device set;
   `reserved_gpu_count` comes from Slurm `AllocTRES`. Four selected GPUs in an
   eight-GPU allocation means four-board power and eight-GPU compute billing.

`workload.plan` freezes prompts, seeds, clip geometry/duration/audio format,
steps, repetitions and warmup count. `workload.server` retains runtime settings.
`execution` joins CI, Slurm, GPU UUIDs, model manifest and observed runtime source.
The exporter checks the original complete checksum inventory when present,
rejects unsafe paths/symlinks, reuses `verify_measurement_job`, and verifies the
original report's local references. This is verification of trusted-runner
records and media hashes; the exporter does not rerun media decoding or provide
independent hardware attestation. The earlier full-stream analyses remain bound
to their original media bytes. Trusted GitHub metadata, when supplied, must join
the successful H3 job, run, attempt, URL and execution commit. During same-run
export, only that exact exporter run/attempt/repository/commit may still be
`in_progress`; `workflow_status_at_export` records this pending container workflow
while the completed H3 job and downloaded artifact identities are checked.

## Metric definitions

| Metric | Unit and boundary | Validity and limits |
| --- | --- | --- |
| Request latency | Seconds from submission through downloaded and technically validated media | Valid measured clips only; startup and warmup excluded. Retains terminal, download and validation timings. |
| Valid clips/sec | Valid measured clips / serial measured-block wall seconds | Wall time includes failed attempts; not concurrent saturation capacity. |
| Completion | Scheduled, attempted, completed, valid, failed, not-started clips | Completed can still be technically invalid. Warmup records remain separate. |
| GPU memory | Observed device-used MiB per selected UUID | Existing role-wide and client-including-warmup peaks retain their original boundaries; not exact allocator peaks. |
| Technical integrity | Full-stream video/audio checks with per-check units | Decode, geometry, duration, timestamps/cadence, motion and sound defects; not semantic or perceptual quality. |
| Paired fidelity | Video PSNR dB, audio spectral cosine and absolute RMS ratio error | Matching original decoded outputs; exact video match has null finite PSNR and `exact_match=true`. |
| GPU power | Timestamped W per GPU and summed selected-GPU W | Board sensor readings, including device memory; exclude host power and unselected GPUs. |
| Average / observed peak power | Integrated J / phase seconds; maximum in-window sensor W | Averages are time weighted. Peaks are sampled observations, not instantaneous electrical peaks. |
| GPU energy / valid clip | Trapezoidal integrated J / technically valid measured clips | Includes all attempted generation windows, even failed or invalid outputs; excludes download/local decoding. Null if any contributing window is invalid or valid count is zero. |

Each power file has versioned `sample_series`, `windows`, `phases`, `semantics`
and `clock_alignment`. Windows separate startup, each warmup, and each measured
submission-to-observed-provider-terminal interval. They retain the timing source
and uncertainty, exact monotonic bounds, per-GPU sample counts/gaps, covered
seconds/fraction, boundary bracketing, and invalid reasons. Phase aggregates
combine only when every requested contributing window is valid. Full time
series remain downloadable even when derived measurements are withheld.

Integration uses the shared InferenceX trapezoidal power integrator with linear
boundary interpolation and no extrapolation. UUID/ownership, finite readings,
ordered timestamps, phase overlap, clock agreement and maximum-gap checks gate
power. The allowed gap is `3 × requested sampling interval` (3 seconds for these runs).
Legacy UTC event reconstruction must agree with recorded monotonic durations;
its startup window can be withheld when a boundary is not covered. H200 NVML
power readings have a trailing averaging window, so phase edges also have sensor
averaging uncertainty. No energy counter is inferred from sampled watts.

## TDP and architectural claims

A generic `NVIDIA H200` name does not prove SXM form factor or 700 W TDP.
`hardware.tdp` stays unavailable until an explicitly verified, sourced hardware
profile joins the same physical UUIDs. When supplied, the exporter records
measured mean/observed-peak fractions of aggregate specification TDP. These are
descriptive ratios, not a configured power limit or a calibrated decision gate.

A later read-only inventory remains under `later_hardware_observation`, with its
own CI/Slurm identity and time. Its configured/default/enforced/maximum power
limits **do not backfill historical generation settings**. Historical limits
remain unavailable when the original run did not record them. New executions
retain their own per-role `configured_power_limits.by_role` before/after snapshots
of configured, enforced, default and maximum W. Each snapshot has its observation
time and validity; a missing or mismatched UUID/value/time withholds that snapshot.
`same_observed_values` compares the endpoints only and never proves continuous
power-limit stability between them.

High observed H3 GPU-board power supports a statement about this workload and
hardware configuration only. Claiming architectural differences from LLMs still
requires matched hardware/topology, precision, power limits, sampling/windows,
warmup and load, including LLM prefill/decode separation and repeated comparable
measurements. These A/A points do not establish significance, general video
quality, performance improvement, or release readiness.

## Export API

From the repository root, put the repository and `experimental/video-generation`
on `PYTHONPATH`, then call:

```python
from pathlib import Path
from evaluator.mvp_result import write_result

write_result(
    Path("/absolute/path/to/copied-source-artifact"),
    producer={"git_commit": "<40-character exporter commit>", "ci": {"run_id": "<export CI>"}},
    source_ci=trusted_github_run_metadata,
    hardware_profile=optional_later_inventory,
)
```

Use a fresh copy. `source_ci` uses GitHub CLI fields `databaseId`, `runAttempt`,
`headSha`, `url`, `status`, `conclusion`, and `jobs`; obtain them independently of
the artifact. The optional profile must bind the same GPU UUIDs, and a verified
TDP needs `status`, `watts_per_gpu`, `hardware_variant`, `source_url`, and evidence.
The publisher owns final checksums, upload, and download acceptance. Do not
modify old CI or measurement identities to label an export as a new benchmark.

## Observed A/A evidence

These results reuse the retained eight-second clockwork-fox execution from
[CI run 34293342829](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34293342829),
commit `65699f7c6`, Slurm **82261.0**, with
[original media, telemetry and report](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34293342829/artifacts/10082823150).
Four selected H200 GPUs ran the same runtime revision sequentially: one warmup
and one measured clip per role.

| Eight-second measured clip | Baseline | Candidate |
| --- | ---: | ---: |
| Submit-to-validated-media latency (s) | 149.768568 | 149.355090 |
| Aggregate mean GPU-board power (W) | 2737.371816 | 2731.735261 |
| GPU energy per valid clip (J) | 406794.062298 | 404620.528709 |
| Sampling coverage fraction | 1.0 | 1.0 |
| Maximum observed sampling gap (s) | 1.3303 | 1.3567 |

Per-GPU mean power was approximately **680–688 W**. Both measured generation
windows were bracketed, with maximum gaps below the 3-second validity limit.
Power and energy cover submission to observed provider completion; the latency
row also includes media transfer and validation. These are different boundaries.
TDP ratios remain pending a verified hardware profile, and the original
executions did not record their configured power limits.

The earlier four-second A/A execution remains available in
[CI run 34291306687](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34291306687),
Slurm **82260.0**, and its
[original artifact](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34291306687/artifacts/10081961245).
Its baseline measured power and energy are withheld because the ending boundary
is not bracketed by telemetry. This does not invalidate the retained workload
execution or latency/media results.

Each duration is a separate frozen workload. Do not pool these durations or
interpret their latency difference as a regression. One measured clip per role
provides point estimates only; workload execution passed while regression
remains uncalibrated and inconclusive.
