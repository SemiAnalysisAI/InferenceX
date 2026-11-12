# Submit an H3 video deployment

Follow these steps to add or change an H3 (VideoGenX) deployment, from choosing a
config to a published result. Each step links the section that holds its detail.
[Native video delivery](../inferencex-e2e/docs/video-generation.md) explains how the
sweep runs, what a result bundle contains, and how reuse is validated.

| Step | Owner |
| --- | --- |
| [1. Choose the deployment](#1-choose-the-deployment) | You |
| [2. Change the recipe and config row](#2-change-the-recipe-and-config-row) | You |
| [3. Check locally](#3-check-locally) | You |
| [4. Append the changelog entry](#4-append-the-changelog-entry) | You |
| [5. Open the PR and start the sweep](#5-open-the-pr-and-start-the-sweep) | You open the PR. A maintainer labels a fork PR. |
| [6. Confirm the run qualifies](#6-confirm-the-run-qualifies) | You |
| [7. Get review and sign-off](#7-get-review-and-sign-off) | One eligible CODEOWNER, then a core maintainer |
| [8. Reuse and merge](#8-reuse-and-merge) | An authorized maintainer |
| [9. Verify publication](#9-verify-publication) | Merge Ingest publishes. You verify. |

## 1. Choose the deployment

The configured deployments are the `*-video` keys in
[`nvidia-master.yaml`](../inferencex-e2e/configs/nvidia-master.yaml) and
[`amd-master.yaml`](../inferencex-e2e/configs/amd-master.yaml). Each key names its
recipe directory under
[`srt-slurm-recipes/h3/`](../inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/h3/).

Start from an existing combination. Keep its frozen workload, model revision,
concurrency population, and measurement boundary. To change a deployment, edit its
row. To add a deployment that must stay separately selectable, copy the row under a
distinct key with its own recipe. A new model, engine API, topology, or model-view
format needs maintainer adapter validation before it can be a recipe-only
contribution.

## 2. Change the recipe and config row

Change the native recipe and the master config row. Do not add a launcher or modify
the Python client. Keep the native serve and client entrypoints and the telemetry
binding.

- Put the complete engine command after the `--` in `services[].args`.
- Put nonsecret process settings in `services[].env` or the recipe-wide
  `environment`. [Native video delivery](../inferencex-e2e/docs/video-generation.md)
  lists the placeholders the binder substitutes and the names it rejects.
- Set `services[].env.VIDEO_RUNTIME_REVISION` to the engine's 40-character commit.
  The binder rejects any other value, and the server records it as the engine
  revision.
- Pin the image by SHA256 digest, and use its real interpreter and architecture.
  The A100, H100, H200, and B200 SGLang rows and the B200 and A100 vLLM-Omni rows use
  the public CUDA amd64 image. The MI355X row uses its ROCm MI35x build.

A shared image does not qualify a different topology or SKU. Each selected
deployment qualifies through its own sweep. Maintainers stage model snapshots in the
runner's existing Hugging Face cache. Do not patch the cache or pass launch flags
outside the recipe.

## 3. Check locally

Render the changed deployments and list the changed config keys as
[Render a deployment locally](../inferencex-e2e/docs/video-generation.md#render-a-deployment-locally)
describes. If you changed code under `video-generation/`, run its
[local checks](README.md#local-checks). For code under `inferencex-e2e/infx/`, run
the [repository local checks](../AGENTS.md#local-checks).

## 4. Append the changelog entry

Append one entry to the physical end of
[`perf-changelog.yaml`](../inferencex-e2e/perf-changelog.yaml). List every affected
config key, and include `video-generation` in `scenario-type`. The planner rejects
an entry whose named video configs would otherwise plan no jobs.

```yaml
- config-keys:
    - h3-bf16-h100-sglang-video
  scenario-type: [video-generation]
  description:
    - "<what changes in the deployment>"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

Leave `append-only` unset, because video runs only as a regular sweep. The
[performance changelog rule](../CONTRIBUTING.md#pr-review-flow) covers the one-entry
limit and byte preservation.

## 5. Open the PR and start the sweep

Write the title and body as [PR descriptions](../CONTRIBUTING.md#pr-descriptions)
requires. Then start the sweep.

- **Same-repository PR.** Apply exactly one primary label, normally
  `full-sweep-fail-fast`. `run-sweep.yml` then sweeps the changelog matrix.
- **Fork PR.** Mark the PR ready for review and resolve merge conflicts. A
  maintainer applies the label, which approves only the current head and dispatches
  the [trusted sweep](../.github/workflows/README.md#trusted-external-fork-sweep-dispatch-poc).
  After each push, the maintainer removes and re-adds the label.

Do not add `evals-only` or `agentx-fast`, because a run with either label cannot be
reused. [PR sweep labels](../.github/workflows/README.md#pr-sweep-labels) describes
every label and modifier.

## 6. Confirm the run qualifies

The qualifying run is a full sweep of the final changelog matrix on a commit in the
PR. Its `validate-video-results` job must pass. That job checks the
[reuse conditions](../inferencex-e2e/docs/video-generation.md) against the retained
evidence of every selected deployment. A valid `release-v1` deployment needs 20
valid measured C1 clips plus its separate warmup. Keep failed attempts. An ad hoc
diagnostic run cannot replace the qualifying sweep.

Inspect the run's artifacts:

- `video-sweep-manifest` identifies the selected attempt and deployments.
- Each `h3-results-*` bundle holds `manifest.json`, `ci.json`, `SHA256SUMS`, and the
  request, media, and power evidence under `gpu/c*/`.

Never hand-edit a bundle to make its result eligible. If a deployment fails, see
[Troubleshoot a run](#troubleshoot-a-run).

## 7. Get review and sign-off

The master configs and the recipes have vendor owners in
[CODEOWNERS](../.github/CODEOWNERS). One eligible CODEOWNER posts the
[PR Review Checklist](../CONTRIBUTING.md#the-pr-review-checklist-codeowner-sign-off).
A core maintainer gives final approval.

## 8. Reuse and merge

After the final sweep passes, an authorized maintainer comments `/use <run_id>` to
select it. [Reuse at merge](../CONTRIBUTING.md#reusing-your-prs-green-sweep-at-merge-with-use)
describes the rules.

- **Same-repository PR.** The maintainer merges with
  `infx.workflows.merge_with_reuse`.
- **Fork PR.** The maintainer pins the trusted `e2e-tests.yml` run by its ID, then
  merges. Automatic selection and `merge_with_reuse` find only `run-sweep.yml` runs.

## 9. Verify publication

On the merge to `main`, Merge Ingest publishes the reused run as
[native video delivery](../inferencex-e2e/docs/video-generation.md) describes.
Confirm that the post-merge jobs pass, as [After merging](../CONTRIBUTING.md#after-merging)
requires. Then use the App's published History and read-only video API to retrieve
the original source. An unpublished source is not an official result, and opening a
comparison must not publish it. Do not copy media by hand.

## Reference quality (PSNR)

After a same-repository Run Sweep finishes, the
[Video Quality](../.github/workflows/video-quality.yml) workflow scores each measured
clip against the public Diffusers reference videos. It uploads the scores as the
`h3-quality-<run>-<attempt>` artifact. It does not score a trusted fork sweep
automatically. A maintainer dispatches it with that run's `run_id` and
`run_attempt`.

PSNR measures how closely each clip matches the reference implementation's output
for the same prompt and seed. Other hardware and engines legitimately diverge from
that output, so the score is not a quality grade. `quality.json` keeps
`quality_status: unjudged` and `release_qualified: false`, and no workflow gates
reuse or merge on the score.
[Reference PSNR](README.md#reference-psnr) describes the method and the reasons a
clip stays unscored.

## Troubleshoot a run

During execution, the native `recipe.lock.yaml` and the scheduler job record identify
the submitted configuration. Inspect `logs/video/server/launch.json` for the actual
argv, environment, and assigned GPU UUIDs. `spec.json` retains the selected model
and workload. Service startup failures are in `logs/service_video-server.out` and
`logs/sweep_<job>.log`. Client failures remain in each cell's `baseline/run.json`.
Native power failures retain `power-validation.json`. When you report a problem,
keep the run ID, attempt, source head, executed checkout, recipe fingerprint, and
plan hash distinct.
