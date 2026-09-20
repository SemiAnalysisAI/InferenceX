# Contributing to InferenceX

<div align="center">

**English** | [中文](./CONTRIBUTING_zh.md)

</div>

Thanks for contributing! PRs are welcome. This page covers the review process every PR goes through before it can be merged.

## PR review flow

Every PR description must include an **AI model disclosure** section. Name the exact model/version used to prepare the PR and each model's role, including delegated agents. Tool names such as Claude Code, Cursor, or Perplexity Computer alone are insufficient. Use the identifier exposed by the runtime; never guess an unavailable identifier. If the runtime does not expose the exact model, explicitly state that it could not be verified. Human-only PRs must state `No AI used`. Update the disclosure when later edits use another model.

1. Open your PR and get it through PR validation. Add the `full-sweep-fail-fast` label (strongly recommended because a broken change wastes one job per matrix rather than the whole fan-out). Use `full-sweep-enabled` only if you need jobs to keep running past a failure. Let the benchmark sweep run and get a green full sweep, including evals, on a commit in your PR.
2. For changes owned by a non-admin CODEOWNER other than `@SemiAnalysisAI/core`, ask one eligible [CODEOWNER](.github/CODEOWNERS) to review and post the **PR Review Checklist** sign-off (see below) in their approval comment.
3. Ping a core maintainer on Slack for final approval, after obtaining the checklist sign-off when required.
4. An authorized maintainer posts `/use <run_id>` (see below) and the PR is merged via the reuse path.

**Performance changelog requirement:** Every change that can affect benchmark performance and every recipe addition or modification **MUST** append a new entry to the physical end of `perf-changelog.yaml`. Historical entries **MUST NOT** be edited.

## Pareto coverage

At least **5 measured points on each affected throughput-versus-E2EL Pareto
frontier are highly recommended** before merging a performance submission.
Five is not an unconditional minimum: a smaller frontier may merge with an
explicit, recorded admin bypass after additional review. Missing or unverifiable
coverage follows the same exception path. Do not treat a warning as ordinary
sign-off approval.

The CODEOWNER must report each affected curve's count and evidence in the
checklist's additional detail section. A shortfall produces a visible `⚠️`
warning and tags @functionstackx (Oren), @cquil11 (Cam), @Oseltamivir (Bryan),
and @adibarra (Alec).
Record the admin's rationale, affected curves and assessed SHA in the PR; verify
the author's repository `permission: admin` and `role_name: admin`. Ordinary
approval, `/use`, membership in Core, or a bot's comment is not this exception.
An exception remains visible as WARN, even after it is authorized.

This is a review policy, not a new required GitHub status. The existing sign-off
workflow remains advisory and updates one comment; it does not prevent GitHub
from merging or grant bypass authority. Existing branch rules and sweep/eval
requirements remain unchanged. An agent must not exercise admin bypass itself.

### Count the dashboard frontier, not the sweep matrix

- Use total token throughput per chip (`tput_per_gpu`, chart `y_tpPerGpu`) versus
  E2EL in seconds. Fixed-sequence uses `median_e2el`; AgentX uses the reviewed
  percentile, P90 by default. Report P75 separately if submitted. Do not substitute
  TTFT, interactivity, output-only throughput or cluster throughput.
- Count each affected model/scenario and hardware/framework/precision curve
  separately, at the assessed image and run/date. Fixed-sequence speculative
  methods are separate series; AgentX permits topology, speculative methods and
  KV offload to vary within a curve. Do not add counts across curves or history.
- Count measured frontier vertices, not configured concurrencies, raw successful
  jobs, repeated coordinates, interpolated line segments or failed measurements.
  E2EL and throughput must be finite and positive. Invalid/missing evidence warns.
- For append-only changes, count the resulting whole curve, including verified
  unchanged same-image points with reusable artifact provenance. Five newly added
  points are not required. Do not import points from an incompatible old image.
- Link the sweep run/attempt and raw artifacts, source SHA, image, selected
  scenario, percentile, metric and per-curve counts. For docs/tooling-only changes
  that cannot affect benchmark curves, record N/A with a reason.

The implementation was traced at InferenceX-app commit
[`d507f3689274c82972709abb531bdedcb2e77946`](https://github.com/SemiAnalysisAI/InferenceX-app/commit/d507f3689274c82972709abb531bdedcb2e77946):

- [`metric-registry.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/metric-registry.ts)
  selects `upper_right` for throughput versus E2EL.
- [`paretoFrontUpperRight`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/lib/chart-utils.ts)
  sorts latency ascending and throughput descending on ties, then retains
  increasing throughput and equal-throughput plateaus at distinct latencies.
  Exact coordinate duplicates count once. The plateau rule is the app's current
  behavior, not strict mathematical non-dominance; do not silently replace it.
- [`chartFrontier`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/utils/powerCurves.ts)
  uses [`canonicalParetoIntersection`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/utils/canonicalFrontier.ts)
  only when canonical flags are present. Compute the E2EL frontier first, then
  intersect with verified flags; filtering beforehand can promote dominated
  points. The current
  [`ChartDisplay`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/d507f3689274c82972709abb531bdedcb2e77946/packages/app/src/components/inference/ui/ChartDisplay.tsx)
  does not stamp those flags on ordinary official E2EL points. Do not infer an
  unconditional AgentX intersection from the unused canonical-ID helper.

The trusted review helper accepts already scoped measured coordinates:

```bash
node .github/scripts/pareto-coverage.cjs < /tmp/pareto-curves.json
```

Example input shape (one measured point, therefore WARN):

```json
[{"key":"model/scenario/hwKey/precision/run/percentile/image","points":[{"x":1.0,"y":100.0}]}]
```

Include every affected curve, including empty ones. Preserve verified
`isOnNormalizedInteractivityFrontier` flags if the assessed app path supplies
them. The helper does not fetch artifacts, prove provenance, discover omitted
curves, or approve exceptions; the verifier must do those checks independently.
If the app has changed its frontier semantics since the pinned implementation,
report WARN and resolve the discrepancy rather than claim equivalent counts.

## Draft-model precision

Speculative-decoding submissions must use the original, unquantized draft weights
and their native precision. This applies to embedded MTP/NextN/EAGLE draft heads
and standalone draft models, including DSpark, on every hardware vendor and
framework. Do not change draft precision at all relative to the reference:
no quantization, downcasts, upcasts, same-width dtype conversions (such as
BF16 to FP16), or mixed-precision overrides. This covers draft weights,
activations, computation, and draft KV cache, whether changed offline, at load
time, or during serving. Do not substitute a precision-converted draft checkpoint.

Reviewers must check the effective draft precision, not just the launch flags.
Inspect checkpoint metadata and quantization exclusions, environment variables,
framework defaults in the pinned image, and any inherited target-model
quantization. A quantized target/verifier is allowed under the existing eval
requirements, but its quantization must not also quantize the draft components.
Do not infer draft precision from the target checkpoint's name or precision label.

This includes settings such as `--speculative-draft-model-quantization quark_mxfp4`
and `SGLANG_GLM_NEXTN_MOE_PTPC=1` when they quantize draft computation. An upstream
recipe, passing evals, a claimed unchanged acceptance length (AL), or a newly
measured AL curve does not exempt a submission from this rule. Synthetic AgentX
acceptance is not evidence that draft precision was preserved.

For speculative-decoding changes, the CODEOWNER's additional detail section must
identify the draft checkpoint/revision (or embedded head), its native and effective
precision, and the metadata or pinned implementation used to verify that no draft
weights or precision were changed. If the precision cannot be verified, the
criterion is not satisfied. See the [review checklist](docs/PR_REVIEW_CHECKLIST.md) and
[verifier Check 13](.github/codeowner-signoff-verify-prompt.md#check-13--draft-weights-and-precision-are-unchanged).

This follows the same reference-head precision principle as
[MLPerf Inference Rules, Appendix C: Speculative Decoding](https://github.com/mlcommons/inference_policies/blob/ff7edba545fded369e7e7e3d5a2f0bab4a95eece/inference_rules.adoc#appendix-c-speculative-decoding),
which requires the reference MTP head "at the same precision as provided" and
prohibits reference-head weight quantization and other acceptance-rate manipulation.
That MLPerf revision includes a workload-specific quantized-edge exception;
InferenceX does not adopt that exception. This comparison concerns preserving
draft weights and precision, not adopting MLPerf's allowed-model list,
speculative-decoding configuration, or acceptance methodology.

## The PR Review Checklist (CODEOWNER sign-off)

Automated CODEOWNER verification is advisory for now. The workflow checks submitted checklists and updates one verdict comment without publishing commit statuses. GitHub's separate Core-team and CODEOWNER approval requirements remain in effect unless bypassed by an authorized maintainer.

Sign-off is required only when a changed file has a CODEOWNER other than a repository admin or `@SemiAnalysisAI/core`. Ownership comes from the current tip of the PR target branch, resolved once and pinned to the same SHA for CODEOWNERS validation and content reads, using the last matching rule; renames check both old and new paths. The PR head and its potentially stale recorded base SHA do not supply ownership rules. A matching core owner does not exempt another owner on the same file. Individual admins must have both repository `permission: admin` and `role_name: admin`; other teams and email owners require sign-off. Missing ownership data or failed permission lookups cannot grant an exemption. Changes without a qualifying owner skip verification.

One eligible CODEOWNER reviewer fills in the latest [PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) template in their approval comment.

**Only one eligible CODEOWNER reviewer needs to post the checklist for each PR.** Check for an existing checklist before posting; additional reviewers do not need to post their own copies. For corrections, missing evidence, or verification retries, the original reviewer must **edit their existing checklist comment** instead of adding a new one. Create a replacement only if the original comment was deleted.

A friendly reminder. Please follow the latest checklist template **correctly**:

- Always copy the template from the **current** [docs/PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) on `main`. The checklist evolves, and a sign-off made from a stale copy will be flagged as missing items.
- Keep the template's opening phrase intact:

  > As a PR reviewer and CODEOWNER, I have reviewed this and have:

  Our CI verification workflow, [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml), triggers on exactly this phrase. **If your approval comment omits that phrase, the workflow will not verify the checklist.**
- The sign-off can be posted as a regular conversation comment, a review summary, or an inline review comment. All three trigger verification.
- Submit or edit the checklist when the PR is open and ready. Pushes, reopening, and leaving draft do not trigger verification. If a review event was missed during a merge conflict, retry after resolving it by editing the existing checklist or using manual dispatch.
- Starting Claude requires an eligible human actor with repository write access.
- Fill in the "Additional detail section" with the links the checklist asks for (validation/eval workflow runs, the corresponding [vLLM recipe](https://github.com/vllm-project/recipes) / [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new) PR, and any exception reasoning).

Once the sign-off is posted, CI independently re-verifies the review checklist claims, including CODEOWNER status, a green sweep and evals on a commit in the PR, the linked recipe, the reuse command, use of the latest checklist template, upstream [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) images, no architecture-changing benchmark hacks, chat-template usage for speculative decoding, unchanged draft-model/head weights and precision, and Pareto coverage. It then creates or updates one verdict comment for the PR, including the SHA actually assessed. Failing criteria and Pareto warnings stay visible; passing and N/A criteria appear together in a collapsed section. An existing verdict comment from the older per-commit format is reused; if the comment was deleted, the next verification creates a replacement. Checkmarks are not taken on trust, so please only check items you have actually verified.

The verdict records only the commit actually assessed; it does not carry approval forward to later commits. To request a new assessment, the original reviewer edits their existing checklist, or an authorized collaborator dispatches `codeowner-signoff-verify.yml` with `pr-number` and its `comment_url` (both must identify the same PR). This updates the same verdict comment.

## Reusing your PR's green sweep at merge with `/use`

A full benchmark sweep is expensive GPU time, and the runners are shared by every open PR. Without reuse, an approved PR's sweep would run **twice**, once for PR validation and again on `main` after merge. The reuse path avoids that:

- After your PR has an eligible green full sweep, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) comments `/use <run_id>` on the PR to select that run. Keep the command and run ID on the same line.
- `/reuse-sweep-run <run_id>` remains supported with identical behavior. Bare `/reuse-sweep-run` selects automatically; bare `/use` is rejected.
- The merge-to-`main` run then validates and ingests the PR sweep's artifacts instead of re-running the whole sweep on `main`.
- **This reduces CI queue time for everyone.** Each reused merge frees hours of GPU runner time for other PRs, so please prefer the reuse path over merging without it. A green sweep alone is not enough. The reuse command must be on record (the sign-off verification checks for it), otherwise `main` silently re-runs the full sweep.
- Reuse does not require retaining a sweep label. The bot reacts to the command with 👍 when accepted or 👎 when rejected, with details in the Actions run summary; source artifacts are revalidated at merge.
- `utils/merge_with_reuse.sh <pr-number>` is the supported merge path. It posts the command, syncs the branch with `main`, waits for checks, and squash-merges. See the [workflows README](.github/workflows/README.md#reusing-an-approved-pr-full-sweep) for eligibility details.

## Adding points to the latest curve with `append-only`

When a PR only adds generated points to an existing curve, mark every new changelog
entry with `append-only: true`. Additions may introduce new concurrency values or new
recipe variants, such as another tensor-parallelism value. Sweep setup compares the
generated matrices at the base and head revisions, runs only the newly added points,
and emits metadata that lets InferenceX-app extend the most recent matching curve
instead of presenting the partial run as a separate curve.

This mode is intentionally narrow, but it is not based on a file allowlist. Supporting
code, benchmark scripts, launchers, and other files may change when their behavioral
effect is exclusive to the newly appended points named by the changelog. No changed
benchmark path may execute for or alter an existing point. Every selected config and
scenario must already exist, and every point generated at the base revision must
remain present with the same recipe. The head may contain any additional generated
recipes or points inside that scope, including new topology or other recipe dimensions;
the sweep schedules the generated set difference. Additions must use the same non-null
image and belong to an existing dashboard visual series. Each generated recipe carries
a deterministic fingerprint so two distinct recipes at the same concurrency remain
distinct database points without splitting the visual curve. Removing or modifying an
existing point, or changing shared logic that can affect one, is rejected. Append-only
entries cannot be mixed with regular entries or eval-selection modifiers in the same
sweep. The matrix validator enforces the additive generated-matrix invariant; the human
and AI reviewers must inspect the complete diff and verify behavioral isolation. The
mechanical comparison renders each config revision with its own generator, validation
code, and runner metadata. Launcher and benchmark-script changes still rely on
complete-diff review because matrix equality alone cannot prove their runtime
control-flow isolation.

```yaml
- config-keys:
    - dsv4-fp4-b300-vllm-mtp
  description:
    - "Add TP8 at concurrency 12 and 16 to the existing curve"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
  append-only: true
```

## AMD cluster: never leave root-owned files in runner workspaces

Multi-node benchmarks on the AMD MI355X TW cluster submit Slurm jobs whose containers often run as **root**. If those containers write files (typically `benchmark_logs/logs/slurm_job-*`) into the GitHub Actions runner workspace and the job is **cancelled** before teardown runs, the root-owned directories are stranded. The runner user cannot delete them, so `actions/checkout` fails with:

```
Error: File was unable to be removed
Error: EACCES: permission denied, rmdir '.../benchmark_logs/logs/slurm_job-<id>'
```

**This bricks every subsequent job on that runner** until someone with `sudo` on the shared `/it-share` storage manually removes the files. Because all AMD MI355X sweeps share the same runner pool, a single stranded root-owned directory blocks the entire queue for everyone.

**Rules for benchmark scripts and Slurm containers:**

1. **Never write as root into the runner workspace.** If your container must run as root, write outputs to a separate scratch directory outside `_work/` (e.g. `/tmp` or a dedicated staging path).
2. **If root writes are unavoidable**, add a cleanup trap or teardown step that `chown`s or `rm`s all root-owned files under the workspace **before** the job exits, including on cancellation (`trap cleanup EXIT`).
3. **Test your teardown path.** Cancel a running benchmark mid-flight and verify no root-owned files remain in the workspace.

If you find a stranded root-owned file blocking runners, the recovery procedure is documented in [`.claude/commands/clean-amd-mi355-runner-root-files.md`](.claude/commands/clean-amd-mi355-runner-root-files.md): SSH into the hop host with `sudo`, scan the `_work` directories, and delete the offending files.

## After merging

**PR authors are responsible for ensuring that after merging, all GitHub Action jobs fully pass.** A lot of the time, failures are just flakes and simply re-running the failed jobs will fix it. [See GitHub's docs on re-running failed jobs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs#re-running-failed-jobs-in-a-workflow).
