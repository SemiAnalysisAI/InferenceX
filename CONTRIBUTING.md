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

## Draft-model precision

The rule is simple: **serve the draft as it ships.** A speculative-decoding
submission must run the draft head or draft model that ships with the checkpoint
it serves, at the precision it ships in, through the pinned upstream framework's
default handling of that checkpoint. This applies to embedded MTP/NextN/EAGLE heads
and standalone draft models, including DSpark, on every hardware vendor and framework.

"As it ships" does not mean BF16 and does not mean "the unquantized release". It
means the default: what the checkpoint's authors published for that checkpoint, and
what the pinned upstream image does with it out of the box. The baseline is the
same checkpoint loaded by the same pinned image with no draft-related settings from
the submission. A submission is compliant when its effective draft precision matches
that baseline. It is not compliant when it makes the draft cheaper than that baseline.

Allowed (this is the baseline, not an exception):

- The draft head or draft weights embedded in, or released alongside, the served
  checkpoint, in the precision they are stored in. If the served FP8 checkpoint ships
  an FP8 MTP head, the FP8 head is the correct draft. Example: the Qwen FP8 checkpoints
  store their MTP weights in FP8, so the embedded FP8 head is compliant, and swapping in
  the BF16 release's head or forcing an unquantized-draft override is the violation,
  not the fix.
- Load-time handling that the pinned upstream framework applies by default to that
  checkpoint, including dtype conversions. Example: the pinned SGLang image loads the
  DeepSeek V4.1 DSpark `wo_a` projections, stored as FP8, as BF16, matching DeepSeek's
  reference implementation. That conversion is the shipped path. A local patch that
  keeps them FP8 changes the draft computation and is the violation.
- The framework's default draft KV-cache dtype for that checkpoint, and an
  upstream-supported KV-cache dtype applied consistently to target and draft (for
  example FP8 draft KV inherited from an FP8 target).
- Genuine upstream optimizations that ship in the pinned image and are used in
  production by accuracy-sensitive customers: fused or lower-precision kernels that
  upstream enables by default for that model, scheduling, and anything else that runs
  the same draft computation faster without lowering the precision of its weights or
  activations below what ships.

Forbidden (submission-side precision hacking of the draft):

- Enabling `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` (`=1` or any other enabling
  value recognized by the pinned implementation) is explicitly prohibited going
  forward, including inherited environment, launcher, container, or image defaults.
  The baseline allowances above do not exempt this flag. Reviewers must verify
  that it is disabled in the effective recipe; an unset value or `=0` is acceptable
  only when the pinned implementation confirms it is disabled. Historical runs
  are not precedent or an exception for submissions under review, including
  image-only bumps and re-enabled recipes. This rule does not retroactively
  invalidate runs that predate it.
- Online or offline quantization of draft weights, activations, or computation below
  the shipped precision, whether through a flag, environment variable, config file, or
  conversion step. This includes `--speculative-draft-model-quantization quark_mxfp4`
  applied to a BF16 MTP head, `SGLANG_GLM_NEXTN_MOE_PTPC=1`, and an ATOM
  `--online_quant_config` whose `exclude_layer` patterns do not cover the entire draft
  head.
- Dtype or KV-cache dtype overrides aimed at the draft that lower its precision below
  the framework's default for that checkpoint.
- Substituting a precision-converted or differently quantized draft checkpoint, or a
  draft head taken from a different release than the served target.
- Patching the pinned image so that it loads or computes the draft at a different
  precision than it does by default, in either direction. The engine-patch rule
  already forbids this; a waiver does not exempt draft precision.
- Reducing draft FLOPs in ways the checkpoint's authors did not ship, such as
  pruning draft layers or experts.

A quantized target/verifier remains allowed under the existing eval requirements.
Target quantization is fine as long as it does not also quantize the draft below what
ships; check inherited quantization and `exclude_layer` coverage rather than assuming.

Matching an upstream recipe, passing evals, a claimed unchanged acceptance length
(AL), or a newly measured AL curve does not exempt a submission that lowers draft
precision. Synthetic AgentX acceptance is not evidence either way. Draft-precision
changes shift the accuracy hit onto acceptance rate, which the target-model evals do
not measure.

Reviewers must check the effective draft precision against the shipped baseline, not
just the launch flags. Inspect checkpoint metadata and quantization exclusions,
environment variables, framework defaults in the pinned image, and any inherited
target-model quantization. Do not infer draft precision from the target checkpoint's
name or precision label.

For speculative-decoding changes, the CODEOWNER's additional detail section must
identify the draft checkpoint/revision (or embedded head), the precision it ships in,
how the pinned upstream image handles it by default, and its effective serving
precision, so the reviewer can confirm the last two match. If this cannot be
verified, the criterion is not satisfied. See the [review checklist](docs/PR_REVIEW_CHECKLIST.md) and
[verifier Check 13](.github/codeowner-signoff-verify-prompt.md#check-13--draft-runs-as-shipped).

This follows the same principle as
[MLPerf Inference Rules, Appendix C: Speculative Decoding](https://github.com/mlcommons/inference_policies/blob/ff7edba545fded369e7e7e3d5a2f0bab4a95eece/inference_rules.adoc#appendix-c-speculative-decoding),
which requires the reference MTP head "at the same precision as provided" and
prohibits reference-head weight quantization and other acceptance-rate manipulation.
InferenceX does not adopt that revision's workload-specific quantized-edge exception,
its allowed-model list, or its speculative-decoding configuration and acceptance
methodology.

## The PR Review Checklist (CODEOWNER sign-off)

Automated CODEOWNER verification is advisory for now. The workflow checks submitted and edited checklists and associates one verdict comment with each sign-off resource without publishing commit statuses. GitHub's separate Core-team and CODEOWNER approval requirements remain in effect unless bypassed by an authorized maintainer.

Sign-off is required only when a changed file has a CODEOWNER other than a repository admin or `@SemiAnalysisAI/core`. Ownership comes from the current tip of the PR target branch, resolved once and pinned to the same SHA for CODEOWNERS validation and content reads, using the last matching rule; renames check both old and new paths. The PR head and its potentially stale recorded base SHA do not supply ownership rules. A matching core owner does not exempt another owner on the same file. Individual admins must have both repository `permission: admin` and `role_name: admin`; other teams and email owners require sign-off. Missing ownership data or failed permission lookups cannot grant an exemption. Changes without a qualifying owner skip verification.

One eligible CODEOWNER reviewer fills in the latest [PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) template in their approval comment.

**Only one eligible CODEOWNER reviewer needs to post the checklist for each PR.** Check for an existing checklist before posting; additional reviewers do not need to post their own copies. For corrections or missing evidence, the original reviewer must **edit their existing checklist comment** instead of adding a new one. Create a replacement only if the original comment was deleted.

A friendly reminder. Please follow the latest checklist template **correctly**:

- Always copy the template from the **current** [docs/PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) on `main`. The checklist evolves, and a sign-off made from a stale copy will be flagged as missing items.
- Keep the template's opening phrase intact:

  > As a PR reviewer and CODEOWNER, I have reviewed this and have:

  Our CI verification workflow, [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml), triggers on exactly this phrase. **If your approval comment omits that phrase, the workflow will not verify the checklist.**
- The sign-off can be posted as a regular conversation comment, a review summary, or an inline review comment. All three trigger verification.
- Submit a new checklist when the PR is open and ready. Editing that checklist triggers verification again. Pushes, reopening, and leaving draft do not trigger verification. If a review event was missed during a merge conflict, retry after resolving it using manual dispatch.
- Starting Claude requires an eligible human actor with repository write access.
- Fill in the "Additional detail section" with the links the checklist asks for (validation/eval workflow runs, the corresponding [vLLM recipe](https://github.com/vllm-project/recipes) / [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new) PR, and any exception reasoning).

Once the sign-off is posted, CI independently re-verifies the review checklist claims, including CODEOWNER status, a green sweep and evals on a commit in the PR, the linked recipe, the reuse command, use of the latest checklist template, upstream [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) images, no architecture-changing benchmark hacks, chat-template usage for speculative decoding, and unchanged draft-model/head weights and precision. It creates one verdict comment for that sign-off resource, including the SHA actually assessed. Editing the same checklist updates only its associated verdict. A replacement or additional checklist receives a separate verdict, and verdicts associated with older sign-offs remain unchanged. Failing criteria stay visible; passing and N/A criteria appear together in a collapsed section. Checkmarks are not taken on trust, so please only check items you have actually verified.

The verdict records only the commit actually assessed; it does not carry approval forward to later commits. To request a new assessment after correcting the existing checklist as needed, an authorized collaborator dispatches `codeowner-signoff-verify.yml` with `pr-number` and its `comment_url` (both must identify the same PR). Manual reassessment updates the verdict associated with that sign-off resource.

## Reusing your PR's green sweep at merge with `/use`

A full benchmark sweep is expensive GPU time, and the runners are shared by every open PR. Without reuse, an approved PR's sweep would run **twice**, once for PR validation and again on `main` after merge. The reuse path avoids that:

- After your PR has an eligible green full sweep, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) comments `/use <run_id>` on the PR to select that run. Keep the command and run ID on the same line.
- `/reuse-sweep-run <run_id>` remains supported with identical behavior. Bare `/reuse-sweep-run` selects automatically; bare `/use` is rejected.
- The merge-to-`main` run then validates and ingests the PR sweep's artifacts instead of re-running the whole sweep on `main`.
- **This reduces CI queue time for everyone.** Each reused merge frees hours of GPU runner time for other PRs, so please prefer the reuse path over merging without it. A green sweep alone is not enough. The reuse command must be on record (the sign-off verification checks for it), otherwise `main` silently re-runs the full sweep.
- Reuse does not require retaining a sweep label. The bot reacts to the command with 👍 when accepted or 👎 when rejected, with details in the Actions run summary; source artifacts are revalidated at merge.
- A missing authorized reuse command produces a Check 4 **WARN**, not a rejection. The warning stays visible in the sign-off verdict; posting an authorized command is still required to reuse artifacts.
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
