# Contributing to InferenceX

<div align="center">

**English** | [中文](./CONTRIBUTING_zh.md)

</div>

Thanks for contributing! PRs are welcome. This page covers the review process every PR goes through before it can be merged.

## PR review flow

1. Open your PR and get it through PR validation: add the `full-sweep-enabled` (or `full-sweep-fail-fast`) label so the benchmark sweep runs, and get a green full sweep — including evals — on a commit in your PR.
2. Request a review from your respective company's [CODEOWNER](.github/CODEOWNERS).
3. The CODEOWNER reviews and posts the **PR Review Checklist** sign-off (see below) in their approval comment.
4. Only after the checklist sign-off is posted should you ping a core maintainer on Slack for final approval.
5. An authorized maintainer posts `/reuse-sweep-run` (see below) and the PR is merged via the reuse path.

## The PR Review Checklist (CODEOWNER sign-off)

When a CODEOWNER approves a PR, they must fill in the latest [PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) template in their approval comment.

A friendly reminder — please follow the latest checklist template **correctly**:

- Always copy the template from the **current** [docs/PR_REVIEW_CHECKLIST.md](docs/PR_REVIEW_CHECKLIST.md) on `main`. The checklist evolves; a sign-off made from a stale copy will be flagged as missing items.
- Keep the template's opening phrase intact:

  > As a PR reviewer and CODEOWNER, I have reviewed this and have:

  Our CI verification workflow, [`codeowner-signoff-verify.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/workflows/codeowner-signoff-verify.yml), triggers on exactly this phrase. **If your approval comment does not follow the checklist template — including that phrase — the sign-off verification CI will not trigger at all**, and your sign-off won't count toward merge.
- The sign-off can be posted as a regular conversation comment, a review summary, or an inline review comment — all three trigger verification.
- Fill in the "Additional detail section" with the links the checklist asks for (validation/eval workflow runs, the corresponding [vLLM recipe](https://github.com/vllm-project/recipes) / [SGLang cookbook](https://github.com/sgl-project/sglang/tree/main/docs_new) PR, and any exception reasoning).

Once the sign-off is posted, CI independently re-verifies the claims that gate a merge — CODEOWNER status, a green sweep + evals on a commit in the PR, the linked recipe, the `/reuse-sweep-run` command, use of the latest checklist template, upstream [vLLM](https://hub.docker.com/u/vllm)/[SGLang](https://hub.docker.com/u/lmsysorg) images, no architecture-changing benchmark hacks, and chat-template usage for speculative decoding — and posts a verdict comment on the PR. Checkmarks are not taken on trust, so please only check items you have actually verified.

## `/reuse-sweep-run` — reusing your PR's green sweep at merge

A full benchmark sweep is expensive GPU time, and the runners are shared by every open PR. Without reuse, an approved PR's sweep would run **twice** — once for PR validation and again on `main` after merge. The reuse path avoids that:

- After your PR has an eligible green full sweep, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) comments `/reuse-sweep-run` on the PR (optionally pinning a specific run: `/reuse-sweep-run <run_id>`).
- The merge-to-`main` run then validates and ingests the PR sweep's artifacts instead of re-running the whole sweep on `main`.
- **This reduces CI queue time for everyone** — each reused merge frees hours of GPU runner time for other PRs, so please prefer the reuse path over merging without it. A green sweep alone is not enough: the `/reuse-sweep-run` comment must be on record (the sign-off verification checks for it), otherwise `main` silently re-runs the full sweep.
- `utils/merge_with_reuse.sh <pr-number>` is the supported merge path; it posts the command, syncs the branch with `main`, waits for checks, and squash-merges. See the [workflows README](.github/workflows/README.md#reusing-an-approved-pr-full-sweep) for eligibility details.

## After merging

**PR authors are responsible for ensuring that after merging, all GitHub Action jobs fully pass.** A lot of the time, failures are just flakes and simply re-running the failed jobs will fix it. [See GitHub's docs on re-running failed jobs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs#re-running-failed-jobs-in-a-workflow).
