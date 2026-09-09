# Documentation Procedures

<div align="center">

**English** | [中文](./documentation-procedures_zh.md)

</div>

Use this guide when adding or updating contributor-facing documentation, preparing the accompanying GitHub text, or reviewing a documentation change. Repository source files and workflows remain authoritative for behavior. Documentation explains their contracts, rationale, and safe operating procedures.

> **Required:** The English page is the primary source version. Write or update the English page first, then translate it into the matching Simplified Chinese `_zh.md` page in the same change. Keep headings, structure, links, commands, tables, and code blocks synchronized. Every pair has an `English | 中文` switcher.

## Procedure index

| Procedure | Use when |
| --- | --- |
| [Choose the documentation scope](#choose-the-documentation-scope) | Deciding whether to update a page, create a focused page, or change a source-adjacent reference |
| [Add a contributor-facing page](#add-a-contributor-facing-page) | Creating a new bilingual guide |
| [Update an existing bilingual pair](#update-an-existing-bilingual-pair) | Changing a contract, command, link, image, or explanation |
| [Update the documentation index](#update-the-documentation-index) | Making a new page discoverable or changing its purpose |
| [Keep source-of-truth links current](#keep-source-of-truth-links-current) | Documenting implementation behavior without creating a second authority |
| [Write bilingual GitHub content](#write-bilingual-github-content) | Preparing PRs, issues, comments, reviews, or commits |
| [Complete CODEOWNER sign-off](#complete-codeowner-sign-off) | Approving a PR under the repository review policy |
| [Review documentation changes](#review-documentation-changes) | Checking correctness, parity, navigation, and readability |

## Governing sources

Read the sources that apply before editing:

| Source | What it governs |
| --- | --- |
| [`AGENTS.md`](../AGENTS.md) | Bilingual documentation and GitHub-writing rules, commit format, and agent policy |
| [`.github/AGENT_OPERATIONS.md`](../.github/AGENT_OPERATIONS.md#translation-terminology) | Preferred translation terminology |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | PR review order, CODEOWNER approval, sweep reuse, and post-merge responsibility |
| [`docs/index.md`](./index.md) and [`docs/index_zh.md`](./index_zh.md) | Maintained documentation map and navigation wording |
| [`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) | Current CODEOWNER sign-off template and merge standard |
| [`.github/CODEOWNERS`](../.github/CODEOWNERS) | Owners for the paths changed by a PR |
| [`.github/codeowner-signoff-verify-prompt.md`](../.github/codeowner-signoff-verify-prompt.md) | Independent checks encoded by the sign-off verifier |
| [`.github/workflows/codeowner-signoff-verify.yml`](../.github/workflows/codeowner-signoff-verify.yml) | Sign-off trigger events, exact-phrase detection, and status publication |
| [`.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`](../.github/PULL_REQUEST_TEMPLATE/pull_request_template.md) | PR fields and author checklist |
| [`.github/ISSUE_TEMPLATE/`](../.github/ISSUE_TEMPLATE/) | Bug and feature issue prompts |

The information architecture follows the useful pattern in the [`InferenceX-app` documentation hub](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/index.md). The hub links focused pages. [Architecture pages](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/architecture.md) explain decisions and tradeoffs, [testing pages](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/testing.md) state mandatory gates and review standards, and [adding-entities pages](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/docs/adding-entities.md) use ordered workflows, prerequisites, checklists, and explicit verification.

## Choose the documentation scope

1. **Identify the audience and task.** A page should answer one recurring question for contributors or operators. If readers must load unrelated subsystems to use it, split it.
2. **Search the documentation map first.** Update the nearest existing page when it already owns the topic. Do not create a second convention or duplicate procedure.
3. **Find the implementation source before drafting.** Read the exact config schema, workflow, script, or policy file behind each operational claim. A neighboring guide is context, not proof of current behavior.
4. **Choose the page type:**
   - **Architecture/rationale:** ownership, data flow, invariants, alternatives, and why the current design exists.
   - **Procedure:** prerequisites, ordered actions, verification, failure/stop conditions, and authoritative links.
   - **Reference:** stable schema, terminology, supported values, or a navigation map.
   - **Troubleshooting:** observable symptom, evidence to collect, root-cause branches, safe remediation, and escalation boundary.
5. **Keep source-adjacent references source-adjacent.** Detailed schema or command documentation may belong beside the code, such as [`configs/CONFIGS.md`](../configs/CONFIGS.md), [`.github/workflows/README.md`](../.github/workflows/README.md), or [`utils/evals/EVALS.md`](../utils/evals/EVALS.md). Link it from `docs/` instead of copying it.
6. **Define completion before editing.** Name the bilingual pair, index location, implementation links, and the review evidence required.

## Add a contributor-facing page

### 1. Name the pair

Use descriptive kebab-case names under `docs/`:

```text
docs/<topic>.md
docs/<topic>_zh.md
```

The `_zh.md` suffix is required for the Simplified Chinese page. Agent-only instructions (`AGENTS.md`, `CLAUDE.md`, `KLAUD_DEBUG.md`), internal references under `.github/` or `utils/`, and implementation-local references such as `configs/CONFIGS.md` and `experimental/README.md` are the documented English-only exceptions. Do not invent new exceptions.

### 2. Add reciprocal switchers

Place the switcher immediately below each title.

English page:

```html
<div align="center">

**English** | [中文](./<topic>_zh.md)

</div>
```

Chinese page:

```html
<div align="center">

[English](./<topic>.md) | **中文**

</div>
```

Use relative links so the pair works in forks, branches, and local renderers.

### 3. Build a focused, indexed page

Use only the sections the topic needs, normally in this order:

1. One-paragraph purpose and authority boundary.
2. A critical prerequisite or invariant, when missing it causes silent data loss, wasted compute, broken review, or unsafe operation.
3. A procedure index for a page with several workflows.
4. Governing sources or architecture context.
5. Ordered procedures with prerequisites before edits.
6. Verification and stop conditions.
7. A final review checklist.

Prefer short decision tables, numbered procedures, checklists, and exact links. Explain **why** for non-obvious constraints. Do not turn the page into a file inventory or paste long implementation blocks that will drift.

### 4. Write English first, then translate and review

- Treat the English page as the primary source version: finish its claims, structure, headings, links, commands, tables, and code blocks before translating the matching `_zh.md` page.
- Keep the same heading hierarchy, procedure order, tables, links, badges, images, warnings, and code blocks in both pages.
- Translate intent, not syntax. Keep commands, paths, environment variables, flags, model names, hardware SKUs, framework names, logs, and stack traces unchanged.
- Use natural technical Chinese rather than word-for-word translation. Follow the [preferred terminology](../.github/AGENT_OPERATIONS.md#translation-terminology).
- Preserve meaningful anchors by keeping corresponding headings semantically aligned. Test cross-page and same-page links after translation.
- If one language cannot express a claim accurately, fix the source claim first. Do not allow the pages to diverge.

### 5. Add navigation and validate

Update both documentation indexes in the same change, then run the checks in [Review documentation changes](#review-documentation-changes). A new page is not complete merely because its files exist.

## Update an existing bilingual pair

1. Read both pages before editing. Confirm they are currently aligned. Record any pre-existing mismatch separately from the requested change.
2. Read the current authoritative implementation or policy. For repository-wide documentation rules, start with [`AGENTS.md`](../AGENTS.md). For review policy, start with [`CONTRIBUTING.md`](../CONTRIBUTING.md) and the live checklist on `main`.
3. Update the English primary source page first. Finalize every changed claim, heading, link, warning, table row, badge, image, and code block there before translating.
4. Translate the finalized English changes into the matching `_zh.md` page in the same PR. Preserve code and evidence verbatim. Translate the surrounding explanation, not a CLI flag, pathname, log, or stack trace.
5. Remove statements invalidated by the new contract. Do not leave a deprecated path, alias, or contradictory procedure unless the implementation still supports it and readers need the migration rule.
6. Update the documentation index pair if the page title, scope, or recommended entrypoint changes.
7. Recheck every operational claim against source. A documentation-only diff can still be behaviorally wrong.

### Special case: the PR review checklist

[`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) is the source of truth for the merge standard. Its template must remain English-verbatim in **both** checklist language files because the verifier detects exact English wording.

If a checklist item is added, removed, or materially reworded, update [`.github/codeowner-signoff-verify-prompt.md`](../.github/codeowner-signoff-verify-prompt.md) so the independent checks encode the same policy, ideally in the same PR. Formatting fixes, typo fixes, and `_zh` translation synchronization do not require verifier logic changes. Then follow [Validate verifier changes](#validate-verifier-changes).

### Validate verifier changes

Verifier changes are not proven by local Markdown or YAML checks. After the change is merged:

1. Open a throwaway `[DO NOT MERGE]` test PR.
2. Post a CODEOWNER sign-off comment containing the exact phrase `As a PR reviewer and CODEOWNER`. Otherwise the workflow will not trigger.
3. Read the verifier's posted verdict and confirm it exercises the expected checks.
4. Close the test PR.

## Update the documentation index

Every new contributor-facing page must be discoverable from the maintained hub.

1. Add and finalize the entry in the primary English [`docs/index.md`](./index.md) first. Link it to `<topic>.md` and write the concise English “use it for” description.
2. Translate that finalized entry into [`docs/index_zh.md`](./index_zh.md) in the equivalent location, linking it to `<topic>_zh.md`.
3. Describe the question the page answers, not merely its title.
4. Put the entry in the existing family or navigation table. Do not add a second index section for the same category.
5. If a page replaces an old entrypoint, update all index links in the same change and remove the obsolete entry. Do not leave two “canonical” pages.
6. Keep both indexes structurally aligned: same entry order, destination set, guide families, and system boundaries.
7. Follow links from the rendered indexes to both language pages and back through their switchers.

## Keep source-of-truth links current

Documentation is an interface to source, not a replacement for it.

### Link claims to the owning source

| Claim | Preferred source |
| --- | --- |
| Repository contribution or bilingual rule | [`AGENTS.md`](../AGENTS.md) or [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| CODEOWNER identity | [`.github/CODEOWNERS`](../.github/CODEOWNERS) |
| Review checklist wording | [`docs/PR_REVIEW_CHECKLIST.md`](./PR_REVIEW_CHECKLIST.md) on `main` |
| CI trigger, permissions, or job behavior | Exact file under [`.github/workflows/`](../.github/workflows/) |
| Config field or accepted value | Master YAML plus [`configs/CONFIGS.md`](../configs/CONFIGS.md) and the validation/generator code |
| Runtime flag or environment variable | The benchmark script, launcher, or shared helper that consumes it |
| Artifact schema or eval gate | The collector/validator code and its source-adjacent reference |
| External framework behavior | Versioned upstream documentation, recipe, release, or source link |

### Link-writing rules

- Prefer repository-relative links for files in InferenceX. They survive forks and branch previews.
- Link the exact file or focused source-adjacent guide, not only a repository root or broad directory.
- Use stable upstream URLs. Pin a commit or release when a claim depends on a particular version. Use the upstream default branch for intentionally live policy.
- State which source wins if two representations exist. For example, workflow YAML defines CI behavior while its README explains how to invoke it.
- Copy only short syntax examples needed to act. Link the implementation for defaults, supported values, branches, and failure behavior.
- When source moves or a contract changes, update inbound documentation links in the same PR. Search for the old path and terminology rather than fixing only the page you opened.
- Never document a command as verified unless it was actually run in the stated environment. Distinguish a static syntax check from a runtime or CI result.

## Write bilingual GitHub content

The bilingual rule applies to every PR and issue and to human-authored PR comments. Bot-generated comments follow their workflow templates. Never post an external comment, review, label, or merge command without authorization.

### PRs and issues

Use this title format:

```text
<English title> / <中文标题>
```

For bodies:

1. Complete the existing PR or issue template. Do not delete required fields or checklists.
2. Write the English summary, motivation, changes, verification, risks, and source links.
3. Add a `## 中文说明` section that mirrors the substantive English content.
4. Keep code blocks, commands, logs, stack traces, paths, identifiers, and URLs unchanged. Explain their meaning in Chinese before or after the unchanged block.
5. Keep evidence symmetric. If the English section links a workflow, artifact, issue, or source, the Chinese section must preserve that link.
6. Do not mark a checklist item complete unless the underlying action was observed.

Minimal shape:

```markdown
## Summary
- Explain what changed and why.

## Verification
- Link or state the exact check that ran.

## 中文说明
### 摘要
- 说明改动内容及原因。

### 验证
- 保留相同的检查结果或链接。
```

### Comments and review summaries

- Short comment: `<English> / <中文>` on one line.
- Longer comment: English paragraphs followed by a `中文：` paragraph or a clearly labeled Chinese section.
- Inline review comments, conversation comments, and review summaries all require Chinese translation.
- Preserve code excerpts and evidence unchanged. Translate the diagnosis, impact, and requested fix.
- The CODEOWNER checklist sign-off is the exception: copy it in exact English and do not append a translated checklist inside the sign-off.

### Commits

Use a conventional English subject and put the Simplified Chinese translation of the subject and key body points in the commit body:

```text
docs: add documentation procedures

Explain the source-of-truth and review impact in English when a body is needed.

中文：新增文档维护流程，并说明权威来源与审阅影响。
```

A squash-merge commit inherits the bilingual PR title and therefore satisfies the subject requirement automatically. Do not use documentation wording to bypass sweep or merge policy.

## Complete CODEOWNER sign-off

Follow [`CONTRIBUTING.md`](../CONTRIBUTING.md) before requesting or posting sign-off.

### Author preparation

1. Finish the bilingual documentation pair and any required index/source updates.
2. Complete the PR template and link the exact verification evidence.
3. Get PR validation through the required path. Under current policy, a CODEOWNER sign-off relies on a green full sweep including evals on a commit in the PR. Do not substitute a stale or unrelated run.
4. Request review from the owner for the changed paths in [`.github/CODEOWNERS`](../.github/CODEOWNERS).
5. Do not ask a core maintainer for final approval until the CODEOWNER has posted a valid checklist sign-off.

### CODEOWNER procedure

1. Open the latest [`docs/PR_REVIEW_CHECKLIST.md`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/docs/PR_REVIEW_CHECKLIST.md) from `main` immediately before signing. Never reuse a saved copy.
2. Confirm you are an applicable CODEOWNER for the changed paths. Read the current [`.github/CODEOWNERS`](../.github/CODEOWNERS). Ownership is evaluated from the current patterns, not company affiliation alone.
3. Copy the entire current template without translating or rewriting it. Keep this opening phrase exactly:

   > As a PR reviewer and CODEOWNER, I have reviewed this and have:

4. Independently verify every item before checking it. Review the diff, source behavior, full-sweep and eval evidence, upstream recipe status, image provenance, architecture constraints, patch/waiver status, chat-template requirements, and AgentX acceptance evidence when applicable.
5. Fill the additional-detail section with the exact validation and eval workflow links, the merged upstream vLLM recipe/SGLang cookbook PR or published recipe link, and explicit reasoning for every exception or non-applicable item.
6. Fill `Signed:` with the actual GitHub username. Do not sign for another reviewer.
7. Post the exact English template as a conversation comment, review summary, or inline review comment. All three event types are supported by the verifier.
8. Confirm [`.github/workflows/codeowner-signoff-verify.yml`](../.github/workflows/codeowner-signoff-verify.yml) triggered and read its verdict. The verifier re-derives merge-gating claims. Checkmarks alone are not accepted.
9. If the PR head advances after sign-off, reassess the new diff and post a fresh sign-off. The previous evidence was tied to the reviewed commit.
10. Only an authorized maintainer may record `/reuse-sweep-run` and use the supported merge path. A CODEOWNER approval does not grant that authorization.

Stop instead of signing when a required source, workflow link, recipe, exception rationale, or verification result is missing. Use unchecked boxes and concrete follow-up requests. Never convert an unknown into an approval claim.

## Review documentation changes

Review documentation as an executable interface: readers will follow its commands and policy statements.

### Correctness and authority

- [ ] Every operational claim matches the current owning source.
- [ ] Commands, paths, flags, config keys, workflow names, labels, and URLs are exact.
- [ ] The page names the source of truth and does not create a competing contract.
- [ ] Prerequisites appear before actions. Verification and stop conditions are explicit.
- [ ] Examples are realistic and do not claim checks or runs that were not performed.
- [ ] A policy change updates all affected policy surfaces, including checklist/verifier synchronization when required.

### Bilingual parity

- [ ] Both files exist and use the required `<name>.md` / `<name>_zh.md` names.
- [ ] Both switchers are at the top, reciprocal, relative, and point to existing files.
- [ ] Heading hierarchy, procedure order, warnings, tables, links, badges, images, and code blocks match semantically.
- [ ] Chinese is natural technical Simplified Chinese and follows repository terminology.
- [ ] Identifiers, commands, logs, model names, hardware SKUs, and framework names remain unchanged.
- [ ] The exact English CODEOWNER template remains verbatim wherever it appears.

### Navigation and usability

- [ ] Both documentation indexes contain equivalent entries in the correct family.
- [ ] The index descriptions say when to use the page.
- [ ] Every relative link and heading anchor resolves from both language pages.
- [ ] The page is focused. Repeated implementation detail was replaced by an exact source link.
- [ ] Decision tables and checklists expose prerequisites, ownership, and completion criteria without requiring readers to load an unrelated manual.

### Review evidence

1. Inspect the complete changed sections in both languages, not only isolated diff lines.
2. Render or preview both pages and check headings, lists, tables, blockquotes, code fences, and switchers.
3. Follow every new or changed link. For external links, confirm the target is the intended version or live policy.
4. Run the narrow documentation checks supported by the repository. At minimum, inspect whitespace errors with:

   ```bash
   git diff --check -- docs/
   ```

5. If the documentation describes a changed command or runtime contract, run the narrow source-owned verification for that contract. A Markdown render does not prove runtime behavior.
6. Summarize review results bilingually in the PR. Separate verified facts from assumptions and name any check that could not run.

Do not approve merely because Markdown renders. Broken navigation, stale commands, mismatched translations, and inaccurate merge policy are blocking documentation defects.
