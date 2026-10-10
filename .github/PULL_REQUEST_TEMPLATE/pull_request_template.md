<!-- Title: <English title> / <中文标题>. Keep English visible; put the Chinese translation in the collapsed section below. -->

## Summary

<!-- Briefly explain the problem, what changes, and why it matters. A human reviewer should understand the PR without expanding any sections. Keep material risks, breaking changes, and unresolved failures visible here. Add related issue links only when relevant; remove unused placeholders. -->

<!-- Validation reporting: report only actual integration or end-to-end runs, with a concise outcome and a run link when available. Put supporting validation evidence, verbose check output, or logs in a <details><summary>Validation details</summary> block without the open attribute. Keep material failures and regressions visible. Omit routine local-check inventories and empty or pending validation sections; continue running appropriate checks. -->

<details>
<summary>AI model disclosure</summary>

<!-- Required: name the exact model/version used to prepare this PR and its role. List every contributing model, including delegated agents; a tool name alone (Claude Code, Cursor, Perplexity Computer) is insufficient. Use the identifier exposed by the runtime, never a guessed identifier. If unavailable, explicitly state that the exact model could not be verified. For human-only PRs, write "No AI used". Update this section if later edits use another model. -->

- Model/version:
- Role:

</details>

<details>
<summary>PR checklist and change type</summary>

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Configuration change
- [ ] Documentation update
- [ ] Other (please describe)

## Checklist

- [ ] I have completed the AI model disclosure and kept it current
- [ ] I have tested my changes locally
- [ ] I have updated documentation if necessary
- [ ] **If this PR should produce new published results, I have appended one new entry to the physical end of `inferencex-e2e/perf-changelog.yaml`; otherwise I have added none. I have not edited historical entries**
- [ ] **If this PR appends an `inferencex-e2e/perf-changelog.yaml` entry, it carries exactly one primary sweep label** (a maintainer applies it on fork PRs): `full-sweep-fail-fast` (recommended), `full-sweep-enabled`, or `non-canary-full-sweep-enabled`. Optional modifiers `all-evals`, `evals-only`, and `agentx-fast` require a primary label; the last two block reuse while applied.
- [ ] **Before merging via reuse, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) has commented `/use <run_id>` (or the legacy `/reuse-sweep-run`) on this PR**. Do this **only once there is a final full sweep that is all green with evals passing**, since after this comment the primary sweep label will no longer automatically kick off new sweeps. Remove and re-add the primary sweep label to force a new sweep.

</details>

<details>
<summary>中文</summary>

<!-- 翻译上方的改动说明、AI 模型使用说明、关联 issue、改动类型、验证结果及注意事项。AI 模型使用说明必须列出实际使用的完整模型名称/版本及各自的工作内容（包括委派给其他 agent 的工作），不能只写工具名，也不能猜测运行环境未提供的模型标识；无法确认时须明确说明。未使用 AI 时填写 “No AI used”。后续修改使用其他模型时须更新说明。引用共用的表格、代码和日志，保留证据链接；检查清单只需在上方填写一次。 -->

</details>
