<!-- Title: <English title> / <中文标题>. Keep English visible; put the Chinese translation in the collapsed section below. -->

## Description

<!-- Provide a brief description of your changes -->

## AI model disclosure

<!-- Required: name the exact model/version used to prepare this PR and its role. List every contributing model, including delegated agents; a tool name alone (Claude Code, Cursor, Perplexity Computer) is insufficient. Use the identifier exposed by the runtime, never a guessed identifier. If unavailable, explicitly state that the exact model could not be verified. For human-only PRs, write "No AI used". Update this section if later edits use another model. -->

- Model/version:
- Role:

## Related Issue

<!-- Link to related issue(s) if applicable -->
Fixes #

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
- [ ] **For every change that can affect benchmark performance and every recipe addition or modification, I have appended a new entry to the physical end of `perf-changelog.yaml` and have not edited historical entries**
- [ ] **Before merging via reuse, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) has commented `/use <run_id>` (or the legacy `/reuse-sweep-run`) on this PR**. Do this **only once there is a final full sweep that is all green with evals passing**, since after this comment the sweep label will no longer automatically kick off new sweeps. Remove and re-add the label to force one.

<details>
<summary>中文</summary>

<!-- 翻译上方的改动说明、AI 模型使用说明、关联 issue、改动类型、验证结果及注意事项。AI 模型使用说明必须列出实际使用的完整模型名称/版本及各自的工作内容（包括委派给其他 agent 的工作），不能只写工具名，也不能猜测运行环境未提供的模型标识；无法确认时须明确说明。未使用 AI 时填写 “No AI used”。后续修改使用其他模型时须更新说明。引用共用的表格、代码和日志，保留证据链接；检查清单只需在上方填写一次。 -->

</details>
