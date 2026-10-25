# AGENTS.md

Guidance for AI agents working with InferenceX. This file covers repo-wide rules; each area has its own guide that loads when you work there.

## Start here

1. Read this file, the guide for the area you are changing, and its source. Load only the relevant command or skill.
2. Source, schemas, workflows, launchers, and collectors are authoritative. If docs disagree, follow the implementation and fix the nearest English guide.
3. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening or reviewing a PR or changing review, sweep, or merge policy. Sweep labels: [PR sweep labels](.github/workflows/README.md#pr-sweep-labels).
4. Klaud-Cold or `claude/*` image-bump work follows the [candidate instructions](.github/klaud-candidate-prompt.md) and the relevant `.claude/commands/` entry.
5. Model-invoked skills live canonically in `.agents/skills/` (`.claude/skills/` holds compatibility symlinks); add or update them there. User-invoked procedures, such as Klaud workflows and [ingest recovery](.claude/commands/recover-failed-ingest.md), live in `.claude/commands/`.

| Area | Guide |
| --- | --- |
| Python package, tests, local checks | [`inferencex-e2e/infx/AGENTS.md`](inferencex-e2e/infx/AGENTS.md) |
| Launchers, clusters, host hooks | [`inferencex-e2e/infx/launch/AGENTS.md`](inferencex-e2e/infx/launch/AGENTS.md) |
| Benchmark scripts, Bash, srt-slurm recipes | [`inferencex-e2e/benchmarks/AGENTS.md`](inferencex-e2e/benchmarks/AGENTS.md) |
| Master configs, deprecation | [`inferencex-e2e/configs/AGENTS.md`](inferencex-e2e/configs/AGENTS.md) |
| Workflows and runner labels | [`.github/AGENTS.md`](.github/AGENTS.md) |
| Power model | [`power_model/AGENTS.md`](power_model/AGENTS.md) |

The Python project is `inferencex-e2e/` (`pyproject.toml`, `uv.lock`, `.python-version`). Run `uv` from there, or use `uv run --project inferencex-e2e` from the root.

## PRs, issues, and commits

- **PR and issue titles MUST be bilingual:** `<English title> / <中文标题>`, e.g. `[Klaud Cold] Remove SWE-bench Lite eval / [Klaud Cold] 移除 SWE-bench Lite 评测`. Keep prefixes on both halves. Fix English-only titles before review or merge.
- PR and issue bodies and human-authored PR comments keep English visible and put natural Simplified Chinese in one collapsed `<details><summary>中文</summary>` section. Leave code, commands, logs, model names, SKUs, flags, and identifiers untranslated. The CODEOWNER sign-off template is English-only. See [translation terminology](.github/AGENT_OPERATIONS.md#translation-terminology).
- **Descriptions are high signal.** Lead with the problem, the change, and why it matters; keep risks, breaking changes, and unresolved failures visible. Put boilerplate (AI disclosure, change-type lists, checklists) in closed `<details>` blocks after the summary. Report validation only from real integration or end-to-end runs, with a run link; collapse supporting evidence; omit local-check inventories and empty or pending sections. Keep running appropriate checks. See [PR descriptions](CONTRIBUTING.md#pr-descriptions).
- Every PR body includes a collapsed `<details><summary>AI model disclosure</summary>` block naming each contributing model (including delegated agents) and its role, copying the identifier the runtime exposes; never guess one. Tool names are not models. If the exact model is unavailable, state that it could not be verified; human-only PRs state `No AI used`. Keep it current when another model edits the PR.
- Only one eligible CODEOWNER posts the PR Review Checklist; check for an existing one first, and edit it in place rather than posting another; post a replacement only if the original was deleted. See [sign-off](CONTRIBUTING.md#the-pr-review-checklist-codeowner-sign-off).
- Commit subjects are conventional English; commit bodies include the Chinese translation. Contributor docs are English-only except READMEs; keep existing `README_zh.md` files in sync and preserve their language switchers; add no other translations.
- Klaud Cold reports use `infx.klaud`'s typed reporting commands and renderer per the [candidate instructions](.github/klaud-candidate-prompt.md); never hand-build baselines, deltas, receipts, or cleanup reports.

## Changelog and merge (non-negotiable)

- A change that alters what any benchmark runs needs an `inferencex-e2e/perf-changelog.yaml` entry for the affected config keys; the entry triggers result collection. Refactors, recipe moves, and docs need none. `python -m infx.matrix.changed --base origin/main` (from `inferencex-e2e/`) lists affected config and recipe keys; if it lists none, add no entry. It does not see script or launcher code, so judge script and launcher changes by what they run.
- `perf-changelog.yaml` is append-only and byte-sensitive: preserve every existing byte and separator, append only at the tail, and resolve conflicts only with the [changelog conflict helper](.github/workflows/README.md#changelog-conflict-recovery). One PR has at most one block (listing one or more `config-keys`); revise it rather than adding another. New entries are English-only (the bilingual rules do not apply); leave historical entries unchanged.
- **`main` never runs sweeps.** A PR with a changelog entry needs an accepted `/use <run_id>` before merge, or its results are never ingested. Merge with `infx.workflows.merge_with_reuse`, except external PRs (pin their `e2e-tests.yml` run explicitly) and ingest-recovery PRs (merge directly, per the [recovery command](.claude/commands/recover-failed-ingest.md)). See [`/use`](CONTRIBUTING.md#reusing-your-prs-green-sweep-at-merge-with-use).
- Generated configuration is not runtime proof. Run the exact config, inspect its generated fields, then collect smoke, sweep, and eval evidence via the [workflow guide](.github/workflows/README.md), [evals](inferencex-e2e/infx/evals/EVALS.md), or [AgentX replay](inferencex-e2e/docs/agentx-standalone.md).

## Every file

- Follow the nearest existing pattern. YAML uses kebab-case fields.
- **Comments (mandatory):**
  - No header comments describing a file, topology, config, recipe, override, or section.
  - No history: how code evolved, what it replaced, where it came from, or past results. Git and `perf-changelog.yaml` are the record.
  - Only technical explanation of a specific line or block: why a value was chosen, or context the code cannot show (hardware quirk, upstream bug, measured failure). One short line is the norm; never restate the code.
- Pareto logic changes update both InferenceX and InferenceX-app with matching regression tests and cross-linked PRs. `power_model/` changes update the app's TypeScript port; see [`power_model/AGENTS.md`](power_model/AGENTS.md).

See also: [config fields](inferencex-e2e/configs/CONFIGS.md), [GitHub operations](.github/AGENT_OPERATIONS.md), and [ingest recovery](.claude/commands/recover-failed-ingest.md).
