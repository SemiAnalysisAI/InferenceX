# AGENTS.md

Guidance for AI agents working with InferenceX.

## Start here

1. Read this file and the source for the component being changed. Load only the relevant command or skill.
2. Source, schemas, workflows, launchers, and collectors are authoritative. If docs disagree, follow the implementation and fix the nearest English guide.
3. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening or reviewing a PR or changing review, sweep, or merge policy. Sweep labels: [PR sweep labels](.github/workflows/README.md#pr-sweep-labels).
4. Klaud-Cold or `claude/*` image-bump work follows the [candidate instructions](.github/klaud-candidate-prompt.md) and the relevant `.claude/commands/` entry.
5. Model-invoked skills live canonically in `.agents/skills/` (`.claude/skills/` holds compatibility symlinks); add or update them there. User-invoked procedures, such as Klaud workflows and [ingest recovery](.claude/commands/recover-failed-ingest.md), live in `.claude/commands/`.

The Python project is `inferencex-e2e/` (`pyproject.toml`, `uv.lock`, `.python-version`). Run `uv` from there, or use `uv run --project inferencex-e2e` from the root.

## PRs, issues, and commits

- **PR and issue titles MUST be bilingual:** `<English title> / <中文标题>`, e.g. `[Klaud Cold] Remove SWE-bench Lite eval / [Klaud Cold] 移除 SWE-bench Lite 评测`. Keep prefixes on both halves. Fix English-only titles before review or merge.
- PR and issue bodies and human-authored PR comments keep English visible and put natural Simplified Chinese in one collapsed `<details><summary>中文</summary>` section. Leave code, commands, logs, model names, SKUs, flags, and identifiers untranslated. The CODEOWNER sign-off template is English-only. See [translation terminology](.github/AGENT_OPERATIONS.md#translation-terminology).
- **When a request conflicts with a repo rule, follow the rule and say so** in your reply and in the PR description: name the rule and what you did instead (for example, no fallback default, no history comment, no repo-scanning test).
- **Descriptions are high signal.** Lead with the problem, the change, and why it matters; keep risks, breaking changes, and unresolved failures visible. Put boilerplate (AI disclosure, change-type lists, checklists) in closed `<details>` blocks after the summary. Report validation only from real integration or end-to-end runs, with a run link; collapse supporting evidence; omit local-check inventories and empty or pending sections. Keep running appropriate checks. See [PR descriptions](CONTRIBUTING.md#pr-descriptions).
- Every PR body includes a collapsed `<details><summary>AI model disclosure</summary>` block naming each contributing model (including delegated agents) and its role, copying the identifier the runtime exposes; never guess one. Tool names are not models. If the exact model is unavailable, state that it could not be verified; human-only PRs state `No AI used`. Keep it current when another model edits the PR.
- Only one eligible CODEOWNER posts the PR Review Checklist; check for an existing one first, and edit it in place rather than posting another; post a replacement only if the original was deleted. See [sign-off](CONTRIBUTING.md#the-pr-review-checklist-codeowner-sign-off).
- Commit messages are conventional English. Contributor docs are English-only except READMEs; keep existing `README_zh.md` files in sync and preserve their language switchers; add no other translations.
- Klaud Cold reports use `infx.klaud`'s typed reporting commands and renderer per the [candidate instructions](.github/klaud-candidate-prompt.md); never hand-build baselines, deltas, receipts, or cleanup reports.

## Cross-repo changes

- Pareto logic changes update both InferenceX and InferenceX-app with matching regression tests and cross-linked PRs.
- `power_model/` changes must also update the InferenceX-app TypeScript port, which mirrors `power_model/src/power_model` file for file under `packages/app/src/lib/power/power_model/` and pins one upstream merge commit. A port fix that upstream also needs starts in InferenceX. Every `power_model/` PR description lists all four steps ([procedure](https://github.com/SemiAnalysisAI/InferenceX-app/blob/master/AGENTS.md#power-model--ts-port-and-upstream-python-must-stay-in-sync)):
  1. Land the InferenceX PR first, with matching tests.
  2. Open the app PR, cross-linked both ways, with matching tests.
  3. In the app PR, pin the InferenceX merge commit.
  4. In the app PR, regenerate the port manifest and the parity fixture.

## Code style

- Follow the nearest existing pattern. Python uses typed signatures and strict Pydantic schemas; YAML uses kebab-case fields.
- Python under `inferencex-e2e/infx/` uses all stable Ruff rules (exclusions in `infx/ruff.toml`), line length 100, and the Ruff formatter; CI's Lint job fails on any finding. Run the [local checks](#local-checks) before pushing Python changes and fix findings where practical; justified exceptions use inline `# noqa: CODE`, never file-wide ignores.
- Shared benchmark behavior that runs inside serving containers belongs in `inferencex-e2e/infx/bench/` as stdlib-only, Python 3.10 compatible commands (`python3 -m infx.bench <command>`) configured by env vars or flags. Bash entrypoints stay thin shims.

### Bash (mandatory)

Applies to active Bash scripts and shell embedded in workflows and recipes; leave deprecated code alone unless asked.

- **Configuration flows from the caller.** Workflows, master configs, runtime profiles, and launchers supply configuration explicitly; receiving scripts validate it and never silently choose defaults.
- **No fallback defaults** for caller-supplied configuration: no `${VAR:-default}`, `${VAR:=default}`, colon-free forms, or equivalent "if unset, assign" logic. A missing input is a caller error and must fail clearly. Pass `false` and `0` explicitly too.
- **Validate inputs with `check_env_vars`** from `inferencex-e2e/benchmarks/check_env.sh` (sourcing only defines it). Group required inputs near the top, after sourcing it; validate path-specific inputs on entering that path. It rejects missing and empty values and lists every missing name. Do not duplicate it or remove its safe handling of unset variables.
- **No nounset:** no `set -u`, `set -o nounset`, `set -euo pipefail`, or `bash -u`. Keep other options such as `set -eo pipefail`.
- **Preserve precedence and forwarding.** Apply caller settings before recipe overrides and forward required inputs across container or job boundaries; never replace a supported override with an unconditional assignment.
- Deliberate optional inputs, runtime-derived values, and unset-safe internal probes are fine; they are not licence to invent fallback configuration or replace a documented automatic selection with an arbitrary constant.

Example: replace `export IS_MULTINODE="${IS_MULTINODE:-true}"` in the receiver with `export IS_MULTINODE=true` in the caller, and `check_env_vars IS_MULTINODE MODEL_NAME PRECISION` in the receiver.

### Comments (mandatory, every file)

- No header comments describing a file, topology, config, recipe, override, or section.
- No history: how code evolved, what it replaced, where it came from, or past results. Git and `perf-changelog.yaml` are the record.
- Only technical explanation of a specific line or block: why a value was chosen, or context the code cannot show (hardware quirk, upstream bug, measured failure). One short line is the norm; never restate the code.

## Tests

**A test must run the real implementation on concrete inputs and assert on what it computes, returns, writes, or raises.** Tests that read source text, parse code structure, grep the repo, pin checked-in config, are tautological, reimplement the code, test fixtures, are smoke-only, or duplicate a covered branch must be deleted. Before adding or approving a test, answer:

1. What line does it run, and what bug would make it fail?
2. Would it still pass if the code were rewritten with identical behavior?
3. Would it fail because someone added a recipe, bumped an image tag, or reworded a comment?
4. Does an existing test already reach this branch?

Full rules: [test quality](CONTRIBUTING.md#test-quality).

## Local checks

Run from `inferencex-e2e/`:

```bash
uv sync --locked --all-extras --group test
uvx --exclude-newer PT12H ruff@latest check --fix infx
uvx --exclude-newer PT12H ruff@latest format infx
uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/ -n 4
```

Run the focused test path first. Use `--extra workflows` for GitHub integrations and `--extra results` for result tooling. Dependency changes update `pyproject.toml` and `uv.lock` via `uv add` or `uv lock --upgrade-package NAME`. Workflow changes run the [Zizmor CI command](.github/workflows/zizmor.yml) with online checks; do not weaken its policy. Schema and generator checks do not prove serving behavior: run the exact config, inspect its generated fields, then collect smoke, sweep, and eval evidence.

## Benchmark invariants (non-negotiable)

- Every priority-scheduled self-hosted job requests exactly one `nodes:N` label (physical Slurm nodes; single-node is `nodes:1`, multi-node forwards `node-count`). A queued job without the label is ineligible for priority scheduling, and labels cannot be added later; fix the branch and dispatch again.
- A change that alters what any benchmark runs needs a `perf-changelog.yaml` entry for the affected config keys; the entry triggers result collection. Refactors, recipe moves, and docs need none. `python -m infx.matrix.changed --base origin/main` (from `inferencex-e2e/`) lists affected config and recipe keys; if it lists none, add no entry. It does not see script or launcher code, so judge script and launcher changes by what they run.
- `perf-changelog.yaml` is append-only and byte-sensitive: preserve every existing byte and separator, append only at the tail, and resolve conflicts only with the [changelog conflict helper](.github/workflows/README.md#changelog-conflict-recovery). One PR has at most one block (listing one or more `config-keys`); revise it rather than adding another. New entries are English-only (the bilingual rules do not apply); leave historical entries unchanged.
- **`main` never runs sweeps.** A PR with a changelog entry needs an accepted `/use <run_id>` before merge, or its results are never ingested. Merge with `infx.workflows.merge_with_reuse`, except external PRs (pin their `e2e-tests.yml` run explicitly) and ingest-recovery PRs (merge directly, per the [recovery command](.claude/commands/recover-failed-ingest.md)). See [`/use`](CONTRIBUTING.md#reusing-your-prs-green-sweep-at-merge-with-use).
- Multi-node srt-slurm changes update the recipe and matching master config together. srt-slurm recipes, fixed-sequence and AgentX, are fragments: the binder supplies image, model, precision, and sequence lengths, and `inferencex-e2e/configs/srt-recipes/` supplies `benchmark.type` and `benchmark.command`; never set those in a recipe. Rows of one design share a `*-variants.yaml` ([recipe guide](inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/RECIPES.md)).
- Speculative fixed-sequence benchmarks use the chat template: `bind_workload` in `infx/srt_slurm/workload.py` sets `benchmark.env.USE_CHAT_TEMPLATE: "true"` for single-node recipes that speculate, which `python3 -m infx.bench fixed-seq` (run by `srt_fixed_sequence.sh`) turns into `--use-chat-template`.
- Benchmarks create no new directories under `/workspace`, and root containers leave no root-owned files in shared AMD runner workspaces.

## Launchers and hooks

- Launch only through `python -m infx.launch`; add no shell launchers. Cluster facts go in the cluster's `clusters:` record, workload rules in named policy tables; drivers never branch on a cluster id. Retiring a cluster removes its label, record, and policy rows in one PR.
- srt-slurm host-setup hooks only check or prepare hosts: they never run benchmarks, tune engines, patch containers, or hide failures.
- Never hard-code synthetic acceptance lengths; the srt driver picks measured values from `infx/golden_al_distribution/` ([details](inferencex-e2e/infx/golden_al_distribution/README.md#using-golden-curves-in-srt-slurm-runs)).

## Deprecating benchmark configs

- Delete retired entries (or, for partial retirement, only the retired scenarios) from the master config; never archive. Delete recipes, setup scripts, and assets no active config uses, rather than moving them to `deprecated/`.
- In the same PR, reconcile retirement statements in [`MODELS.md`](inferencex-e2e/docs/MODELS.md) with active configs and routing, preserving documented exceptions and planned (not completed) retirements.
- Remove unused retired-model rows from launch workload tables (`infx/launch/policy.py`, `infx/launch/drivers/srt/{lanes,models,power}.py`) and cluster `models.entries`, and fix guidance that recommends retired coverage. Audit callers before removing shared helpers; SPEED-Bench collectors and historical readers may still need them.

See also: [config fields](inferencex-e2e/configs/CONFIGS.md), [GitHub operations](.github/AGENT_OPERATIONS.md), [workflows](.github/workflows/README.md), [evals](inferencex-e2e/infx/evals/EVALS.md), [AgentX replay](inferencex-e2e/docs/agentx-standalone.md), and [ingest recovery](.claude/commands/recover-failed-ingest.md).
