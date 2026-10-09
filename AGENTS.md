# AGENTS.md

Guidance for AI agents working with InferenceX.

## Start here

1. **Start every task with [`inferencex-e2e/docs/index.md`](inferencex-e2e/docs/index.md).** Choose the one focused guide that matches the task. Do not load every documentation page.
2. Repository source, schemas, workflows, launchers, and collectors are authoritative. If documentation disagrees with implementation, follow the implementation and update the nearest English guide plus its Chinese counterpart.
3. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening or reviewing a PR or changing review, sweep, or merge policy. Sweep labels and modifiers: see [PR primary and modifier labels](inferencex-e2e/docs/ci-procedures.md#pr-primary-and-modifier-labels).
4. Before debugging a Klaud-Cold or `claude/*` image-bump PR, read [`inferencex-e2e/docs/klaud.md`](inferencex-e2e/docs/klaud.md) and the [known failure signatures](inferencex-e2e/docs/troubleshooting.md#known-failure-signatures).

The end-to-end Python project owns `inferencex-e2e/pyproject.toml`, `inferencex-e2e/uv.lock`, and `inferencex-e2e/.python-version`. Run its `uv` commands from `inferencex-e2e/`; root-level automation can select it with `uv run --project inferencex-e2e`.

## Agent-specific policy

- Pareto logic changes must update both InferenceX and InferenceX-app with matching regression tests and cross-linked PRs.
- **Keep PR descriptions high signal and low noise.** Lead with a short explanation of the problem, what changes, and why it matters to a human reviewer; keep material risks, breaking changes, and unresolved failures visible. Put AI disclosure, change-type lists, author checklists, and other administrative boilerplate in clearly named `<details><summary>…</summary>` blocks, without `open`, after the summary. Report validation only from actual integration or end-to-end runs, with a concise outcome and run link when available; collapse supporting evidence. Omit routine local-check inventories and empty or pending validation sections. Continue running appropriate checks. See [PR descriptions](CONTRIBUTING.md#pr-descriptions).
- Every PR description must include an **AI model disclosure** section inside a collapsed-by-default `<details><summary>AI model disclosure</summary>` block, naming the exact model/version used to prepare the PR. List each contributing model and its role, including delegated agents. Tool names such as Claude Code, Cursor, or Perplexity Computer are not model identities. Copy the model identifier exposed by the runtime; do not guess an unavailable identifier. If the runtime does not expose the exact model, explicitly state that it could not be verified. Human-only PRs must state `No AI used`. Keep the disclosure current when later edits use another model.
- Repository skills are canonical under `.agents/skills/`. Add or update skills there. `.claude/skills/` contains compatibility symlinks for Claude discovery.
- **PR titles MUST be bilingual:** every PR title MUST use `<English title> / <中文标题>`, e.g. `[Klaud Cold] Remove SWE-bench Lite eval / [Klaud Cold] 移除 SWE-bench Lite 评测`. Keep any prefix such as `[Klaud Cold]` on both halves. An English-only title is non-compliant; fix it before requesting review or merging. Issue titles follow the same format.
- PR and issue descriptions and human-authored PR comments must include English and natural Simplified Chinese. In bodies and comments, keep English visible and put Chinese in one collapsed `<details><summary>中文</summary>` section. Keep code, commands, logs, stack traces, model names, hardware SKUs, framework names, flags, and identifiers unchanged. The exact CODEOWNER sign-off template is English-only. See [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#translation-terminology).
- **One reviewer checklist per PR:** Only one eligible CODEOWNER reviewer needs to post the completed PR Review Checklist. Check for an existing checklist before posting; other reviewers do not need to duplicate it. The original reviewer must edit their existing checklist comment when correcting items or adding evidence, rather than post a new checklist. Create a replacement only if the original was deleted. See [`CONTRIBUTING.md`](CONTRIBUTING.md#the-pr-review-checklist-codeowner-sign-off).
- **Klaud Cold reports:** Follow the compact body/comment templates in [`inferencex-e2e/docs/klaud-reporting.md`](inferencex-e2e/docs/klaud-reporting.md), including cleanup and completion reports.
- Commit subjects use conventional English style, while commit bodies include the Chinese translation. Contributor-facing docs use English as the source version and ship with a synchronized `_zh.md` page and language switcher.
- Python under `inferencex-e2e/infx/` uses all stable Ruff rules with reviewed exclusions in `inferencex-e2e/infx/ruff.toml`, line length 100, and the Ruff formatter. The Lint job in `.github/workflows/ci.yml` runs whenever Python files change and fails on any finding. Before pushing Python changes, run the [commands in the testing guide](inferencex-e2e/docs/testing.md#python-lint-and-formatting). Fix findings where practical; justified exceptions use inline `# noqa: CODE` rather than file-wide ignores.
- Follow the nearest existing pattern. Python uses typed signatures and strict Pydantic schemas. YAML uses kebab-case fields. Shared benchmark behavior that runs inside serving containers belongs in `inferencex-e2e/infx/bench/` as stdlib-only, Python 3.10 compatible commands (`python3 -m infx.bench <command>`), with parameters passed through environment variables or flags. Bash entrypoints stay thin shims.

## Bash conventions (mandatory)

These rules apply to active Bash scripts and shell commands embedded in workflows and recipes. Follow them when adding, changing, or reviewing Bash code. Leave deprecated code alone unless explicitly asked to update it.

- **Configuration flows from the caller.** Workflows, master configs, runtime profiles, and launchers explicitly supply configuration to the scripts they invoke. Receiving scripts consume and validate those inputs; they must not silently choose defaults.
- **No fallback defaults for caller-supplied configuration.** Avoid `${VAR:-default}`, `${VAR:=default}`, their colon-free equivalents, and equivalent "if unset, assign a default" logic. A missing input is a caller error and must fail clearly. Pass values such as `false` and `0` explicitly too.
- **Validate every required environment input with `check_env_vars` before use.** Source the shared helper from `inferencex-e2e/benchmarks/check_env.sh`; sourcing it only defines the function. Group required inputs near the beginning, after sourcing the helper; validate inputs used only by a particular execution path when entering that path. The helper rejects both missing and empty values and lists every missing name. Do not duplicate it or remove its safe handling of unset variables.
- **Do not enable nounset.** No `set -u`, `set -o nounset`, combined flags such as `set -euo pipefail`, or `bash -u` invocation flags. Use explicit validation; preserve other intended shell options, for example `set -eo pipefail`.
- **Preserve configuration precedence and forwarding.** Apply caller-owned settings before recipe-specific overrides, and explicitly forward required inputs across container or job boundaries. Do not replace a supported override with an unconditional assignment in the receiving script.
- Preserve deliberate optional-input handling, runtime-derived values, and unset-safe internal-state probes. These are not permission to invent fallback configuration or replace a documented automatic selection with an arbitrary constant.

For example, remove this from the receiving script:

```bash
export IS_MULTINODE="${IS_MULTINODE:-true}"
```

Set it in the responsible caller:

```bash
export IS_MULTINODE=true
```

Then validate it in the receiving script after sourcing the shared helper:

```bash
check_env_vars IS_MULTINODE MODEL_NAME PRECISION
```

## Deprecating benchmark configs

- Delete retired entries from the active master config; do not archive them. Git history and `inferencex-e2e/perf-changelog.yaml` are the record of past settings. For a partial deprecation, remove only the retired scenarios and retain the supported scenarios in the active entry.
- Check retirement statements in [`inferencex-e2e/docs/MODELS.md`](inferencex-e2e/docs/MODELS.md) against active configs and script routing in the same PR, and update `inferencex-e2e/docs/MODELS.md` plus `inferencex-e2e/docs/MODELS_zh.md` together. Preserve explicitly documented exceptions and conditional retirement policies; do not treat planned retirement as completed.
- Remove unused retired-model rows from the launch workload tables (shared ones in `inferencex-e2e/infx/launch/policy.py`, srt-slurm ones in `inferencex-e2e/infx/launch/drivers/srt/{lanes,models,power}.py`) and cluster `models.entries`, and update workflow/agent guidance that still recommends retired coverage. Audit callers before removing shared helpers; retained SPEED-Bench collectors and historical result readers may still need model-specific support.
- Delete recipes, setup scripts and other assets that no active config uses any more rather than moving them to a `deprecated/` directory.

## Launchers, hooks, and synthetic acceptance

- Launch only through `python -m infx.launch`; do not add shell launchers. Cluster facts go in the cluster's `clusters:` record, workload rules in named policy tables, and drivers never branch on a cluster id. Retiring a cluster removes its label, record, and policy rows in the same PR. Details: [launch mechanics](inferencex-e2e/docs/architecture.md#launch-mechanics-stay-in-cluster-records).
- srt-slurm host-setup hooks only check or prepare hosts. They never run benchmarks, tune engines, patch containers, or hide failures. Details: [host-setup hooks](inferencex-e2e/docs/architecture.md#srt-slurm-host-setup-hooks).
- Never hard-code synthetic acceptance lengths. The srt driver selects the measured value from `inferencex-e2e/infx/golden_al_distribution/` automatically. Details: [using golden curves](inferencex-e2e/infx/golden_al_distribution/README.md#using-golden-curves-in-srt-slurm-runs).

## Test quality

**The one rule: a test must exercise the real implementation with concrete inputs and assert on what it computes, returns, writes, or raises. A test that inspects the code, the repo, or a config file instead of running behavior is not a test and must be deleted.** These rules are mandatory for every test added, modified, or reviewed in this repository. When in doubt, delete the test.

### Forbidden: tests about the code rather than its behavior

Never write, and always delete on sight, a test that does any of the following:

1. **Reads source text and asserts on it.** Opening a `.sh`, `.py`, `.yml`, `.yaml`, `.cjs`, or `.md` file and asserting that a string, flag, regex, command, or line is present or absent, counting occurrences, or checking line order. This includes launchers, workflow files, skill files, and docs. Grepping is not testing.
2. **Parses source structure.** Using `ast.parse`, `inspect.getsource`, `inspect.signature`, `hasattr`, `callable`, `__doc__`, or import-succeeds checks to assert that a function, class, constant, argument, or flag exists or has a given shape.
3. **Git-greps the repo.** Asserting which files contain a literal, how many files match, or that a pin appears in exactly N places.
4. **Pins checked-in config or data.** Asserting the contents of a recipe, master config, `runners.yaml`, `platform_config.json`, a registry dict, an enum, an image tag, a SHA, a port number, or the current count of recipes, SKUs, backends, or models. Validate config through the real validation code with controlled inputs instead.
5. **Is tautological.** Asserting a constant equals its own literal; asserting a dict or fixture equals what the test just built; asserting only that a mock was called with the arguments the test itself passed; or computing the expected value with the same helper, formula, or algorithm the test is supposed to check.
6. **Reimplements the code under test.** Any parser, filter, jq/YAML expression, argparse tree, formula, or state machine copied into the test file so the test can run against the copy. This also covers "mirror" parsers and "reference specs" cross-checked against a second in-test implementation.
7. **Tests the test infrastructure.** Tests of fixtures, conftest helpers, in-test expression evaluators, or "this test has teeth" self-checks.
8. **Is smoke-only.** Module imports, `--help` exits 0, or "does not raise" with no assertion on output.
9. **Duplicates a covered path.** Several tests that reach the same branch with trivially different inputs. Keep one, or use `pytest.mark.parametrize` / `subTest`. A second test is justified only by a distinct branch, error path, or boundary.

### Required: what every kept test looks like

- Feeds small, controlled inputs into the real function, CLI, or script and asserts on the computed output, written artifact, exit code, or raised error.
- Uses expected values worked out independently by hand, never derived by calling the implementation or its helpers.
- Covers a specific branch, boundary, malformed input, or failure path that no other test already covers.
- Mocks only external collaborators (network, GitHub, Slurm, clocks, GPUs), never the behavior under test. Shell scripts are tested by running them with stubbed binaries on `PATH` and checking what they produced, not by reading their text.
- Would fail on a plausible regression in observable behavior, and would not fail on a harmless refactor, a rename, or the addition of a valid recipe or SKU.

### Before adding or approving a test, answer all four

1. Which line of the real implementation does this run, and what bug in it would make the assertion fail?
2. Would this test still pass if the code were rewritten with identical behavior? If not, it is testing structure and must go.
3. Would this test fail because someone added a recipe, bumped an image tag, or reworded a comment? If yes, it is pinning config or source and must go.
4. Does an existing test already reach this branch? If yes, extend it or drop the new one.

Deleting a test that fails these questions needs no replacement. Do not preserve test counts. See [the testing guide](inferencex-e2e/docs/testing.md#test-quality) for timing and contract tests, and [Randy Coulman's Tautological Tests](https://randycoulman.com/blog/2016/12/20/tautological-tests/) for the distinction between independent expectations and assertions that repeat the implementation.

## Non-negotiable benchmark invariants

- Every priority-scheduled benchmark job on a self-hosted cluster must request exactly one `nodes:N` label, where `N` is the positive integer number of physical Slurm nodes required. Single-node jobs use `nodes:1`; generated multi-node jobs must forward their computed `node-count`. A queued job missing this label is ineligible for priority scheduling, and labels cannot be added retroactively, so fix the source branch and dispatch a new run.
- Every change that can affect benchmark performance and every recipe addition or modification requires a new `inferencex-e2e/perf-changelog.yaml` entry. The file is append-only and byte-sensitive. Preserve all existing bytes and separator whitespace, and append only at the tail.
- One PR maps to one `inferencex-e2e/perf-changelog.yaml` block. By convention, a PR with a perf changelog entry is a change that affects inference performance or otherwise requires a rerun to collect up-to-date results. That block may list one or more `config-keys` to run, but do not add a second block for the same PR, even when the change evolved across commits. Revise the PR's existing block so it describes the final state of the change.
- New `inferencex-e2e/perf-changelog.yaml` entries must be English-only. Do not add Chinese translations or bilingual descriptions; the bilingual documentation and GitHub-content rules do not apply to these entries. Leave historical entries unchanged.
- Multi-node srt-slurm changes update the recipe YAML and matching master config together. srt-slurm recipes, fixed-sequence and AgentX, are fragments that get the master image, model, precision (and fixed-sequence lengths) from the binder.
- Every speculative fixed-sequence benchmark renders prompts with the chat template: the binder (`inferencex-e2e/infx/srt_slurm/workload.py::bind_workload`) sets `benchmark.env.USE_CHAT_TEMPLATE: "true"` for single-node recipes that speculate, which `python3 -m infx.bench fixed-seq` (run by `srt_fixed_sequence.sh`) turns into `--use-chat-template` for the benchmark client.
- Benchmarks create no new directories under `/workspace`. Root containers must not leave root-owned files in shared AMD runner workspaces.
- Generated configuration is not runtime proof. Run the narrowest local check, then the applicable smoke, sweep, or eval procedure from the [task routing table](inferencex-e2e/docs/index.md#task-routing).

All repository maps, task routes, commands, schemas, sweep semantics, artifact contracts, recovery steps, and detailed conventions live behind [`inferencex-e2e/docs/index.md`](inferencex-e2e/docs/index.md).
