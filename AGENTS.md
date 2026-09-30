# AGENTS.md

Guidance for AI agents working with InferenceX.

## Start here

1. **Start every task with [`inferencex-e2e/docs/index.md`](inferencex-e2e/docs/index.md).** Choose the one focused guide that matches the task. Do not load every documentation page.
2. Repository source, schemas, workflows, launchers, and collectors are authoritative. If documentation disagrees with implementation, follow the implementation and update the nearest English guide plus its Chinese counterpart.
3. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening or reviewing a PR or changing review, sweep, or merge policy.
4. Read [`inferencex-e2e/docs/KLAUD_DEBUG.md`](inferencex-e2e/docs/KLAUD_DEBUG.md) before debugging a Klaud-Cold or `claude/*` image-bump PR.

The end-to-end Python project owns `inferencex-e2e/pyproject.toml`, `inferencex-e2e/uv.lock`, and `inferencex-e2e/.python-version`. Run its `uv` commands from `inferencex-e2e/`; root-level automation can select it with `uv run --project inferencex-e2e`.

## Agent-specific policy

- Pareto logic changes must update both InferenceX and InferenceX-app with matching regression tests and cross-linked PRs.
- Every PR description must include an **AI model disclosure** section naming the exact model/version used to prepare the PR. List each contributing model and its role, including delegated agents. Tool names such as Claude Code, Cursor, or Perplexity Computer are not model identities. Copy the model identifier exposed by the runtime; do not guess an unavailable identifier. If the runtime does not expose the exact model, explicitly state that it could not be verified. Human-only PRs must state `No AI used`. Keep the disclosure current when later edits use another model.
- Repository skills are canonical under `.agents/skills/`. Add or update skills there. `.claude/skills/` contains compatibility symlinks for Claude discovery.
- **PR titles MUST be bilingual:** every PR title MUST use `<English title> / <中文标题>`, e.g. `[Klaud Cold] Remove SWE-bench Lite eval / [Klaud Cold] 移除 SWE-bench Lite 评测`. Keep any prefix such as `[Klaud Cold]` on both halves. An English-only title is non-compliant; fix it before requesting review or merging. Issue titles follow the same format.
- PR and issue descriptions and human-authored PR comments must include English and natural Simplified Chinese. In bodies and comments, keep English visible and put Chinese in one collapsed `<details><summary>中文</summary>` section. Keep code, commands, logs, stack traces, model names, hardware SKUs, framework names, flags, and identifiers unchanged. The exact CODEOWNER sign-off template is English-only. See [`inferencex-e2e/docs/documentation-procedures.md`](inferencex-e2e/docs/documentation-procedures.md) and [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#translation-terminology).
- **One reviewer checklist per PR:** Only one eligible CODEOWNER reviewer needs to post the completed PR Review Checklist. Check for an existing checklist before posting; other reviewers do not need to duplicate it. The original reviewer must edit their existing checklist comment when correcting items or adding evidence, rather than post a new checklist. Create a replacement only if the original was deleted. See [`CONTRIBUTING.md`](CONTRIBUTING.md#the-pr-review-checklist-codeowner-sign-off).
- **Klaud Cold reports:** Follow the compact body/comment templates in [`inferencex-e2e/docs/klaud-reporting.md`](inferencex-e2e/docs/klaud-reporting.md), including cleanup and completion reports.
- Commit subjects use conventional English style, while commit bodies include the Chinese translation. Contributor-facing docs use English as the source version and ship with a synchronized `_zh.md` page and language switcher.
- Python under `inferencex-e2e/infx/` uses all stable Ruff rules with reviewed exclusions in `inferencex-e2e/infx/ruff.toml`, line length 100, and the Ruff formatter. The Lint job in `.github/workflows/ci.yml` runs whenever Python files change and fails on any finding. Before pushing Python changes, run the [commands in the testing guide](inferencex-e2e/docs/testing.md#python-lint-and-formatting). Fix findings where practical; justified exceptions use inline `# noqa: CODE` rather than file-wide ignores.
- Follow the nearest existing pattern. Python uses typed signatures and strict Pydantic schemas. YAML uses kebab-case fields. Shared benchmark Bash behavior belongs in `benchmark_lib.sh`, with parameters passed through environment variables.

## Bash conventions (mandatory)

These rules apply to active Bash scripts and shell commands embedded in workflows and recipes. Follow them when adding, changing, or reviewing Bash code. Leave deprecated code alone unless explicitly asked to update it.

- **Configuration flows from the caller.** Workflows, master configs, runtime profiles, and launchers explicitly supply configuration to the scripts they invoke. Receiving scripts consume and validate those inputs; they must not silently choose defaults.
- **No fallback defaults for caller-supplied configuration.** Avoid `${VAR:-default}`, `${VAR:=default}`, their colon-free equivalents, and equivalent "if unset, assign a default" logic. A missing input is a caller error and must fail clearly. Pass values such as `false` and `0` explicitly too.
- **Validate every required environment input with `check_env_vars` before use.** Use the shared helper in `inferencex-e2e/benchmarks/benchmark_lib.sh`. Group required inputs near the beginning, after sourcing the helper; validate inputs used only by a particular execution path when entering that path. The helper rejects both missing and empty values. Do not duplicate it or remove its safe handling of unset variables. Callers needing validation without benchmark initialization can source the library with `--validation-only`.
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

## Launchers (one Python entrypoint, cluster records, named policy)

- The reusable workflows run `python -m infx.launch run` from the measured project root. It resolves the cluster from `RUNNER_NAME` through the `cluster:<id>` labels in [`inferencex-e2e/configs/runners.yaml`](inferencex-e2e/configs/runners.yaml). Then [`launch_path`](inferencex-e2e/infx/launch/policy.py) picks one driver under [`inferencex-e2e/infx/launch/drivers/`](inferencex-e2e/infx/launch/drivers): `srt` (srt-slurm recipes), `script` (explicit `BENCH_SCRIPT_OVERRIDE` collectors such as SPEED-Bench), or `legacy` (the remaining pre-srt-slurm lanes, slated for removal). See [Stage 4 in `inferencex-e2e/docs/architecture.md`](inferencex-e2e/docs/architecture.md#stage-4-launcher-and-runtime-execution).
- Cluster facts (node shape, the workload `env`, staged models, and the scheduler's own settings: for Slurm the partition, account, volumes, squash cache and srt-slurm profile under `slurm:`) belong in that cluster's `clusters:` record, not in driver code. Workload rules keyed by model, framework, precision, or recipe (model aliases, `/ix` workspaces, power eligibility, time bumps, TileRT UCX settings) belong in named tables: shared ones in [`inferencex-e2e/infx/launch/policy.py`](inferencex-e2e/infx/launch/policy.py), ones a single driver reads beside it (for srt-slurm `drivers/srt/lanes.py`, `models.py`, `power.py`). Never branch on a cluster id inside a driver. Drivers reach the scheduler only through the cluster's backend ([`inferencex-e2e/infx/launch/backends/`](inferencex-e2e/infx/launch/backends)). A new scheduler is new files plus two registry entries: its settings model (with its own volume type) under `inferencex-e2e/infx/clusters/`, registered in `infx.clusters.SCHEDULERS`, and its backend under `inferencex-e2e/infx/launch/backends/`, registered in `BACKENDS`. Clusters on it run script-driver points (`BENCH_SCRIPT_OVERRIDE`, such as SPEED-Bench) only; srt-slurm and legacy points need Slurm and fail there before any work.
- Every revision launches through `python -m infx.launch`; there is no shell-launcher fallback. Do not add shell launchers.
- When a cluster is retired, delete its `cluster:<id>` label, its `clusters:` record, and any policy rows keyed by its id in the same PR.

## SRT Slurm cluster hooks

- Put reusable host-check functions in `inferencex-e2e/runners/srt-slurm/hooks/common.sh`. Sourcing it must only define functions, without running checks, changing environment variables, or initializing benchmarks. Cluster-only helpers stay beside their setup script.
- Keep cluster-specific host prerequisites in `inferencex-e2e/runners/srt-slurm/hooks/<cluster>/setup.sh`, invoked explicitly by that cluster's `srt-slurm.host-setup` record in `inferencex-e2e/configs/runners.yaml`. These run after allocation, before services and workers start.
- Hooks are only for checks and setup required by that cluster's hosts or fabric. Keep them small, workload-independent, and safe to run repeatedly. Prefer native srt-slurm configuration whenever it can express the requirement.
- Do not put benchmark execution, model selection, engine flags, concurrency tuning, evaluation, result collection, or job orchestration in hooks. Those belong in recipes, benchmark scripts, or the existing orchestration layer.
- Do not use hooks to patch engines or containers, bypass failed checks, or hide runtime bugs behind retries and ad hoc workarounds. Fix problems in the component that owns them.
- Pass settings explicitly from the cluster record (`srt-slurm.host-setup.env`). Scope mutations to the allocated nodes, preserve other jobs' resources, and register teardown for temporary state that needs restoring. See [cluster profiles](inferencex-e2e/docs/configuration-procedures.md#cluster-profiles).

## SRT Slurm synthetic acceptance

- **Do not hard-code synthetic acceptance lengths in SRT recipes, master configs, or launchers.** InferenceX automatically selects the measured value from [`inferencex-e2e/infx/golden_al_distribution/`](inferencex-e2e/infx/golden_al_distribution) for speculative AgentX throughput runs. Do not add manual `SYNTHETIC_ACCEPTANCE_LENGTH`, vLLM `synthetic_acceptance_length`, SGLang `SGLANG_SIMULATE_ACC_LEN`, or TRT-LLM `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` settings.
- Submit recipes through the srt driver ([`inferencex-e2e/infx/launch/drivers/srt/`](inferencex-e2e/infx/launch/drivers/srt)). It runs the [`inferencex-e2e/infx/srt_slurm` connector](inferencex-e2e/infx/srt_slurm/synthetic_acceptance.py), which applies native SRT `--set` / `--unset` overrides. Calling upstream `srtctl` directly does not perform InferenceX's automatic selection.
- Keep the actual speculative method, draft model, draft-token count, and relevant sampling settings explicit in the recipe. The connector combines the generation role's settings (decode, otherwise aggregated), after caller overrides, with `MODEL_PREFIX` and `THINKING_MODE` to select the golden curve. For Kimi DSpark, explicitly set `draft_sample_method` to `greedy` or `probabilistic`.
- Eval-only and non-AgentX runs use real verification; the connector removes stale synthetic settings. Non-speculative roles do not receive simulation settings. `RUN_EVAL` does not disable simulation for the throughput portion.
- Missing golden curves or unmeasured draft lengths fail before submission. Add the corresponding measured golden data when supporting a new combination; do not work around the error with a guessed or hard-coded acceptance length.

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

Deleting a test that fails these questions needs no replacement. Do not preserve test counts. See [the testing guide](inferencex-e2e/docs/testing.md#test-quality) for the reasoning and [Randy Coulman's Tautological Tests](https://randycoulman.com/blog/2016/12/20/tautological-tests/) for the distinction between independent expectations and assertions that repeat the implementation.

## Non-negotiable benchmark invariants

- Every priority-scheduled benchmark job on a self-hosted cluster must request exactly one `nodes:N` label, where `N` is the positive integer number of physical Slurm nodes required. Single-node jobs use `nodes:1`; generated multi-node jobs must forward their computed `node-count`. A queued job missing this label is ineligible for priority scheduling, and labels cannot be added retroactively, so fix the source branch and dispatch a new run.
- Every change that can affect benchmark performance and every recipe addition or modification requires a new `inferencex-e2e/perf-changelog.yaml` entry. The file is append-only and byte-sensitive. Preserve all existing bytes and separator whitespace, and append only at the tail.
- Multi-node srt-slurm changes update the recipe YAML and matching master config together. For image bumps, `model.container` must equal `image`.
- Every speculative fixed-sequence benchmark renders prompts with the chat template: single-node srt-slurm recipes that speculate set `benchmark.env.USE_CHAT_TEMPLATE: "true"` (enforced by `inferencex-e2e/infx/srt_slurm/single_node.py::validate_recipe`), which `srt_fixed_sequence.sh` turns into `--use-chat-template` for `run_benchmark_serving`.
- Benchmarks create no new directories under `/workspace`. Root containers must not leave root-owned files in shared AMD runner workspaces.
- Generated configuration is not runtime proof. Run the narrowest local check, then the applicable smoke, sweep, or eval procedure from [`inferencex-e2e/docs/procedures.md`](inferencex-e2e/docs/procedures.md).

All repository maps, task routes, commands, schemas, sweep semantics, artifact contracts, recovery steps, and detailed conventions live behind [`inferencex-e2e/docs/index.md`](inferencex-e2e/docs/index.md).
