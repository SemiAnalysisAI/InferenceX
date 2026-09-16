# AGENTS.md

Guidance for AI agents working with InferenceX.

## Start here

1. **Start every task with [`docs/index.md`](docs/index.md).** Choose the one focused guide that matches the task. Do not load every documentation page.
2. Repository source, schemas, workflows, launchers, and collectors are authoritative. If documentation disagrees with implementation, follow the implementation and update the nearest English guide plus its Chinese counterpart.
3. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening or reviewing a PR or changing review, sweep, or merge policy.
4. Read [`KLAUD_DEBUG.md`](KLAUD_DEBUG.md) before debugging a Klaud-Cold or `claude/*` image-bump PR.

## Agent-specific policy

- Repository skills are canonical under `.agents/skills/`. Add or update skills there. `.claude/skills/` contains compatibility symlinks for Claude discovery.
- PR and issue titles, descriptions, and human-authored PR comments must include English and natural Simplified Chinese. Titles use `<English title> / <中文标题>`. In bodies and comments, keep English visible and put Chinese in one collapsed `<details><summary>中文</summary>` section. Keep code, commands, logs, stack traces, model names, hardware SKUs, framework names, flags, and identifiers unchanged. The exact CODEOWNER sign-off template is English-only. See [`docs/documentation-procedures.md`](docs/documentation-procedures.md) and [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#translation-terminology).
- **One reviewer checklist per PR:** Only one eligible CODEOWNER reviewer needs to post the completed PR Review Checklist. Check for an existing checklist before posting; other reviewers do not need to duplicate it. The original reviewer must edit their existing checklist comment when correcting items or adding evidence, rather than post a new checklist. Create a replacement only if the original was deleted. See [`CONTRIBUTING.md`](CONTRIBUTING.md#the-pr-review-checklist-codeowner-sign-off).
- **Klaud Cold reports:** Follow the compact body/comment templates in [`docs/klaud-reporting.md`](docs/klaud-reporting.md), including cleanup and completion reports.
- Commit subjects use conventional English style, while commit bodies include the Chinese translation. Contributor-facing docs use English as the source version and ship with a synchronized `_zh.md` page and language switcher.
- Python under `infx/` uses all stable Ruff rules with reviewed exclusions in `infx/ruff.toml`, line length 100, and the Ruff formatter. The Lint job in `.github/workflows/ci.yml` runs whenever Python files change and fails on any finding. Before pushing Python changes, run the [commands in the testing guide](docs/testing.md#python-lint-and-formatting). Fix findings where practical; justified exceptions use inline `# noqa: CODE` rather than file-wide ignores.
- Follow the nearest existing pattern. Python uses typed signatures and strict Pydantic schemas. YAML uses kebab-case fields. Shared benchmark Bash behavior belongs in `benchmark_lib.sh`, with parameters passed through environment variables.

## Bash conventions (mandatory)

These rules apply to active Bash scripts and shell commands embedded in workflows and recipes. Follow them when adding, changing, or reviewing Bash code. Leave deprecated code alone unless explicitly asked to update it.

- **Configuration flows from the caller.** Workflows, master configs, runtime profiles, and launchers explicitly supply configuration to the scripts they invoke. Receiving scripts consume and validate those inputs; they must not silently choose defaults.
- **No fallback defaults for caller-supplied configuration.** Avoid `${VAR:-default}`, `${VAR:=default}`, their colon-free equivalents, and equivalent "if unset, assign a default" logic. A missing input is a caller error and must fail clearly. Pass values such as `false` and `0` explicitly too.
- **Validate every required environment input with `check_env_vars` before use.** Use the shared helper in `benchmarks/benchmark_lib.sh`. Group required inputs near the beginning, after sourcing the helper; validate inputs used only by a particular execution path when entering that path. The helper rejects both missing and empty values. Do not duplicate it or remove its safe handling of unset variables. Callers needing validation without benchmark initialization can source the library with `--validation-only`.
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

Deleting a test that fails these questions needs no replacement. Do not preserve test counts. See [the testing guide](docs/testing.md#test-quality) for the reasoning and [Randy Coulman's Tautological Tests](https://randycoulman.com/blog/2016/12/20/tautological-tests/) for the distinction between independent expectations and assertions that repeat the implementation.

## Non-negotiable benchmark invariants

- Every priority-scheduled benchmark job on a self-hosted cluster must request exactly one `nodes:N` label, where `N` is the positive integer number of physical Slurm nodes required. Single-node jobs use `nodes:1`; generated multi-node jobs must forward their computed `node-count`. A queued job missing this label is ineligible for priority scheduling, and labels cannot be added retroactively, so fix the source branch and dispatch a new run.
- Every change that can affect benchmark performance and every recipe addition or modification requires a new `perf-changelog.yaml` entry. The file is append-only and byte-sensitive. Preserve all existing bytes and separator whitespace, and append only at the tail.
- Multi-node srt-slurm changes update the recipe YAML and matching master config together. For image bumps, `model.container` must equal `image`.
- Every `*_mtp.sh` passes `--use-chat-template` to `run_benchmark_serving`.
- Benchmarks create no new directories under `/workspace`. Root containers must not leave root-owned files in shared AMD runner workspaces.
- Generated configuration is not runtime proof. Run the narrowest local check, then the applicable smoke, sweep, or eval procedure from [`docs/procedures.md`](docs/procedures.md).

All repository maps, task routes, commands, schemas, sweep semantics, artifact contracts, recovery steps, and detailed conventions live behind [`docs/index.md`](docs/index.md).
