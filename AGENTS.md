# AGENT.md

Guidance for AI agents working with InferenceX.

> **Mandatory reading: [`CONTRIBUTING.md`](CONTRIBUTING.md)** — read it before opening or reviewing any PR. It covers the full PR review flow, the CODEOWNER sign-off process, the `/reuse-sweep-run` merge path, post-merge responsibilities, and critical cluster rules (e.g. never leaving root-owned files on AMD runners).

> **Translation quality bar (for the bilingual `_zh` docs — see Code Conventions › Docs):** write natural technical Chinese as used by ML infra engineers, not word-for-word machine translation. Follow the style of [`vllm-project/vllm-ascend` `README.zh.md`](https://github.com/vllm-project/vllm-ascend/blob/main/README.zh.md): translate concepts into idiomatic Chinese while preserving model names, hardware SKUs (MI355X, B300, GB200 ...), framework names (vLLM, SGLang, ATOM ...), flags, and CLI/env-var identifiers in English. Use parenthetical English clarification for acronyms on first use, e.g. 混合专家(MOE), 专家并行(EP). Preferred term mappings:
>
> | English | Chinese |
> |---|---|
> | benchmark | 基准测试 |
> | image (Docker) | 镜像 |
> | config / configuration | 配置 |
> | single-node / multi-node | 单节点 / 多节点 |
> | speculative decoding | 投机解码 |
> | inference | 推理 |
> | throughput | 吞吐量 |
> | latency | 延迟 |
> | prefill / decode | 预填充 / 解码 |
> | disaggregated (serving) | 分离式（推理） |
> | expert parallelism | 专家并行 |
> | sweep | 扫描 |
> | launcher | 启动器 |
> | artifact | 产物 |
> | evaluation / eval | 评估 |

> **Before debugging a failing Klaud-Cold / claude/* image-bump PR, read [`KLAUD_DEBUG.md`](KLAUD_DEBUG.md).** It captures recurring failure modes (vLLM CUDA-graph OOM, B300 sglang regressions, cluster docker/perms/disk issues), the exact workarounds, and gh-CLI gotchas — most cron-PR failures are already cataloged there.

## Project Overview

InferenceX is an open-source automated benchmarking system that tracks LLM inference performance across hardware (NVIDIA B200/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results published to https://inferencex.com/.

## Directory Structure

Run `ls` for details. Key paths:

- `perf-changelog.yaml` - benchmark trigger log; append-only; preserve whitespace.
- `benchmarks/` - `benchmark_lib.sh` (shared helpers); `single_node/` and `multi_node/` entrypoints; `*_mtp.sh` for MTP/spec-decoding; `multi_node/srt-slurm-recipes/` checked-in external recipe YAMLs.
- `runners/` - hardware launcher scripts.
- `utils/matrix_logic/` - `generate_sweep_configs.py`, `validation.py` Pydantic schemas, tests.
- `utils/bench_serving/` - `benchmark_serving.py` and backends.
- `utils/evals/` - lm-eval task configs, thresholds, `validate_scores.py` (see `EVALS.md`).
- `utils/` - `process_result.py`, `process_changelog.py` (incl. `trim_conc`), `collect_*.py`, `compare_results.py`.
- `experimental/` - non-core experiments.

## Terminology

STP (Single Token Prediction): vanilla autoregressive decoding, one token per forward pass, no speculative decoding. MTP (Multi-Token Prediction): predicts multiple tokens per forward pass via speculative decoding (EAGLE, NEXTN, etc.).

## Development Workflow

Tests: `python -m pytest utils/matrix_logic/ -v` (markers: `slow`, `integration`).

Generate configs:

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml \
  [--model-prefix dsr1|gptoss|dsv4|...] \
  [--framework sglang|trt|vllm|atom|dynamo-trt|dynamo-sglang] \
  [--precision fp4|fp8|...] \
  [--runner-type b200|h100|h200|gb200|...]
```

Process results: `python utils/process_result.py`.

## Supported Configuration Values

Frameworks: `sglang`, `trt`, `vllm`, `atom`, `dynamo-trt`, `dynamo-sglang`, `sglang-disagg`.
Sequence lengths (ISL/OSL): `1k1k` (1024/1024), `8k1k` (8192/1024).

## Code Conventions

Python: type hints (`list[str]`, `Optional[int]`), Pydantic with `extra='forbid'`, field aliases `Field(alias="model-prefix")`, docstrings on functions.

YAML: kebab-case field names (`model-prefix`, `conc-start`, `dp-attn`). Master configs define all benchmark configurations. `perf-changelog.yaml` triggers which configs to benchmark and is read chronologically (oldest at top, newest at bottom) - new entries MUST be appended to the END, never inserted in the middle or prepended.

Bash: source shared utilities via `source benchmark_lib.sh` (`check_env_vars`, `wait_for_server_ready`, `run_benchmark_serving`, `run_eval`, `append_lm_eval_summary`); parameters passed via env vars. **MTP scripts MUST pass `--use-chat-template` to `run_benchmark_serving`** - EAGLE-style spec decoding is trained against chat-formatted inputs; benchmarking against raw prompts silently regresses acceptance rate. Applies to every `*_mtp.sh`.

Git: conventional commit messages. `[skip-sweep]` in the latest PR head commit skips that PR's benchmark setup after changelog validation. It is ignored on pushes to `main`. Changes to `perf-changelog.yaml` trigger benchmark runs.

Docs: all contributor-facing docs are bilingual — **every such Markdown doc MUST have a Simplified Chinese version** named `<name>_zh.md` alongside it, with an `English | 中文` switcher at the top. Current pairs: `README.md`/`README_zh.md`, `CONTRIBUTING.md`/`CONTRIBUTING_zh.md`, `docs/PR_REVIEW_CHECKLIST.md`/`docs/PR_REVIEW_CHECKLIST_zh.md`. **Any edit to an English doc MUST be mirrored in its `_zh` counterpart (and vice versa) in the same PR** — same sections, links, badges, images — and a new doc must ship with its `_zh` version in the same PR. Exceptions: agent-instruction files (`AGENTS.md`, `CLAUDE.md`, `KLAUD_DEBUG.md`) and internal references under `.github/`/`utils/` are English-only; the sign-off template inside `docs/PR_REVIEW_CHECKLIST*.md` stays in English verbatim in BOTH versions, because `codeowner-signoff-verify.yml` triggers on its exact English opening phrase.

Checklist ↔ sign-off verifier sync: `docs/PR_REVIEW_CHECKLIST.md` is the source of truth for the merge standard, and the verifier prompt in `.github/codeowner-signoff-verify-prompt.md` encodes it as independently-verified checks (the prompt lives in that standalone template — rendered by `.github/workflows/codeowner-signoff-verify.yml` via envsubst — because GitHub caps inline workflow expressions at 21000 chars; do NOT move it back inline). **Whenever `docs/PR_REVIEW_CHECKLIST.md` is updated — an item added, removed, or materially reworded — agents are allowed and expected to update the verifier prompt to match, ideally in the same PR.** Cosmetic edits (formatting, typos, `_zh` translation sync) need no verifier change. The verifier's Check 5 already compares sign-offs against the live checklist file, so stale sign-off templates are caught automatically — but a new or removed policy item needs its own check logic added to / removed from the workflow prompt. To validate a verifier change: merge it, open a throwaway `[DO NOT MERGE]` test PR, post a sign-off comment (it must contain the exact phrase `As a PR reviewer and CODEOWNER` or the workflow won't trigger), read the posted verdict comment, then close the test PR.

### Pull Request Sweep Labels

PRs do not run the sweep automatically - `run-sweep.yml` is gated on a primary sweep label. Pick exactly one of the five primary labels below; setting multiple primary labels is rejected by the workflow. **For full sweeps, `full-sweep-fail-fast` is the strongly recommended default** - a broken change burns one canary job plus at most one job per matrix instead of the whole fan-out. Reach for `full-sweep-enabled` only when you specifically need every matrix job to run to completion despite failures (e.g. a flaky config where one flake would kill a matrix's in-flight results).

- `sweep-enabled` - runs the sweep with `--trim-conc` (each parallelism config reduced to its single lowest concurrency). Default for most PRs.
- `full-sweep-enabled` - runs the full intermediate concurrency sweep behind a sequential single-node canary gate, with every matrix job running to completion regardless of failures. **Not the recommended default** - prefer `full-sweep-fail-fast`; use this only when a single flaky job killing its matrix's in-flight results is worse than burning GPU time on a broken change.
- `non-canary-full-sweep-enabled` - runs the full intermediate concurrency sweep without the canary gate. Use when the canary is flaky or not representative of the affected configuration.
- `full-sweep-fail-fast` - runs the full intermediate concurrency sweep behind the same sequential single-node canary gate as `full-sweep-enabled` (so a globally broken change burns one job, not the whole fan-out), and with `strategy.fail-fast` enabled on every matrix: the first failure in a matrix cancels that matrix's remaining jobs. Fail-fast is matrix-scoped, so the other matrices (1k1k vs 8k1k vs agentic vs evals) keep running and self-terminate on their own first failure; their completed results remain valid. The failing job keeps its red *failure* conclusion and the run concludes failed. **This is the strongly recommended default for full sweeps** (image bumps, recipe changes, bring-up) - a failure means the rest of that matrix is wasted GPU time. Caveat: one flaky job kills its matrix's in-flight results; if that repeatedly bites a specific config, fall back to `full-sweep-enabled` for that PR.
- `full-sweep-fail-fast-no-canary` - same as `full-sweep-fail-fast` but without the canary gate: all matrices fan out immediately. Use when the canary is flaky or not representative of the affected configuration but you still want per-matrix fail-fast.

`all-evals` and `evals-only` are optional modifier labels. Combine either or both with one primary sweep label. `all-evals` expands eval selection to every generated fixed-sequence configuration without changing throughput. `evals-only` suppresses throughput while keeping the default eval subset; combining both runs every eval and no throughput. `all-evals` remains eligible for artifact reuse when paired with an eligible full-sweep label. Runs with `evals-only`, including runs with both modifiers, are not eligible.

**The sweep does not trigger while the PR has merge conflicts.** Even with a sweep label applied, the `run-sweep.yml` workflow will not start until the PR cleanly merges into main — a stale claude/* or update-* branch with a `perf-changelog.yaml` conflict (the common case) will sit in NO_SWEEP / NO_SUCCESS until rebased. Resolution recipe is documented in `KLAUD_DEBUG.md §1.1`: `git merge origin/main`, then `git checkout origin/main -- perf-changelog.yaml`, then re-append the PR's own changelog entry at the tail. Don't 3-way merge `perf-changelog.yaml`; whitespace edits silently re-trigger the deletion check.

Push-to-main always enters sweep setup: it either reuses approved full-sweep artifacts or runs the full untrimmed sweep. `[skip-sweep]` never suppresses a main-branch sweep. For PR runs, the marker in the latest head commit skips benchmark setup while still allowing changelog validation and reuse authorization checks. Trim logic lives in `trim_conc()` in `utils/process_changelog.py`: single-node entries are grouped by every non-`conc` field and only the lowest-`conc` entry per group is kept; multi-node entries have their `conc` list collapsed to `[min(conc)]`.

## Common Tasks

### Dispatching jobs

Sweeps and one-offs dispatch against `.github/workflows/e2e-tests.yml` (`workflow_dispatch`). `run-sweep.yml` is push/PR-triggered, not dispatchable.

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=my-feature-branch' \
  -f 'inputs[test-name]=DSR1 fp8 H200 sglang smoke' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files configs/nvidia-master.yaml --model-prefix dsr1 --framework sglang --runner-type h200 --min-conc 4 --max-conc 4 --seq-lens 1k1k' \
  -f 'inputs[duration-override]='
```

Inputs: top-level `ref` (required) is the workflow ref to dispatch from, almost always `main`. `inputs[ref]` is the repo ref under test (defaults to the dispatch ref's `github.sha`). `inputs[generate-cli-command]` (required) is passed verbatim to `generate_sweep_configs.py` - test locally first. `inputs[test-name]` is the display name in the Actions UI. `inputs[duration-override]` overrides per-config duration (seconds); empty = use matrix value.

The POST returns no body and no run ID - find the run with `gh run list` below.

### Monitoring jobs

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status   # block, non-zero on failure
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed     # inspect failures
gh run cancel "$RUN_ID" --repo SemiAnalysisAI/InferenceX                # cancel
```

Artifacts: see "Fetching GitHub Actions Benchmark Results" below.

### Adding a benchmark configuration

Add entries to `configs/nvidia-master.yaml` or `amd-master.yaml` (agentic-coding entries live in the Agentic benchmark configurations section at the bottom), append to `perf-changelog.yaml`, then validate with `generate_sweep_configs.py full-sweep`.

### Adding a runner

Add to `configs/runners.yaml`, create launcher in `runners/`, add the runner type to the relevant master config.

### Registering recipes from srtslurm

For `dynamo-sglang` / `dynamo-trt` disaggregated multi-node configs, see `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md` for the full mapping from srtslurm recipe YAML to `nvidia-master.yaml` entries.

Multi-node srt-slurm changes must edit the recipe yaml AND `nvidia-master.yaml` together. `srtctl` reads only the recipe (`model.container`, resources, prefill/decode workers); the sweep generator (`utils/matrix_logic/generate_sweep_configs.py`) reads `nvidia-master.yaml` for frontend labels - its prefill/decode numbers never reach `srtctl`. Recipe-only edits mislabel results, master-only edits don't take effect. For image bumps, `model.container` must equal `image:`, since the launcher uses the latter as the container-alias key.

### Updating Docker images

Update the image tag in the relevant `configs/*-master.yaml` and/or `benchmarks/*.sh`, update any related env vars / config params, and append a `perf-changelog.yaml` entry (required - triggers benchmarks):

```yaml
- config-keys:
    - dsr1-fp8-*-vllm  # wildcards match multiple configs
  description:
    - "Update vLLM image from v0.11.2 to v0.13.0"
    - "Add VLLM_MXFP4_USE_MARLIN=1 environment variable"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

## Evals (Accuracy Validation)

Optional accuracy checks ensuring inference optimizations do not degrade outputs. See `utils/evals/EVALS.md` for the full reference.

Eval selection is marked by `mark_eval_entries()` in `utils/matrix_logic/generate_sweep_configs.py`; evals run by default on the 8k1k subset. Workflow jobs run separately from throughput jobs in `EVAL_ONLY=true` mode. Flags on `generate_sweep_configs.py`: `--no-evals` to skip, `--evals-only` for the selected eval subset only, and `--all-evals` to expand eval-only selection across every generated fixed-sequence config. For multi-node configs, `--all-evals` creates one eval job per engine topology and runs every distinct value in its `conc-list` sequentially against that same engine. `--all-evals` composes with `--evals-only` and remains a standalone shorthand. Changelog `all-evals: true` suppresses throughput for that entry. The PR modifier label `all-evals` only expands selection, while the PR modifier label `evals-only` suppresses throughput across appended entries. Aggregated output produced by `utils/collect_eval_results.py`.

## Key Files

`utils/matrix_logic/validation.py` (config schemas), `generate_sweep_configs.py` (config generation), `utils/bench_serving/benchmark_serving.py` (benchmark client), `configs/nvidia-master.yaml` / `configs/amd-master.yaml` (benchmark definitions, with agentic sections at the bottom), `.github/workflows/run-sweep.yml` (main CI/CD), `.github/workflows/collect-evals.yml` (eval collection), `benchmarks/benchmark_lib.sh` (shared utilities), `utils/evals/` (eval task definitions), `utils/collect_eval_results.py` (aggregator).

## Important Notes

- No new directories in `/workspace` during a benchmark (files are fine).
- **Never delete or modify whitespace in `perf-changelog.yaml`** - CI depends on exact whitespace (including trailing spaces on blank separator lines). Altering it breaks CI.

## Fetching GitHub Actions Benchmark Results

```bash
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/<RUN_ID>/artifacts --jq '.artifacts[].name'
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
```

### Parsing results (don't dump raw JSON)

`agg_bmk.json` is large with many decimals - never `cat` raw. Use `jq` to extract and round:

```bash
cat ./results/agg_bmk.json | jq -r '
  .[] | [.hw, .infmax_model_prefix, "\(.isl)/\(.osl)", (.tput_per_gpu | round)]
  | @tsv' | column -t

cat ./results/agg_bmk.json | jq '[.[] | select(.infmax_model_prefix == "gptoss")]'
```

### Key metrics

`tput_per_gpu` (total throughput per GPU, tok/s), `output_tput_per_gpu` (output token throughput), `mean_ttft` / `p99_ttft` (time to first token), `mean_tpot` (time per output token), `mean_e2el` (end-to-end latency).

### Artifacts

`results_bmk` → `agg_bmk.json` (aggregated). `results_all` → all results aggregated (may not exist). `eval_results_all` → `agg_eval_all.json` (may not exist). `run-stats` → `run_stats.json` (which nodes ran and succeeded).
