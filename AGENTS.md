# AGENTS.md

Guidance for AI agents working with InferenceX.

> **Mandatory reading: [`CONTRIBUTING.md`](CONTRIBUTING.md)** — read it before opening or reviewing any PR. It covers the full PR review flow, the CODEOWNER sign-off process, the `/reuse-sweep-run` merge path, post-merge responsibilities, and critical cluster rules (e.g. never leaving root-owned files on AMD runners).

> **PR and GitHub-issue titles & descriptions must be bilingual — include a Simplified Chinese version in addition to English.** Title format: `<English title> / <中文标题>`. In the PR/issue body, follow the English content with its Chinese translation (e.g. a `## 中文说明` section mirroring the summary; don't translate code blocks, logs, or stack traces — summarize around them). **PR comments must include a Chinese translation too** — conversation comments, review summaries, and inline review comments alike: short comments as a single `<English> / <中文>` line, longer ones with the Chinese translation as a trailing paragraph (`中文：...`). Exception: the CODEOWNER sign-off template stays English-verbatim (the sign-off verifier triggers on its exact phrase); bot-generated comments follow their own workflow templates. This applies to every PR and every issue, matching the bilingual docs rule in Code Conventions.

> **Translation quality bar:** write natural technical Chinese as used by ML infra engineers, not word-for-word machine translation. Follow the style of [`vllm-project/vllm-ascend` `README.zh.md`](https://github.com/vllm-project/vllm-ascend/blob/main/README.zh.md): translate concepts into idiomatic Chinese while preserving model names, hardware SKUs (MI355X, B300, GB200 ...), framework names (vLLM, SGLang, ATOM ...), flags, and CLI/env-var identifiers in English. Use parenthetical English clarification for acronyms on first use, e.g. 混合专家(MOE), 专家并行(EP). Preferred term mappings are in [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#translation-terminology).

> **Before debugging a Klaud-Cold / `claude/*` image-bump PR, read [`KLAUD_DEBUG.md`](KLAUD_DEBUG.md).**

## Project Overview

InferenceX is an open-source automated benchmarking system that tracks LLM inference performance across hardware (NVIDIA B200/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results are published at https://inferencex.com/, which will route to https://inferencex.semianalysis.com.

## Frontend Results API

Use the public API for published dashboard results; use GitHub Actions artifacts for un-ingested runs, raw output, logs, or debugging evidence.

```bash
INFERENCEX_API=https://inferencex.semianalysis.com/api/v1
curl --fail --compressed \
  "$INFERENCEX_API/benchmarks?model=DeepSeek-V4-Pro" \
  | jq '.[] | select(.benchmark_type == "single_turn" and .isl == 8192 and .osl == 1024)'
```

`model=` takes the frontend display name; `InferenceX-app/packages/constants/src/models.ts` is authoritative. Fixed-sequence rows use numeric `isl`/`osl`; `agentic_traces` rows use null lengths, so do not filter them out accidentally. Use `view=calculator&sequence=8k/1k` for compact interpolation data, and `date`, `runId`, `exact`, or `exactRun` for historical/run-scoped reads. Discovery endpoints are `/availability`, `/workflow-info`, `/evaluations`, and `/reliability`.

Always use `--compressed` and `jq`; never dump raw benchmark JSON into logs or agent context. Do not cache-bust or repeatedly poll CDN-cached results.

## Repository Map

- `perf-changelog.yaml`: append-only benchmark trigger log; preserve every existing byte and append new entries at the end.
- `configs/`: master benchmark definitions and runner inventory.
- `benchmarks/`: shared `benchmark_lib.sh` plus single-node, multi-node, and srt-slurm recipe entrypoints.
- `runners/`: hardware launchers.
- `utils/matrix_logic/`: config generation, Pydantic validation, and tests.
- `utils/bench_serving/`: benchmark client and backends.
- `utils/evals/`: evaluation configs, thresholds, validation, and [`EVALS.md`](utils/evals/EVALS.md).
- `utils/`: result, changelog, power, collection, and comparison tooling.

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
Active fixed-sequence workload: `8k1k` (8192/1024). `1k1k` (1024/1024) remains accepted by tooling for archived or targeted runs but was deprecated from the regular benchmark matrix on 2026-07-17; see [`MODELS.md`](MODELS.md).

## Code Conventions

Python: type hints (`list[str]`, `Optional[int]`), Pydantic with `extra='forbid'`, field aliases `Field(alias="model-prefix")`, docstrings on functions.

YAML: kebab-case field names (`model-prefix`, `conc-start`, `dp-attn`). Master configs define all benchmark configurations. `perf-changelog.yaml` triggers which configs to benchmark and is read chronologically (oldest at top, newest at bottom) - new entries MUST be appended to the END, never inserted in the middle or prepended.

Bash: source shared utilities via `source benchmark_lib.sh` (`check_env_vars`, `wait_for_server_ready`, `run_benchmark_serving`, `run_eval`, `append_lm_eval_summary`); parameters passed via env vars. **MTP scripts MUST pass `--use-chat-template` to `run_benchmark_serving`** - EAGLE-style spec decoding is trained against chat-formatted inputs; benchmarking against raw prompts silently regresses acceptance rate. Applies to every `*_mtp.sh`.

Git: conventional commit messages. **Commit messages must include a Simplified Chinese translation in addition to English** — keep the subject line in English (conventional-commit style), then include the Chinese translation of the subject and key body points in the commit body (e.g. a trailing `中文：<translation>` paragraph), following the same translation quality bar as PRs/issues. Squash-merge commits inherit the bilingual PR title, which satisfies the subject requirement automatically. `[skip-sweep]` in the latest PR head commit skips that PR's benchmark setup after changelog validation. It is ignored on pushes to `main`. Changes to `perf-changelog.yaml` trigger benchmark runs.

Docs: all contributor-facing docs are bilingual — **every such Markdown doc MUST have a Simplified Chinese version** named `<name>_zh.md` alongside it, with an `English | 中文` switcher at the top. Current pairs: `README.md`/`README_zh.md`, `CONTRIBUTING.md`/`CONTRIBUTING_zh.md`, `MODELS.md`/`MODELS_zh.md`, `docs/PR_REVIEW_CHECKLIST.md`/`docs/PR_REVIEW_CHECKLIST_zh.md`, and `golden_al_distribution/README.md`/`golden_al_distribution/README_zh.md`. **Any edit to an English doc MUST be mirrored in its `_zh` counterpart (and vice versa) in the same PR** — same sections, links, badges, images — and a new doc must ship with its `_zh` version in the same PR. Exceptions: agent-instruction files (`AGENTS.md`, `CLAUDE.md`, `KLAUD_DEBUG.md`), internal references under `.github/`/`utils/`, and implementation-local references such as `configs/CONFIGS.md` and `experimental/README.md` are English-only; the sign-off template inside `docs/PR_REVIEW_CHECKLIST*.md` stays in English verbatim in BOTH versions, because `codeowner-signoff-verify.yml` triggers on its exact English opening phrase.

Checklist ↔ sign-off verifier sync: `docs/PR_REVIEW_CHECKLIST.md` is the source of truth for the merge standard, and the verifier prompt in `.github/codeowner-signoff-verify-prompt.md` encodes it as independently-verified checks (the prompt lives in that standalone template — rendered by `.github/workflows/codeowner-signoff-verify.yml` via envsubst — because GitHub caps inline workflow expressions at 21000 chars; do NOT move it back inline). **Whenever `docs/PR_REVIEW_CHECKLIST.md` is updated — an item added, removed, or materially reworded — agents are allowed and expected to update the verifier prompt to match, ideally in the same PR.** Cosmetic edits (formatting, typos, `_zh` translation sync) need no verifier change. The verifier's Check 5 already compares sign-offs against the live checklist file, so stale sign-off templates are caught automatically — but a new or removed policy item needs its own check logic added to / removed from the workflow prompt. To validate a verifier change: merge it, open a throwaway `[DO NOT MERGE]` test PR, post a sign-off comment (it must contain the exact phrase `As a PR reviewer and CODEOWNER` or the workflow won't trigger), read the posted verdict comment, then close the test PR.

### Pull Request Sweep Labels

A PR sweep requires exactly one primary label:

| Use case | Label |
|---|---|
| Lightweight, minimum-concurrency validation | `sweep-enabled` |
| Full sweep; recommended default | `full-sweep-fail-fast` |
| Full sweep where flakes must not cancel a matrix | `full-sweep-enabled` |
| Full sweep with fail-fast but no canary | `full-sweep-fail-fast-no-canary` |
| Full sweep without fail-fast or canary | `non-canary-full-sweep-enabled` |

Modifiers are `all-evals`, `evals-only`, and `agentx-fast`. Runs using `evals-only` or `agentx-fast` cannot be reused. Sweeps do not start while a PR has merge conflicts. `[skip-sweep]` affects PR setup only and never suppresses a `main` sweep. See [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#sweep-labels-and-reuse) for exact semantics and conflict recovery.

## Common Tasks

### Dispatching and monitoring jobs

One-offs use `.github/workflows/e2e-tests.yml`; `run-sweep.yml` is not dispatchable. The top-level dispatch `ref` selects the workflow definition (normally `main`), while `inputs[ref]` selects the revision under test. See [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#workflow-dispatch-and-monitoring) for commands and input semantics.

### Adding a benchmark configuration

Add entries to `configs/nvidia-master.yaml` or `amd-master.yaml` (agentic-coding entries live in the Agentic benchmark configurations section at the bottom), append to `perf-changelog.yaml`, then validate with `generate_sweep_configs.py full-sweep`.

### Adding a runner

Add to `configs/runners.yaml`, create launcher in `runners/`, add the runner type to the relevant master config.

### Registering recipes from srtslurm

For `dynamo-sglang` / `dynamo-trt` disaggregated multi-node configs, see `benchmarks/multi_node/srt-slurm-recipes/RECIPES.md` for the full mapping from srtslurm recipe YAML to `nvidia-master.yaml` entries.

Multi-node srt-slurm changes must edit the recipe yaml AND `nvidia-master.yaml` together. `srtctl` reads only the recipe (`model.container`, resources, prefill/decode workers); the sweep generator (`utils/matrix_logic/generate_sweep_configs.py`) reads `nvidia-master.yaml` for frontend labels - its prefill/decode numbers never reach `srtctl`. Recipe-only edits mislabel results, master-only edits don't take effect. For image bumps, `model.container` must equal `image:`, since the launcher uses the latter as the container-alias key.

Power lanes have a recipe/launcher pinning contract; read [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#power-telemetry) before changing telemetry or official energy collection.

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

Evals default to the 8k1k subset and run separately from throughput. The generator supports `--no-evals`, `--evals-only`, and `--all-evals`; the last two compose. Read [`utils/evals/EVALS.md`](utils/evals/EVALS.md) for task behavior and [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#evaluation-selection) for matrix and label semantics.

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

### Metrics, power, and artifact schema

Core metrics:

| Metric | Meaning |
|---|---|
| `tput_per_gpu` | Total input plus output tokens per second per GPU |
| `output_tput_per_gpu` | Output tokens per second per GPU |
| `mean_ttft` | Mean time to first token |
| `p99_ttft` | 99th-percentile time to first token |
| `mean_tpot` | Mean time per output token after the first |
| `mean_e2el` | Mean end-to-end request latency |

Power fields and artifact schemas are documented in [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#power-telemetry). Never dump raw result JSON.
