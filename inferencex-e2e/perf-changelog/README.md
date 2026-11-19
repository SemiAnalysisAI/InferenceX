# Performance changelog

A PR that changes what a benchmark runs adds exactly one file here, named after its PR number, for example `perf-changelog/5101.yaml`. The file selects the PR sweep and is the record Merge Ingest publishes with the results. Refactors, recipe moves and docs add no file.

Because each PR adds its own file, changelog entries never conflict.

## Writing the file

Open the PR first, then run from `inferencex-e2e/`.

```bash
uv run --extra recipes python -m infx.matrix.changelog new \
  -m "Update the image to vllm/vllm-openai:v0.21.0." \
  -m "Raise max-num-seqs from 64 to 128 at concurrency 64."
```

The command reads the PR number from `gh pr view` (pass `--pr N` to set it) and fills `config-keys`, `scenario-type` and `retired-keys` from the config diff against `origin/main`. Each `-m` is one description bullet. Script and launcher changes are invisible to the diff, so add `--key <config>` for each config they change. Commit the file, then add any flags by hand.

## Format

The file holds one YAML mapping with the `ChangelogEntry` fields. Leave out `pr-link`, which comes from the file name.

```yaml
config-keys:
  - dsv41flash-fp4-b300-dynamo-vllm-agentic-dspark-agg
  - dsv41flash-fp4-b300-dynamo-vllm-agentic-dspark-disagg
retired-keys:
  - dsv41flash-fp4-b300-dynamo-vllm-agentic-agg
scenario-type:
  - agentic-coding
description:
  - "Move the B300 DeepSeek-V4.1-Flash FP4 Dynamo + vLLM AgentX recipes to vllm/vllm-openai:cu134-nightly-96710245ece8."
  - "Retire the non-DSpark aggregated recipe, which the DSpark recipe replaces."
```

- `retired-keys` lists configs the PR deletes alongside the configs it sweeps.
- A PR that only deletes configs writes `retired: true` and lists the deleted configs in `config-keys`. It runs no sweep and merges without `/use`.
- Optional flags are `append-only`, `evals-only`, `all-evals`, `no-evals` and `eval-min-prefill-ep`, as described in [CONTRIBUTING.md](../../CONTRIBUTING.md) and [EVALS.md](../infx/evals/EVALS.md).

## Rules

- One file per PR, named `<PR number>.yaml`. Revise it while the PR is open rather than adding a second one.
- Merged files are immutable. Never edit, rename or delete another PR's file.
- The `changelog-guard` check fails when configs change with no file, when the file is misnamed or there is more than one, or when the file does not cover every changed config, removed config and changed scenario. It warns when shared scripts or launchers change with no file.
- [`perf-changelog.yaml`](../perf-changelog.yaml) is frozen history. A PR opened before this directory existed may still append one entry there until it merges, using the [legacy conflict helper](../../.github/workflows/README.md#changelog-conflict-recovery) if it conflicts. Every other PR uses this directory, and a PR never writes both.

This README also marks when the directory was introduced, so keep it in place.
