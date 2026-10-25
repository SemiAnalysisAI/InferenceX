# Benchmarks

Repo-wide rules: [root `AGENTS.md`](../../AGENTS.md).

## Bash (mandatory)

Applies to active Bash scripts and shell embedded in workflows and recipes; leave deprecated code alone unless asked.

- **Configuration flows from the caller.** Workflows, master configs, runtime profiles, and launchers supply configuration explicitly; receiving scripts validate it and never silently choose defaults.
- **No fallback defaults** for caller-supplied configuration: no `${VAR:-default}`, `${VAR:=default}`, colon-free forms, or equivalent "if unset, assign" logic. A missing input is a caller error and must fail clearly. Pass `false` and `0` explicitly too.
- **Validate inputs with `check_env_vars`** from `check_env.sh` (sourcing only defines it). Group required inputs near the top, after sourcing it; validate path-specific inputs on entering that path. It rejects missing and empty values and lists every missing name. Do not duplicate it or remove its safe handling of unset variables.
- **No nounset:** no `set -u`, `set -o nounset`, `set -euo pipefail`, or `bash -u`. Keep other options such as `set -eo pipefail`.
- **Preserve precedence and forwarding.** Apply caller settings before recipe overrides and forward required inputs across container or job boundaries; never replace a supported override with an unconditional assignment.
- Deliberate optional inputs, runtime-derived values, and unset-safe internal probes are fine; they are not licence to invent fallback configuration or replace a documented automatic selection with an arbitrary constant.

Example: replace `export IS_MULTINODE="${IS_MULTINODE:-true}"` in the receiver with `export IS_MULTINODE=true` in the caller, and `check_env_vars IS_MULTINODE MODEL_NAME PRECISION` in the receiver.

## Recipes and runtime

- Multi-node srt-slurm changes update the recipe and matching master config together. srt-slurm recipes, fixed-sequence and AgentX, are fragments: the binder supplies image, model, precision, and sequence lengths, and `../configs/srt-recipes/` supplies `benchmark.type` and `benchmark.command`; never set those in a recipe. Rows of one design share a `*-variants.yaml` ([recipe guide](multi_node/srt-slurm-recipes/RECIPES.md)).
- Speculative fixed-sequence benchmarks use the chat template: `bind_workload` in `infx/srt_slurm/workload.py` sets `benchmark.env.USE_CHAT_TEMPLATE: "true"` for single-node recipes that speculate, which `python3 -m infx.bench fixed-seq` (run by `srt_fixed_sequence.sh`) turns into `--use-chat-template`.
- Benchmarks create no new directories under `/workspace`, and root containers leave no root-owned files in shared AMD runner workspaces.
