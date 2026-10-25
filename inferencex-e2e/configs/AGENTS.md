# Configs

Repo-wide rules, including the changelog: [root `AGENTS.md`](../../AGENTS.md). Field reference: [`CONFIGS.md`](CONFIGS.md).

- Multi-node srt-slurm master-config changes update the matching recipe together ([benchmarks guide](../benchmarks/AGENTS.md#recipes-and-runtime)).

## Deprecating benchmark configs

- Delete retired entries (or, for partial retirement, only the retired scenarios) from the master config; never archive. Delete recipes, setup scripts, and assets no active config uses, rather than moving them to `deprecated/`.
- In the same PR, reconcile retirement statements in [`MODELS.md`](../docs/MODELS.md) with active configs and routing, preserving documented exceptions and planned (not completed) retirements.
- Remove unused retired-model rows from launch workload tables (`infx/launch/policy.py`, `infx/launch/drivers/srt/{lanes,models,power}.py`) and cluster `models.entries`, and fix guidance that recommends retired coverage. Audit callers before removing shared helpers; SPEED-Bench collectors and historical readers may still need them.
