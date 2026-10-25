# infx.launch

Repo-wide rules: [root `AGENTS.md`](../../../AGENTS.md). Python rules: [`infx/AGENTS.md`](../AGENTS.md).

- Launch only through `python -m infx.launch`; add no shell launchers. Cluster facts go in the cluster's `clusters:` record, workload rules in named policy tables; drivers never branch on a cluster id. Retiring a cluster removes its label, record, and policy rows in one PR.
- srt-slurm host-setup hooks only check or prepare hosts: they never run benchmarks, tune engines, patch containers, or hide failures.
- When retiring a model, remove its unused rows from the workload tables (`policy.py`, `drivers/srt/{lanes,models,power}.py`) and cluster `models.entries`. See [deprecating configs](../../configs/AGENTS.md#deprecating-benchmark-configs).
