# Workflows

Repo-wide rules: [root `AGENTS.md`](../AGENTS.md). Operations: [`AGENT_OPERATIONS.md`](AGENT_OPERATIONS.md), [workflows README](workflows/README.md).

- Every priority-scheduled self-hosted job requests exactly one `nodes:N` label (physical Slurm nodes; single-node is `nodes:1`, multi-node forwards `node-count`). A queued job without the label is ineligible for priority scheduling, and labels cannot be added later; fix the branch and dispatch again.
- Shell embedded in workflows follows the [Bash rules](../inferencex-e2e/benchmarks/AGENTS.md#bash-mandatory): configuration comes from the caller, no fallback defaults, no nounset.
- Workflow changes run the [Zizmor CI command](workflows/zizmor.yml) with online checks; do not weaken its audit or global suppression policy.
