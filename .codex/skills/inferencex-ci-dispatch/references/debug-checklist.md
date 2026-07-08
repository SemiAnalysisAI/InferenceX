# CI Debug Checklist

## Success in GitHub UI but Benchmark Failed

Treat missing throughput or `server internal error` as a failure even if the workflow conclusion is green.

Inspect:

- `results_bmk/agg_bmk.json` or agentic result JSON for missing `tput` and failed request counts.
- `run-stats/run_stats.json` for node allocation and success flags.
- Agentic raw artifacts under `LOGS/agentic` or uploaded `bmk_agentic_*` artifacts.
- `multinode_server_logs*` or `benchmark_artifacts` for router, prefill, decode, and LMCache logs.

## Checkout EACCES

Root-owned stale outputs can break `actions/checkout clean: true`.

Ensure `.github/workflows/benchmark-multinode-tmpl.yml` has pre-checkout cleanup before `actions/checkout`:

```yaml
- name: Workspace cleanup (pre-checkout)
  run: |
    sudo rm -rf "$GITHUB_WORKSPACE/benchmark_logs" || true
    sudo rm -rf "$GITHUB_WORKSPACE/benchmark_artifacts" || true
    sudo rm -rf "$GITHUB_WORKSPACE/LOGS" || true
    sudo rm -rf "$GITHUB_WORKSPACE/agentic_checkpoints" || true
    sudo rm -f "$GITHUB_WORKSPACE"/samples*.jsonl || true
    sudo rm -f "$GITHUB_WORKSPACE"/sample*.jsonl || true
    sudo rm -f "$GITHUB_WORKSPACE"/results*.json || true
    sudo rm -f "$GITHUB_WORKSPACE"/meta_env.json || true
    sudo chown -R "$USER":"$USER" "$GITHUB_WORKSPACE" || true
```

## Agentic Matrix Split

In `.github/workflows/e2e-tests.yml`, throughput matrices should exclude `eval-only` entries. Eval-only multinode agentic jobs need `scenario-type`, `kv-offloading`, `kv-offload-backend`, `total-cpu-dram-gb`, and a concrete `conc`.

## LMCache Offload Fields

For multinode agentic LMCacheMP entries, generated matrix entries must preserve:

- `kv-offloading: dram`
- `kv-offload-backend: lmcache`
- `total-cpu-dram-gb`
- `exp-name` suffix `_kvdram-lmcache`

## Runner Assignment

If removing a runner from `runners.yaml` does not affect a job, check whether the workflow uses a stale generated matrix, a different runner key, or a cluster launcher that maps `runner` differently from the YAML label.

## PR Sweep Triggering

PR sweeps require exactly one primary sweep label and no merge conflicts. `run-sweep.yml` does not run from manual dispatch; use `e2e-tests.yml` for manual repro.
