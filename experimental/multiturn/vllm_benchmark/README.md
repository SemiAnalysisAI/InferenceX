# vLLM Benchmark (Experimental)

This directory tracks the PR #993 parity surface for multi-turn trace replay and KV stress experiments.

## Trace sources

- **ISB-1 exports**: existing committed replay exports.
- **kv-cache-tester**: `kv-cache-tester/` is a placeholder for the external trace replay repo.
- **AIPerf synthetic traces**: `aiperf_traces/` provides fallback synthetic traces.

## Analysis tools

The parity analysis scripts live under `datasets/isb1/scripts/`:

- `plot_pareto.py`
- `analyze_benchmark_distributions.py`
- `collect_sweep_results.py`
- `adapt_trace_replay_result.py`

## LMCache variants

LMCache launch helpers are under `launch/`:

- `lmcache_vllm_h200.sh`
- `lmcache_vllm_b200.sh`

## Per-hardware replay scripts

Trace replay scripts are under `scripts/` for per-model/per-engine/per-hardware combinations.

---

**Experimental infrastructure. Not part of official ISB-1 support matrix.**
