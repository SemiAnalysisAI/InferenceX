---
version: 1.1.0
date: 2026-04-16
status: proposed
---

# ISB1 ↔ kv-cache-tester Coexistence Plan

## The Two Systems

| | kv-cache-tester (PR #993) | ISB1 (PR #1032) |
|---|---|---|
| **Location** | `experimental/multiturn/vllm_benchmark/kv-cache-tester/` | `datasets/isb1/exports/` |
| **Traces** | 522 real Claude Code sessions | 35 synthetic multi-turn traces |
| **Source** | Real production agentic workloads | Synthetic with controlled stress patterns |
| **Replay** | `trace_replay_tester.py` | `benchmark_export_replay.py` |
| **Config** | `multiturn-agentic-trace.yaml` | `isb1-kv-stress.yaml` |
| **Metrics** | Prometheus sidecar (`metrics_collector.py`) | `process_result_isb1.py` |

## Why Both Are Needed

**kv-cache-tester** shows how chips perform under **real workloads** — actual Claude Code
sessions with natural token distributions. This is the ground truth for "how does inference
actually work in production?"

**ISB1** shows how chips perform under **controlled stress conditions** — specific KV cache
behaviors that real workloads rarely trigger but production systems must handle:

| Stress Pattern | kv-cache-tester | ISB1 |
|---|---|---|
| Natural agentic workload distribution | ✅ (522 real traces) | ❌ |
| Targeted prefix reuse testing | ❌ | ✅ (high_prefix stress class) |
| Forced KV offload cliff | ❌ (depends on trace) | ✅ (offload_cliff stress, 128K-1M context) |
| Session reactivation after idle | ❌ | ✅ (reactivation stress, idle windows) |
| KV compaction under long sessions | ❌ | ✅ (compaction_heavy stress, 25+ turns) |
| Shared prefix fanout | ❌ | ✅ (fanout stress, branching requests) |
| 500K-1M context depth | ❌ (real traces are shorter) | ✅ (xlc2/ulc1/ulc2 bands) |

Together they cover the Pareto frontier from realistic operating points (kv-cache-tester)
through stress-test extremes (ISB1).

## How They Coexist

### Configs (no conflict)
```yaml
# kv-cache-tester config (PR #993)
# .github/configs/multiturn-agentic-trace.yaml
h200-fp8-llama70b:
  trace-file: experimental/multiturn/vllm_benchmark/kv-cache-tester/traces/...

# ISB1 config (PR #1032)
# .github/configs/isb1-kv-stress.yaml
dsr1-fp8-h200-isb1-kv-stress-vllm:
  export-file: datasets/isb1/exports/extension_131k/code_131k1k.json
```

### Workflows (no conflict)
```yaml
# kv-cache-tester workflow (PR #993)
# .github/workflows/multiturn-sweep.yml → benchmark-multiturn-tmpl.yml
#   Uses: trace_replay_tester.py

# ISB1 workflow (PR #1032)
# .github/workflows/run-isb1-kv-stress-sweep.yml → benchmark-isb1-tmpl.yml
#   Uses: benchmark_export_replay.py
```

### Data directories (no conflict)
```
experimental/multiturn/vllm_benchmark/    ← kv-cache-tester (PR #993, untouched by PR #1032)
  kv-cache-tester/                          522 real traces + replayer
  aiperf/                                   AIPerf submodule
  bench/metrics_collector.py                Prometheus sidecar
  analysis/plot_pareto.py                   Pareto charts

datasets/isb1/                            ← ISB1 (PR #1032)
  exports/                                  ISB1 replay bundles
    core/                                   8K baseline
    extension_32k/                          32K context (flat)
    extension_64k/                          64K context (flat)
    extension_131k/                         131K context (flat)
    preview/long_context_500k/              500K reviewed_preview
    preview/long_context_1m/                1M gated preview
```

### Shared infrastructure ISB1 USES from PR #993
- vLLM offload API flags (`--kv_offloading_backend native`, etc.)
- Prometheus metrics collector pattern (ISB1 ships its own `process_result_isb1.py` pipeline)
- Offload mode sweep pattern (on/off/noprefix)
- Runner launch scripts (`runners/launch_*.sh`)
- Concurrency sweep structure

### What PR #1032 does NOT touch
- `experimental/multiturn/vllm_benchmark/kv-cache-tester/` — kv-cache-tester tree
- `aiperf/` submodule — alternative benchmark, unchanged
- `benchmark-multiturn-tmpl.yml` — kv-cache-tester workflow template, unchanged
- `multiturn-agentic-trace.yaml` — kv-cache-tester config, unchanged

## Support-status vocabulary

ISB1 replay surfaces in PR #1032 classify under the five-class support vocabulary:

- `supported` — core 8K replay path
- `reviewed_preview` — 32K / 64K / 131K extensions, 500K preview
- `gated` — 1M preview (manual config `isb1-qwen-1m-preview.yaml` only)
- `artifact_only` — retained artifacts without live replay
- `unsupported` — not a valid path

No ISB1 surface in PR #1032 claims `live_benchmark_certification`; all claims are bounded
to `dataset_replay_verified`.
