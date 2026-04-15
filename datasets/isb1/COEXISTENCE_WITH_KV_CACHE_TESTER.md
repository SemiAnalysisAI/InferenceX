---
version: 1.0.0
date: 2026-04-14
author: William Chen
status: proposed
---

# ISB1 ↔ kv-cache-tester Coexistence Plan

## The Two Systems

| | kv-cache-tester (Cameron's) | ISB1 (ours) |
|---|---|---|
| **Location** | `experimental/multiturn/vllm_benchmark/kv-cache-tester/` | `datasets/isb1/exports/` |
| **Traces** | 522 real Claude Code sessions | 35 synthetic multi-turn traces |
| **Source** | Real production agentic workloads | Synthetic with controlled stress patterns |
| **Replay** | `trace_replay_tester.py` | `benchmark_export_replay.py` |
| **Config** | `multiturn-agentic-trace.yaml` | `isb1-kv-stress-pr993.yaml` |
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

Together they give the Pareto frontier Cameron wants: kv-cache-tester at realistic operating
points, ISB1 at stress-test extremes.

## How They Coexist in PR #993

### Configs (no conflict)
```yaml
# Cameron's existing config — uses kv-cache-tester traces
# .github/configs/multiturn-agentic-trace.yaml
h200-fp8-llama70b:
  trace-file: experimental/multiturn/vllm_benchmark/kv-cache-tester/traces/...

# Our config — uses ISB1 export traces  
# .github/configs/isb1-kv-stress-pr993.yaml
dsr1-fp8-h200-isb1-kv-stress-vllm-pr993:
  export-file: datasets/isb1/exports/extension_131k/vllm/code_131k1k.json
```

### Workflows (no conflict)
```yaml
# Cameron's workflow
# .github/workflows/multiturn-sweep.yml → benchmark-multiturn-tmpl.yml
#   Uses: trace_replay_tester.py

# Our workflow  
# .github/workflows/run-isb1-sweep.yml → benchmark-isb1-tmpl.yml
#   Uses: benchmark_export_replay.py
```

### Data directories (no conflict)
```
experimental/multiturn/vllm_benchmark/    ← Cameron's (untouched)
  kv-cache-tester/                          522 real traces + replayer
  aiperf/                                   AIPerf submodule
  bench/metrics_collector.py                Prometheus sidecar
  analysis/plot_pareto.py                   Pareto charts

datasets/isb1/                            ← Ours (separate directory)
  exports/                                  ISB1 replay bundles
    extension_131k/                         131K context (DSR1, GPT-OSS, Qwen)
    preview/long_context_500k/              500K Qwen preview
    preview/long_context_1m/                1M Qwen preview
```

### Shared infrastructure we USE from PR #993
- vLLM offload API flags (`--kv_offloading_backend native`, etc.)
- Prometheus metrics collector (could share `metrics_collector.py`)
- Offload mode sweep pattern (on/off/noprefix)
- Runner launch scripts (`runners/launch_*.sh`)
- Concurrency sweep structure

### What we DO NOT touch
- `experimental/multiturn/vllm_benchmark/` — entirely Cameron's
- `kv-cache-tester/` submodule — real traces, don't modify
- `aiperf/` submodule — alternative benchmark, don't modify
- `benchmark-multiturn-tmpl.yml` — Cameron's workflow template

## Recommended PR Structure

### Option A: Single PR with two benchmark lanes (cleanest)
PR #993 ships with BOTH:
- Lane 1: kv-cache-tester (real traces) — Cameron's existing work
- Lane 2: ISB1 (synthetic stress traces) — our addition

Both use the same vLLM server configs, offload modes, and concurrency sweeps.
Results are compared side by side — real vs stress.

### Option B: ISB1 as follow-up PR (safest)
PR #993 ships with kv-cache-tester only (Cameron's work).
We submit a follow-up PR that adds ISB1 as a second benchmark lane.
Uses the same runner infrastructure and offload configs.

### Recommendation: Option A
Cameron explicitly asked for "realistic multi-turn benchmarks" at GTC. Having both
real traces AND synthetic stress traces in the same PR makes a stronger story:
"Here's how chips perform under real workloads AND here's where they break under
targeted KV stress." That's the complete Pareto frontier.

## What We Need From Cameron's Team
1. Confirm ISB1 configs don't conflict with multiturn-agentic-trace.yaml
2. Confirm datasets/isb1/exports/ is the right location for our files
3. Decide: do we share metrics_collector.py or use process_result_isb1.py?
4. Agree on result format for combined Pareto visualization
