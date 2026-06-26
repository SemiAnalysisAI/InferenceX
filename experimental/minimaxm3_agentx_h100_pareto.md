# MiniMax M3 H100 AgentX Pareto Audit

## Data source

- Direct PostgreSQL access was unavailable because `DATABASE_URL` was not set. The repository query schema is `benchmark_results` joined to `configs` and `workflow_runs` in `utils/compare_results.py`.
- The authoritative fallback is the reproducible `results_bmk/agg_bmk.json` artifact from full run `28204059405`: `gh run download 28204059405 --repo SemiAnalysisAI/InferenceX -n results_bmk`.
- Interactivity is computed as `1 / p90_tpot`. Stored `p90_intvty` is not substituted because it represents a different percentile convention.
- The frontier query rejects a point when another point has at least as much `tput_per_gpu` and either no more `p90_e2el` or at least as much `1 / p90_tpot`, with one strict inequality.
- CPU TP c3 is excluded because its aggregate covered only 975 seconds rather than the required profiling interval. Attempt 2 remains in progress and will be audited before the final same-run rerun.

## Full-hour frontier evidence

| Objective | Topology | Conc | KV offload | Tput/GPU | p90 E2E | 1/p90 TPOT |
|---|---:|---:|---|---:|---:|---:|
| Interactivity | TP8 | 1 | none | 231.6 | 186.24s | 14.45 tok/s |
| Both | TP8 | 2 | none | 331.6 | 157.34s | 8.86 tok/s |
| E2E | TP8 | 3 | none | 628.6 | 220.36s | 5.61 tok/s |
| Interactivity | TP8 | 6 | CPU | 1077.3 | 349.36s | 6.38 tok/s |
| Both | TP8 | 5 | none | 1099.2 | 276.67s | 6.37 tok/s |

The CPU c6 interactivity advantage over TP8 c5 is 0.01 tok/s, while CPU points repeatedly OOM or stall. It is noise-sized and does not justify DRAM KV offload in the final matrix.

## Pruning and probes

- Retain TP8 no-offload c1/c2/c3/c5 as the measured union frontier.
- Remove TEP8 and CPU-KV points unless a new genuinely complete result overturns domination or reliability evidence.
- Prior H100 DEP evidence is invalid for selection because it forced a shortened model context. Native-context DEP c4/c8 probes use 80 GiB/GPU model-weight CPU offload, one sequence per DP rank, and never set `max_model_len`.
- Add DEP only if a native-context probe is non-dominated and completes without hidden AIPerf or server errors.
