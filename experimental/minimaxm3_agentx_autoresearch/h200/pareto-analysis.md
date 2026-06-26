# MiniMax M3 H200 AgentX Pareto Audit

## Data source

- Direct PostgreSQL access was unavailable because `DATABASE_URL` was not set. The repository query schema is `benchmark_results` joined to `configs` and `workflow_runs` in `utils/compare_results.py`.
- The authoritative fallback is `results_bmk/agg_bmk.json` from full run `28204060437`: `gh run download 28204060437 --repo SemiAnalysisAI/InferenceX -n results_bmk`.
- Interactivity is computed as `1 / p90_tpot`; stored `p90_intvty` is not treated as equivalent.
- The frontier query rejects a point when another has at least as much `tput_per_gpu` and either no more `p90_e2el` or at least as much `1 / p90_tpot`, with one strict inequality.
- TEP8 CPU c12/c16/c20 are excluded: profiling stopped after 785.79s, 2575.06s, and 87.25s with `ProfileAborted(failed_request_threshold)`, `replay_rc=1`, and missing canonical profile exports.

## Genuine full-hour union frontier

| Objective | Topology | Conc | KV offload | Tput/GPU | p90 E2E | 1/p90 TPOT |
|---|---:|---:|---|---:|---:|---:|
| Interactivity | TP8 | 1 | none | 1070.8 | 48.23s | 96.57 tok/s |
| Interactivity | TP8 | 2 | none | 1476.5 | 38.51s | 77.89 tok/s |
| Interactivity | TEP8 | 2 | none | 1532.7 | 52.91s | 60.53 tok/s |
| Interactivity | TP8 | 4 | none | 2828.0 | 57.45s | 58.58 tok/s |
| E2E | TP8 | 5 | none | 1242.9 | 28.73s | 52.98 tok/s |
| E2E | TEP8 | 5 | CPU | 2135.3 | 30.37s | 51.94 tok/s |
| E2E | TEP8 | 5 | none | 2535.0 | 40.50s | 46.89 tok/s |
| Both | TP8 | 6 | none | 2905.8 | 54.21s | 47.63 tok/s |
| Both | TP8 | 7 | none | 3968.4 | 56.25s | 44.20 tok/s |
| Both | TP8 | 8 | none | 4046.5 | 58.40s | 39.13 tok/s |
| E2E | TP8 | 9 | none | 4412.6 | 67.58s | 30.60 tok/s |
| Both | TEP8 | 10 | CPU | 4719.5 | 68.81s | 31.07 tok/s |

## Pruning and probes

- Remove TEP8 CPU c12, c16, and c20. They are invalid and nowhere near the p90-interactivity frontier.
- Retain the measured union candidates TP8 c1/c2/c4/c5/c6/c7/c8/c9, TEP8 no-offload c2/c5, and TEP8 CPU c5/c10. Probe TP8 c3 to close the low-latency gap.
- Prune all other TEP8 and post-cliff TP8 points unless a later genuine result is non-dominated.
- Native-context DEP c4/c8 probes use 24 GiB/GPU model-weight CPU offload and one sequence per DP rank. They never set `max_model_len`.
- Full run `28204060437` has zero failed conclusions, so `--failed` cannot select its three false-green successes. A rerun-all attempt on the same run is the non-replacement path if authorized after focused validation.
