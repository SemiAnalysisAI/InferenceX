# operatorx

Multi-platform inference operator benchmark suite. Times one op at a time
(gemm, attention, moe, collectives, ...) on NVIDIA / AMD / TPU / Trainium and
emits one JSON per run under `results/<platform>/<cluster>/`.

See `CLUSTERS.md` for how to reach each cluster and the per-host quirks.

## Running

One process per GPU runs `python -m operatorx` inside the backend's container
(`containers.toml`); on SLURM clusters, launch it with `srun` from an
allocation. Configuration is by environment:

| Var | Notes |
|-----|-------|
| `OPERATORX_CLUSTER` | Cluster id; routes to the platform runner (`operatorx.clusters.CLUSTER_PLATFORMS`). |
| `OPERATORX_BACKENDS` | CSV of backends to run (e.g. `vllm`, `torch`). |
| `OPERATORX_TESTLISTS` | CSV of testlist names from `testlists/`; default all. |
| `WORLD_SIZE`, `RANK` | Process group size and rank (1 / 0 for single-GPU GEMM). |
| `OPERATORX_TELEMETRY_DIR` | Where per-op clock/power samples are written. |
| `OPERATORX_PROFILE` | `0` disables the cold profiler replay (per-kernel breakdown) attached to each result. |

```bash
srun --container-image=<vllm image .sqsh> bash -c '
  cd <checkout>/experimental && \
  OPERATORX_CLUSTER=h200_hgx_8x OPERATORX_BACKENDS=vllm OPERATORX_TESTLISTS=gemm_serving_8k1k_min \
  WORLD_SIZE=1 RANK=0 python -m operatorx'
```

The GitHub Actions sweep (`CI.md`) runs the same entry point on the shared pools.

### TPU / Trainium

These hosts have no SLURM. Run the benchmark directly on the VM or instance:

```bash
OPERATORX_CLUSTER=v6e_4x   python -m operatorx   # TPU     (default tpu cluster)
OPERATORX_CLUSTER=trn3_16x python -m operatorx   # Trainium (default trainium cluster)
```

The TPU `maxtext` backend depends on Google's MaxText library. Install it
once per TPU VM (the `jax` backend works without it):

```bash
git clone https://github.com/AI-Hypercomputer/maxtext ~/maxtext
pip install -e ~/maxtext
```

If MaxText isn't installed, `moe_forward` on TPU emits `unsupported` rows
rather than running our old single-device dense fallback.

## Adding a backend / op

- Backend impl: `operatorx/runners/<platform>/backends/<name>.py`, exports an
  `IMPLS = [BackendImpl(...)]` list.
- Op spec: `operatorx/ops/<name>.py`, calls `register(OpSpec(..., flops=, bytes=))`.
- Add the container image to `containers.toml`.
- Add shapes to `testlists/<name>.json`.
