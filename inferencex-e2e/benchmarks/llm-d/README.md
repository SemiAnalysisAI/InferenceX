# llmd-vllm framework artifacts

This directory holds the static, baked-into-the-image pieces of the
`llmd-vllm` benchmark framework.

| File | Purpose |
|---|---|
| `Dockerfile` | Combined image: vLLM (DeepEP-enabled), EPP, routing-sidecar, Envoy. One image, every node uses what its role requires. |
| `epp-config.yaml` | Fallback EPP scheduling config. Used when no recipe overrides it via `CONFIG_FILE`. `disagg-profile-handler` + `kv-cache-utilization-scorer` + `random-picker` over the file-discovery endpoint set. |
| `envoy.yaml` | Static Envoy: listener `:8080`, ext_proc to `127.0.0.1:9002`, ORIGINAL_DST cluster reading `x-gateway-destination-endpoint`. |

The runtime pieces (per-node `server.sh`, the SLURM job script, recipe
files, and the endpoint discovery mechanism) live under
`benchmarks/multi_node/llm-d/` and `benchmarks/multi_node/llm-d-recipes/`.
See the README in `benchmarks/multi_node/llm-d/` for the endpoints-file
generation flow.
