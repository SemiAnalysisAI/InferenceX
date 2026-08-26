# Qwen3.8 two-node vLLM Ray runtime (MI355X)

Slurm/Docker/Ray orchestration for `qwen3.8-fp8-mi355x-vllm-multi-nodes-agentic`:
two exclusive nodes, vLLM TP8/PP2 via Ray, AgentX client through `agentic_srt.sh`.

| File | Role |
|---|---|
| `qwen3.8_env.sh` | Shared defaults (`QWEN38_MODEL_PATH`, `MODEL_PATH`) |
| `submit.sh` | `sbatch` wrapper; returns job ID |
| `job.slurm` | Two-node allocation, snapshot preflight, Ray + vLLM + client |
| `node_control.sh` | Per-node Docker/Ray/vLLM actions via `srun` |
| `client.sh` | AgentX entrypoint inside the head container |
| `snapshot_manifest.py` | Validates local snapshot layout and prints a digest |
| `verify_model_staging.sh` | Ops helper to confirm snapshots exist and match |

## Cluster prerequisites (required before CI)

1. **Model snapshot** — stage `Qwen/Qwen3.8-2.4T-A95B-FP8` on **every** MI355X node
   Slurm can assign (not only the nodes used in a one-off smoke):

   ```
   /models/Qwen/Qwen3.8-2.4T-A95B-FP8/
   ```

   Each node must include Hugging Face download metadata under
   `.cache/huggingface/download/` (used by `snapshot_manifest.py`).

2. **Docker image** — `vllm/vllm-openai-rocm:qwen38` reachable from each node.

3. **Network** — `eno0` must carry the Ray/NCCL address used by Slurm-assigned hosts.

4. **Overrides** — set before launch if snapshots live elsewhere during bring-up:

   ```bash
   export QWEN38_MODEL_PATH=/path/to/Qwen3.8-2.4T-A95B-FP8
   ```

## Pre-CI verification

```bash
# Local node check:
bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/verify_model_staging.sh --local

# Two-node digest match (replace with your hosts):
bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/verify_model_staging.sh \
    --nodes mia1-p01-g16,mia1-p01-g19

# Slurm preflight-only (no vLLM launch):
export GITHUB_WORKSPACE="$PWD"
export BENCHMARK_LOGS_DIR="$PWD/benchmark_logs"
export IMAGE=vllm/vllm-openai-rocm:qwen38
export QWEN38_SCENARIO=agentic-coding
export RUNNER_NAME=qwen38-staging-check
export SLURM_ACCOUNT="$USER"
export SLURM_PARTITION=compute
QWEN38_PREFLIGHT_ONLY=1 bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/submit.sh
```

## CI path

`benchmarks/multi_node/agentic/qwen3.8_fp8_mi355x_vllm-multi-nodes.sh` → `submit.sh` →
`job.slurm`. The MI355X launcher routes `framework: vllm-multi-nodes` through
`multi_node/agentic/` and validates `${RESULT_FILENAME}_conc*.json` in
`benchmark-multinode-tmpl.yml`.
