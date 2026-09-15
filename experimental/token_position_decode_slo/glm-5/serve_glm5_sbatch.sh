#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:nightly
#SBATCH --container-mounts=/home/kimbo/inferperf/glm-5:/workspace
#SBATCH --no-container-entrypoint
#SBATCH --job-name=vllm-glm5
#SBATCH --output=/home/kimbo/inferperf/glm-5/logs/vllm-server-tp8.log

# Persistent vLLM server for GLM-5 FP8
set -eo pipefail

MODEL="zai-org/GLM-5-FP8"
PORT=8000
SERVER_INFO_FILE=/workspace/server_info.txt

HOSTNAME=$(hostname)
echo "http://${HOSTNAME}:${PORT}" > "$SERVER_INFO_FILE"
echo "Server will be available at: http://${HOSTNAME}:${PORT}"
echo "Server info written to: $SERVER_INFO_FILE"

cleanup() {
    echo "Cleaning up..."
    rm -f "$SERVER_INFO_FILE"
}
trap cleanup EXIT

# transformers main is required for the glm_moe_dsa architecture.
echo "Installing latest transformers from GitHub main branch..."
pip install https://github.com/huggingface/transformers/archive/refs/heads/main.zip

# Profiling is activated via the /start_profile endpoint.
mkdir -p /workspace/traces
export VLLM_TORCH_PROFILER_DIR=/workspace/traces
export VLLM_CACHE_ROOT=/workspace/model_weights/vllm_cache
export DG_JIT_CACHE_DIR=/workspace/model_weights/deep_gemm_cache

echo "Starting vLLM server..."

exec vllm serve $MODEL \
    --download-dir /workspace/model_weights \
    --tensor-parallel-size 8 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port $PORT
