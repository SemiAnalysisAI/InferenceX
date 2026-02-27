#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:nightly
#SBATCH --container-mounts=/mnt/home/kimbo/inferperf/minimax-m2.5:/workspace,/mnt/vast/model_weights/minimax-m2.5:/model_storage
#SBATCH --no-container-entrypoint
#SBATCH --job-name=vllm-minimax-m25-tp
#SBATCH --output=/mnt/home/kimbo/inferperf/minimax-m2.5/logs/vllm-server-tep8.log
#SBATCH --open-mode=append

# Persistent vLLM server for MiniMax M2.5 (tensor-parallel + expert-parallel)
set -euo pipefail

MODEL="MiniMaxAI/MiniMax-M2.5"
PORT=8000
SERVER_INFO_FILE=/workspace/logs/server_info.txt

# Write server address to shared file for client jobs
HOSTNAME=$(hostname)
echo "http://${HOSTNAME}:${PORT}" > "$SERVER_INFO_FILE"
echo "Server will be available at: http://${HOSTNAME}:${PORT}"
echo "Server info written to: $SERVER_INFO_FILE"

# Trap to clean up on exit
cleanup() {
    echo "Cleaning up..."
    rm -f "$SERVER_INFO_FILE"
}
trap cleanup EXIT

# Install specific vLLM commit
VLLM_COMMIT=dea63512bb9bdf7521d591546c52138d9d79e8ce
echo "Installing vLLM commit ${VLLM_COMMIT}..."
pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}

export VLLM_CACHE_ROOT=/model_storage/vllm_cache
export DG_JIT_CACHE_DIR=/model_storage/deep_gemm_cache

# Start vLLM server (foreground - keeps job alive)
echo "Starting vLLM server..."

exec vllm serve $MODEL \
    --download-dir /model_storage \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/traces"}' \
    --host 0.0.0.0 \
    --port $PORT
