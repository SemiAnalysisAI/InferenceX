#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:nightly
#SBATCH --container-mounts=/mnt/home/kimbo/inferperf/qwen3.5:/workspace,/mnt/vast/model_weights/qwen3.5:/model_storage
#SBATCH --no-container-entrypoint
#SBATCH --job-name=vllm-qwen35-tp
#SBATCH --output=/mnt/home/kimbo/inferperf/qwen3.5/logs/vllm-server-tp8.log
#SBATCH --open-mode=append

# Persistent vLLM server for Qwen3.5-397B-A17B (tensor-parallel)
set -euo pipefail

MODEL="Qwen/Qwen3.5-397B-A17B"
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

export VLLM_CACHE_ROOT=/model_storage/vllm_cache

# Start vLLM server (foreground - keeps job alive)
echo "Starting vLLM server..."

exec vllm serve $MODEL \
    --download-dir /model_storage \
    --tensor-parallel-size 8 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port $PORT
