#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --container-image=vllm/vllm-openai:nightly-8711b216766bb5d3cbe15161061c3a7d9fffe59c
#SBATCH --container-mounts=/home/kimbo/inferperf_minimax-m2.1/model_weights:/model_weights,/home/kimbo/inferperf_minimax-m2.1:/workspace,/home/kimbo/inferperf_minimax-m2.1/results:/results
#SBATCH --no-container-entrypoint
#SBATCH --job-name=vllm-minimax-m2.1
#SBATCH --output=/home/kimbo/inferperf_minimax-m2.1/vllm-server-tp4.log
#SBATCH --mem=128G

# Persistent vLLM server for MiniMax M2.1
set -euo pipefail

MODEL="MiniMaxAI/MiniMax-M2.1"
PORT=8000
SERVER_INFO_FILE=/workspace/server_info.txt

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

# Start vLLM server (foreground - keeps job alive)
echo "Starting vLLM server..."

exec vllm serve $MODEL \
    --download-dir /model_weights \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port $PORT