#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:8
#SBATCH --container-image=vllm/vllm-openai:nightly-8711b216766bb5d3cbe15161061c3a7d9fffe59c
#SBATCH --container-mounts=/home/kimbo/inferperf_kimi-k2/model_weights:/model_weights,/home/kimbo/inferperf_kimi-k2:/workspace,/home/kimbo/inferperf_kimi-k2/results:/results,/home/kimbo/inferperf_kimi-k2/logs:/logs
#SBATCH --no-container-entrypoint
#SBATCH --job-name=server-kimi-k2
#SBATCH --output=/home/kimbo/inferperf_kimi-k2/logs/vllm-server.log
#SBATCH --mem=256G

# Persistent vLLM server for Kimi K2 Thinking
set -euo pipefail

MODEL="moonshotai/Kimi-K2-Thinking"
PORT=8000
SERVER_INFO_FILE=/logs/server_info.txt

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
    --tensor-parallel-size 8 \
    --enable-auto-tool-choice \
    --max-num-batched-tokens 32768 \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --host 0.0.0.0 \
    --port $PORT
