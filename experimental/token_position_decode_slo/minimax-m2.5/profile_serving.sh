#!/usr/bin/env bash
# Profile a serving workload at ISL=16384 conc=16
# Usage: ./profile_serving.sh <server_url>
# Example: ./profile_serving.sh http://slurm-h200-205-057:8000

set -euo pipefail

SERVER_URL="${1:?Usage: $0 <server_url>}"
MODEL="MiniMaxAI/MiniMax-M2.5"
ISL=16384
OSL=128
CONC=16
NUM_WARMUPS=32
NUM_PROFILE=32

echo "=== Profiling Config ==="
echo "Server:    $SERVER_URL"
echo "ISL:       $ISL"
echo "OSL:       $OSL"
echo "Conc:      $CONC"
echo "Warmups:   $NUM_WARMUPS"
echo "Profiled:  $NUM_PROFILE requests"
echo ""

# Wait for server health
echo "Checking server health..."
while ! curl -sf "${SERVER_URL}/health" > /dev/null; do
    echo "Server not ready, retrying in 10s..."
    sleep 10
done
echo "Server is healthy."

# Single run: warmup happens before profiler starts (handled by benchmark script)
echo ""
echo "=== Running warmup ($NUM_WARMUPS) + profiled ($NUM_PROFILE) requests ==="
python3 benchmark_serving_random.py \
    --model $MODEL \
    --base-url "$SERVER_URL" \
    --random-input-len $ISL \
    --random-output-len $OSL \
    --num-warmups $NUM_WARMUPS \
    --num-prompts $NUM_PROFILE \
    --max-concurrency $CONC \
    --request-rate inf \
    --ignore-eos \
    --profile \
    --result-filepath results/profile_tep8_isl${ISL}_osl${OSL}_conc${CONC}.json

echo ""
echo "=== Done ==="
echo "Traces saved to traces/ directory"
echo "View at https://ui.perfetto.dev/"
