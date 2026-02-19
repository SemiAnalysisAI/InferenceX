#!/usr/bin/env bash
#SBATCH -p h200
#SBATCH --gres=gpu:0
#SBATCH --container-image=vllm/vllm-openai:nightly
#SBATCH --container-mounts=/mnt/home/kimbo/inferperf/qwen3.5:/workspace
#SBATCH --no-container-entrypoint
#SBATCH --job-name=bmk-qwen35
#SBATCH --output=/mnt/home/kimbo/inferperf/qwen3.5/logs/bmk-client.log

set -euo pipefail

MODEL="Qwen/Qwen3.5-397B-A17B"
PARALLEL_TAG="${PARALLEL_TAG:-tp8}"

wait_for_server() {
    # Wait for server_info.txt to appear (server may still be starting)
    echo "Waiting for server_info.txt..."
    while [ ! -f /workspace/logs/server_info.txt ]; do
        echo "server_info.txt not found, retrying in 60s..."
        sleep 60
    done
    SERVER_URL=$(cat /workspace/logs/server_info.txt)
    echo "Waiting for vLLM server at $SERVER_URL..."
    while ! curl -sf "${SERVER_URL}/health" > /dev/null; do
        # Re-read in case server_info.txt was updated
        SERVER_URL=$(cat /workspace/logs/server_info.txt 2>/dev/null || echo "$SERVER_URL")
        echo "Server not ready, retrying in 60s..."
        sleep 60
    done
    echo "Server is ready at $SERVER_URL"
}

wait_for_server

# === Sweep Configuration ===
INPUT_LENS=(1024 2048 4096 6144 8192 10240 12288 14336 16384)
OUTPUT_LEN=128

get_concurrency_levels() {
    local isl=$1
    echo "4 8 16 32 64"
}

# === Run Sweep ===
for INPUT_LEN in "${INPUT_LENS[@]}"; do
    CONCURRENCY_LEVELS=($(get_concurrency_levels $INPUT_LEN))
    for MAX_CONC in "${CONCURRENCY_LEVELS[@]}"; do
        NUM_WARMUPS=$((MAX_CONC * 2))
        NUM_PROMPTS=$((MAX_CONC * 10))
        RESULT_FILENAME="qwen35_vllm_${PARALLEL_TAG}_isl${INPUT_LEN}_osl${OUTPUT_LEN}_conc${MAX_CONC}.json"

        # Skip if result already exists
        if [ -f /workspace/results/$RESULT_FILENAME ]; then
            echo "Skipping (exists): $RESULT_FILENAME"
            continue
        fi

        # Check server health before each run
        if ! curl -sf "${SERVER_URL}/health" > /dev/null; then
            echo "Server went down, waiting for restart..."
            wait_for_server
        fi

        echo "=== Benchmark: ISL=${INPUT_LEN} OSL=${OUTPUT_LEN} CONC=${MAX_CONC} ==="

        python3 /workspace/benchmark_serving_random.py \
            --model $MODEL \
            --base-url "$SERVER_URL" \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --num-warmups $NUM_WARMUPS \
            --num-prompts $NUM_PROMPTS \
            --max-concurrency $MAX_CONC \
            --request-rate inf \
            --ignore-eos \
            --result-filepath /workspace/results/$RESULT_FILENAME

        # Validate result - remove if not 100% completion
        if [ -f /workspace/results/$RESULT_FILENAME ]; then
            COMPLETED=$(python3 -c "import json; print(json.load(open('/workspace/results/$RESULT_FILENAME'))['completed'])")
            if [ "$COMPLETED" -ne "$NUM_PROMPTS" ]; then
                echo "Incomplete: $COMPLETED/$NUM_PROMPTS completed, removing result and retrying after server restart..."
                rm -f /workspace/results/$RESULT_FILENAME
                wait_for_server
                # Retry this configuration
                python3 /workspace/benchmark_serving_random.py \
                    --model $MODEL \
                    --base-url "$SERVER_URL" \
                    --random-input-len $INPUT_LEN \
                    --random-output-len $OUTPUT_LEN \
                    --num-warmups $NUM_WARMUPS \
                    --num-prompts $NUM_PROMPTS \
                    --max-concurrency $MAX_CONC \
                    --request-rate inf \
                    --ignore-eos \
                    --result-filepath /workspace/results/$RESULT_FILENAME
            fi
        fi

        echo "Done: $RESULT_FILENAME"
    done
done

echo "All benchmarks complete!"
