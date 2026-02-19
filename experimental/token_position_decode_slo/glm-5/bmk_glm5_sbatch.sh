#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:0
#SBATCH --container-image=vllm/vllm-openai:nightly
#SBATCH --container-mounts=/home/kimbo/inferperf/glm-5:/workspace
#SBATCH --no-container-entrypoint
#SBATCH --job-name=bmk-glm5
#SBATCH --output=/home/kimbo/inferperf/glm-5/logs/bmk-client.log

set -euo pipefail

# Install latest transformers (required for GLM-5 tokenizer)
echo "Installing latest transformers from GitHub main branch..."
pip install https://github.com/huggingface/transformers/archive/refs/heads/main.zip

MODEL="zai-org/GLM-5-FP8"

wait_for_server() {
    # Re-read server URL in case server restarted on a different node
    SERVER_URL=$(cat /workspace/server_info.txt)
    echo "Waiting for vLLM server at $SERVER_URL..."
    while ! curl -sf "${SERVER_URL}/health" > /dev/null; do
        # Re-read in case server_info.txt was updated
        SERVER_URL=$(cat /workspace/server_info.txt 2>/dev/null || echo "$SERVER_URL")
        echo "Server not ready, retrying in 60s..."
        sleep 60
    done
    echo "Server is ready at $SERVER_URL"
}

wait_for_server

# === Sweep Configuration ===
INPUT_LENS=(1024 2048 4096 6144 8192 10240 12288 14336 16384)
OUTPUT_LEN=128
CONCURRENCY_LEVELS=(4 8 16 32 48 64)

# === Run Sweep ===
for INPUT_LEN in "${INPUT_LENS[@]}"; do
    for MAX_CONC in "${CONCURRENCY_LEVELS[@]}"; do
        NUM_WARMUPS=$((MAX_CONC * 2))
        NUM_PROMPTS=$((MAX_CONC * 10))
        RESULT_FILENAME="glm5_fp8_vllm_tp8_isl${INPUT_LEN}_osl${OUTPUT_LEN}_conc${MAX_CONC}.json"

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

        # Validate result - remove if all requests failed
        if [ -f /workspace/results/$RESULT_FILENAME ]; then
            COMPLETED=$(python3 -c "import json; print(json.load(open('/workspace/results/$RESULT_FILENAME'))['completed'])")
            if [ "$COMPLETED" -eq 0 ]; then
                echo "All requests failed, removing result and retrying after server restart..."
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
