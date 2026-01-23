#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:0
#SBATCH --container-image=vllm/vllm-openai:nightly-8711b216766bb5d3cbe15161061c3a7d9fffe59c
#SBATCH --container-mounts=/home/kimbo/inferperf_kimi-k2:/workspace,/home/kimbo/inferperf_kimi-k2/results:/results,/home/kimbo/inferperf_kimi-k2/logs:/logs
#SBATCH --no-container-entrypoint
#SBATCH --job-name=client-kimi-k2
#SBATCH --output=/home/kimbo/inferperf_kimi-k2/logs/bmk-client.log
#SBATCH --mem=16G

set -euo pipefail

MODEL="moonshotai/Kimi-K2-Thinking"
SERVER_URL=$(cat /logs/server_info.txt)
echo "Server URL: $SERVER_URL"

# === Sweep Configuration ===
INPUT_LENS=(1024 2048 4096 6144 8192 10240 12288 14336 16384)
OUTPUT_LEN=128
CONCURRENCY_LEVELS=(4 8 16 32 48 64)

# === Run Sweep ===
for INPUT_LEN in "${INPUT_LENS[@]}"; do
    for MAX_CONC in "${CONCURRENCY_LEVELS[@]}"; do
        NUM_WARMUPS=$((MAX_CONC * 2))
        NUM_PROMPTS=$((MAX_CONC * 10))
        RESULT_FILENAME="kimi_k2_vllm_tp8_isl${INPUT_LEN}_osl${OUTPUT_LEN}_conc${MAX_CONC}.json"

        echo "=== Benchmark: ISL=${INPUT_LEN} OSL=${OUTPUT_LEN} CONC=${MAX_CONC} ==="

        python3 /workspace/benchmark_serving_random.py \
            --model $MODEL \
            --base-url "$SERVER_URL" \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --num-prompts $NUM_PROMPTS \
            --max-concurrency $MAX_CONC \
            --num-warmups $NUM_WARMUPS \
            --request-rate inf \
            --ignore-eos \
            --disable-tqdm \
            --result-filepath /results/$RESULT_FILENAME

        echo "Done: $RESULT_FILENAME"
    done
done

echo "All benchmarks complete!"
