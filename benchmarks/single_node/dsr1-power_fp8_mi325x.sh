#!/usr/bin/bash

# Source benchmark utilities early
source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

SERVER_LOG=/workspace/server.log
PORT=8888
hf download $MODEL

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-sglang-deepseek-r1-fp8.html#run-the-inference-benchmark

export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1

# --- GPU Power Monitoring via amd-smi ---
GPU_POWER_CSV="/workspace/gpu_power_${RESULT_FILENAME}.csv"

# Write CSV header
echo "timestamp,gpu,power_w,power_limit_w,temp_junction_c,temp_edge_c,temp_mem_c,gfx_clock_mhz,mem_clock_mhz,gpu_util_pct,vram_used_mib,vram_total_mib" > "$GPU_POWER_CSV"

# Start background amd-smi polling loop (1-second intervals)
(
  while true; do
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    # Use amd-smi metric --csv to get power, temperature, clock, usage, and VRAM
    csv_output=$(amd-smi metric --power --clock --temperature --usage --vram --csv 2>/dev/null)
    if [ -n "$csv_output" ]; then
      # Parse the CSV output (skip header line)
      echo "$csv_output" | tail -n +2 | while IFS=',' read -r gpu power_socket power_gfx power_limit temp_edge temp_junction temp_mem temp_vram gfx_clk mem_clk gpu_busy mem_busy enc_busy dec_busy vram_used vram_total rest; do
        gpu=$(echo "$gpu" | xargs)
        power_socket=$(echo "$power_socket" | xargs)
        power_limit=$(echo "$power_limit" | xargs)
        temp_junction=$(echo "$temp_junction" | xargs)
        temp_edge=$(echo "$temp_edge" | xargs)
        temp_mem=$(echo "$temp_mem" | xargs)
        gfx_clk=$(echo "$gfx_clk" | xargs)
        mem_clk=$(echo "$mem_clk" | xargs)
        gpu_busy=$(echo "$gpu_busy" | xargs)
        vram_used=$(echo "$vram_used" | xargs)
        vram_total=$(echo "$vram_total" | xargs)
        echo "${ts},${gpu},${power_socket},${power_limit},${temp_junction},${temp_edge},${temp_mem},${gfx_clk},${mem_clk},${gpu_busy},${vram_used},${vram_total}"
      done >> "$GPU_POWER_CSV"
    fi
    sleep 1
  done
) &
POWER_MONITOR_PID=$!
echo "GPU power monitor started (PID=$POWER_MONITOR_PID) using amd-smi, logging to $GPU_POWER_CSV"

set -x
python3 -m sglang.launch_server \
--model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
--tensor-parallel-size=$TP \
--mem-fraction-static=0.8 \
--cuda-graph-max-bs=128 \
--chunked-prefill-size=131072 \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=131072 \
--kv-cache-dtype fp8_e4m3 \
--attention-backend aiter \
--disable-radix-cache \
> $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# Stop GPU power monitor
kill $POWER_MONITOR_PID 2>/dev/null || true
wait $POWER_MONITOR_PID 2>/dev/null || true
echo "GPU power monitor stopped"

POWER_LINE_COUNT=$(wc -l < "$GPU_POWER_CSV")
echo "GPU power CSV: $POWER_LINE_COUNT lines written to $GPU_POWER_CSV"

# Copy power CSV to workspace root for artifact upload
cp "$GPU_POWER_CSV" /workspace/ 2>/dev/null || true

set +x
