#!/usr/bin/env bash

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

hf download "$MODEL"

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-sglang-deepseek-r1-fp8.html#run-the-inference-benchmark

# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
# Disable that features to avoid crashes.
# This is related to the changes in the driver at:
# https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# --- GPU Power Monitoring ---
GPU_POWER_CSV="/workspace/gpu_power_${RESULT_FILENAME}.csv"

# Write CSV header
echo "timestamp,device,temperature_junction_C,temperature_edge_C,temperature_memory_C,power_draw_W,power_limit_W,gpu_util_pct,gpu_clock_MHz,mem_clock_MHz,vram_used_MiB,vram_total_MiB" > "$GPU_POWER_CSV"

# Start background rocm-smi polling loop (1-second intervals)
(
  while true; do
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    # Collect rocm-smi --csv output for power, temp, utilization, clocks, memory
    raw=$(rocm-smi --showpower --showtemp --showuse --showclocks --showmeminfo vram --csv 2>/dev/null)
    if [ -n "$raw" ]; then
      # Parse CSV output: skip header line, extract fields
      echo "$raw" | tail -n +2 | while IFS=',' read -r device temp_edge temp_junction temp_mem cur_power avg_power power_cap gpu_use gpu_clk mem_clk vram_total vram_used rest; do
        # Clean up values - strip whitespace
        device=$(echo "$device" | xargs)
        temp_junction=$(echo "$temp_junction" | xargs)
        temp_edge=$(echo "$temp_edge" | xargs)
        temp_mem=$(echo "$temp_mem" | xargs)
        cur_power=$(echo "$cur_power" | xargs)
        power_cap=$(echo "$power_cap" | xargs)
        gpu_use=$(echo "$gpu_use" | xargs)
        gpu_clk=$(echo "$gpu_clk" | xargs)
        mem_clk=$(echo "$mem_clk" | xargs)
        vram_used=$(echo "$vram_used" | xargs)
        vram_total=$(echo "$vram_total" | xargs)
        echo "${ts},${device},${temp_junction},${temp_edge},${temp_mem},${cur_power},${power_cap},${gpu_use},${gpu_clk},${mem_clk},${vram_used},${vram_total}"
      done >> "$GPU_POWER_CSV"
    fi
    sleep 1
  done
) &
POWER_MONITOR_PID=$!
echo "GPU power monitor started (PID=$POWER_MONITOR_PID), logging to $GPU_POWER_CSV"

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
--disable-radix-cache > $SERVER_LOG 2>&1 &

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
