#!/usr/bin/env bash
#
# Experimental: gptoss fp4 mi300x benchmark with GPU power monitoring
# Uses amd-smi or rocm-smi --csv for native CSV power/temp/utilization logging
#

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# --- GPU Power Monitoring (CSV, 1-second intervals) ---
GPU_POWER_CSV="/workspace/gpu_power_${RESULT_FILENAME}.csv"
echo "Starting GPU power monitoring -> $GPU_POWER_CSV"

if command -v amd-smi &>/dev/null; then
    echo "Using amd-smi for power monitoring"
    amd-smi metric -p -t -u -b --csv -w 1 > "$GPU_POWER_CSV" 2>/dev/null &
    POWER_MONITOR_PID=$!
elif command -v rocm-smi &>/dev/null; then
    echo "Using rocm-smi for power monitoring (amd-smi not found)"
    (
        # Write header once, then append data rows every second
        rocm-smi --showpower --showtemp --showuse --showmeminfo vram --csv | head -1 > "$GPU_POWER_CSV"
        while true; do
            rocm-smi --showpower --showtemp --showuse --showmeminfo vram --csv | tail -n +2 >> "$GPU_POWER_CSV"
            sleep 1
        done
    ) &
    POWER_MONITOR_PID=$!
else
    echo "WARNING: Neither amd-smi nor rocm-smi found. Skipping GPU power monitoring."
    POWER_MONITOR_PID=""
fi
echo "Power monitor PID: $POWER_MONITOR_PID"

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--compilation-config  '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
--block-size=64 \
--no-enable-prefix-caching \
--disable-log-requests > $SERVER_LOG 2>&1 &

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
set +x

# Stop power monitoring
if [ -n "$POWER_MONITOR_PID" ]; then
    kill $POWER_MONITOR_PID 2>/dev/null
    wait $POWER_MONITOR_PID 2>/dev/null
fi

# Print power summary
if [ -f "$GPU_POWER_CSV" ]; then
    LINES=$(wc -l < "$GPU_POWER_CSV")
    echo ""
    echo "=== GPU Power Summary ==="
    echo "Collected $((LINES - 1)) power samples"
    echo "CSV: $GPU_POWER_CSV"
    echo "First 3 lines:"
    head -3 "$GPU_POWER_CSV"
    echo "Last 3 lines:"
    tail -3 "$GPU_POWER_CSV"
else
    echo "WARNING: gpu_power.csv not found!"
fi

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
