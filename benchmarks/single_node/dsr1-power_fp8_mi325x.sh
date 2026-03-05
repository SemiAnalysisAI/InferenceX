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
GPU_POWER_LOG="/workspace/gpu_power_debug.log"

# Diagnostic: check amd-smi availability and output format
echo "=== amd-smi diagnostics ===" | tee "$GPU_POWER_LOG"
echo "which amd-smi: $(which amd-smi 2>&1)" | tee -a "$GPU_POWER_LOG"
echo "amd-smi version: $(amd-smi version 2>&1)" | tee -a "$GPU_POWER_LOG"
echo "--- amd-smi metric --csv sample ---" | tee -a "$GPU_POWER_LOG"
amd-smi metric --csv 2>&1 | head -5 | tee -a "$GPU_POWER_LOG"
echo "--- amd-smi metric --power --csv sample ---" | tee -a "$GPU_POWER_LOG"
amd-smi metric --power --csv 2>&1 | head -5 | tee -a "$GPU_POWER_LOG"
echo "--- amd-smi monitor sample ---" | tee -a "$GPU_POWER_LOG"
amd-smi monitor -ptcug 2>&1 | head -5 | tee -a "$GPU_POWER_LOG"
echo "--- amd-smi static --csv sample ---" | tee -a "$GPU_POWER_LOG"
amd-smi static --csv 2>&1 | head -3 | tee -a "$GPU_POWER_LOG"
echo "=== end diagnostics ===" | tee -a "$GPU_POWER_LOG"

# Write CSV header
echo "timestamp,gpu,power_w,power_limit_w,temp_junction_c,temp_edge_c,temp_mem_c,gfx_clock_mhz,mem_clock_mhz,gpu_util_pct,vram_used_mib,vram_total_mib" > "$GPU_POWER_CSV"

# Start background amd-smi polling loop (1-second intervals)
# We use a Python script for robust CSV parsing since amd-smi output format varies
(
  first_iter=true
  while true; do
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Try amd-smi metric with --csv
    csv_output=$(amd-smi metric --power --clock --temperature --usage --vram --csv 2>/tmp/amdsmi_err.log)
    rc=$?

    if [ "$first_iter" = true ]; then
      echo "[power-monitor] amd-smi metric --csv rc=$rc" >> "$GPU_POWER_LOG"
      echo "[power-monitor] stderr: $(cat /tmp/amdsmi_err.log 2>/dev/null)" >> "$GPU_POWER_LOG"
      echo "[power-monitor] stdout lines: $(echo "$csv_output" | wc -l)" >> "$GPU_POWER_LOG"
      echo "[power-monitor] stdout head:" >> "$GPU_POWER_LOG"
      echo "$csv_output" | head -3 >> "$GPU_POWER_LOG"
      first_iter=false
    fi

    if [ $rc -eq 0 ] && [ -n "$csv_output" ] && [ "$(echo "$csv_output" | wc -l)" -gt 1 ]; then
      # Parse using Python for robustness — handles varying column counts
      python3 -c "
import sys, csv, io
reader = csv.DictReader(io.StringIO(sys.argv[1]))
for row in reader:
    gpu = row.get('gpu', row.get('GPU', ''))
    # Power fields
    power = row.get('power_socket_power', row.get('SOCKET_POWER', row.get('power', '')))
    power_limit = row.get('power_socket_power_cap', row.get('POWER_CAP', row.get('power_cap', '')))
    # Temperature fields
    temp_junction = row.get('temperature_hotspot', row.get('TEMPERATURE_HOTSPOT', row.get('temperature_junction', '')))
    temp_edge = row.get('temperature_edge', row.get('TEMPERATURE_EDGE', ''))
    temp_mem = row.get('temperature_mem', row.get('TEMPERATURE_MEM', row.get('temperature_vram', '')))
    # Clock fields
    gfx_clk = row.get('clock_gfx', row.get('GFX_SCLK', row.get('sclk', '')))
    mem_clk = row.get('clock_mem', row.get('MEM_MCLK', row.get('mclk', '')))
    # Usage fields
    gpu_busy = row.get('usage_gfx_activity', row.get('GFX_ACTIVITY', row.get('gpu_busy_percent', '')))
    # VRAM fields
    vram_used = row.get('vram_used', row.get('VRAM_USED', ''))
    vram_total = row.get('vram_total', row.get('VRAM_TOTAL', ''))
    print(f'${ts},{gpu},{power},{power_limit},{temp_junction},{temp_edge},{temp_mem},{gfx_clk},{mem_clk},{gpu_busy},{vram_used},{vram_total}')
" "$csv_output" >> "$GPU_POWER_CSV" 2>/dev/null
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

# Print debug log
echo "=== Power monitor debug log ==="
cat "$GPU_POWER_LOG" 2>/dev/null || true
echo "=== End debug log ==="

# Copy power CSV to workspace root for artifact upload
cp "$GPU_POWER_CSV" /workspace/ 2>/dev/null || true

set +x
