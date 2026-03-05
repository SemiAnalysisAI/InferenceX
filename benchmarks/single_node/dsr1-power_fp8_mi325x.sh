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
echo "timestamp,gpu,socket_power_w,gfx_clock_mhz,mem_clock_mhz,temp_edge_c,temp_hotspot_c,temp_mem_c,gfx_activity_pct,umc_activity_pct,vram_used_mib,vram_total_mib" > "$GPU_POWER_CSV"

# Start background amd-smi polling loop (1-second intervals)
# Uses separate amd-smi metric calls to avoid embedded-list CSV parsing issues
(
  while true; do
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    # Use Python to query amd-smi for each metric category separately and merge
    python3 -c "
import subprocess, csv, io, sys

def parse_csv(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    reader = csv.DictReader(io.StringIO(result.stdout))
    rows = {}
    for row in reader:
        gpu = row.get('gpu', '')
        rows[gpu] = row
    return rows

try:
    power = parse_csv(['amd-smi', 'metric', '--power', '--csv'])
    clock = parse_csv(['amd-smi', 'metric', '--clock', '--csv'])
    temp = parse_csv(['amd-smi', 'metric', '--temperature', '--csv'])
    usage = parse_csv(['amd-smi', 'metric', '--usage', '--csv'])
    vram = parse_csv(['amd-smi', 'metric', '--vram', '--csv'])

    for gpu_id in sorted(power.keys()):
        p = power.get(gpu_id, {})
        c = clock.get(gpu_id, {})
        t = temp.get(gpu_id, {})
        u = usage.get(gpu_id, {})
        v = vram.get(gpu_id, {})

        socket_power = p.get('socket_power', '')
        gfx_clk = c.get('gfx_0_clk', '')
        mem_clk = c.get('mem_0_clk', '')
        temp_edge = t.get('edge', '')
        temp_hotspot = t.get('hotspot', '')
        temp_mem = t.get('mem', '')
        gfx_activity = u.get('gfx_activity', '')
        umc_activity = u.get('umc_activity', '')
        vram_used = v.get('used_vram', '')
        vram_total = v.get('total_vram', '')

        print(f'$ts,{gpu_id},{socket_power},{gfx_clk},{mem_clk},{temp_edge},{temp_hotspot},{temp_mem},{gfx_activity},{umc_activity},{vram_used},{vram_total}')
except Exception as e:
    pass
" >> "$GPU_POWER_CSV" 2>/dev/null
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
