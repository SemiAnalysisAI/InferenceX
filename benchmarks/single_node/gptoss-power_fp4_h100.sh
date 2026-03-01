#!/usr/bin/env bash
#
# Experimental: gptoss fp4 h100 benchmark with GPU power monitoring
# Tracks nvidia-smi power consumption every second and saves to CSV
#

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

# ── Start GPU power monitoring (1-second interval) ──
echo "=== Starting GPU power monitoring ==="
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1 > /workspace/gpu_power.csv 2>&1 &
POWER_MONITOR_PID=$!
echo "Power monitor PID: $POWER_MONITOR_PID"

# ── Benchmark logic (same as gptoss_fp4_h100.sh) ──
hf download "$MODEL"

cat > config.yaml << EOF
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

export PYTHONNOUSERSITE=1
export VLLM_MXFP4_USE_MARLIN=1
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

set -x
vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config config.yaml \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC  \
--disable-log-requests > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

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

# ── Stop GPU power monitoring ──
echo "=== Stopping GPU power monitoring ==="
kill $POWER_MONITOR_PID 2>/dev/null || true
wait $POWER_MONITOR_PID 2>/dev/null || true

# Print summary
if [ -f /workspace/gpu_power.csv ]; then
    LINES=$(wc -l < /workspace/gpu_power.csv)
    echo "GPU power data collected: $LINES samples"
    echo "First 5 lines:"
    head -5 /workspace/gpu_power.csv
    echo "Last 5 lines:"
    tail -5 /workspace/gpu_power.csv
else
    echo "WARNING: gpu_power.csv not found!"
fi
