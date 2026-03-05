#!/usr/bin/env bash
# Experimental: DeepSeek R1 FP8 H200 SGLang benchmark with GPU metrics collection
# Collects power consumption, temperature, and GPU clock every second via nvidia-smi

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

GPU_METRICS_CSV=/workspace/gpu_metrics.csv

# --- GPU Monitoring ---
start_gpu_monitor() {
    echo "Starting GPU metrics collection (1s interval) -> $GPU_METRICS_CSV"
    nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory \
        --format=csv -l 1 > "$GPU_METRICS_CSV" 2>&1 &
    GPU_MONITOR_PID=$!
    echo "GPU monitor PID: $GPU_MONITOR_PID"
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        echo "GPU monitoring stopped"
        if [[ -f "$GPU_METRICS_CSV" ]]; then
            local lines
            lines=$(wc -l < "$GPU_METRICS_CSV")
            echo "GPU metrics: $lines rows saved to $GPU_METRICS_CSV"
        fi
    fi
}

trap stop_gpu_monitor EXIT

# Start monitoring before anything else
start_gpu_monitor

# --- Benchmark Setup ---
pip3 install --user sentencepiece

hf download "$MODEL"
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export TORCH_CUDA_ARCH_LIST="9.0"

nvidia-smi

set -x
if [[ $ISL -eq 1024 && $OSL -eq 1024 ]]; then
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 512 --cuda-graph-max-bs 512 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
else
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 256 --cuda-graph-max-bs 256 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
fi

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

# Stop GPU monitoring (also handled by trap)
stop_gpu_monitor
