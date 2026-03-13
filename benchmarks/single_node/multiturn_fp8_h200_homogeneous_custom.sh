#!/usr/bin/env bash
set -euo pipefail
set -x

# Homogeneous multi-turn benchmark for FP8 models on H200 using custom client.
# Every session has ~10 turns, ~2000 new ISL/turn, ~500 OSL/turn.
# Uses benchmark_serving_multi_turn.py with --synthetic and --steady-state-prefill.
#
# Required env vars:
#   MODEL, TP, USERS, OFFLOAD_MODE, TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   PORT (default 8888), REQUEST_TIMEOUT (default 3600)
#   DURATION (if set, runs for this many seconds; otherwise uses fixed request count)

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    USERS \
    OFFLOAD_MODE \
    TOTAL_CPU_DRAM_GB \
    RESULT_DIR

PORT=${PORT:-8888}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-3600}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ---- Download model --------------------------------------------------------
hf download "$MODEL"

nvidia-smi

# ---- Paths -----------------------------------------------------------------
MULTITURN_DIR=/workspace/experimental/multiturn/vllm_benchmark

pip install --quiet urllib3 requests 2>/dev/null || true

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- Generate vLLM config --------------------------------------------------
cat > "$RESULT_DIR/config.yaml" << 'EOF'
kv-cache-dtype: fp8
async-scheduling: true
max-num-batched-tokens: 8192
EOF

# ---- Build vLLM command -----------------------------------------------------
offload_size=$TOTAL_CPU_DRAM_GB
max_seqs=$USERS

VLLM_CMD="vllm serve $MODEL --host 0.0.0.0 --port $PORT"
VLLM_CMD+=" --config $RESULT_DIR/config.yaml"
VLLM_CMD+=" --max-num-seqs $max_seqs"
VLLM_CMD+=" --gpu-memory-utilization 0.9"
VLLM_CMD+=" --tensor-parallel-size $TP"

if [ "$OFFLOAD_MODE" = "on" ]; then
    VLLM_CMD+=" --kv_offloading_backend native"
    VLLM_CMD+=" --kv_offloading_size $offload_size"
    VLLM_CMD+=" --disable-hybrid-kv-cache-manager"
elif [ "$OFFLOAD_MODE" = "noprefix" ]; then
    VLLM_CMD+=" --no-enable-prefix-caching"
fi

echo "$VLLM_CMD" > "$RESULT_DIR/vllm_command.txt"

# ---- Start vLLM server ------------------------------------------------------
echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1

$VLLM_CMD > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready \
    --port "$PORT" \
    --server-log "$SERVER_LOG" \
    --server-pid "$SERVER_PID"

# ---- Install dependencies ---------------------------------------------------
set -x
pip install -q -r "$MULTITURN_DIR/requirements.txt"
set +x

# ---- Start server metrics collector -----------------------------------------
export PYTHONPATH="$MULTITURN_DIR:${PYTHONPATH:-}"

echo "Starting server metrics collector..."
python3 -m bench.run_metrics_collector \
    --url "http://localhost:$PORT" \
    --output-prefix "$RESULT_DIR/metrics" \
    --pid-file "$RESULT_DIR/metrics_collector.pid" &
METRICS_PID=$!
echo "Metrics collector PID: $METRICS_PID"

sleep 2

# ---- Run custom multi-turn benchmark ----------------------------------------
BENCH_CMD="python3 $MULTITURN_DIR/benchmark/benchmark_serving_multi_turn.py"
BENCH_CMD+=" --model $MODEL"
BENCH_CMD+=" --url http://localhost:$PORT"
BENCH_CMD+=" --backend vllm"
BENCH_CMD+=" --synthetic"
BENCH_CMD+=" --synthetic-isl-mean 2000"
BENCH_CMD+=" --synthetic-isl-stddev 200"
BENCH_CMD+=" --synthetic-osl-mean 500"
BENCH_CMD+=" --synthetic-osl-stddev 50"
BENCH_CMD+=" --synthetic-turns-mean 10"
BENCH_CMD+=" --synthetic-turns-stddev 1"
BENCH_CMD+=" --synthetic-num-convos 10000"
BENCH_CMD+=" --num-clients $USERS"
BENCH_CMD+=" --max-active-conversations 1"
BENCH_CMD+=" --steady-state-prefill"
BENCH_CMD+=" --ignore-eos"
BENCH_CMD+=" --request-timeout-sec $REQUEST_TIMEOUT"
BENCH_CMD+=" --seed 42"
if [ -n "${DURATION:-}" ]; then
    BENCH_CMD+=" --duration $DURATION"
fi
BENCH_CMD+=" --save-result --result-dir $RESULT_DIR --result-filename benchmark_results"
BENCH_CMD+=" --output-file $RESULT_DIR/conversations.json"

echo "$BENCH_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
if $BENCH_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log"; then
    echo "SUCCESS" > "$RESULT_DIR/status.txt"
    echo "Benchmark completed successfully"
else
    echo "FAILED" > "$RESULT_DIR/status.txt"
    echo "Benchmark failed"
fi
set +x

# ---- Stop metrics collector -------------------------------------------------
echo "Stopping metrics collector..."
if [ -n "$METRICS_PID" ] && kill -0 "$METRICS_PID" 2>/dev/null; then
    kill -TERM "$METRICS_PID" 2>/dev/null || true
    wait "$METRICS_PID" 2>/dev/null || true
fi

# ---- Cleanup -----------------------------------------------------------------
echo "Stopping vllm server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "Experiment finished at $(date)"
