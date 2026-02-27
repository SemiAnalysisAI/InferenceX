#!/usr/bin/env bash
set -euo pipefail

# Multi-turn benchmark script for FP4 models on B200.
# Runs a single experiment (one TP x users x offload combination) inside
# the enroot container.  Called by runners/launch_b200-nv.sh.
#
# Required env vars (set by benchmark-multiturn-tmpl.yml → runner):
#   MODEL, TP, USERS, OFFLOAD_MODE, DURATION, THINK_TIME,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   PORT (default 8888), MAX_RETRIES (default 3), REQUEST_TIMEOUT (default 3600)

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    USERS \
    OFFLOAD_MODE \
    DURATION \
    THINK_TIME \
    TOTAL_CPU_DRAM_GB \
    RESULT_DIR

PORT=${PORT:-8888}
MAX_RETRIES=${MAX_RETRIES:-3}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-3600}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ---- Download model --------------------------------------------------------
hf download "$MODEL"

nvidia-smi

# ---- Paths -----------------------------------------------------------------
MULTITURN_DIR=/workspace/experimental/multiturn/vllm_benchmark
INPUT_FILE="$MULTITURN_DIR/sample_20k_realistic.json"
SERVER_LOG="$RESULT_DIR/server.log"

mkdir -p "$RESULT_DIR"

# ---- Generate vLLM config --------------------------------------------------
cat > "$RESULT_DIR/config.yaml" << 'EOF'
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true},"custom_ops":["+quant_fp8","+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}'
async-scheduling: true
max-num-batched-tokens: 8192
EOF

# ---- Build vLLM command -----------------------------------------------------
offload_size=$((TOTAL_CPU_DRAM_GB / TP))
max_seqs=$USERS

VLLM_CMD="vllm serve $MODEL --host 0.0.0.0 --port $PORT"
VLLM_CMD+=" --config $RESULT_DIR/config.yaml"
VLLM_CMD+=" --max-num-seqs $max_seqs"
VLLM_CMD+=" --gpu-memory-utilization 0.9"
VLLM_CMD+=" --tensor-parallel-size $TP"
VLLM_CMD+=" --attention-config.use_trtllm_attention=0"

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
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

$VLLM_CMD > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready \
    --port "$PORT" \
    --server-log "$SERVER_LOG" \
    --server-pid "$SERVER_PID"

# ---- Install benchmark deps -------------------------------------------------
pip install -q -r "$MULTITURN_DIR/requirements.txt"

# ---- Run benchmark -----------------------------------------------------------
cd "$MULTITURN_DIR"
export PYTHONPATH="$MULTITURN_DIR:${PYTHONPATH:-}"

BENCH_CMD="python3 benchmark/benchmark_serving_multi_turn.py"
BENCH_CMD+=" -i $INPUT_FILE"
BENCH_CMD+=" -m $MODEL"
BENCH_CMD+=" -u http://localhost:$PORT"
BENCH_CMD+=" -p $USERS"
BENCH_CMD+=" --duration $DURATION"
BENCH_CMD+=" --think-time-lognormal $THINK_TIME"
BENCH_CMD+=" --max-retries $MAX_RETRIES"
BENCH_CMD+=" --request-timeout $REQUEST_TIMEOUT"
BENCH_CMD+=" --metrics-output $RESULT_DIR/metrics"
BENCH_CMD+=" --metrics-csv"
BENCH_CMD+=" --responses-file $RESULT_DIR/responses.json"

echo "$BENCH_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
if $BENCH_CMD > "$RESULT_DIR/benchmark.log" 2>&1; then
    echo "SUCCESS" > "$RESULT_DIR/status.txt"
    echo "Benchmark completed successfully"
else
    echo "FAILED" > "$RESULT_DIR/status.txt"
    echo "Benchmark failed"
fi
set +x

# ---- Cleanup -----------------------------------------------------------------
echo "Stopping vllm server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "Experiment finished at $(date)"
