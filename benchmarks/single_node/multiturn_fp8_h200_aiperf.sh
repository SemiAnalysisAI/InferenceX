#!/usr/bin/env bash
set -euo pipefail
set -x

# Multi-turn benchmark script for FP4 models on B200 using AIPerf with Mooncake traces.
# Uses AIPerf as the benchmark client with server-side metrics collection.
#
# Required env vars (set by benchmark-multiturn-tmpl.yml → runner):
#   MODEL, TP, USERS, OFFLOAD_MODE,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   PORT (default 8888), REQUEST_TIMEOUT (default 3600)
#   CONVOS_PER_USER (default 5) — conversations per concurrent user

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
CONVOS_PER_USER=${CONVOS_PER_USER:-5}
MIN_CONVERSATIONS=${MIN_CONVERSATIONS:-100}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ---- Download model --------------------------------------------------------
hf download "$MODEL"

nvidia-smi

# ---- Paths -----------------------------------------------------------------
MULTITURN_DIR=/workspace/experimental/multiturn/vllm_benchmark
AIPERF_DIR="$MULTITURN_DIR/aiperf"

# Ensure HF CLI dependencies are available (some container builds strip urllib3)
pip install --quiet urllib3 requests 2>/dev/null || true

# Patch vLLM bug: local_cache_hit counter can go negative under high load
STATS_FILE=$(python3 -c "import vllm; import os; print(os.path.join(os.path.dirname(vllm.__file__), 'v1', 'metrics', 'stats.py'))" 2>/dev/null || echo "")
if [ -n "$STATS_FILE" ] && [ -f "$STATS_FILE" ] && grep -q 'self.local_cache_hit += (' "$STATS_FILE"; then
    echo "Patching vLLM stats.py: $STATS_FILE"
    python3 -c "
import re, sys
with open(sys.argv[1]) as f:
    src = f.read()
src = src.replace(
    'self.local_cache_hit += (\n            num_cached_tokens + recomputed - num_external_computed_tokens\n        )',
    'self.local_cache_hit += max(0,\n            num_cached_tokens + recomputed - num_external_computed_tokens\n        )',
)
with open(sys.argv[1], 'w') as f:
    f.write(src)
" "$STATS_FILE"
fi

# Generate synthetic multi-turn dataset
# Duration mode: generate a large pool and run for a fixed time
# Count mode: generate exactly USERS * CONVOS_PER_USER conversations
if [ -n "${DURATION:-}" ]; then
    CONV_COUNT=10000
    echo "Duration mode: ${DURATION}s with $CONV_COUNT conversation pool"
else
    CONV_COUNT=$((USERS * CONVOS_PER_USER))
    if [ "$CONV_COUNT" -lt "$MIN_CONVERSATIONS" ]; then
        CONV_COUNT=$MIN_CONVERSATIONS
    fi
    echo "Count mode: $CONV_COUNT conversations ($USERS users x $CONVOS_PER_USER convos)"
fi
INPUT_FILE="$MULTITURN_DIR/synthetic_multiturn.jsonl"
SYNTH_CONFIG="$MULTITURN_DIR/scripts/configs/qwen_trace_profile.yaml"
echo "Generating synthetic dataset ($CONV_COUNT conversations)..."
python3 "$MULTITURN_DIR/scripts/generate_synthetic_dataset.py" \
    --config "$SYNTH_CONFIG" \
    --num-conversations "$CONV_COUNT" \
    --seed 42 \
    --output "$INPUT_FILE"

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
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

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

# Install aiperf in a separate venv to avoid dependency conflicts with vLLM
echo "Installing aiperf in isolated venv..."
python3 -m venv /tmp/aiperf-venv --system-site-packages
/tmp/aiperf-venv/bin/pip install -q -e "$AIPERF_DIR" 2>&1 | tail -10
AIPERF_BIN="/tmp/aiperf-venv/bin/aiperf"

# Verify aiperf is importable
/tmp/aiperf-venv/bin/python -c "import aiperf; print('aiperf installed OK')"
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

# Give the collector a moment to start polling
sleep 2

# ---- Run AIPerf benchmark ----------------------------------------------------
export AIPERF_LOG_CONVERSATIONS="$RESULT_DIR/conversations.jsonl"

AIPERF_CMD="$AIPERF_BIN profile"
AIPERF_CMD+=" --model $MODEL"
AIPERF_CMD+=" --url http://localhost:$PORT"
AIPERF_CMD+=" --endpoint-type chat"
AIPERF_CMD+=" --streaming"
AIPERF_CMD+=" --input-file $INPUT_FILE"
AIPERF_CMD+=" --custom-dataset-type mooncake_trace"
AIPERF_CMD+=" --shared-system-prompt-length 100"
AIPERF_CMD+=" --concurrency $USERS"
AIPERF_CMD+=" --conversation-num $CONV_COUNT"
if [ -n "${DURATION:-}" ]; then
    AIPERF_CMD+=" --benchmark-duration $DURATION"
    AIPERF_CMD+=" --benchmark-grace-period 0"
fi
AIPERF_CMD+=" --request-timeout-seconds $REQUEST_TIMEOUT"
AIPERF_CMD+=" --output-artifact-dir $RESULT_DIR/aiperf_artifacts"
AIPERF_CMD+=" --extra-inputs ignore_eos:true"
AIPERF_CMD+=" --export-level records"
AIPERF_CMD+=" --ui-type simple"
AIPERF_CMD+=" --random-seed 42"

echo "$AIPERF_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
if $AIPERF_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log"; then
    echo "SUCCESS" > "$RESULT_DIR/status.txt"
    echo "Benchmark completed successfully"
else
    echo "FAILED" > "$RESULT_DIR/status.txt"
    echo "Benchmark failed"
fi
set +x

# ---- Analyze workload distributions -----------------------------------------
echo "Analyzing workload distributions..."
python3 "$MULTITURN_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/aiperf_artifacts" -o "$RESULT_DIR" 2>&1 || true

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
