#!/usr/bin/env bash
set -euo pipefail
set -x

# Multi-turn benchmark script for FP4 models on B200 using AIPerf with Mooncake traces.
# Uses AIPerf as the benchmark client with server-side metrics collection.
#
# Required env vars (set by benchmark-multiturn-tmpl.yml → runner):
#   MODEL, TP, USERS, OFFLOAD_MODE, DURATION,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   TRACE_FILE (default: mooncake trace from HuggingFace)
#   PORT (default 8888), REQUEST_TIMEOUT (default 3600)

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    USERS \
    OFFLOAD_MODE \
    DURATION \
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

# Download WildChat dataset from HuggingFace
INPUT_FILE="$MULTITURN_DIR/sample_20k_realistic.json"
echo "Downloading sample_20k_realistic.json from HuggingFace..."
hf download inferencemax/multiturn-benchmark-data sample_20k_realistic.json \
    --repo-type dataset --local-dir "$MULTITURN_DIR"

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

# ---- Install dependencies ---------------------------------------------------
pip install -q -r "$MULTITURN_DIR/requirements.txt"

# Install aiperf
echo "Installing aiperf..."
cd "$AIPERF_DIR"
pip install -q -e . || pip install -q .
cd "$MULTITURN_DIR"

# ---- Start server metrics collector -----------------------------------------
export PYTHONPATH="$MULTITURN_DIR:${PYTHONPATH:-}"

echo "Starting server metrics collector..."
python3 -m bench.run_metrics_collector \
    --url "http://localhost:$PORT" \
    --output-prefix "$RESULT_DIR/metrics" \
    --duration "$DURATION" \
    --pid-file "$RESULT_DIR/metrics_collector.pid" &
METRICS_PID=$!
echo "Metrics collector PID: $METRICS_PID"

# Give the collector a moment to start polling
sleep 2

# ---- Run AIPerf benchmark ----------------------------------------------------
# User-centric rate: QPS scales with USERS to maintain ~1 req/s per user
# with a turn gap of ~1s. Adjust USER_CENTRIC_QPS to control load.
USER_CENTRIC_QPS=${USER_CENTRIC_QPS:-$USERS}

AIPERF_CMD="aiperf profile"
AIPERF_CMD+=" --model $MODEL"
AIPERF_CMD+=" --url http://localhost:$PORT"
AIPERF_CMD+=" --endpoint-type chat"
AIPERF_CMD+=" --streaming"
AIPERF_CMD+=" --input-file $INPUT_FILE"
AIPERF_CMD+=" --custom-dataset-type wildchat"
AIPERF_CMD+=" --user-centric-rate $USER_CENTRIC_QPS"
AIPERF_CMD+=" --num-users $USERS"
AIPERF_CMD+=" --concurrency $USERS"
AIPERF_CMD+=" --benchmark-duration $DURATION"
AIPERF_CMD+=" --request-timeout-seconds $REQUEST_TIMEOUT"
AIPERF_CMD+=" --output-artifact-dir $RESULT_DIR/aiperf_artifacts"
AIPERF_CMD+=" --export-level raw"
AIPERF_CMD+=" --ui-mode none"
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
