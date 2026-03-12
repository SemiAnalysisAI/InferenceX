#!/usr/bin/env bash
set -euo pipefail
set -x

# Multi-turn benchmark script for FP4 models on B200 using LMCache KV connector.
# Uses LMCache for CPU KV cache offloading instead of vLLM's native offloader.
#
# Required env vars (set by benchmark-multiturn-tmpl.yml → runner):
#   MODEL, TP, USERS, OFFLOAD_MODE, DURATION,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   REQUEST_RATE (default 0, Poisson req/s per client)
#   LMCACHE_CHUNK_SIZE (default 256, tokens per KV chunk)
#   PORT (default 8888), MAX_RETRIES (default 3), REQUEST_TIMEOUT (default 3600)

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
MAX_RETRIES=${MAX_RETRIES:-3}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-3600}
REQUEST_RATE=${REQUEST_RATE:-0}
LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-256}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ---- Download model --------------------------------------------------------
hf download "$MODEL"

nvidia-smi

# ---- Paths -----------------------------------------------------------------
MULTITURN_DIR=/workspace/experimental/multiturn/vllm_benchmark
INPUT_FILE="$MULTITURN_DIR/sample_20k_realistic.json"

# Ensure HF CLI dependencies are available (some container builds strip urllib3)
pip install --quiet urllib3 requests 2>/dev/null || true

# ---- Install LMCache -------------------------------------------------------
echo "Installing LMCache..."
pip install --quiet lmcache || {
    echo "WARNING: pip install lmcache failed, trying from source..."
    pip install --quiet git+https://github.com/LMCache/LMCache.git
}
python3 -c "import lmcache; print(f'LMCache version: {lmcache.__version__}')" || echo "WARNING: Could not import lmcache"

# Patch vLLM bug: local_cache_hit counter can go negative under high load
# (causes "Counters can only be incremented by non-negative amounts" crash)
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

# Download dataset from HuggingFace
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

# ---- Generate LMCache config -----------------------------------------------
offload_size=$TOTAL_CPU_DRAM_GB

cat > "$RESULT_DIR/lmcache_config.yaml" << EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
local_cpu: true
max_local_cpu_size: ${offload_size}
EOF

echo "LMCache config:"
cat "$RESULT_DIR/lmcache_config.yaml"

# ---- Build vLLM command -----------------------------------------------------
max_seqs=$USERS

VLLM_CMD="vllm serve $MODEL --host 0.0.0.0 --port $PORT"
VLLM_CMD+=" --config $RESULT_DIR/config.yaml"
VLLM_CMD+=" --max-num-seqs $max_seqs"
VLLM_CMD+=" --gpu-memory-utilization 0.9"
VLLM_CMD+=" --tensor-parallel-size $TP"
VLLM_CMD+=" --attention-config.use_trtllm_attention=0"

if [ "$OFFLOAD_MODE" = "on" ]; then
    # Use LMCache connector for KV offloading instead of native
    VLLM_CMD+=" --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}'"
elif [ "$OFFLOAD_MODE" = "noprefix" ]; then
    VLLM_CMD+=" --no-enable-prefix-caching"
fi

echo "$VLLM_CMD" > "$RESULT_DIR/vllm_command.txt"

# ---- Start vLLM server ------------------------------------------------------
echo "Starting vllm server with LMCache KV connector..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

# LMCache config via environment
export LMCACHE_CONFIG_FILE="$RESULT_DIR/lmcache_config.yaml"
export LMCACHE_USE_EXPERIMENTAL="True"

eval $VLLM_CMD > "$SERVER_LOG" 2>&1 &
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
BENCH_CMD+=" --request-rate $REQUEST_RATE"
BENCH_CMD+=" --max-retries $MAX_RETRIES"
BENCH_CMD+=" --request-timeout $REQUEST_TIMEOUT"
BENCH_CMD+=" --metrics-output $RESULT_DIR/metrics"
BENCH_CMD+=" --metrics-csv"
BENCH_CMD+=" --responses-file $RESULT_DIR/responses.json"

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

# ---- Cleanup -----------------------------------------------------------------
echo "Stopping vllm server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo "Experiment finished at $(date)"
