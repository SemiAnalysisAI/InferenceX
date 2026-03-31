#!/usr/bin/env bash
set -euo pipefail
set -x

# Trace replay benchmark for FP8 models on MI355X.
# Replays real agentic coding traces at a fixed number of concurrent users.
# Uses kv-cache-tester/trace_replay_tester.py with realistic cache patterns.
#
# Required env vars:
#   MODEL, TP, USERS, OFFLOAD_MODE, TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   PORT (default 8888), REQUEST_TIMEOUT (default 3600)
#   DURATION (default 1800, benchmark duration in seconds)
#   MAX_DELAY (default 60, max gap between requests in seconds)
#   ADVANCE_MIN (default 0.0, min trace advancement fraction)
#   ADVANCE_MAX (default 0.7, max trace advancement fraction)

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
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ---- Download model --------------------------------------------------------
hf download "$MODEL"

nvidia-smi 2>/dev/null || rocm-smi 2>/dev/null || true

# ---- Paths -----------------------------------------------------------------
MULTITURN_DIR=/workspace/experimental/multiturn/vllm_benchmark
KV_CACHE_TESTER_DIR="$MULTITURN_DIR/kv-cache-tester"
TRACE_DIR="$KV_CACHE_TESTER_DIR/traces"

pip install --quiet urllib3 requests 2>/dev/null || true

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

# Patch vLLM bug: stale KV transfer callback after request cleanup (PR #37859)
# (causes "AssertionError: assert req_id in self.requests" crash under KV offloading)
SCHED_FILE=$(python3 -c "import vllm; import os; print(os.path.join(os.path.dirname(vllm.__file__), 'v1', 'core', 'sched', 'scheduler.py'))" 2>/dev/null || echo "")
if [ -n "$SCHED_FILE" ] && [ -f "$SCHED_FILE" ] && grep -q 'assert req_id in self.requests' "$SCHED_FILE"; then
    echo "Patching vLLM scheduler.py: $SCHED_FILE"
    python3 << 'PYEOF' "$SCHED_FILE"
import sys
with open(sys.argv[1]) as f:
    src = f.read()
src = src.replace(
    'assert req_id in self.requests\n            req = self.requests[req_id]\n            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:',
    'req = self.requests.get(req_id)\n            if req is None:\n                logger.debug("Ignoring finished recving KV transfer for unknown request %s", req_id)\n                self.finished_recving_kv_req_ids.discard(req_id)\n                continue\n            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:',
)
src = src.replace(
    'assert req_id in self.requests\n            self._free_blocks(self.requests[req_id])',
    'req = self.requests.get(req_id)\n            if req is None:\n                logger.debug("Ignoring finished sending KV transfer for unknown request %s", req_id)\n                continue\n            self._free_blocks(req)',
)
with open(sys.argv[1], 'w') as f:
    f.write(src)
PYEOF
fi

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- Generate vLLM config --------------------------------------------------
# cat > "$RESULT_DIR/config.yaml" << 'EOF'
# kv-cache-dtype: fp8
# async-scheduling: true
# max-num-batched-tokens: 8192
# EOF

cat > "$RESULT_DIR/config.yaml" << 'EOF'
kv-cache-dtype: fp8
async-scheduling: true
EOF

# ---- Build vLLM command -----------------------------------------------------
offload_size=$TOTAL_CPU_DRAM_GB
# max_seqs=$USERS

VLLM_CMD="vllm serve $MODEL --host 0.0.0.0 --port $PORT"
VLLM_CMD+=" --config $RESULT_DIR/config.yaml"
# VLLM_CMD+=" --max-num-seqs $max_seqs"
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
# MI355X is ROCm — no CUDA arch needed
# export TORCH_CUDA_ARCH_LIST="9.0"
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
pip install -q -r "$KV_CACHE_TESTER_DIR/requirements.txt"
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

# ---- Run trace replay benchmark ---------------------------------------------
REPLAY_CMD="python3 $KV_CACHE_TESTER_DIR/trace_replay_tester.py"
REPLAY_CMD+=" --api-endpoint http://localhost:$PORT"
REPLAY_CMD+=" --trace-directory $TRACE_DIR"
REPLAY_CMD+=" --output-dir $RESULT_DIR/trace_replay"
REPLAY_CMD+=" --start-users $USERS"
REPLAY_CMD+=" --max-users $USERS"
REPLAY_CMD+=" --max-ttft 9999"
REPLAY_CMD+=" --test-duration $DURATION"
REPLAY_CMD+=" --recycle"
REPLAY_CMD+=" --max-delay $MAX_DELAY"
REPLAY_CMD+=" --max-concurrent-requests 0"
REPLAY_CMD+=" --max-new-tokens-per-period 999999999"
REPLAY_CMD+=" --advance-min $ADVANCE_MIN"
REPLAY_CMD+=" --advance-max $ADVANCE_MAX"
REPLAY_CMD+=" --advance-all-users"
REPLAY_CMD+=" --seed 42"
REPLAY_CMD+=" --no-color"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
if $REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log"; then
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
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true

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
