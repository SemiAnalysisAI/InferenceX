#!/usr/bin/env bash
set -euo pipefail
set -x

# LMCache agentic trace benchmark for FP8 models on H200 using AIPerf.
# Replays SWE-bench/GAIA/WildClaw agentic traces via mooncake_trace format.
# Dataset: https://huggingface.co/datasets/sammshen/lmcache-agentic-traces
#
# Required env vars:
#   MODEL, TP, USERS, OFFLOAD_MODE, TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   PORT (default 8888), REQUEST_TIMEOUT (default 3600)
#   DURATION (if set, runs for this many seconds; otherwise runs to completion)

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
AIPERF_DIR="$MULTITURN_DIR/aiperf"
TRACE_FILE="$RESULT_DIR/lmcache_traces.jsonl"

pip install --quiet urllib3 requests orjson datasets 2>/dev/null || true

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

mkdir -p "$RESULT_DIR"

# ---- Convert LMCache traces to mooncake format -----------------------------
echo "Downloading and converting LMCache traces..."
python3 -c "
import json, os
try:
    from datasets import load_dataset
    ds = load_dataset('sammshen/lmcache-agentic-traces', split='train')
    out_path = '$TRACE_FILE'
    sessions = set()
    with open(out_path, 'w') as f:
        for row in ds:
            # vLLM v0.18+ follows the newer OpenAI API spec where 'system' role
            # was renamed to 'developer'. Convert to avoid 400 validation errors.
            messages = []
            for msg in row['input']:
                if msg.get('role') == 'system':
                    msg = {**msg, 'role': 'developer'}
                messages.append(msg)
            entry = {
                'session_id': row['session_id'],
                'messages': messages,
                'output_length': row['output_length'],
            }
            f.write(json.dumps(entry) + '\n')
            sessions.add(row['session_id'])
    print(f'Converted {len(ds)} iterations from {len(sessions)} sessions to {out_path}')
except Exception as e:
    print(f'ERROR converting traces: {e}')
    exit(1)
"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- Generate vLLM config --------------------------------------------------
cat > "$RESULT_DIR/config.yaml" << 'EOF'
kv-cache-dtype: fp8
async-scheduling: true
EOF

# ---- Build vLLM command -----------------------------------------------------
offload_size=$TOTAL_CPU_DRAM_GB

VLLM_CMD="vllm serve $MODEL --host 0.0.0.0 --port $PORT"
VLLM_CMD+=" --config $RESULT_DIR/config.yaml"
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

echo "Installing aiperf in isolated venv..."
python3 -m venv /tmp/aiperf-venv --system-site-packages
/tmp/aiperf-venv/bin/pip install -q -e "$AIPERF_DIR" 2>&1 | tail -10
AIPERF_BIN="/tmp/aiperf-venv/bin/aiperf"

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

sleep 2

# ---- Run AIPerf benchmark ----------------------------------------------------
export AIPERF_LOG_CONVERSATIONS="$RESULT_DIR/conversations.jsonl"

AIPERF_CMD="$AIPERF_BIN profile"
AIPERF_CMD+=" --model $MODEL"
AIPERF_CMD+=" --url http://localhost:$PORT"
AIPERF_CMD+=" --endpoint-type chat"
AIPERF_CMD+=" --streaming"
AIPERF_CMD+=" --input-file $TRACE_FILE"
AIPERF_CMD+=" --custom-dataset-type mooncake_trace"
AIPERF_CMD+=" --concurrency $USERS"
if [ -n "${DURATION:-}" ]; then
    AIPERF_CMD+=" --benchmark-duration $DURATION"
    AIPERF_CMD+=" --benchmark-grace-period 0"
fi
AIPERF_CMD+=" --request-timeout-seconds $REQUEST_TIMEOUT"
AIPERF_CMD+=" --output-artifact-dir $RESULT_DIR/aiperf_artifacts"
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
