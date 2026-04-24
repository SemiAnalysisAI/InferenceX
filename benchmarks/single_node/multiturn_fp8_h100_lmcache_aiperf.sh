#!/usr/bin/env bash
set -euo pipefail
set -x

# LMCache agentic trace benchmark for FP8 models on H100 using AIPerf.
# Replays SWE-bench/GAIA/WildClaw agentic traces via mooncake_trace format.
# Dataset: https://huggingface.co/datasets/sammshen/lmcache-agentic-traces
#
# Required env vars:
#   MODEL, TP, USERS, OFFLOAD_MODE, TOTAL_CPU_DRAM_GB, RESULT_DIR
# Optional:
#   PORT (default 8888), REQUEST_TIMEOUT (default 3600)
#   DURATION (if set, runs for this many seconds; otherwise runs to completion)

source "$(dirname "$0")/../benchmark_lib.sh"

export CUDA_LAUNCH_BLOCKING=1

ulimit -a

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

# Check vLLM version — patches for local_cache_hit and scheduler stale KV
# transfer are fixed in 0.19.0+
VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "vLLM version: $VLLM_VERSION"
if python3 -c "from packaging.version import Version; exit(0 if Version('${VLLM_VERSION}') >= Version('0.19.0') else 1)" 2>/dev/null; then
    echo "vLLM >= 0.19.0: no patches needed"
else
    echo "WARNING: vLLM $VLLM_VERSION < 0.19.0 — local_cache_hit and scheduler patches are no longer applied. Upgrade to 0.19.0+ for stability."
fi

mkdir -p "$RESULT_DIR"

# ---- Download and convert LMCache traces to mooncake format ----------------
echo "Downloading LMCache traces..."
hf download sammshen/lmcache-agentic-traces --repo-type dataset

echo "Converting LMCache traces to mooncake format..."
python3 -c "
import json, glob, os
hf_cache = os.environ.get('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
# Find the downloaded parquet/jsonl files in the HF cache
candidates = glob.glob(os.path.join(hf_cache, 'datasets--sammshen--lmcache-agentic-traces', '**', '*.parquet'), recursive=True)
if not candidates:
    candidates = glob.glob(os.path.join(hf_cache, 'datasets--sammshen--lmcache-agentic-traces', '**', '*.jsonl'), recursive=True)
if not candidates:
    # Fallback: use datasets library to load from cache
    from datasets import load_dataset
    ds = load_dataset('sammshen/lmcache-agentic-traces', split='train')
    rows = list(ds)
else:
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(candidates):
        table = pq.read_table(f)
        rows.extend(table.to_pylist())
    print(f'Loaded {len(rows)} rows from {len(candidates)} cached files')

out_path = '$TRACE_FILE'
sessions = set()
skipped = 0
with open(out_path, 'w') as f:
    for row in rows:
        # Strip None fields — vLLM's Pydantic validation rejects explicit nulls
        messages = [{k: v for k, v in msg.items() if v is not None} for msg in row['input']]
        if not messages:
            skipped += 1
            continue
        entry = {
            'session_id': row['session_id'],
            'messages': messages,
            'output_length': row['output_length'],
        }
        f.write(json.dumps(entry) + '\n')
        sessions.add(row['session_id'])
if skipped:
    print(f'Skipped {skipped} entries with empty messages')
print(f'Converted {len(rows) - skipped} iterations from {len(sessions)} sessions to {out_path}')
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
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
    VLLM_CMD+=" --kv_offloading_backend native"
    VLLM_CMD+=" --kv_offloading_size $offload_size"
    VLLM_CMD+=" --no-disable-hybrid-kv-cache-manager"
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
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

AIPERF_CMD="$AIPERF_BIN profile"
AIPERF_CMD+=" --model $MODEL"
AIPERF_CMD+=" --url http://localhost:$PORT"
AIPERF_CMD+=" --endpoint-type chat"
AIPERF_CMD+=" --streaming"
AIPERF_CMD+=" --use-server-token-count"
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
