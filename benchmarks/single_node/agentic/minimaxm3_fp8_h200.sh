#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

resolve_complete_model_snapshot() {
    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

model_cache_dir = Path(sys.argv[1])
try:
    revision = model_cache_dir.joinpath("refs/main").read_text().strip()
except OSError:
    raise SystemExit

if not revision or Path(revision).name != revision:
    raise SystemExit

snapshot = model_cache_dir / "snapshots" / revision
index_path = snapshot / "model.safetensors.index.json"
required_files = (
    snapshot / "config.json",
    snapshot / "tokenizer_config.json",
    index_path,
)
if not all(path.is_file() for path in required_files):
    raise SystemExit
try:
    weight_map = json.loads(index_path.read_text())["weight_map"]
except (KeyError, json.JSONDecodeError, OSError):
    raise SystemExit
shards = {snapshot / filename for filename in weight_map.values()}
if shards and all(path.is_file() for path in shards):
    print(snapshot)
PY
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    MODEL_CACHE_ROOT="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface/hub}}"
    MODEL_CACHE_DIR="$MODEL_CACHE_ROOT/models--${MODEL//\//--}"
    mkdir -p "$MODEL_CACHE_ROOT"
    MODEL_PATH=$(resolve_complete_model_snapshot "$MODEL_CACHE_DIR")
    if [[ -z "$MODEL_PATH" ]]; then
        exec 9>"$MODEL_CACHE_ROOT/.minimaxm3-download.lock"
        flock -w 3600 9
        MODEL_PATH=$(resolve_complete_model_snapshot "$MODEL_CACHE_DIR")
        if [[ -z "$MODEL_PATH" ]]; then
            DOWNLOADED_MODEL_PATH=$(hf download "$MODEL")
            MODEL_PATH=$(resolve_complete_model_snapshot "$MODEL_CACHE_DIR")
            if [[ -z "$MODEL_PATH" ]]; then
                echo "Downloaded model snapshot is incomplete: $DOWNLOADED_MODEL_PATH" >&2
                exit 1
            fi
        fi
        flock -u 9
    fi
    echo "Using complete cached model snapshot: $MODEL_PATH"
    export MODEL_PATH
fi
nvidia-smi

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONNOUSERSITE=1

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
if require_agentic_kv_offload_backend mooncake; then
        PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / TP))
        MOONCAKE_VERSION=0.3.11.post1
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            --force-reinstall "mooncake-transfer-engine-cuda13==$MOONCAKE_VERSION"
        python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
EOF
        export MOONCAKE_CONFIG_PATH PYTHONHASHSEED=0 MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio=0.80 \
            --eviction_ratio=0.10 > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        kill -0 "$MOONCAKE_MASTER_PID"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        )
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if (( EP_SIZE > 1 )); then
    EP_ARGS=(--enable-expert-parallel)
fi

VLLM_BACKEND_PORT="$PORT"
if [[ "$DP_ATTENTION" == "true" ]]; then
    VLLM_BACKEND_PORT=$((PORT + 1))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet 'vllm-router==0.1.14'
fi

MAX_NUM_SEQS=$((2 * CONC))
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_BACKEND_PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --block-size 128 \
    --language-model-only \
    --enable-prefix-caching \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    "${OFFLOAD_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$DP_ATTENTION" == "true" ]]; then
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy consistent_hash \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$((PORT + 10000))" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
