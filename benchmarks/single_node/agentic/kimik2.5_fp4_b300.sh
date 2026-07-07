#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 NVFP4 on B300 using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=native or KV_OFFLOAD_BACKEND=mooncake.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION


if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
MOONCAKE_MASTER_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$MOONCAKE_MASTER_PID" "Mooncake master"
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
    case "$KV_OFFLOAD_BACKEND" in
    native)
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    mooncake)
        { set +x; } 2>/dev/null
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / TP))

        MOONCAKE_VERSION=0.3.11.post1
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            --force-reinstall "mooncake-transfer-engine-cuda13==$MOONCAKE_VERSION"

        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "tcp",
  "device_name": ""
}
EOF
        export MOONCAKE_CONFIG_PATH
        export WITH_NVIDIA_PEERMEM=0
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1

        echo "Starting Mooncake master on port $MOONCAKE_MASTER_PORT..."
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --default_kv_lease_ttl=1h \
            > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died during startup." >&2
            cat "$MOONCAKE_MASTER_LOG" >&2
            exit 1
        fi
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
        )
        ;;
    *) echo "Error: unsupported KV_OFFLOAD_BACKEND value '$KV_OFFLOAD_BACKEND' (expected one of: native, mooncake)" >&2; exit 1 ;;
    esac
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

DCP_ARGS=()
if [[ "$CONC" -ge 16 ]]; then
    DCP_ARGS=(--decode-context-parallel-size "$TP")
fi

export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --kv-cache-dtype fp8
    --trust-remote-code
    --block-size 64
    --language-model-only
    --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
    --max-cudagraph-capture-size 2048
    --max-num-batched-tokens 16384
    --stream-interval 10
    --enable-prefix-caching
    --tensor-parallel-size "$TP"
    "${DCP_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
