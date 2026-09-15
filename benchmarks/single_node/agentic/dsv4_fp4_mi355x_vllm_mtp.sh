#!/usr/bin/env bash
set -eo pipefail
set -x

# DeepSeek-V4-Pro FP4 on MI355X with vLLM MTP and golden synthetic acceptance.
# Pure TP (DP_ATTENTION=false), TP+EP (EP_SIZE>1), and DEP (DP_ATTENTION=true)
# arms. https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro?hardware=mi355x
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=vllm-native or lmcache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR \
    DURATION EP_SIZE DP_ATTENTION
check_env_vars EVAL_ONLY

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

resolve_trace_source
install_agentic_deps

# The nightly ROCm image lacks these runtime deps.
agentic_pip_install --quiet Pillow fastapi uvicorn

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# vllm-router expands one HTTP backend into a logical worker per DP rank.
# AIPerf's X-Correlation-ID is stable across a conversation's turns; alias it
# to the router's X-Session-ID so every turn lands on the same rank.
USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_VERSION=0.1.14
    VLLM_ROUTER_POLICY=consistent_hash
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet "vllm-router==$VLLM_ROUTER_VERSION"
fi

# AIPerf scrapes the public endpoint's /metrics, which is the router under
# DP-attention; add the engine endpoint explicitly (deduplicated for pure TP).
export AIPERF_SERVER_METRICS_URLS="http://localhost:${VLLM_BACKEND_PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"

# 805 GiB checkpoint; cold Weka loads take about two hours for the 64 shards.
export VLLM_ENGINE_READY_TIMEOUT_S=10800

# vllm-project/vllm#43447 keeps local SWA prefix-cache tails sparsely, while
# vllm-project/vllm#44774 applies the same reachability policy to Mooncake's
# store mask. 32k matches the trace-replay tuning validated for this workload.
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
ROUTER_PID=""

OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
    check_env_vars KV_OFFLOAD_BACKEND
    case "$KV_OFFLOAD_BACKEND" in
      vllm-native)
        require_agentic_kv_offload_backend vllm-native
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        TOTAL_CPU_DRAM_PARTITION_GB="$((TOTAL_CPU_DRAM_GB / (8 / TP)))"
        # OffloadingConnector, not SimpleCPUOffloadConnector: VLLM_USE_SIMPLE_KV_OFFLOAD
        # must stay unset.

        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_PARTITION_GB"
        )

        ;;
      lmcache)
        require_agentic_kv_offload_backend lmcache
        LMCACHE_PID=""

        cleanup_lmcache_server() {
            if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
                kill "$LMCACHE_PID" 2>/dev/null || true
                wait "$LMCACHE_PID" 2>/dev/null || true
            fi
        }

        trap cleanup_lmcache_server EXIT

        cleanup_agentic_services() {
            local exit_code=$?
            trap - EXIT INT TERM
            set +e
            stop_background_process_tree "$ROUTER_PID" "vLLM router"
            stop_background_process_tree "$SERVER_PID" "vLLM server" 60
            exit "$exit_code"
        }
        trap cleanup_agentic_services EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM

        wait_for_lmcache_ready() {
            { set +x; } 2>/dev/null
            local attempts="120"
            local tail_pid=""

            while [ ! -f "$LMCACHE_LOG" ]; do
                if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
                    echo "LMCache server died before creating log file. Exiting." >&2
                    exit 1
                fi
                sleep 10
            done

            tail -f -n +1 "$LMCACHE_LOG" &
            tail_pid=$!

            for ((i = 1; i <= attempts; i++)); do
                if curl --output /dev/null --silent --fail "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck"; then
                    kill "$tail_pid" 2>/dev/null || true
                    wait "$tail_pid" 2>/dev/null || true
                    return 0
                fi
                if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
                    echo "LMCache server died before becoming healthy. Log follows:" >&2
                    kill "$tail_pid" 2>/dev/null || true
                    wait "$tail_pid" 2>/dev/null || true
                    cat "$LMCACHE_LOG" >&2 || true
                    exit 1
                fi
                sleep 1
            done

            echo "Timed out waiting for LMCache server healthcheck. Log follows:" >&2
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        }
            { set +x; } 2>/dev/null
            unset VLLM_USE_SIMPLE_KV_OFFLOAD

            git clone https://github.com/LMCache/LMCache.git
            cd LMCache
            # https://github.com/LMCache/LMCache/pull/3853
            git checkout 9229067cec0b3a63bb8a39368d101db7ac0bc3c1
            pip install -r requirements/build.txt
            pip install grpcio==1.78.0
            CXX=hipcc BUILD_WITH_HIP=1 pip install -e .   --no-build-isolation
            cd ..

            python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

            TOTAL_CPU_DRAM_PARTITION_GB="$((TOTAL_CPU_DRAM_GB / (8 / TP)))"
            # The external MP server owns the pool so vLLM does not split
            # --kv-offloading-size across TP ranks.
            LMCACHE_HOST="127.0.0.1"
            LMCACHE_PORT="5555"
            LMCACHE_HTTP_PORT="8080"
            # LMCacheMPConnector concatenates lmcache.mp.host and port into the
            # ZMQ endpoint, so the connector gets a ZMQ-style host string.
            LMCACHE_CONNECT_HOST="tcp://$LMCACHE_HOST"
            LMCACHE_L1_SIZE_GB="${TOTAL_CPU_DRAM_PARTITION_GB}"
            if [ "$LMCACHE_L1_SIZE_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
                echo "Error: LMCACHE_L1_SIZE_GB=$LMCACHE_L1_SIZE_GB exceeds configured capacity $TOTAL_CPU_DRAM_GB" >&2
                exit 1
            fi
            LMCACHE_L1_INIT_SIZE_GB="20"
            # Read locks are leases on chunks lookup promised vLLM can retrieve.
            # TP8/conc32 can spend >300 s between lookup and retrieve while GPU
            # KV is saturated, leaving the object in L1 but unreadable.
            LMCACHE_L1_READ_TTL_SECONDS="7200"
            LMCACHE_CHUNK_SIZE="256"
            LMCACHE_MAX_WORKERS="$TP"
            export PYTHONHASHSEED="0"
            export LMCACHE_BLOCKING_TIMEOUT_SECS=1200
            LMCACHE_TX_MODE="lmcache_driven"

            echo "Starting LMCache MP server..."
            LMCACHE_CMD=(
                lmcache server
                --host "$LMCACHE_HOST"
                --port "$LMCACHE_PORT"
                --http-host "$LMCACHE_HOST"
                --http-port "$LMCACHE_HTTP_PORT"
                --l1-size-gb "$LMCACHE_L1_SIZE_GB"
                --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
                --l1-read-ttl-seconds "$LMCACHE_L1_READ_TTL_SECONDS"
                --chunk-size "$LMCACHE_CHUNK_SIZE"
                --max-workers "$LMCACHE_MAX_WORKERS"
                --eviction-policy LRU
                --supported-transfer-mode "$LMCACHE_TX_MODE"
            )
            printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
            printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
            "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
            LMCACHE_PID=$!
            echo "LMCache server PID: $LMCACHE_PID"
            wait_for_lmcache_ready

            PREFIX_CACHE_ARGS=(--enable-prefix-caching)
            OFFLOAD_ARGS=(
                --kv-transfer-config
                "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
            )
        ;;
      *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: vllm-native, lmcache)" >&2
        exit 1
        ;;
    esac
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

DP_SCHED_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    DP_SCHED_ARGS=(
        --prefill-schedule-interval 8
        --long-prefill-token-threshold 16384
    )
fi

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_NUM_SEQS=$((2 * CONC))
if [ "$DP_ATTENTION" = "true" ]; then
    MAX_NUM_SEQS="$CONC"
fi

# Golden AL 2.49: committed thinking-on curve for a three-token MTP draft.
# Eval-only runs use real target verification.
NUM_SPEC_TOKENS=3
SYNTHETIC_ACCEPT_LEN=2.49
if [ "${EVAL_ONLY}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}"
else
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

echo "Starting vllm server..."
set -x
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_MOE=1
# This checkpoint mixes packed MXFP4 routed experts with a full-width FP8
# shared expert. The latest nightly otherwise admits the combination into the
# fused path and fails while loading incompatible scales/shapes.
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
# vLLM only clamps torch threads after weight loading; cap from process start.
export OMP_NUM_THREADS=1

sleep 180

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --trust-remote-code
    --async-scheduling
    --distributed-executor-backend mp
    --kv-cache-dtype fp8
    --max-num-batched-tokens 8192
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${DP_SCHED_ARGS[@]}"
    --gpu-memory-utilization 0.86
    --moe-backend aiter
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --speculative-config "$SPEC_CONFIG"
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    --enable-auto-tool-choice
    --enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs "$MAX_NUM_SEQS"
    "${OFFLOAD_ARGS[@]}"
)

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_VLLM_ROUTER" = "true" ]; then
    echo "Starting native vLLM router on port $PORT for $TP DP ranks..."
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy "$VLLM_ROUTER_POLICY" \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$VLLM_ROUTER_METRICS_PORT" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
