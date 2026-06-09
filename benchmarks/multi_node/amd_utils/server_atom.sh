#!/bin/bash
# ATOM Disaggregated Server Launcher
# =============================================================================
# Uses atom.entrypoints.openai_server with mooncake RDMA KV transfer.
# Mirrors server_sglang.sh topology (dynamic xP/yD) but adapts to ATOM's
# explicit kv-transfer-config and atomesh router.
#
# Key differences from server_sglang.sh:
#   - Engine: atom.entrypoints.openai_server  (not sglang.launch_server)
#   - KV transfer: mooncake (--kv-transfer-config JSON)
#   - Router: atomesh  (not sglang_router)
#   - Prefill port: $PREFILL_PORT (default 8010) / Decode port: $DECODE_PORT (default 8020)
#   - Router port: $ROUTER_PORT (default 8000)
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}"
yD="${yD:-1}"

IPADDRS="${IPADDRS:-localhost}"

# Parallelism
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"

# ATOM server ports (different from SGLang which uses 8000 for all)
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"

# ATOM server tuning (from reference script defaults)
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

DRY_RUN="${DRY_RUN:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

source $ATOM_WS_PATH/setup_deps.sh
source $ATOM_WS_PATH/env_atom.sh

host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
if [[ -z "$host_ip" ]]; then
    host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
host_name=$(hostname)

# =============================================================================
# Cluster Topology Configuration
# =============================================================================

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

# Build prefill IP list and atomesh --prefill args
PREFILL_ARGS=""
PREFILL_IPS=()
for i in $(seq 0 $((xP - 1))); do
    idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_IPS[$i]="${IP_ARRAY[$idx]}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$idx]}:${PREFILL_PORT}"
done

# Build decode IP list and atomesh --decode args
DECODE_ARGS=""
DECODE_IPS=()
for i in $(seq 0 $((yD - 1))); do
    idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_IPS[$i]="${IP_ARRAY[$idx]}"
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$idx]}:${DECODE_PORT}"
done

echo "Prefill IPs : ${PREFILL_IPS[*]}"
echo "Decode  IPs : ${DECODE_IPS[*]}"

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $ATOM_WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 3000

# =============================================================================
# Node Role Assignment
#
# Role mapping (same as server_sglang.sh):
#   rank 0                          -> prefill node 0 + router
#   rank 1 .. (NODE_OFFSET-1)       -> remaining prefill nodes
#   rank NODE_OFFSET ..             -> decode nodes
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    # ──────────────────────────────────────────────────────────────────────────
    # Node 0: prefill server (producer) + atomesh router
    # ──────────────────────────────────────────────────────────────────────────
    echo "NODE INFO ======================================="
    echo "${host_name}:${host_ip} is Prefill Node 0 + Router"
    echo "Prefill TP=${PREFILL_TP_SIZE}, Decode TP=${DECODE_TP_SIZE}"
    echo "Prefill servers: ${PREFILL_ARGS}"
    echo "Decode  servers: ${DECODE_ARGS}"
    echo "================================================"

    PREFILL_CMD="python3 -m atom.entrypoints.openai_server \
        --model ${MODEL_DIR}/${MODEL_NAME} \
        --host 0.0.0.0 --server-port ${PREFILL_PORT} \
        --trust-remote-code \
        -tp ${PREFILL_TP_SIZE} \
        --enable-dp-attention \
        --kv_cache_dtype ${KV_CACHE_DTYPE} \
        --block-size ${BLOCK_SIZE} \
        --gpu-memory-utilization ${MEM_FRACTION} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --kv-transfer-config '{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}' \
        ${EXTRA_SERVER_ARGS}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log &
        set +x
        prefill0_pid=$!
    fi

    # Wait for all prefill and decode servers to be ready
    echo "Waiting for all servers to be up..."
    BARRIER_CMD="python3 $ATOM_WS_PATH/sync.py barrier \
        --node-ips ${IPADDRS} \
        --node-ports ${PREFILL_PORT} \
        --wait-for-all-ports \
        --timeout 3000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi
    echo "All servers up. Starting atomesh router..."

    ROUTER_CMD="/usr/local/bin/atomesh launch \
        --host 0.0.0.0 --port ${ROUTER_PORT} \
        --pd-disaggregation \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS} \
        --policy random \
        --backend atom \
        --log-level info \
        --disable-health-check \
        --disable-circuit-breaker \
        --prometheus-port 29100"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $ROUTER_CMD"
    else
        ROUTER_LOG_FILE="/tmp/slurm_job-${SLURM_JOB_ID}_router_${host_name}.log"
        set -x
        eval "$ROUTER_CMD" 2>&1 | tee "$ROUTER_LOG_FILE" &
        set +x
        proxy_pid=$!

        # Wait for router to accept connections
        HEALTH_BARRIER_CMD="python3 $ATOM_WS_PATH/sync.py barrier \
            --node-ips ${NODE0_ADDR} \
            --node-ports ${ROUTER_PORT} \
            --wait-for-all-ports \
            --timeout 3000"
        eval "$HEALTH_BARRIER_CMD"
        echo "Router is ready for benchmarking"
    fi

    echo "Ready for benchmarking on ${host_name}:${host_ip}"

    cd $ATOM_WS_PATH

    BENCH_CMD="bash $ATOM_WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"

    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        echo "EVAL_ONLY mode: skipping throughput benchmark"
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    # Run evaluation if requested (before killing router)
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        echo "Running lm-eval evaluation on Node 0..."

        # Health check: verify the router is still serving before running eval.
        EVAL_HEALTH_OK=false
        for _attempt in 1 2 3; do
            if curl -sf --max-time 10 "http://0.0.0.0:${ROUTER_PORT}/health" >/dev/null 2>&1; then
                EVAL_HEALTH_OK=true
                break
            fi
            echo "Eval health check attempt $_attempt failed, retrying in 10s..."
            sleep 10
        done

        if [[ "$EVAL_HEALTH_OK" != "true" ]]; then
            echo "WARNING: Router health check failed after 3 attempts. Skipping eval."
        else
            pushd /workspace

            source /workspace/benchmarks/benchmark_lib.sh

            if [[ -n "${EVAL_CONC:-}" ]]; then
                export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
            else
                export EVAL_CONCURRENT_REQUESTS=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
            fi

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "DRY RUN: run_eval --framework lm-eval --port ${ROUTER_PORT} (conc=${EVAL_CONCURRENT_REQUESTS})"
            else
                run_eval --framework lm-eval --port ${ROUTER_PORT}
                eval_rc=$?

                if [[ $eval_rc -ne 0 ]]; then
                    echo "ERROR: run_eval exited rc=$eval_rc; skipping metadata write and eval artifact staging" >&2
                    EVAL_FAILED=1
                else
                    export TP="${PREFILL_TP_SIZE}"
                    export CONC="${EVAL_CONCURRENT_REQUESTS}"
                    export PREFILL_TP="${PREFILL_TP_SIZE}"
                    export PREFILL_EP=1
                    export PREFILL_NUM_WORKERS="${xP}"
                    export DECODE_TP="${DECODE_TP_SIZE}"
                    export DECODE_EP=1
                    export DECODE_NUM_WORKERS="${yD}"
                    export ISL="${BENCH_INPUT_LEN}"
                    export OSL="${BENCH_OUTPUT_LEN}"

                    append_lm_eval_summary

                    EVAL_COPY_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/eval_results"
                    mkdir -p "$EVAL_COPY_DIR"
                    for f in meta_env.json; do
                        [ -e "/workspace/$f" ] && cp -f "/workspace/$f" "$EVAL_COPY_DIR/"
                    done
                    find /workspace -maxdepth 1 -name 'results*.json' -exec cp -f {} "$EVAL_COPY_DIR/" \;
                    find /workspace -maxdepth 1 -name 'sample*.jsonl' -exec cp -f {} "$EVAL_COPY_DIR/" \;

                    echo "Eval completed. Artifacts staged in $EVAL_COPY_DIR"
                fi
            fi

            popd
        fi
    fi

    # Copy results
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing router and prefill server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $proxy_pid
        kill $prefill0_pid
    fi

    if [[ "${EVAL_FAILED:-0}" -eq 1 ]]; then
        echo "ERROR: eval failed; exiting node-0 with rc=1"
        exit 1
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$NODE_OFFSET" ]; then
    # ──────────────────────────────────────────────────────────────────────────
    # Prefill nodes 1..N (kv_producer)
    # ──────────────────────────────────────────────────────────────────────────
    echo "${host_name}:${host_ip} is Prefill Node (rank ${NODE_RANK})"

    # Determine which prefill worker this node belongs to, and its headnode IP
    prefill_worker_idx=$((NODE_RANK / PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_IP="${PREFILL_IPS[$prefill_worker_idx]}"

    PREFILL_CMD="python3 -m atom.entrypoints.openai_server \
        --model ${MODEL_DIR}/${MODEL_NAME} \
        --host 0.0.0.0 --server-port ${PREFILL_PORT} \
        --trust-remote-code \
        -tp ${PREFILL_TP_SIZE} \
        --enable-dp-attention \
        --kv_cache_dtype ${KV_CACHE_DTYPE} \
        --block-size ${BLOCK_SIZE} \
        --gpu-memory-utilization ${MEM_FRACTION} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --kv-transfer-config '{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}' \
        ${EXTRA_SERVER_ARGS}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for router to be up..."
    BARRIER_CMD="python3 $ATOM_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 3600"
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $BARRIER_CMD"; else eval "$BARRIER_CMD"; fi

    echo "Waiting until router closes..."
    WAIT_CMD="python3 $ATOM_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $WAIT_CMD"; else eval "$WAIT_CMD"; fi

    echo "Killing prefill server (rank ${NODE_RANK})"
    if [[ "$DRY_RUN" -eq 0 ]]; then kill $prefill_pid; fi

else
    # ──────────────────────────────────────────────────────────────────────────
    # Decode nodes (kv_consumer)
    # ──────────────────────────────────────────────────────────────────────────
    RANK=$((NODE_RANK - NODE_OFFSET))
    echo "${host_name}:${host_ip} is Decode Node (rank ${RANK})"

    DECODE_CMD="python3 -m atom.entrypoints.openai_server \
        --model ${MODEL_DIR}/${MODEL_NAME} \
        --host 0.0.0.0 --server-port ${DECODE_PORT} \
        --trust-remote-code \
        -tp ${DECODE_TP_SIZE} \
        --enable-dp-attention \
        --kv_cache_dtype ${KV_CACHE_DTYPE} \
        --block-size ${BLOCK_SIZE} \
        --gpu-memory-utilization ${MEM_FRACTION} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --kv-transfer-config '{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}' \
        --cudagraph-capture-sizes '[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]' \
        ${EXTRA_SERVER_ARGS}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        eval "$DECODE_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log &
        set +x
        decode_pid=$!
    fi

    echo "Waiting for router to be up..."
    BARRIER_CMD="python3 $ATOM_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 3600"
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $BARRIER_CMD"; else eval "$BARRIER_CMD"; fi

    echo "Waiting until router closes..."
    WAIT_CMD="python3 $ATOM_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $WAIT_CMD"; else eval "$WAIT_CMD"; fi

    echo "Killing decode server (rank ${RANK})"
    if [[ "$DRY_RUN" -eq 0 ]]; then kill $decode_pid; fi
fi

echo "Script completed successfully"
exit 0
