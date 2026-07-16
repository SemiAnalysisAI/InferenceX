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
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP}"

# MTP
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# ATOM server ports (different from SGLang which uses 8000 for all)
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"

# ATOM server tuning — defaults applied after YAML load (env var > YAML > shell default)
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

# Raise FD limit — lm-eval with high num_concurrent can exhaust the default 1024
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true
echo "ulimit -n (open files): $(ulimit -n)"

host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
if [[ -z "$host_ip" ]]; then
    host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
host_name=$(hostname)

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================
# Load model-specific config from YAML (single parse for all fields)
set -x
_yaml_tmp=$(mktemp)
python3 << PYEOF > "$_yaml_tmp"
import yaml
with open('${ATOM_WS_PATH}/models_atom.yaml') as f:
    m = yaml.safe_load(f).get('${MODEL_NAME}', {})
def sh(v): return v.replace("'", "'\\''")
print(f"MODEL_ENVS='{sh(m.get('env', ''))}'")
_tp_dp = m.get('tp_dp_flags', '')
print(f"PREFILL_MODEL_TP_DP_FLAGS='{sh(m.get('prefill_tp_dp_flags', _tp_dp))}'")
print(f"DECODE_MODEL_TP_DP_FLAGS='{sh(m.get('decode_tp_dp_flags', _tp_dp))}'")
_ep_dp = m.get('ep_dp_flags', '')
print(f"PREFILL_MODEL_EP_DP_FLAGS='{sh(m.get('prefill_ep_dp_flags', _ep_dp))}'")
print(f"DECODE_MODEL_EP_DP_FLAGS='{sh(m.get('decode_ep_dp_flags', _ep_dp))}'")
print(f"MODEL_TP_DP_ENV='{sh(m.get('tp_dp_env', ''))}'")
print(f"MODEL_EP_DP_ENV='{sh(m.get('ep_dp_env', ''))}'")
print(f"MODEL_MTP_FLAGS='{sh(m.get('mtp_flags', ''))}'")
print(f"MODEL_KV_ARG='{sh(m.get('kv_cache_flags', ''))}'")
print(f"_ONLINE_QUANT_CONFIG='{sh(m.get('online_quant_config', ''))}'")
print(f"_ONLINE_QUANT_DPA_CONFIG='{sh(m.get('online_quant_dpa_config', m.get('online_quant_config', '')))}'")
print(f"_YAML_BLOCK_SIZE='{sh(m.get('block_size', ''))}'")
print(f"_YAML_MEM_FRAC_STATIC='{sh(m.get('mem_frac_static', ''))}'")
print(f"_YAML_MAX_MODEL_LEN='{sh(m.get('max_model_len', ''))}'")
print(f"_YAML_MAX_NUM_SEQS='{sh(m.get('max_num_seqs', ''))}'")
print(f"_YAML_MAX_NUM_BATCHED_TOKENS='{sh(m.get('max_num_batched_tokens', ''))}'")
print(f"_YAML_SCHEDULER_DELAY_FACTOR='{sh(m.get('scheduler_delay_factor', ''))}'")
PYEOF
# shellcheck source=/dev/null
source "$_yaml_tmp"
rm -f "$_yaml_tmp"
unset _yaml_tmp

# Apply server-tuning: YAML > env var > shell default
# (job.slurm injects BLOCK_SIZE/MEM_FRAC_STATIC/MAX_NUM_SEQS with hardcoded
#  defaults into the Docker env, so env-first would always shadow the YAML.)
BLOCK_SIZE="${_YAML_BLOCK_SIZE:-${BLOCK_SIZE:-16}}"
MEM_FRAC_STATIC="${_YAML_MEM_FRAC_STATIC:-${MEM_FRAC_STATIC:-0.85}}"
MAX_MODEL_LEN="${_YAML_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-}}"
MAX_NUM_SEQS="${_YAML_MAX_NUM_SEQS:-${MAX_NUM_SEQS:-256}}"
MAX_NUM_BATCHED_TOKENS="${_YAML_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-}}"
SCHEDULER_DELAY_FACTOR="${_YAML_SCHEDULER_DELAY_FACTOR:-${SCHEDULER_DELAY_FACTOR:-}}"
unset _YAML_BLOCK_SIZE _YAML_MEM_FRAC_STATIC _YAML_MAX_MODEL_LEN _YAML_MAX_NUM_SEQS _YAML_MAX_NUM_BATCHED_TOKENS _YAML_SCHEDULER_DELAY_FACTOR

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

PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP}"




# Parallel args
PREFILL_PARALLEL_ARGS=(-tp "$PREFILL_TP_SIZE") #TP
ONLINE_QUANT_ARG=""
if [ "$PREFILL_ENABLE_DP" = "true" ]; then
    if [ "$PREFILL_ENABLE_EP" = "true" ]; then #EP+DPA
        PREFILL_PARALLEL_ARGS=(-tp "$PREFILL_TP_SIZE" ${PREFILL_MODEL_EP_DP_FLAGS})
        for _dp_env_pair in ${MODEL_EP_DP_ENV}; do export "$_dp_env_pair"; done
    else #TP+DPA
        PREFILL_PARALLEL_ARGS=(-tp "$PREFILL_TP_SIZE" ${PREFILL_MODEL_TP_DP_FLAGS})
        for _dp_env_pair in ${MODEL_TP_DP_ENV}; do export "$_dp_env_pair"; done
    fi
    if [[ -n "$_ONLINE_QUANT_DPA_CONFIG" ]]; then
        ONLINE_QUANT_ARG="--online_quant_config '${_ONLINE_QUANT_DPA_CONFIG}'"
    fi
else
    if [[ -n "$_ONLINE_QUANT_CONFIG" ]]; then
        ONLINE_QUANT_ARG="--online_quant_config '${_ONLINE_QUANT_CONFIG}'"
    fi
fi

DECODE_PARALLEL_ARGS=(-tp "$DECODE_TP_SIZE") #TP
if [ "$DECODE_ENABLE_DP" = "true" ]; then
    if [ "$DECODE_ENABLE_EP" = "true" ]; then #EP+DPA
        DECODE_PARALLEL_ARGS=(-tp "$DECODE_TP_SIZE" ${DECODE_MODEL_EP_DP_FLAGS})
        for _dp_env_pair in ${MODEL_EP_DP_ENV}; do export "$_dp_env_pair"; done
    else #TP+DPA
        DECODE_PARALLEL_ARGS=(-tp "$DECODE_TP_SIZE" ${DECODE_MODEL_TP_DP_FLAGS})
        for _dp_env_pair in ${MODEL_TP_DP_ENV}; do export "$_dp_env_pair"; done
    fi
fi
unset _dp_env_pair
unset _ONLINE_QUANT_CONFIG _ONLINE_QUANT_DPA_CONFIG

for _env_pair in ${MODEL_ENVS}; do
    export "$_env_pair"
done
unset _env_pair

# MTP args
SPEC_ARGS=()
if [[ -n "$MODEL_MTP_FLAGS" && "${DECODE_MTP_SIZE:-0}" -gt 0 ]]; then
    SPEC_ARGS=(${MODEL_MTP_FLAGS} "$DECODE_MTP_SIZE")
fi

# KV cache arg - full flag string from YAML
KV_CACHE_ARG="${MODEL_KV_ARG}"

# Optional model length / batched-token cap
MODEL_LEN_ARGS=""
if [[ -n "$MAX_MODEL_LEN" ]]; then
    MODEL_LEN_ARGS="${MODEL_LEN_ARGS} --max-model-len ${MAX_MODEL_LEN}"
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    MODEL_LEN_ARGS="${MODEL_LEN_ARGS} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}"
fi
if [[ -n "$SCHEDULER_DELAY_FACTOR" ]]; then
    MODEL_LEN_ARGS="${MODEL_LEN_ARGS} --scheduler-delay-factor ${SCHEDULER_DELAY_FACTOR}"
fi


cat <<INFO
=== Configuration ===
PREFILL  : ${PREFILL_IPS[*]} (TP=${PREFILL_TP_SIZE}, EP=${PREFILL_ENABLE_EP:-false}, DP=${PREFILL_ENABLE_DP:-false}, port=${PREFILL_PORT})
DECODE   : ${DECODE_IPS[*]}  (TP=${DECODE_TP_SIZE},  EP=${DECODE_ENABLE_EP:-false},  DP=${DECODE_ENABLE_DP:-false},  port=${DECODE_PORT})
ROUTER   : port=${ROUTER_PORT}
MODEL    : ${MODEL_NAME}
BACKEND  : atom (PD mooncake KV transfer)
MTP      : method=mtp num_speculative_tokens=${DECODE_MTP_SIZE}
xP/yD    : ${xP} / ${yD}
KV cache : ${KV_CACHE_ARG:-none} block_size=${BLOCK_SIZE} mem_frac=${MEM_FRAC_STATIC}
Model len: max_model_len=${MAX_MODEL_LEN:-unset} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-unset}
Prefill args : ${PREFILL_PARALLEL_ARGS[*]}
Decode  args : ${DECODE_PARALLEL_ARGS[*]}
Spec    args : ${SPEC_ARGS[*]}
Opt     args : ${ONLINE_QUANT_ARG}
=====================
INFO

set -x
echo "::group::Environment Variables"
env
echo "::endgroup::"

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
        ${PREFILL_PARALLEL_ARGS[*]} \
        ${SPEC_ARGS[*]} \
        ${KV_CACHE_ARG} \
        --block-size ${BLOCK_SIZE} \
        --gpu-memory-utilization ${MEM_FRAC_STATIC} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        ${MODEL_LEN_ARGS} \
        --no-enable_prefix_caching \
        ${ONLINE_QUANT_ARG} \
        --kv-transfer-config '{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}' \
        ${EXTRA_SERVER_ARGS}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill0_${host_name}.log &
        set +x
        prefill0_pid=$!
    fi

    # Wait for all prefill and decode servers to be ready
    WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-2500}"
    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Waiting for all servers to be up (timeout=${WAIT_SERVER_TIMEOUT}s)..."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: wait for prefill/decode /health endpoints"
    else
        _deadline=$(( $(date +%s) + WAIT_SERVER_TIMEOUT ))
        for _ip in "${PREFILL_IPS[@]}"; do
            echo "[wait] prefill http://${_ip}:${PREFILL_PORT}/health"
            while ! curl -sf --max-time 10 "http://${_ip}:${PREFILL_PORT}/health" >/dev/null 2>&1; do
                if [[ $(date +%s) -ge $_deadline ]]; then
                    echo "[wait][FAIL] prefill ${_ip}:${PREFILL_PORT} not ready after ${WAIT_SERVER_TIMEOUT}s" >&2
                    exit 1
                fi
                sleep 10
            done
            echo "[wait][OK] prefill ${_ip}:${PREFILL_PORT} ready"
        done
        for _ip in "${DECODE_IPS[@]}"; do
            echo "[wait] decode http://${_ip}:${DECODE_PORT}/health"
            while ! curl -sf --max-time 10 "http://${_ip}:${DECODE_PORT}/health" >/dev/null 2>&1; do
                if [[ $(date +%s) -ge $_deadline ]]; then
                    echo "[wait][FAIL] decode ${_ip}:${DECODE_PORT} not ready after ${WAIT_SERVER_TIMEOUT}s" >&2
                    exit 1
                fi
                sleep 10
            done
            echo "[wait][OK] decode ${_ip}:${DECODE_PORT} ready"
        done
    fi
    echo "[-------]" NODE $NODE_RANK "[--------]"
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
        WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"
        echo "[wait] router http://0.0.0.0:${ROUTER_PORT}/v1/models (timeout=${WAIT_ROUTER_TIMEOUT}s)"
        _router_deadline=$(( $(date +%s) + WAIT_ROUTER_TIMEOUT ))
        while ! curl -sf --max-time 10 "http://0.0.0.0:${ROUTER_PORT}/v1/models" >/dev/null 2>&1; do
            if [[ $(date +%s) -ge $_router_deadline ]]; then
                echo "[wait][FAIL] router ${ROUTER_PORT}/v1/models not ready after ${WAIT_ROUTER_TIMEOUT}s" >&2
                exit 1
            fi
            sleep 10
        done
        echo "[wait][OK] router /v1/models ready"

        echo "Router is ready for benchmarking"
    fi

    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Ready for benchmarking on ${host_name}:${host_ip}"

    cd $ATOM_WS_PATH

    export IS_MTP="false"
    if [[ -n "$MODEL_MTP_FLAGS" && "${DECODE_MTP_SIZE:-0}" -gt 0 ]]; then
        export IS_MTP="true"
    fi

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
                MODEL_NAME="${MODEL_DIR}/${MODEL_NAME}" run_eval --framework lm-eval --port ${ROUTER_PORT}
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

                    MODEL_NAME="${MODEL_DIR}/${MODEL_NAME}" append_lm_eval_summary

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

    echo "Waiting 60s before killing router and prefill server..."
    sleep 60

    echo "[-------]" NODE $NODE_RANK "[--------]"
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
        ${PREFILL_PARALLEL_ARGS[*]} \
        ${SPEC_ARGS[*]} \
        ${KV_CACHE_ARG} \
        --block-size ${BLOCK_SIZE} \
        --gpu-memory-utilization ${MEM_FRAC_STATIC} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        ${MODEL_LEN_ARGS} \
        --no-enable_prefix_caching \
        ${ONLINE_QUANT_ARG} \
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
        trap 'echo "Caught signal, killing prefill (pid=$prefill_pid)"; kill $prefill_pid 2>/dev/null; exit 0' SIGTERM SIGINT
    fi

    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Waiting for router to be up..."
    WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-2800}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: wait for router ${NODE0_ADDR}:${ROUTER_PORT}/health"
    else
        _router_deadline=$(( $(date +%s) + WAIT_ROUTER_TIMEOUT ))
        while ! curl -sf --max-time 10 "http://${NODE0_ADDR}:${ROUTER_PORT}/health" >/dev/null 2>&1; do
            if [[ $(date +%s) -ge $_router_deadline ]]; then
                echo "[wait][FAIL] router ${NODE0_ADDR}:${ROUTER_PORT} not ready after ${WAIT_ROUTER_TIMEOUT}s" >&2
                exit 1
            fi
            sleep 10
        done
        echo "[wait][OK] router ${NODE0_ADDR}:${ROUTER_PORT} ready"
    fi

    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Waiting until router closes..."
    trap 'echo "Caught signal, killing prefill (pid=$prefill_pid)"; kill $prefill_pid 2>/dev/null; exit 0' SIGTERM SIGINT
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: wait until router ${NODE0_ADDR}:${ROUTER_PORT} closes"
    else
        while curl -sf --max-time 10 "http://${NODE0_ADDR}:${ROUTER_PORT}/health" >/dev/null 2>&1; do
            sleep 10 &
            wait $!
        done
        echo "[wait] router ${NODE0_ADDR}:${ROUTER_PORT} closed"
    fi

    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Killing prefill server (rank ${NODE_RANK})"
    if [[ "$DRY_RUN" -eq 0 ]]; then kill $prefill_pid 2>/dev/null; fi

else
    # ──────────────────────────────────────────────────────────────────────────
    # Decode nodes (kv_consumer)
    # ──────────────────────────────────────────────────────────────────────────
    RANK=$((NODE_RANK - NODE_OFFSET))
    echo "${host_name}:${host_ip} is Decode Node (rank ${RANK})"

    _MAX_CONC=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
    CUDAGRAPH_SIZES='[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]'

    DECODE_MAX_NUM_SEQS="${_MAX_CONC}"

    DECODE_CMD="python3 -m atom.entrypoints.openai_server \
        --model ${MODEL_DIR}/${MODEL_NAME} \
        --host 0.0.0.0 --server-port ${DECODE_PORT} \
        --trust-remote-code \
        ${DECODE_PARALLEL_ARGS[*]} \
        ${SPEC_ARGS[*]} \
        ${KV_CACHE_ARG} \
        --block-size ${BLOCK_SIZE} \
        --gpu-memory-utilization ${MEM_FRAC_STATIC} \
        --max-num-seqs ${DECODE_MAX_NUM_SEQS} \
        ${MODEL_LEN_ARGS} \
        --no-enable_prefix_caching \
        ${ONLINE_QUANT_ARG} \
        --kv-transfer-config '{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${host_ip}\",\"handshake_port\":${HANDSHAKE_PORT}}' \
        --cudagraph-capture-sizes "${CUDAGRAPH_SIZES}" \
        ${EXTRA_SERVER_ARGS}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        eval "$DECODE_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log &
        set +x
        decode_pid=$!
        trap 'echo "Caught signal, killing decode (pid=$decode_pid)"; kill $decode_pid 2>/dev/null; exit 0' SIGTERM SIGINT
    fi

    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Waiting for router to be up..."
    WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-2800}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: wait for router ${NODE0_ADDR}:${ROUTER_PORT}/health"
    else
        _router_deadline=$(( $(date +%s) + WAIT_ROUTER_TIMEOUT ))
        while ! curl -sf --max-time 10 "http://${NODE0_ADDR}:${ROUTER_PORT}/health" >/dev/null 2>&1; do
            if [[ $(date +%s) -ge $_router_deadline ]]; then
                echo "[wait][FAIL] router ${NODE0_ADDR}:${ROUTER_PORT} not ready after ${WAIT_ROUTER_TIMEOUT}s" >&2
                exit 1
            fi
            sleep 10
        done
        echo "[wait][OK] router ${NODE0_ADDR}:${ROUTER_PORT} ready"
    fi

    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Waiting until router closes..."
    trap 'echo "Caught signal, killing decode (pid=$decode_pid)"; kill $decode_pid 2>/dev/null; exit 0' SIGTERM SIGINT
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: wait until router ${NODE0_ADDR}:${ROUTER_PORT} closes"
    else
        while curl -sf --max-time 10 "http://${NODE0_ADDR}:${ROUTER_PORT}/health" >/dev/null 2>&1; do
            sleep 10 &
            wait $!
        done
        echo "[wait] router ${NODE0_ADDR}:${ROUTER_PORT} closed"
    fi

    echo "[-------]" NODE $NODE_RANK "[--------]"
    echo "Killing decode server (rank ${RANK})"
    if [[ "$DRY_RUN" -eq 0 ]]; then kill $decode_pid 2>/dev/null; fi
fi

echo "Script completed successfully"
exit 0