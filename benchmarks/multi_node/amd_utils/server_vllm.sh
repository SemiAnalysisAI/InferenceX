#!/bin/bash
# vLLM Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================
#
# Node role assignment (by NODE_RANK):
#   0           -> Proxy/Router + first Prefill node  (kv_producer)
#   1..xP-1     -> Additional Prefill nodes            (kv_producer)
#   xP..xP+yD-1 -> Decode nodes                        (kv_consumer)
#
# Total nodes = xP + yD (router co-located with first prefill, like SGLang).

# =============================================================================
# Dependency Setup (idempotent; required when using base vLLM image)
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/setup_deps.sh"

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

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

DRY_RUN="${DRY_RUN:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-$GPUS_PER_NODE}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-$GPUS_PER_NODE}"

ROUTER_PORT="${ROUTER_PORT:-30000}"
SERVER_PORT="${SERVER_PORT:-2584}"
ENGINE_ID="${ENGINE_ID:-${MODEL_NAME}-pd-run}"
VLLM_SERVER_READY_TIMEOUT="${VLLM_SERVER_READY_TIMEOUT:-800}"

# Prefer MODEL_PATH from job.slurm (handles HF cache snapshot resolution)
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_NAME}}"
if [[ ! -f "${MODEL_PATH}/config.json" && -d "${MODEL_PATH}/snapshots" ]]; then
    snapshot_path="$(find "${MODEL_PATH}/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | head -1)"
    if [[ -n "$snapshot_path" && -f "${snapshot_path}/config.json" ]]; then
        echo "[model] Resolved HF cache root ${MODEL_PATH} -> ${snapshot_path}"
        MODEL_PATH="$snapshot_path"
    fi
fi

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
source $WS_PATH/env.sh

host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
# RDMA IP for Nixl KV transfer (prefer 192.168.x.x subnet if available)
rdma_ip=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.' | head -1)
rdma_ip="${rdma_ip:-$host_ip}"
host_name=$(hostname)

echo "[INFO] Management IP (barriers/proxy): $host_ip"
echo "[INFO] RDMA IP (Nixl KV transfer): $rdma_ip"

stage_run_logs() {
    if [[ "${DRY_RUN:-0}" == "1" || -z "${SLURM_JOB_ID:-}" ]]; then
        return 0
    fi
    local src="/run_logs/slurm_job-${SLURM_JOB_ID}"
    local dst="${BENCHMARK_LOGS_DIR:-/run_logs}/logs/slurm_job-${SLURM_JOB_ID}"
    if [[ -d "$src" ]]; then
        mkdir -p "$dst"
        cp -a "$src"/. "$dst"/ 2>/dev/null || true
    fi
}

trap stage_run_logs EXIT

# =============================================================================
# RDMA / Nixl Workarounds
# =============================================================================

setup_rdma_env() {
    # Pensando ionic (RoCEv2) point-to-point /31 route fix.
    # Each benic interface has a /31 to the TOR switch. Without explicit routes,
    # traffic to other nodes' RDMA IPs falls through to the management network.
    if [[ "$rdma_ip" =~ ^192\.168\.([0-9]+)\.([0-9]+)$ ]]; then
        local rdma_subnet="${BASH_REMATCH[1]}"
        local rdma_host="${BASH_REMATCH[2]}"
        local rdma_gw="192.168.${rdma_subnet}.$(( rdma_host | 1 ))"
        local rdma_iface
        rdma_iface=$(ip -o addr show | awk -v ip="$rdma_ip" '$4 ~ ip {print $2}' | head -1)
        if [[ -n "$rdma_iface" ]]; then
            ip route replace "192.168.${rdma_subnet}.0/24" via "$rdma_gw" dev "$rdma_iface" 2>/dev/null && \
                echo "[RDMA-ROUTE] Added 192.168.${rdma_subnet}.0/24 via $rdma_gw dev $rdma_iface" || \
                echo "[RDMA-ROUTE] Route add failed for 192.168.${rdma_subnet}.0/24"
        fi
    fi

    # Patch Nixl UCX backend: set ucx_error_handling_mode=none.
    # Required for ALL NIC types under high concurrency (C512+). Without this,
    # UCX's default UCP_ERR_HANDLING_MODE_PEER triggers transport-level error
    # recovery on ibv_post_send failures, preventing RIXL RDMA READ retries from
    # recovering gracefully. This causes the prefill KV cache to fill to 100%
    # and deadlock the pipeline. On ionic NICs this was already applied (rdmacm
    # incompatibility); on mlx5 NICs it was incorrectly skipped.
    local nixl_api
    nixl_api=$(python3 -c "import rixl._api; print(rixl._api.__file__)" 2>/dev/null)
    if [[ -n "$nixl_api" ]]; then
        if ! grep -q 'ucx_error_handling_mode' "$nixl_api"; then
            sed -i '/self\.create_backend(bknd, init)/i\                init["ucx_error_handling_mode"] = "none"' "$nixl_api"
            echo "[PATCH] Added ucx_error_handling_mode=none to $nixl_api (IBDEVICES=${IBDEVICES:-unset})"
        else
            echo "[PATCH] ucx_error_handling_mode already set in $nixl_api"
        fi
    fi
}

setup_rdma_env

if [[ -z "$UCX_NET_DEVICES" ]]; then
    echo "Error: UCX_NET_DEVICES is empty after env.sh detection" >&2
    exit 1
fi

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================
MODELS_YAML="${WS_PATH}/models_vllm.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME is not set"; exit 1
fi

eval "$(python3 -c "
import yaml, sys

with open('${MODELS_YAML}') as f:
    models = yaml.safe_load(f)

model_name = '${MODEL_NAME}'
if model_name not in models:
    print(f'echo \"ERROR: Model {model_name} not in models.yaml\"; exit 1')
    sys.exit(0)

m = models[model_name]

def bash_escape(s):
    \"\"\"Escape a value for safe embedding in a bash double-quoted assignment.\"\"\"
    return s.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"').replace('\$', '\\\\\$').replace('\`', '\\\\\`')

pf = bash_escape(m.get('prefill_flags', '--tensor-parallel-size 8'))
df = bash_escape(m.get('decode_flags', '--tensor-parallel-size 8'))
ev = bash_escape(m.get('env', ''))
dev = bash_escape(m.get('decode_env', ''))
pev = bash_escape(m.get('prefill_env', ''))
print(f'PREFILL_SERVER_CONFIG=\"{pf}\"')
print(f'DECODE_SERVER_CONFIG=\"{df}\"')
print(f'MODEL_ENVS=\"{ev}\"')
print(f'DECODE_MODEL_ENVS=\"{dev}\"')
print(f'PREFILL_MODEL_ENVS=\"{pev}\"')
")"

echo "Loaded model configuration for: $MODEL_NAME"

# Apply tensor-parallel size and EP/DP flags from submit pipeline.
if [[ -n "${PREFILL_TP_SIZE:-}" ]]; then
    if echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--tensor-parallel-size'; then
        PREFILL_SERVER_CONFIG=$(echo "$PREFILL_SERVER_CONFIG" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size ${PREFILL_TP_SIZE}/g")
    else
        PREFILL_SERVER_CONFIG+=" --tensor-parallel-size ${PREFILL_TP_SIZE}"
    fi
fi
if [[ -n "${DECODE_TP_SIZE:-}" ]]; then
    if echo "$DECODE_SERVER_CONFIG" | grep -q -- '--tensor-parallel-size'; then
        DECODE_SERVER_CONFIG=$(echo "$DECODE_SERVER_CONFIG" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size ${DECODE_TP_SIZE}/g")
    else
        DECODE_SERVER_CONFIG+=" --tensor-parallel-size ${DECODE_TP_SIZE}"
    fi
fi
if [[ "${PREFILL_ENABLE_EP:-false}" == "true" ]] && ! echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    PREFILL_SERVER_CONFIG+=" --enable-expert-parallel"
fi
if [[ "${PREFILL_ENABLE_DP:-false}" == "true" ]] && ! echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--enable-dp-attention'; then
    PREFILL_SERVER_CONFIG+=" --enable-dp-attention"
fi
if [[ "${DECODE_ENABLE_EP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    DECODE_SERVER_CONFIG+=" --enable-expert-parallel"
fi
if [[ "${DECODE_ENABLE_DP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enable-dp-attention'; then
    DECODE_SERVER_CONFIG+=" --enable-dp-attention"
fi

echo "PREFILL_SERVER_CONFIG (after TP/EP/DP): $PREFILL_SERVER_CONFIG"
echo "DECODE_SERVER_CONFIG (after TP/EP/DP): $DECODE_SERVER_CONFIG"

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 600 || exit 1

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

for ((i=0; i<xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="${IP_ARRAY[$i]} "
done

for ((i=xP; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="${IP_ARRAY[$i]} "
done

echo "Prefill node IPs: ${PREFILL_ARGS}"
echo "Decode  node IPs: ${DECODE_ARGS}"

# MoRI-IO proxy ZMQ registration port (must match vllm-router --vllm-discovery-address)
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}"

# =============================================================================
# KV transfer config
# =============================================================================
# The P/D transfer itself is always MoRIIO. When the matrix also asks for a CPU
# DRAM tier (kv-offloading: dram), the two are composed with MultiConnector
# instead of picking one, mirroring the shape the NVIDIA agentic recipes use for
# PD-plus-store (see benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/
# agentic/disagg-*-agentic.yaml: outer kv_role kv_both, each child carrying its
# own role).
#
# The tier is attached to the PREFILL side only. That is not a simplification:
# utils/matrix_logic/generate_sweep_configs.py::agentic_dram_offload_gb sizes
# TOTAL_CPU_DRAM_GB from the *prefill* worker's per-node GPU footprint precisely
# because "only prefill offloads KV to CPU DRAM today". Giving the decoder a tier
# too would spend a budget that was never computed for it.
#
# Escaping note: these strings are interpolated into PREFILL_CMD/DECODE_CMD and
# then `eval`-ed, so JSON is wrapped in single quotes. The JSON value itself must
# contain plain double quotes; pre-escaped \" reaches vLLM literally and fails
# pydantic JSON parsing.
build_kv_transfer_configs() {
    local moriio_prefill moriio_decode
    moriio_prefill="{\"kv_connector\": \"MoRIIOConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"proxy_ip\": \"${NODE0_ADDR}\", \"proxy_ping_port\": \"${PROXY_PING_PORT}\", \"http_port\": \"${SERVER_PORT}\", \"read_mode\": true}}"
    moriio_decode="{\"kv_connector\": \"MoRIIOConnector\", \"kv_role\": \"kv_consumer\", \"kv_connector_extra_config\": {\"proxy_ip\": \"${NODE0_ADDR}\", \"proxy_ping_port\": \"${PROXY_PING_PORT}\", \"http_port\": \"${SERVER_PORT}\", \"read_mode\": true}}"

    KV_TRANSFER_PREFILL="'${moriio_prefill}'"
    KV_TRANSFER_DECODE="'${moriio_decode}'"

    if [[ "${KV_OFFLOADING:-none}" == "dram" && "${KV_OFFLOAD_BACKEND:-}" == "vllm-simple" ]]; then
        if [[ -z "${TOTAL_CPU_DRAM_GB:-}" || "${TOTAL_CPU_DRAM_GB}" == "0" ]]; then
            echo "ERROR: kv-offloading=dram with backend vllm-simple but TOTAL_CPU_DRAM_GB is unset." >&2
            echo "       The agentic matrix derives it from dram-utilization; it must be consumed as given." >&2
            exit 1
        fi
        local per_rank
        per_rank=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / PREFILL_TP_SIZE ))
        local simple
        simple="{\"kv_connector\": \"SimpleCPUOffloadConnector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {\"cpu_bytes_to_use_per_rank\": ${per_rank}, \"lazy_offload\": false}}"
        KV_TRANSFER_PREFILL="'{\"kv_connector\": \"MultiConnector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {\"connectors\": [${moriio_prefill}, ${simple}]}}'"
        echo "[kv] prefill = MultiConnector[MoRIIO(kv_producer) + SimpleCPUOffload(${per_rank} B/rank x ${PREFILL_TP_SIZE})]"
        echo "[kv] decode  = MoRIIO(kv_consumer)"
    else
        echo "[kv] MoRIIO only (kv-offloading=${KV_OFFLOADING:-none} backend=${KV_OFFLOAD_BACKEND:-none})"
    fi
}

# =============================================================================
# Model-specific in-container patches
# =============================================================================
# Kimi-K3 needs two patch stacks on this image, and a disagg run needs both for the
# same reason the aggregate arm does:
#
#   1. apply_k3_container_patches.sh -- the model stack the single-node recipe
#      installs at kimik3_fp4_mi355x_mtp.sh:181. Not optional and not P/D-specific:
#      it carries the rocm_aiter_mla small-head multi-token MTP-verify paged-KV
#      path, triton_mla UNIFORM_BATCH cudagraph support for the non-causal DSpark
#      draft group, the KDA fused_recurrent state_indices coercion that PIECEWISE
#      cudagraph boot needs, Triton 3.7.0 + tabulate for Gluon MLA, and the aiter
#      #4521 asm verify kernels. Without it K3 does not survive warmup with a
#      DSpark draft, disagg or not.
#   2. apply_k3_moriio_patches.sh -- the MoRIIO side: hybrid MLA+KDA state
#      transfer, per-layer KV block lengths, the DSpark draft-layer skip.
#
# The two are disjoint file by file, so order does not matter; container patches
# run first to mirror the order the arm was validated in. Both are idempotent, so
# every prefill/decode rank can call this.
apply_model_patches() {
    if [[ "${MODEL_NAME}" == "Kimi-K3" ]]; then
        local here
        here="$(dirname "${BASH_SOURCE[0]}")"
        bash "${here}/../../single_node/agentic/apply_k3_container_patches.sh" \
            || { echo "ERROR: apply_k3_container_patches.sh failed" >&2; exit 1; }
        bash "${here}/apply_k3_moriio_patches.sh" \
            || { echo "ERROR: apply_k3_moriio_patches.sh failed" >&2; exit 1; }
    fi
}

# vLLM runtime environment (static vars moved to env.sh; these depend on per-node state)
setup_vllm_env() {
    export VLLM_NIXL_SIDE_CHANNEL_HOST=${rdma_ip}
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
    for env_pair in ${MODEL_ENVS}; do
        export "$env_pair"
    done
}

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill servers: ${PREFILL_ARGS}"
    echo "Decode  servers: ${DECODE_ARGS}"
    echo "================================================"

    setup_vllm_env

    for env_pair in ${PREFILL_MODEL_ENVS}; do
        export "$env_pair"
        echo "[PREFILL_ENV] $env_pair"
    done

    # Router is started as an external container by job.slurm (VLLM_ROUTER_IMAGE)
    echo "Using external vllm-router container (started by job.slurm on this node)"

    apply_model_patches
    build_kv_transfer_configs
    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config ${KV_TRANSFER_PREFILL} \
        ${PREFILL_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        PREFILL_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log"
        set -x
        eval "$PREFILL_CMD" > "$PREFILL_LOG_FILE" 2>&1 &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for all prefill and decode servers to be up . . ."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: skipping barrier (wait-for-all-ports)"
    else
        python3 $WS_PATH/sync.py barrier \
            --node-ips ${IPADDRS} \
            --node-ports $SERVER_PORT \
            --wait-for-all-ports \
            --timeout ${VLLM_SERVER_READY_TIMEOUT} || exit 1
    fi

    echo "Congratulations!!! All prefill and decode servers are up . . ."

    # Wait for proxy /health to confirm it is accepting requests
    HEALTH_BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-health \
        --health-endpoint /health \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $HEALTH_BARRIER_CMD"
    else
        eval "$HEALTH_BARRIER_CMD"
        [[ $? -eq 0 ]] || exit 1
        echo "MoRI-IO proxy is ready for benchmarking"
    fi

    echo "Ready for benchmarking on ${host_name}:${host_ip}"
    echo "Benchmarking on ${host_name}:${host_ip}"
    cd $WS_PATH

    export ROUTER_PORT=$ROUTER_PORT

    # Select the benchmark runner, mirroring server_sglang.sh:
    #   IS_AGENTIC=1/true  -> agentic trace replay (trace_replay.sh)
    #   IS_AGENTIC unset/0 -> fixed-seq-len throughput benchmark (bench.sh)
    # trace_replay.sh is already engine-aware (it resolves MODEL from MODEL_NAME
    # when ENGINE=vllm-disagg), so the agentic client needs no vLLM-specific
    # runner -- only the endpoint wiring below.
    if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
        # Point aiperf's server-metrics scrape at the per-worker Prometheus
        # endpoints. The router aiperf would otherwise infer from --url does not
        # expose Prometheus, so without this every server-side cache/KV field in
        # the result JSON comes out null. Unlike SGLang this is not gated on
        # ENABLE_METRICS: vLLM serves /metrics unconditionally.
        METRICS_URLS=()
        for _ip in ${PREFILL_ARGS} ${DECODE_ARGS}; do
            METRICS_URLS+=("http://${_ip}:${SERVER_PORT}/metrics")
        done
        if [[ "${#METRICS_URLS[@]}" -gt 0 ]]; then
            AIPERF_SERVER_METRICS_URLS=$(IFS=,; echo "${METRICS_URLS[*]}")
            export AIPERF_SERVER_METRICS_URLS
            echo "AIPERF_SERVER_METRICS_URLS=${AIPERF_SERVER_METRICS_URLS}"
        fi
        # SERVER_FLUSH_URLS_CSV is deliberately left unset: trace_replay.sh's
        # between-concurrency cache clear POSTs SGLang's /flush_cache, which vLLM
        # does not serve (its equivalent, /reset_prefix_cache, only exists under
        # VLLM_SERVER_DEV_MODE=1, which env.sh pins to 0). The clear is
        # best-effort there and an empty list makes it a no-op instead of a wall
        # of failed curls. It costs nothing today because the multinode agentic
        # matrix explodes to one concurrency per allocation
        # (MAX_MULTINODE_AGENTIC_CONCURRENCIES_PER_ALLOCATION=1), so no run
        # measures a second conc point behind a warm cache.

        # trace_replay.sh signature: model_path model_name concurrency_list log_path
        BENCH_CMD="bash $WS_PATH/trace_replay.sh \
            $MODEL_DIR $MODEL_NAME \"${BENCH_MAX_CONCURRENCY}\" /run_logs/slurm_job-${SLURM_JOB_ID}"
        echo "Benchmark runner: trace_replay.sh (agentic, KV_OFFLOADING=${KV_OFFLOADING:-none}, backend=${KV_OFFLOAD_BACKEND:-none}, CONC=${BENCH_MAX_CONCURRENCY}, DURATION=${DURATION:-unset})"
    else
        BENCH_CMD="bash $WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
            $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
            ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
            ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"
        echo "Benchmark runner: bench.sh (fixed-seq-len)"
    fi

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
                echo "DRY RUN: run_eval --framework lm-eval --port $ROUTER_PORT (conc=${EVAL_CONCURRENT_REQUESTS}, ctx=${EVAL_MAX_MODEL_LEN:-auto})"
            else
                run_eval --framework lm-eval --port "$ROUTER_PORT"
                eval_rc=$?

                if [[ $eval_rc -ne 0 ]]; then
                    echo "ERROR: run_eval exited rc=$eval_rc; skipping metadata write and eval artifact staging" >&2
                    EVAL_FAILED=1
                else
                    export TP="${PREFILL_TP_SIZE}"
                    export CONC="${EVAL_CONCURRENT_REQUESTS}"
                    export EP_SIZE=1
                    [[ "${PREFILL_ENABLE_EP}" == "true" ]] && EP_SIZE="${PREFILL_TP_SIZE}"
                    export PREFILL_TP="${PREFILL_TP_SIZE}"
                    export PREFILL_EP=1
                    [[ "${PREFILL_ENABLE_EP}" == "true" ]] && PREFILL_EP="${PREFILL_TP_SIZE}"
                    export PREFILL_NUM_WORKERS="${xP}"
                    export DECODE_TP="${DECODE_TP_SIZE}"
                    export DECODE_EP=1
                    [[ "${DECODE_ENABLE_EP}" == "true" ]] && DECODE_EP="${DECODE_TP_SIZE}"
                    export DECODE_NUM_WORKERS="${yD}"
                    export DP_ATTENTION="${PREFILL_ENABLE_DP}"
                    export PREFILL_DP_ATTENTION="${PREFILL_ENABLE_DP}"
                    export DECODE_DP_ATTENTION="${DECODE_ENABLE_DP}"
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

    # Copy benchmark/eval results to BENCHMARK_LOGS_DIR (mounted from host)
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing the prefill server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        [[ -n "${prefill_pid:-}" ]] && kill $prefill_pid 2>/dev/null || true
        sleep 2
        pkill -f "vllm serve" 2>/dev/null || true
    fi

    if [[ "${EVAL_FAILED:-0}" -eq 1 ]]; then
        echo "ERROR: eval failed; exiting node-0 with rc=1"
        exit 1
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    echo "${host_name}:${host_ip} is Additional Prefill Node (Model: ${MODEL_NAME})"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"

    setup_vllm_env

    for env_pair in ${PREFILL_MODEL_ENVS}; do
        export "$env_pair"
        echo "[PREFILL_ENV] $env_pair"
    done

    apply_model_patches
    build_kv_transfer_configs
    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config ${KV_TRANSFER_PREFILL} \
        ${PREFILL_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        PREFILL_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log"
        set -x
        eval "$PREFILL_CMD" > "$PREFILL_LOG_FILE" 2>&1 &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD" || exit 1
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the prefill server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $prefill_pid 2>/dev/null || true

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"

    setup_vllm_env

    for env_pair in ${DECODE_MODEL_ENVS}; do
        export "$env_pair"
        echo "[DECODE_ENV] $env_pair"
    done

    apply_model_patches
    build_kv_transfer_configs
    SERVED_MODEL="${MODEL_NAME}"
    DECODE_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config ${KV_TRANSFER_DECODE} \
        ${DECODE_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        DECODE_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log"
        set -x
        eval "$DECODE_CMD" > "$DECODE_LOG_FILE" 2>&1 &
        set +x
        decode_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD" || exit 1
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the decode server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $decode_pid 2>/dev/null || true
fi

# echo "Killing the etcd server"
# kill $etcd_pid 2>/dev/null || true
# pkill -f etcd 2>/dev/null || true

echo "Script completed successfully"
exit 0
