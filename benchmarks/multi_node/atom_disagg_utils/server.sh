#!/bin/bash
# ATOM Disaggregated Server Launcher (multi-node enabling phase).
#
# Structurally mirrors benchmarks/multi_node/amd_utils/server.sh but launches
# atom.entrypoints.openai_server in prefill/decode roles instead of
# sglang.launch_server. The orchestration spine — barrier sync, per-rank role
# dispatch (proxy/prefill/decode), router fan-out, benchmark invocation,
# eval staging, log copy — is identical so that fixes/improvements on the
# SGLang side can be re-applied here mechanically.
#
# ENABLING-PHASE CAVEATS:
#   - ATOM's --disaggregation-mode CLI surface is still being upstreamed.
#     The flags emitted below (--disaggregation-mode prefill|decode,
#     --disaggregation-ib-device $IBDEVICES, --disaggregation-transfer-backend
#     mori) mirror the SGLang convention; revisit once the ATOM disagg PRs
#     land and the canonical flag names are confirmed.
#   - The proxy/router for ATOM is also under construction. As a placeholder
#     for 1P1D, we point the client benchmark directly at the prefill server
#     (no router) — this only works for 1P1D and must be replaced with a
#     proper PD router for >1 worker topologies.
#   - The config matrix in amd-master.yaml is therefore restricted to 1P1D
#     CONC=1 at the enabling stage. Multi-worker/router scenarios will be
#     opened up in follow-up PRs.

set -euo pipefail

# =============================================================================
# Environment configuration (set by job.slurm via Docker -e flags)
# =============================================================================
NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}" # Number of Prefill Workers
yD="${yD:-1}" # Number of Decode Workers

IPADDRS="${IPADDRS:-localhost}"
HEADNODE_PORT="${HEADNODE_PORT:-20000}"

PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-false}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-false}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-false}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-false}"

BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-0.8}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-1}"

DRY_RUN="${DRY_RUN:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# =============================================================================
# Dependencies and environment setup
# =============================================================================
source "$ATOM_WS_PATH/env.sh"

# Detect local IP without depending on iproute2 (atom-dev image is minimal).
# Order of preference: `ip` -> `hostname -I` -> python socket trick.
if command -v ip >/dev/null 2>&1; then
    host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
elif host_ip=$(hostname -I 2>/dev/null | awk '{print $1}') && [[ -n "$host_ip" ]]; then
    :
else
    host_ip=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('1.1.1.1', 1)); print(s.getsockname()[0])")
fi
host_name=$(hostname)

# =============================================================================
# Model-specific configuration from atom_disagg_utils/models.yaml
# =============================================================================
MODELS_YAML="${ATOM_WS_PATH}/models.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

eval "$(python3 -c "
import yaml, sys
config_path = '${MODELS_YAML}'
model_name = '${MODEL_NAME}'
with open(config_path) as f:
    models = yaml.safe_load(f)
if model_name not in models:
    print(f'echo \"ERROR: Model {model_name} not in models.yaml\"; exit 1')
    sys.exit(0)
m = models[model_name]
print(f'MODEL_BASE_FLAGS=\"{m.get(\"base_flags\", \"\")}\"')
print(f'MODEL_PREFILL_FLAGS=\"{m.get(\"prefill_flags\", \"\")}\"')
print(f'MODEL_DECODE_FLAGS=\"{m.get(\"decode_flags\", \"\")}\"')
")"

echo "Loaded model configuration for: $MODEL_NAME"

# =============================================================================
# Cluster topology (mirrors amd_utils/server.sh)
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
done

DECODE_HEADNODE_URLS=()
DECODE_ARGS=""
for i in $(seq 0 $((yD - 1))); do
    decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_HEADNODE_URLS[$i]="${IP_ARRAY[$decode_idx]}:${HEADNODE_PORT}"
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$decode_idx]}:8000"
done

echo "Prefill worker headnode list: ${PREFILL_HEADNODE_URLS[*]}"
echo "Decode  worker headnode list: ${DECODE_HEADNODE_URLS[*]}"

# =============================================================================
# ATOM command-line builder (uses atom's actual disagg API)
# =============================================================================
# Per /app/ATOM/atom/kv_transfer/disaggregation/README.md inside the atom-dev
# image, disaggregated serving uses:
#
#   * a dedicated proxy process: python -m atom.kv_transfer.disaggregation.proxy
#     --port <PROXY_PORT>
#     - registers prefill (kv_producer) and decode (kv_consumer) servers
#     - listens on PROXY_PORT for client requests (default 10001)
#     - listens on _DEFAULT_DISCOVERY_PORT=36367 for prefill/decode register
#   * prefill/decode servers each get:
#     --kv-transfer-config '{"kv_role":"kv_producer"|"kv_consumer",
#                            "proxy_ip":"<NODE0_ADDR>",
#                            "proxy_ping_port":36367,
#                            "http_port":<server-port>}'
#   * MORI_SHMEM_MODE=ISOLATION is REQUIRED to keep MoRI (MoE all-to-all) and
#     MORI-IO (KV transfer) on separate symmetric heap pools.
#
# Atom does NOT use sglang's --disaggregation-mode / --disaggregation-ib-device
# / --disaggregation-transfer-backend flags. The IB device list is selected
# via MORI_RDMA_DEVICES env var; transfer backend is implicit ("moriio").
PROXY_PORT="${PROXY_PORT:-30000}"
PROXY_PING_PORT=36367
export MORI_SHMEM_MODE=ISOLATION
# Restrict MORI-IO to the cluster RDMA NICs the runner picked.
export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-${IBDEVICES}}"

build_atom_cmd() {
    local kv_role="$1"     # "kv_producer" or "kv_consumer"
    local tp_size="$2"
    local role_flags="$3"  # MODEL_PREFILL_FLAGS or MODEL_DECODE_FLAGS

    local enable_ep_flag=""
    if [[ "$kv_role" == "kv_producer" && "$PREFILL_ENABLE_EP" == "true" ]] \
        || [[ "$kv_role" == "kv_consumer" && "$DECODE_ENABLE_EP" == "true" ]]; then
        enable_ep_flag="--enable-expert-parallel"
    fi
    local enable_dp_flag=""
    if [[ "$kv_role" == "kv_producer" && "$PREFILL_ENABLE_DP" == "true" ]] \
        || [[ "$kv_role" == "kv_consumer" && "$DECODE_ENABLE_DP" == "true" ]]; then
        enable_dp_flag="--enable-dp-attention"
    fi

    # PR #812 wires V4 PD-disagg on the mooncake connector path. The moriio
    # connector in this image is pre-#812 and trips UnboundLocalError on
    # meta_list for V4 (loop body never runs because V4 doesn't populate
    # kv_cache.k_cache/v_cache in the standard schema). Override the
    # default connector ('moriio') to 'mooncake' via the JSON config.
    local kv_connector="${KV_CONNECTOR:-mooncake}"
    local kv_transfer_config="{\"kv_role\":\"${kv_role}\",\"kv_connector\":\"${kv_connector}\",\"proxy_ip\":\"${NODE0_ADDR}\",\"proxy_ping_port\":${PROXY_PING_PORT},\"http_port\":8000}"

    echo "python3 -m atom.entrypoints.openai_server \
        --model ${MODEL_DIR}/${MODEL_NAME} \
        --host 0.0.0.0 \
        --server-port 8000 \
        --block-size 16 \
        -tp ${tp_size} \
        ${enable_ep_flag} \
        ${enable_dp_flag} \
        ${MODEL_BASE_FLAGS} \
        ${role_flags} \
        --kv-transfer-config '${kv_transfer_config}'"
}

PREFILL_CMD_BASE=$(build_atom_cmd "kv_producer" "$PREFILL_TP_SIZE" "$MODEL_PREFILL_FLAGS")
DECODE_CMD_BASE=$(build_atom_cmd "kv_consumer" "$DECODE_TP_SIZE" "$MODEL_DECODE_FLAGS")

# =============================================================================
# Container synchronization barrier
# =============================================================================
# Bind sync.py's local listener to 0.0.0.0 rather than $host_ip. Inside
# the docker container, `hostname -I` may return docker-bridge IPs (e.g.
# 10.101.x.x) that do NOT match the IPs in $IPADDRS (which are the
# host-side default-route source IPs, e.g. 10.24.112.x). Binding 0.0.0.0
# accepts on every interface so peers can connect on whichever IP they
# resolved $IPADDRS to. Mirrors how sglang_router behaves on this cluster.
echo "Waiting at the container creation barrier on $host_name"
python3 "$ATOM_WS_PATH/sync.py" barrier \
    --local-ip 0.0.0.0 \
    --local-port 5000 \
    --enable-port \
    --node-ips "${IPADDRS}" \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300

# =============================================================================
# Per-rank role dispatch
# =============================================================================
# =============================================================================
# Install atom KV-transfer dependencies on EVERY rank.
# rocm/atom-dev:nightly_202605171131-mooncake_build ships mooncake's
# Python engine but is missing msgpack/quart/msgspec — atom's PD-disagg
# proxy AND both connector modules (moriio_connector.py, mooncake_connector.py)
# import msgpack at module load time. Without this install, every rank
# crashes during register_kv_caches with ModuleNotFoundError.
# Adds ~5s per rank to cold start; install lives in per-container
# site-packages and is discarded with --rm.
# =============================================================================
PROXY_DEPS=()
for _mod in msgpack quart msgspec; do
    if ! python3 -c "import ${_mod}" >/dev/null 2>&1; then
        PROXY_DEPS+=("${_mod}")
    fi
done
if [[ "${#PROXY_DEPS[@]}" -gt 0 ]]; then
    echo "[deps] rank ${NODE_RANK} installing missing: ${PROXY_DEPS[*]} ..."
    pip install --quiet --no-cache-dir "${PROXY_DEPS[@]}" >/dev/null 2>&1 \
        || pip install --quiet "${PROXY_DEPS[@]}"
fi

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME:-'Not specified'}"
    echo "Prefill   : TP=${PREFILL_TP_SIZE} EP=${PREFILL_ENABLE_EP} DP=${PREFILL_ENABLE_DP}"
    echo "Decode    : TP=${DECODE_TP_SIZE} EP=${DECODE_ENABLE_EP} DP=${DECODE_ENABLE_DP}"
    echo "Prefill servers (${PREFILL_NODES_PER_WORKER} nodes each): ${PREFILL_ARGS}"
    echo "Decode  servers (${DECODE_NODES_PER_WORKER} nodes each): ${DECODE_ARGS}"
    echo "================================================"

    # --------------------------------------------------------------------
    # Start the atom PD-disaggregation proxy on rank 0 first. Prefill and
    # decode servers will register with it (on PROXY_PING_PORT=36367) at
    # bring-up. Client traffic goes to PROXY_PORT (default 30000 to keep
    # symmetry with the sglang-disagg track).
    # --------------------------------------------------------------------
    PROXY_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/proxy_${host_name}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: python3 -m atom.kv_transfer.disaggregation.proxy --port ${PROXY_PORT}"
    else
        set -x
        python3 -m atom.kv_transfer.disaggregation.proxy --port "${PROXY_PORT}" \
            > "$PROXY_LOG_FILE" 2>&1 &
        set +x
        proxy_pid=$!
        echo "Started atom proxy pid=${proxy_pid} on port ${PROXY_PORT} (ping ${PROXY_PING_PORT})"
    fi

    # Wait briefly for the proxy ping port to come up before launching
    # prefill/decode servers (they connect during startup).
    eval "python3 $ATOM_WS_PATH/sync.py barrier --node-ips ${NODE0_ADDR} --node-ports ${PROXY_PING_PORT} --wait-for-all-ports --timeout 120" || {
        echo "ERROR: atom proxy ping-port ${PROXY_PING_PORT} did not come up; tail of proxy log:" >&2
        tail -n 50 "$PROXY_LOG_FILE" 2>&1 >&2 || true
        exit 1
    }
    echo "Proxy ping-port ${PROXY_PING_PORT} is ready."

    # --------------------------------------------------------------------
    # Launch the rank-0 prefill server (kv_producer).
    # --------------------------------------------------------------------
    PREFILL_CMD="$PREFILL_CMD_BASE"
    if [[ "$PREFILL_NODES_PER_WORKER" -gt 1 ]]; then
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank 0"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" 2>&1 | tee "/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log" &
        set +x
        prefill0_pid=$!
    fi

    # --------------------------------------------------------------------
    # Wait for prefill+decode HTTP ports (8000) on every node, then for
    # the proxy client port to come up. With atom's proxy the client port
    # is opened only after the proxy receives at least one prefill+decode
    # registration, so we want to barrier on it as a readiness signal.
    # --------------------------------------------------------------------
    echo "Waiting for all prefill and decode servers to be ready (port 8000)..."
    eval "python3 $ATOM_WS_PATH/sync.py barrier --node-ips ${IPADDRS} --node-ports 8000 --wait-for-all-ports --timeout 1800"
    echo "Waiting for proxy client port ${PROXY_PORT} to accept connections..."
    eval "python3 $ATOM_WS_PATH/sync.py barrier --node-ips ${NODE0_ADDR} --node-ports ${PROXY_PORT} --wait-for-all-ports --timeout 600"
    echo "Atom PD-disagg topology is ready for benchmarking."

    echo "Benchmarking on ${host_name}:${host_ip}"
    cd "$ATOM_WS_PATH"

    BENCH_CMD="bash $ATOM_WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} ${BENCH_MAX_CONCURRENCY} ${BENCH_REQUEST_RATE} \
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

    # Copy benchmark results to BENCHMARK_LOGS_DIR (mounted from host).
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r "/run_logs/slurm_job-${SLURM_JOB_ID}" "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing proxy and prefill server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill "$proxy_pid" 2>/dev/null || true
        kill "$prefill0_pid" 2>/dev/null || true
    fi

elif [[ "$NODE_RANK" -gt 0 && "$NODE_RANK" -lt "$NODE_OFFSET" ]]; then
    # Additional prefill workers (only fires when xP > 1; not used by the
    # current 1P2D enabling-phase matrix entry, kept for forward compat).
    echo "${host_name}:${host_ip} is Prefill worker rank $NODE_RANK"
    eval "python3 $ATOM_WS_PATH/sync.py barrier --node-ips ${NODE0_ADDR} --node-ports ${PROXY_PING_PORT} --wait-for-all-ports --timeout 600"
    PREFILL_CMD="$PREFILL_CMD_BASE"
    if [[ "$PREFILL_NODES_PER_WORKER" -gt 1 ]]; then
        rank=$((NODE_RANK % PREFILL_NODES_PER_WORKER))
        prefill_idx=$((NODE_RANK / PREFILL_NODES_PER_WORKER))
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[$prefill_idx]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank $rank"
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" 2>&1 | tee "/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log" &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting until proxy client port ${PROXY_PORT} closes..."
    eval "python3 $ATOM_WS_PATH/sync.py wait --remote-ip ${NODE0_ADDR} --remote-port ${PROXY_PORT}"
    if [[ "$DRY_RUN" -eq 0 ]]; then kill "$prefill_pid" 2>/dev/null || true; fi

else
    # Decode workers
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    echo "${host_name}:${host_ip} is Decode worker rank $RANK"
    # Wait for the proxy ping-port on rank 0 to be up before attempting
    # to register from this decode worker.
    eval "python3 $ATOM_WS_PATH/sync.py barrier --node-ips ${NODE0_ADDR} --node-ports ${PROXY_PING_PORT} --wait-for-all-ports --timeout 600"
    DECODE_CMD="$DECODE_CMD_BASE"
    if [[ "$DECODE_NODES_PER_WORKER" -gt 1 ]]; then
        rank=$((RANK % DECODE_NODES_PER_WORKER))
        decode_idx=$((RANK / DECODE_NODES_PER_WORKER))
        DECODE_CMD="$DECODE_CMD --dist-init-addr ${DECODE_HEADNODE_URLS[$decode_idx]} --nnodes ${DECODE_NODES_PER_WORKER} --node-rank $rank"
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        eval "$DECODE_CMD" 2>&1 | tee "/run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log" &
        set +x
        decode_pid=$!
    fi

    echo "Waiting until proxy client port ${PROXY_PORT} closes..."
    eval "python3 $ATOM_WS_PATH/sync.py wait --remote-ip ${NODE0_ADDR} --remote-port ${PROXY_PORT}"
    if [[ "$DRY_RUN" -eq 0 ]]; then kill "$decode_pid" 2>/dev/null || true; fi
fi

echo "Script completed successfully"
exit 0
