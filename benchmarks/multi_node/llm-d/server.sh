#!/usr/bin/env bash
#
# Per-node entrypoint for the llm-d-vllm wide-EP P/D disagg benchmark.
# NODE_RANK is set by srun (= $SLURM_PROCID) in job.slurm.
#
# Roles:
#   Rank 0                         -> prefill leader (DP rank 0)
#   Ranks 1 .. PREFILL_NODES-1     -> prefill workers
#   Rank PREFILL_NODES             -> decode leader (DP rank 0) + pd-sidecar
#                                     + EPP + Envoy + benchmark client
#                                     (the coordinator, like AMD's decode-0)
#   Ranks PREFILL_NODES+1 ..       -> decode workers
#
# Each "instance" (prefill or decode) is a single vLLM engine spanning
# PREFILL_NODES (or DECODE_NODES) nodes via --data-parallel-hybrid-lb. The
# leader pod accepts external traffic; workers handle their local DP ranks.

set -euo pipefail

source /workspace/benchmarks/benchmark_lib.sh

NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
PREFILL_NODES="${PREFILL_NODES:-1}"
DECODE_NODES="${DECODE_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
VLLM_PORT=8200
SIDECAR_PORT=8000
ENVOY_PORT=8080
EPP_GRPC_PORT=9002
EPP_HEALTH_PORT=9003
EPP_METRICS_PORT=9090

MODEL="${MODEL_DIR}/${MODEL_NAME}"
HOST_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')

VLLM_LOG="/benchmark_logs/vllm_rank${NODE_RANK}.log"
SIDECAR_LOG="/benchmark_logs/sidecar_rank${NODE_RANK}.log"
EPP_LOG="/benchmark_logs/epp.log"
ENVOY_LOG="/benchmark_logs/envoy.log"

echo "=== rank=$NODE_RANK host=$HOST_IP model=$MODEL ==="

# ----------------------------------------------------------------
# Role assignment
# ----------------------------------------------------------------
if [[ "$NODE_RANK" -lt "$PREFILL_NODES" ]]; then
    ROLE="prefill"
    DP_SIZE="$PREFILL_DP_SIZE"
    DP_ADDR="$PREFILL_DP_ADDR"
    LWS_WORKER_INDEX="$NODE_RANK"
    LWS_GROUP_SIZE="$PREFILL_NODES"
elif [[ "$NODE_RANK" -lt $((PREFILL_NODES + DECODE_NODES)) ]]; then
    ROLE="decode"
    DP_SIZE="$DECODE_DP_SIZE"
    DP_ADDR="$DECODE_DP_ADDR"
    LWS_WORKER_INDEX=$((NODE_RANK - PREFILL_NODES))
    LWS_GROUP_SIZE="$DECODE_NODES"
else
    echo "ERROR: NODE_RANK=$NODE_RANK out of range" >&2
    exit 1
fi

DP_SIZE_LOCAL="$GPUS_PER_NODE"
START_RANK=$((LWS_WORKER_INDEX * DP_SIZE_LOCAL))
TP_SIZE=1

echo "ROLE=$ROLE DP_SIZE=$DP_SIZE DP_ADDR=$DP_ADDR LWS_WORKER_INDEX=$LWS_WORKER_INDEX START_RANK=$START_RANK"

# ----------------------------------------------------------------
# Read role-specific extra-args and env from the recipe file.
# ----------------------------------------------------------------
ROLE_EXTRA_ARGS=""
if [[ -n "${CONFIG_FILE:-}" ]]; then
    RECIPE_PATH="/etc/llmd-recipes/${CONFIG_FILE}"
    if [[ -f "$RECIPE_PATH" ]]; then
        echo "Loading $ROLE recipe from $RECIPE_PATH"
        eval "$(python3 - <<PY
import yaml
recipe = yaml.safe_load(open('${RECIPE_PATH}'))
section = recipe.get('${ROLE}', {}) or {}
extra = (section.get('extra-args') or '').strip()
print(f'ROLE_EXTRA_ARGS={extra!r}')
for k, v in (section.get('env') or {}).items():
    print(f'export {k}={v!r}')
PY
)"
    else
        echo "WARNING: CONFIG_FILE=$CONFIG_FILE but $RECIPE_PATH not found; using defaults" >&2
    fi
fi

# ----------------------------------------------------------------
# Wide-EP / P/D env (from the llm-d wide-EP-lws guide manifests).
# ----------------------------------------------------------------
export NVIDIA_GDRCOPY=enabled
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=true
export NVSHMEM_SYMMETRIC_SIZE=16G
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export VLLM_SKIP_P2P_CHECK=1
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_NIXL_SIDE_CHANNEL_HOST="$HOST_IP"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# ----------------------------------------------------------------
# Start vLLM (every node, prefill or decode)
#
# Flags split into:
#   * COMMON_ARGS - always passed.
#   * MULTINODE_DP_ARGS - only when an instance spans more than one node
#     (LWS_GROUP_SIZE > 1, i.e. wide-EP topology). vLLM's
#     --data-parallel-hybrid-lb and the cross-process DP coordination
#     flags are wrong for the single-node-per-instance case where DP is
#     contained inside one engine process.
# ----------------------------------------------------------------
KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

COMMON_ARGS=(
    --port "$VLLM_PORT"
    --trust-remote-code
    --api-server-count 1
    --disable-access-log-for-endpoints=/health,/metrics
    --enable-expert-parallel
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --kv_transfer_config "$KV_TRANSFER_CONFIG"
    --moe-backend deep_gemm
)

if [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
    COMMON_ARGS+=(
        --data-parallel-hybrid-lb
        --data-parallel-size-local "$DP_SIZE_LOCAL"
        --data-parallel-address "$DP_ADDR"
        --data-parallel-rpc-port 5555
        --data-parallel-start-rank "$START_RANK"
    )
fi

echo "Starting vLLM ($ROLE) DP=$DP_SIZE local=$DP_SIZE_LOCAL start_rank=$START_RANK group_size=$LWS_GROUP_SIZE"
# shellcheck disable=SC2086
vllm serve "$MODEL" "${COMMON_ARGS[@]}" $ROLE_EXTRA_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Only the leader of each instance accepts external requests on $VLLM_PORT.
if [[ "$LWS_WORKER_INDEX" -eq 0 ]]; then
    wait_for_server_ready --port "$VLLM_PORT" --server-log "$VLLM_LOG" --server-pid "$VLLM_PID"
    echo "vLLM leader ready on rank $NODE_RANK"

    # ------------------------------------------------------------
    # Start pd-sidecar on each leader (prefill leader and decode leader).
    # The decode-side sidecar is what EPP routes to; the prefill-side
    # sidecar is the target the decode sidecar pulls KVs from.
    # ------------------------------------------------------------
    SIDECAR_CONNECTOR="nixlv2"
    SIDECAR_FLAGS=(--port="$SIDECAR_PORT" --vllm-port="$VLLM_PORT"
                   --kv-connector="$SIDECAR_CONNECTOR" --secure-proxy=false)
    if [[ "$ROLE" == "decode" ]]; then
        SIDECAR_FLAGS+=(--enable-prefiller-sampling)
    fi
    echo "Starting pd-sidecar ($ROLE leader): ${SIDECAR_FLAGS[*]}"
    pd-sidecar "${SIDECAR_FLAGS[@]}" > "$SIDECAR_LOG" 2>&1 &
    wait_for_server_ready --port "$SIDECAR_PORT" --server-log "$SIDECAR_LOG"
    echo "pd-sidecar ready on $HOST_IP:$SIDECAR_PORT"
fi

# ----------------------------------------------------------------
# Coordinator: decode leader runs EPP + Envoy + benchmark client.
# ----------------------------------------------------------------
if [[ "$ROLE" == "decode" && "$LWS_WORKER_INDEX" -eq 0 ]]; then

    # Write endpoints.yaml. See benchmarks/multi_node/llm-d/README.md for
    # the discovery contract.
    # NOTE: endpoint 'namespace' must match EPP's --pool-namespace below
    # (file-discovery filters endpoints by namespace; the schema default
    # 'default' would otherwise drop every entry).
    python3 - <<PY
import os, yaml
NS = 'inferencex'
endpoints = [
    {'name': 'prefill-0',
     'namespace': NS,
     'address': os.environ['PREFILL_LEADER_IP'],
     'port': '$SIDECAR_PORT',
     'labels': {'llm-d.ai/role': 'prefill'}},
    {'name': 'decode-0',
     'namespace': NS,
     'address': os.environ['DECODE_LEADER_IP'],
     'port': '$SIDECAR_PORT',
     'labels': {'llm-d.ai/role': 'decode'}},
]
yaml.safe_dump({'endpoints': endpoints}, open('/tmp/endpoints.yaml', 'w'))
print('endpoints.yaml:')
print(open('/tmp/endpoints.yaml').read())
PY

    # EPP config: recipe override, else the default mounted by job.slurm
    # at /etc/epp/config.yaml (sourced from benchmarks/llm-d/epp-config.yaml).
    if [[ -n "$CONFIG_FILE" && -f "/etc/llmd-recipes/$CONFIG_FILE" ]]; then
        EPP_CONFIG="/etc/llmd-recipes/$CONFIG_FILE"
    else
        EPP_CONFIG="/etc/epp/config.yaml"
    fi
    echo "EPP config: $EPP_CONFIG"

    epp \
        --pool-name=epp \
        --pool-namespace=inferencex \
        --config-file="$EPP_CONFIG" \
        --grpc-port="$EPP_GRPC_PORT" \
        --grpc-health-port="$EPP_HEALTH_PORT" \
        --metrics-port="$EPP_METRICS_PORT" \
        > "$EPP_LOG" 2>&1 &

    envoy -c /etc/envoy/envoy.yaml > "$ENVOY_LOG" 2>&1 &

    wait_for_server_ready --port "$ENVOY_PORT" --server-log "$ENVOY_LOG"

    # Wait for the prefill leader's sidecar before starting the bench.
    wait_for_server_ready --port "$SIDECAR_PORT" --host "$PREFILL_LEADER_IP"

    # Bench against Envoy. EPP routes to decode (and decode sidecar pulls
    # from prefill via NIXL).
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$ENVOY_PORT" \
        --backend openai \
        --input-len "$BENCH_INPUT_LEN" \
        --output-len "$BENCH_OUTPUT_LEN" \
        --random-range-ratio "$BENCH_RANDOM_RANGE_RATIO" \
        --num-prompts "$((BENCH_MAX_CONCURRENCY * BENCH_NUM_PROMPTS_MULTIPLIER))" \
        --max-concurrency "$BENCH_MAX_CONCURRENCY" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir "$BENCHMARK_LOGS_DIR/"

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        run_eval --framework lm-eval --port "$ENVOY_PORT"
        append_lm_eval_summary
    fi

    scancel "$SLURM_JOB_ID"
else
    # Workers (prefill workers, decode workers, prefill leader): just keep vLLM alive.
    wait
fi
