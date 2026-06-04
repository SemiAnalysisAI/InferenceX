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
# Multi-node DP / NIXL P/D env: needed in any topology.
# ----------------------------------------------------------------
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export VLLM_SKIP_P2P_CHECK=1
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_NIXL_SIDE_CHANNEL_HOST="$HOST_IP"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# ----------------------------------------------------------------
# Wide-EP NVSHMEM / ibgda env (from the llm-d wide-EP-lws guide
# manifests). Gated on LWS_GROUP_SIZE > 1 - the simple 1P+1D recipe
# explicitly avoids DeepEP, NVSHMEM ibgda, and full-mesh RDMA, so
# leaving these set on a single-node-per-role topology is misleading
# and could trigger ibgda code paths it does not need.
# ----------------------------------------------------------------
if [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
    export NVIDIA_GDRCOPY=enabled
    export NVSHMEM_REMOTE_TRANSPORT=ibgda
    export NVSHMEM_IB_ENABLE_IBGDA=true
    export NVSHMEM_SYMMETRIC_SIZE=16G
    export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-eth0}
fi

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
    SIDECAR_PID=$!
    wait_for_server_ready --port "$SIDECAR_PORT" --server-log "$SIDECAR_LOG" --server-pid "$SIDECAR_PID"
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
    EPP_PID=$!

    # Wait for EPP to bind its gRPC port before starting Envoy. Envoy's
    # ext_proc filter dials 127.0.0.1:$EPP_GRPC_PORT - if Envoy comes up
    # first the early bench requests hit ext_proc connection errors.
    # gRPC has no plain HTTP /health, so probe the TCP listener directly.
    echo "Waiting for EPP on 127.0.0.1:$EPP_GRPC_PORT"
    EPP_WAIT_DEADLINE=$(( $(date +%s) + 60 ))
    until (echo > "/dev/tcp/127.0.0.1/$EPP_GRPC_PORT") 2>/dev/null; do
        if ! kill -0 "$EPP_PID" 2>/dev/null; then
            echo "ERROR: EPP died before binding $EPP_GRPC_PORT" >&2
            exit 1
        fi
        if [[ "$(date +%s)" -ge "$EPP_WAIT_DEADLINE" ]]; then
            echo "ERROR: EPP did not bind $EPP_GRPC_PORT within 60s" >&2
            exit 1
        fi
        sleep 1
    done
    echo "EPP listening on $EPP_GRPC_PORT"

    envoy -c /etc/envoy/envoy.yaml > "$ENVOY_LOG" 2>&1 &
    ENVOY_PID=$!

    wait_for_server_ready --port "$ENVOY_PORT" --server-log "$ENVOY_LOG" --server-pid "$ENVOY_PID"

    # Wait for the prefill leader's sidecar before starting the bench.
    # wait_for_server_ready can only probe localhost; the prefill leader
    # is on a different node, so poll directly with a deadline.
    echo "Waiting for prefill sidecar at $PREFILL_LEADER_IP:$SIDECAR_PORT/health"
    PREFILL_WAIT_DEADLINE=$(( $(date +%s) + 300 ))
    until curl --output /dev/null --silent --fail \
            "http://$PREFILL_LEADER_IP:$SIDECAR_PORT/health"; do
        if [[ "$(date +%s)" -ge "$PREFILL_WAIT_DEADLINE" ]]; then
            echo "ERROR: prefill sidecar did not become ready within 5 min" >&2
            exit 1
        fi
        sleep 5
    done
    echo "Prefill sidecar at $PREFILL_LEADER_IP:$SIDECAR_PORT is ready"

    # Sweep concurrency. BENCH_MAX_CONCURRENCY arrives from submit.sh as
    # an 'x'-delimited list (e.g. "2048x1024x512"); the runner / sweep
    # configs expect one bench run per level. Same shape as
    # benchmarks/multi_node/amd_utils/bench.sh.
    IFS='x' read -r -a CONCURRENCIES <<< "$BENCH_MAX_CONCURRENCY"
    for max_concurrency in "${CONCURRENCIES[@]}"; do
        num_prompts=$(( max_concurrency * BENCH_NUM_PROMPTS_MULTIPLIER ))
        [[ "$num_prompts" -lt 16 ]] && num_prompts=16
        # Bench against Envoy. EPP routes to decode (and decode sidecar
        # pulls from prefill via NIXL).
        run_benchmark_serving \
            --model "$MODEL" \
            --port "$ENVOY_PORT" \
            --backend openai \
            --input-len "$BENCH_INPUT_LEN" \
            --output-len "$BENCH_OUTPUT_LEN" \
            --random-range-ratio "$BENCH_RANDOM_RANGE_RATIO" \
            --num-prompts "$num_prompts" \
            --max-concurrency "$max_concurrency" \
            --result-filename "${RESULT_FILENAME}_c${max_concurrency}" \
            --result-dir "$BENCHMARK_LOGS_DIR/"
    done

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        run_eval --framework lm-eval --port "$ENVOY_PORT"
        append_lm_eval_summary
    fi

    # Signal job.slurm (running outside the container, where SLURM
    # client tools are available) to scancel the allocation. The image
    # does not bundle scancel, so calling it here would just trip
    # set -e. Workers end server.sh in `wait`; without this signal
    # they would hold the job until TIME_LIMIT.
    touch "$BENCHMARK_LOGS_DIR/.bench_done.$SLURM_JOB_ID"
else
    # Workers (prefill workers, decode workers, prefill leader): just keep vLLM alive.
    wait
fi
