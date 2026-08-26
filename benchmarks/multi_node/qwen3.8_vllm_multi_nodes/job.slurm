#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?This script must run inside a Slurm allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST must be set}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set}"
: "${BENCHMARK_LOGS_DIR:?BENCHMARK_LOGS_DIR must be set}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set}"
: "${SCENARIO_TYPE:?SCENARIO_TYPE must be set}"

RUNTIME_DIR="$GITHUB_WORKSPACE/benchmarks/multi_node/qwen3.8_vllm_multi_nodes"
# shellcheck source=benchmarks/multi_node/qwen3.8_vllm_multi_nodes/qwen3.8_env.sh
source "$RUNTIME_DIR/qwen3.8_env.sh"
JOB_LOG_DIR="$BENCHMARK_LOGS_DIR/logs/slurm_job-${SLURM_JOB_ID}"
SERVER_READY_TIMEOUT="${VLLM_ENGINE_READY_TIMEOUT_S:-10800}"

export MODEL_PATH
export JOB_LOG_DIR
export PORT=8000
export IS_MULTINODE=true
export SCENARIO_TYPE

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected exactly two assigned nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi

HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

run_on_node() {
    local node="$1"
    shift
    srun \
        --overlap \
        --nodes=1 \
        --ntasks=1 \
        --nodelist="$node" \
        "$@"
}

node_ip() {
    local node="$1"
    run_on_node "$node" /bin/bash -lc \
        "ip -o -4 addr show dev eno0 | awk '{print \$4}' | cut -d/ -f1 | head -n1"
}

HEAD_IP="$(node_ip "$HEAD_NODE")"
WORKER_IP="$(node_ip "$WORKER_NODE")"
if [[ -z "$HEAD_IP" || -z "$WORKER_IP" || "$HEAD_IP" == "$WORKER_IP" ]]; then
    echo "ERROR: invalid assigned-node IPs: head=$HEAD_NODE/$HEAD_IP worker=$WORKER_NODE/$WORKER_IP" >&2
    exit 1
fi

echo "Assigned Qwen3.8 Ray head: $HEAD_NODE ($HEAD_IP)"
echo "Assigned Qwen3.8 Ray worker: $WORKER_NODE ($WORKER_IP)"

manifest_for_node() {
    local node="$1"
    local manifest
    if ! manifest="$(run_on_node "$node" \
        python3 "$RUNTIME_DIR/snapshot_manifest.py" "$MODEL_PATH" 2>&1)"; then
        echo "ERROR: Qwen3.8 snapshot check failed on $node for $MODEL_PATH" >&2
        echo "$manifest" >&2
        echo "Run verify_model_staging.sh and see benchmarks/multi_node/qwen3.8_vllm_multi_nodes/README.md" >&2
        exit 1
    fi
    printf '%s' "$manifest"
}

HEAD_MANIFEST="$(manifest_for_node "$HEAD_NODE")"
WORKER_MANIFEST="$(manifest_for_node "$WORKER_NODE")"
echo "Head snapshot manifest: $HEAD_MANIFEST"
echo "Worker snapshot manifest: $WORKER_MANIFEST"
if [[ "$HEAD_MANIFEST" != "$WORKER_MANIFEST" ]]; then
    echo "ERROR: Qwen3.8 snapshots differ between $HEAD_NODE and $WORKER_NODE" >&2
    exit 1
fi

if [[ "${QWEN38_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "Qwen3.8 two-node Slurm, network, and snapshot preflight passed"
    exit 0
fi

mkdir -p "$JOB_LOG_DIR"

control() {
    local node="$1"
    local action="$2"
    local role="$3"
    local ip="$4"
    run_on_node "$node" \
        env \
            MODEL_PATH="$MODEL_PATH" \
            JOB_LOG_DIR="$JOB_LOG_DIR" \
        /bin/bash "$RUNTIME_DIR/node_control.sh" \
            "$action" "$role" "$ip" "$HEAD_IP"
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    control "$HEAD_NODE" archive-ray-logs head "$HEAD_IP"
    control "$WORKER_NODE" archive-ray-logs worker "$WORKER_IP"
    control "$HEAD_NODE" cleanup head "$HEAD_IP"
    control "$WORKER_NODE" cleanup worker "$WORKER_IP"
    sudo chown -R "$(id -u):$(id -g)" "$JOB_LOG_DIR" 2>/dev/null || true
    shopt -s nullglob
    result_files=("$GITHUB_WORKSPACE/${RESULT_FILENAME}"_conc*.json)
    if [[ "${#result_files[@]}" -gt 0 ]]; then
        sudo chown "$(id -u):$(id -g)" "${result_files[@]}" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

control "$HEAD_NODE" start-container head "$HEAD_IP" &
head_start_pid=$!
control "$WORKER_NODE" start-container worker "$WORKER_IP" &
worker_start_pid=$!
wait "$head_start_pid"
wait "$worker_start_pid"

control "$HEAD_NODE" install-ray head "$HEAD_IP" &
head_ray_install_pid=$!
control "$WORKER_NODE" install-ray worker "$WORKER_IP" &
worker_ray_install_pid=$!
wait "$head_ray_install_pid"
wait "$worker_ray_install_pid"

control "$HEAD_NODE" start-ray-head head "$HEAD_IP"
control "$WORKER_NODE" start-ray-worker worker "$WORKER_IP"
control "$HEAD_NODE" verify-ray head "$HEAD_IP"

control "$HEAD_NODE" start-vllm head "$HEAD_IP"

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
until curl --fail --silent --show-error "http://${HEAD_IP}:8000/health" >/dev/null; do
    if ! control "$HEAD_NODE" vllm-running head "$HEAD_IP"; then
        echo "ERROR: vLLM exited before becoming healthy" >&2
        if [[ -f "$JOB_LOG_DIR/vllm-server.log" ]]; then
            tail -200 "$JOB_LOG_DIR/vllm-server.log" >&2
        fi
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "ERROR: vLLM did not become healthy within ${SERVER_READY_TIMEOUT}s" >&2
        if [[ -f "$JOB_LOG_DIR/vllm-server.log" ]]; then
            tail -200 "$JOB_LOG_DIR/vllm-server.log" >&2
        fi
        exit 1
    fi
    sleep 15
done

echo "vLLM is healthy at http://${HEAD_IP}:8000"
control "$HEAD_NODE" run-client head "$HEAD_IP"
echo "Qwen3.8 ${SCENARIO_TYPE} benchmark completed"
