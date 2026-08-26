#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:?usage: node_control.sh ACTION ROLE NODE_IP HEAD_IP}"
ROLE="${2:?role is required}"
NODE_IP="${3:?node IP is required}"
HEAD_IP="${4:?head IP is required}"

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=benchmarks/multi_node/qwen3.8_vllm_multi_nodes/qwen3.8_env.sh
source "$RUNTIME_DIR/qwen3.8_env.sh"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:qwen38}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set}"
JOB_LOG_DIR="${JOB_LOG_DIR:?JOB_LOG_DIR must be set}"
CONTAINER_NAME="vllm-qwen-${ROLE}"

if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo -n docker)
else
    echo "ERROR: cannot access Docker directly or through passwordless sudo" >&2
    exit 1
fi

container_exec() {
    "${DOCKER[@]}" exec "$CONTAINER_NAME" "$@"
}

socket_env=(
    -e GLOO_SOCKET_IFNAME=eno0
    -e NCCL_SOCKET_IFNAME=eno0
    -e RCCL_SOCKET_IFNAME=eno0
    -e VLLM_HOST_IP="$NODE_IP"
)

case "$ACTION" in
    start-container)
        "${DOCKER[@]}" pull "$IMAGE"
        "${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        mkdir -p "$JOB_LOG_DIR"
        "${DOCKER[@]}" run -d \
            --name "$CONTAINER_NAME" \
            --network host \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            --ipc=host \
            --security-opt seccomp=unconfined \
            --cap-add=SYS_PTRACE \
            --ulimit memlock=-1 \
            -v /mnt/hf_hub_cache:/mnt/hf_hub_cache \
            -v /models:/models:ro \
            -v "$GITHUB_WORKSPACE:/workspace" \
            -v "$JOB_LOG_DIR:/run_logs" \
            -e MODEL \
            -e MODEL_PREFIX \
            -e MODEL_PATH \
            -e FRAMEWORK \
            -e PRECISION \
            -e IMAGE \
            -e RUNNER_TYPE \
            -e CONC_LIST \
            -e CONC \
            -e ISL \
            -e OSL \
            -e RANDOM_RANGE_RATIO \
            -e RESULT_FILENAME \
            -e DURATION \
            -e RUN_EVAL \
            -e EVAL_ONLY \
            -e EVAL_CONC \
            -e EVAL_LIMIT \
            -e KV_OFFLOADING \
            -e KV_OFFLOAD_BACKEND \
            -e KV_OFFLOAD_BACKEND_METADATA \
            -e TOTAL_CPU_DRAM_GB \
            -e PREFILL_NUM_WORKERS \
            -e PREFILL_TP \
            -e PREFILL_PP_SIZE \
            -e PREFILL_EP \
            -e PREFILL_DP_ATTN \
            -e DECODE_NUM_WORKERS \
            -e DECODE_TP \
            -e DECODE_PP_SIZE \
            -e DECODE_EP \
            -e DECODE_DP_ATTN \
            -e IS_MULTINODE \
            -e SCENARIO_TYPE \
            -e AIPERF_EXPERIMENTAL_FAST \
            -e AIPERF_DATASET_MMAP_CACHE_DIR=/mnt/hf_hub_cache/aiperf_mmap_cache \
            -e HF_HOME=/mnt/hf_hub_cache \
            -e HF_HUB_CACHE=/mnt/hf_hub_cache \
            -e HF_TOKEN \
            -e PYTHONDONTWRITEBYTECODE=1 \
            -e PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache \
            "${socket_env[@]}" \
            --entrypoint /bin/bash \
            "$IMAGE" \
            -lc 'trap : TERM INT; sleep infinity & wait'
        ;;
    install-ray)
        container_exec python3 -m pip install -U 'ray[default]'
        ;;
    start-ray-head)
        "${DOCKER[@]}" exec "${socket_env[@]}" "$CONTAINER_NAME" \
            ray start --head \
                --include-dashboard=false \
                --node-ip-address="$HEAD_IP" \
                --port=6380 \
                --num-gpus=8 \
                --disable-usage-stats
        ;;
    start-ray-worker)
        "${DOCKER[@]}" exec "${socket_env[@]}" "$CONTAINER_NAME" \
            ray start \
                --address="$HEAD_IP:6380" \
                --node-ip-address="$NODE_IP" \
                --num-gpus=8 \
                --disable-usage-stats
        ;;
    verify-ray)
        "${DOCKER[@]}" exec \
            -e RAY_ADDRESS="$HEAD_IP:6380" \
            "$CONTAINER_NAME" \
            python3 -c '
import ray

ray.init(address="auto")
alive = [node for node in ray.nodes() if node.get("Alive")]
gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"Ray cluster: active_nodes={len(alive)} gpus={gpus}")
if len(alive) != 2 or gpus != 16:
    raise SystemExit("expected exactly two active Ray nodes and 16 GPUs")
'
        ;;
    start-vllm)
        "${DOCKER[@]}" exec -d \
            -e RAY_ADDRESS="$HEAD_IP:6380" \
            -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            -e GLOO_SOCKET_IFNAME=eno0 \
            -e NCCL_SOCKET_IFNAME=eno0 \
            -e RCCL_SOCKET_IFNAME=eno0 \
            -e VLLM_HOST_IP="$HEAD_IP" \
            -e VLLM_ROCM_USE_AITER=1 \
            -e SAFETENSORS_FAST_GPU=1 \
            "$CONTAINER_NAME" \
            /bin/bash -lc "
                exec vllm serve '$MODEL_PATH' \
                    --host 0.0.0.0 \
                    --port 8000 \
                    --trust-remote-code \
                    --tensor-parallel-size 8 \
                    --pipeline-parallel-size 2 \
                    --distributed-executor-backend ray \
                    --max-model-len auto \
                    --gpu-memory-utilization 0.8 \
                    --reasoning-parser qwen3 \
                    --served-model-name 'Qwen/Qwen3.8-2.4T-A95B-FP8' \
                    --language-model-only \
                    --no-enable-prefix-caching \
                    --mamba-cache-mode none \
                    --enable-auto-tool-choice \
                    --tool-call-parser qwen3_coder \
                    > /run_logs/vllm-server.log 2>&1
            "
        ;;
    vllm-running)
        container_exec pgrep -f "vllm serve.*${MODEL_PATH}" >/dev/null
        ;;
    run-client)
        container_exec \
            /bin/bash \
            /workspace/benchmarks/multi_node/qwen3.8_vllm_multi_nodes/client.sh \
            "$SCENARIO_TYPE"
        ;;
    archive-ray-logs)
        container_exec /bin/bash -lc \
            "if [ -d /tmp/ray/session_latest/logs ]; then tar czf /run_logs/ray-${ROLE}-logs.tar.gz -C /tmp/ray/session_latest logs; fi"
        ;;
    cleanup)
        container_exec ray stop --force >/dev/null 2>&1 || true
        "${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        ;;
    *)
        echo "ERROR: unknown action: $ACTION" >&2
        exit 2
        ;;
esac
