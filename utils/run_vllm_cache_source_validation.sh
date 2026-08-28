#!/usr/bin/env bash
set -euo pipefail

# Run the cache-source metric validator against a one-GPU vLLM Docker server.
# Required: IMAGE. Optional: MODE=device|cpu|disk|external, HOST_HF_CACHE,
# DISK_ROOT, PORT.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:?Set IMAGE to the vLLM image under test}
MODE=${MODE:-device}
PORT=${PORT:-18000}
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
PROMPT_MODE=${PROMPT_MODE:-token_ids}
HOST_HF_CACHE=${HOST_HF_CACHE:-/models/gharunners/hf-hub-cache}
DISK_ROOT=${DISK_ROOT:-/tmp}
RUN_KEY=${SLURM_JOB_ID:-$$}
VISIBLE_GPU_LIST=${CUDA_VISIBLE_DEVICES:-0}
GPU_DEVICE=${DOCKER_GPU_DEVICE:-${VISIBLE_GPU_LIST%%,*}}
GPU_DEVICE=${GPU_DEVICE:-0}
CONTAINER="vllm-cache-source-${MODE}-${RUN_KEY}"
SERVER_LOG="${TMPDIR:-/tmp}/${CONTAINER}.log"
DISK_DIR=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if docker inspect "$CONTAINER" >/dev/null 2>&1; then
        docker logs "$CONTAINER" >"$SERVER_LOG" 2>&1 || true
    fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    if [[ -n "$DISK_DIR" && -d "$DISK_DIR" ]]; then
        docker run --rm --volume "$DISK_DIR:/cleanup" --entrypoint /bin/sh \
            "$IMAGE" -c 'rm -rf /cleanup/* /cleanup/.[!.]* /cleanup/..?*' \
            >/dev/null 2>&1 || true
        rmdir -- "$DISK_DIR" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -d "$HOST_HF_CACHE" ]]; then
    echo "Hugging Face cache does not exist: $HOST_HF_CACHE" >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull "$IMAGE"
fi

DOCKER_ARGS=(
    --detach
    --name "$CONTAINER"
    --gpus "device=$GPU_DEVICE"
    --network host
    --ipc host
    --env VLLM_ENABLE_CUDA_COMPATIBILITY=1
    --env VLLM_SERVER_DEV_MODE=1
    --volume "$HOST_HF_CACHE:/root/.cache/huggingface/hub"
)
SERVER_ARGS=(
    "$MODEL"
    --host 127.0.0.1
    --port "$PORT"
    --language-model-only
    --enable-prefix-caching
    --enforce-eager
    --max-model-len 4096
    --max-num-seqs 4
    --kv-cache-memory-bytes 1073741824
)
EXPECTED_TIER=()

case "$MODE" in
    device)
        ;;
    cpu)
        SERVER_ARGS+=(
            --kv-transfer-config
            '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"kv_offload_backend":"cpu","cpu_bytes_to_use":536870912,"lazy_offload":false}}'
        )
        EXPECTED_TIER=(--expected-tier cpu)
        ;;
    disk)
        DISK_DIR=$(mktemp -d "$DISK_ROOT/vllm-cache-source.XXXXXX")
        DOCKER_ARGS+=(--volume "$DISK_DIR:/kv-offload")
        SERVER_ARGS+=(
            --kv-transfer-config
            '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"kv_offload_backend":"disk","disk_path":"/kv-offload/cache.bin","disk_capacity_bytes":536870912,"disk_buffer_slots":4,"lazy_offload":false}}'
        )
        EXPECTED_TIER=(--expected-tier disk)
        ;;
    external)
        DISK_DIR=$(mktemp -d "$DISK_ROOT/vllm-cache-source.XXXXXX")
        DOCKER_ARGS+=(--volume "$DISK_DIR:/external-kv")
        SERVER_ARGS+=(
            --kv-transfer-config
            '{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"/external-kv"}}'
        )
        EXPECTED_TIER=(--expected-tier external)
        ;;
    *)
        echo "Unsupported MODE=$MODE; expected device, cpu, disk, or external" >&2
        exit 1
        ;;
esac

docker run "${DOCKER_ARGS[@]}" "$IMAGE" "${SERVER_ARGS[@]}" >/dev/null

ready=false
for _ in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
        ready=true
        break
    fi
    if ! docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
        echo "vLLM container stopped before becoming ready" >&2
        docker logs "$CONTAINER" >&2 || true
        exit 1
    fi
    sleep 2
done
if [[ "$ready" != true ]]; then
    echo "vLLM server did not become ready" >&2
    docker logs "$CONTAINER" >&2 || true
    exit 1
fi

python3 "$SCRIPT_DIR/validate_vllm_cache_source_metrics.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --prompt-mode "$PROMPT_MODE" \
    "${EXPECTED_TIER[@]}"

curl --fail --silent "http://127.0.0.1:${PORT}/metrics" \
    | grep -E '^vllm:prompt_tokens_(cached|cached_by_source|by_source)' \
    | tee "${SERVER_LOG%.log}.metrics"
docker logs "$CONTAINER" >"$SERVER_LOG" 2>&1
echo "server log: $SERVER_LOG"
