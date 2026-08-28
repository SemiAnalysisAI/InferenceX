#!/usr/bin/env bash
set -euo pipefail

# Run inside a Slurm/Pyxis or Enroot container on clusters without Docker.

MODE=${MODE:-device}
PORT=${PORT:-18000}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PROMPT_MODE=${PROMPT_MODE:-token_ids}
VALIDATION_DIR=${VALIDATION_DIR:-/validation}
RESULT_DIR=${RESULT_DIR:-$VALIDATION_DIR/results}
RUN_KEY=${SLURM_JOB_ID:-$$}
DISK_ROOT=${DISK_ROOT:-/mnt/numa0/enroot/runtime/user-${UID}}
SERVER_LOG="$RESULT_DIR/${MODE}-${RUN_KEY}.server.log"
METRICS_LOG="$RESULT_DIR/${MODE}-${RUN_KEY}.metrics"
DISK_DIR=""
SERVER_PID=""
mkdir -p "$RESULT_DIR"

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$DISK_DIR" && -d "$DISK_DIR" ]]; then
        rm -rf -- "$DISK_DIR"
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SERVER_ARGS=(
    serve "$MODEL"
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
        SERVER_ARGS+=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"kv_offload_backend\":\"disk\",\"disk_path\":\"$DISK_DIR/cache.bin\",\"disk_capacity_bytes\":536870912,\"disk_buffer_slots\":4,\"lazy_offload\":false}}"
        )
        EXPECTED_TIER=(--expected-tier disk)
        ;;
    *)
        echo "Unsupported MODE=$MODE; expected device, cpu, or disk" >&2
        exit 1
        ;;
esac

export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_SERVER_DEV_MODE=1
vllm "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=false
for _ in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
        ready=true
        break
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "vLLM server stopped before becoming ready" >&2
        tail -n 200 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 2
done
if [[ "$ready" != true ]]; then
    echo "vLLM server did not become ready" >&2
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
fi

python3 "$VALIDATION_DIR/validate_vllm_cache_source_metrics.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --prompt-mode "$PROMPT_MODE" \
    "${EXPECTED_TIER[@]}"

curl --fail --silent "http://127.0.0.1:${PORT}/metrics" \
    | grep -E '^vllm:prompt_tokens_(cached|cached_by_source|by_source)' \
    | tee "$METRICS_LOG"
echo "server log: $SERVER_LOG"
echo "metrics: $METRICS_LOG"
