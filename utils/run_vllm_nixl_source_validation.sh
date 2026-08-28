#!/usr/bin/env bash
set -euo pipefail

# Run inside a two-GPU vLLM container. The upstream toy proxy must be mounted
# at NIXL_INTEGRATION_DIR and the cache-source validator at VALIDATION_DIR.

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
VALIDATION_DIR=${VALIDATION_DIR:-/validation}
NIXL_INTEGRATION_DIR=${NIXL_INTEGRATION_DIR:-/nixl-integration}
RESULT_DIR=${RESULT_DIR:-$VALIDATION_DIR/results}
RUN_KEY=${RUN_KEY:-${SLURM_JOB_ID:-$$}}
PREFILL_PORT=${PREFILL_PORT:-18100}
DECODE_PORT=${DECODE_PORT:-18200}
PROXY_PORT=${PROXY_PORT:-18192}
PREFILL_INTERNAL_PORT=${PREFILL_INTERNAL_PORT:-28100}
DECODE_INTERNAL_PORT=${DECODE_INTERNAL_PORT:-28200}
PREFILL_SIDE_PORT=${PREFILL_SIDE_PORT:-15559}
DECODE_SIDE_PORT=${DECODE_SIDE_PORT:-15659}
PREFILL_LOG="$RESULT_DIR/nixl-${RUN_KEY}.prefill.log"
DECODE_LOG="$RESULT_DIR/nixl-${RUN_KEY}.decode.log"
PROXY_LOG="$RESULT_DIR/nixl-${RUN_KEY}.proxy.log"
METRICS_LOG="$RESULT_DIR/nixl-${RUN_KEY}.metrics"
PIDS=()
LAST_SERVER_PID=""
mkdir -p "$RESULT_DIR"

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if ((${#PIDS[@]})); then
        kill "${PIDS[@]}" >/dev/null 2>&1 || true
        wait "${PIDS[@]}" >/dev/null 2>&1 || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
    local gpu=$1
    local port=$2
    local internal_port=$3
    local side_port=$4
    local role=$5
    local log=$6
    CUDA_VISIBLE_DEVICES="$gpu" \
        VLLM_PORT="$internal_port" \
        VLLM_NIXL_SIDE_CHANNEL_PORT="$side_port" \
        VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
        VLLM_SERVER_DEV_MODE=1 \
        VLLM_SSM_CONV_STATE_LAYOUT=DS \
        UCX_NET_DEVICES=all \
        vllm serve "$MODEL" \
            --host 127.0.0.1 \
            --port "$port" \
            --language-model-only \
            --block-size 128 \
            --gpu-memory-utilization 0.2 \
            --enforce-eager \
            --max-model-len 4096 \
            --kv-transfer-config \
            "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$role\"}" \
            >"$log" 2>&1 &
    LAST_SERVER_PID=$!
    PIDS+=("$LAST_SERVER_PID")
}

wait_for_health() {
    local port=$1
    local log=$2
    local pid=$3
    for _ in $(seq 1 300); do
        if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
            return
        fi
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            echo "server on port $port stopped before becoming ready" >&2
            tail -n 200 "$log" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "server on port $port did not become ready" >&2
    tail -n 200 "$log" >&2 || true
    return 1
}

start_server 0 "$PREFILL_PORT" "$PREFILL_INTERNAL_PORT" "$PREFILL_SIDE_PORT" \
    kv_producer "$PREFILL_LOG"
PREFILL_PID=$LAST_SERVER_PID
start_server 1 "$DECODE_PORT" "$DECODE_INTERNAL_PORT" "$DECODE_SIDE_PORT" \
    kv_consumer "$DECODE_LOG"
DECODE_PID=$LAST_SERVER_PID
wait_for_health "$PREFILL_PORT" "$PREFILL_LOG" "$PREFILL_PID"
wait_for_health "$DECODE_PORT" "$DECODE_LOG" "$DECODE_PID"

python3 "$NIXL_INTEGRATION_DIR/toy_proxy_server.py" \
    --host 127.0.0.1 \
    --port "$PROXY_PORT" \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports "$PREFILL_PORT" \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports "$DECODE_PORT" \
    >"$PROXY_LOG" 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 60); do
    if curl --fail --silent "http://127.0.0.1:${PROXY_PORT}/healthcheck" >/dev/null; then
        break
    fi
    sleep 1
done
curl --fail --silent "http://127.0.0.1:${PROXY_PORT}/healthcheck" >/dev/null

PYTHONPATH="$VALIDATION_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "http://127.0.0.1:${PROXY_PORT}" \
    "http://127.0.0.1:${DECODE_PORT}" \
    "$MODEL" <<'PY'
import sys

from validate_vllm_cache_source_metrics import (
    completion,
    snapshot,
    validate_delta,
    wait_for_accounting,
)

proxy_url, decoder_url, model = sys.argv[1:]
prompt = [100] * 3264
before = snapshot(decoder_url)
completion(proxy_url, model, prompt)
after = wait_for_accounting(decoder_url, before, len(prompt), timeout=30)
delta = after.delta(before)
validate_delta("nixl_external_transfer", delta, {"external"})
if delta.prompt_by_source.get("external_kv_transfer", 0) <= 0:
    raise RuntimeError(f"NIXL request had no external KV transfer: {delta}")
PY

curl --fail --silent "http://127.0.0.1:${DECODE_PORT}/metrics" \
    | grep -E '^vllm:prompt_tokens_(cached|cached_by_source|by_source)' \
    | tee "$METRICS_LOG"
echo "NIXL source validation passed"
echo "prefill log: $PREFILL_LOG"
echo "decode log: $DECODE_LOG"
echo "proxy log: $PROXY_LOG"
echo "metrics: $METRICS_LOG"
