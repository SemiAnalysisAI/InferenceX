#!/usr/bin/env bash

source "$(dirname "$0")/../../../../benchmarks/benchmark_lib.sh"

check_env_vars MODEL TP

PORT=${PORT:-8888}
SERVER_LOG=/workspace/server.log
CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-131272}

cat > config.yaml << EOF
kv-cache-dtype: ${KV_CACHE_DTYPE:-fp8}
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: $CALCULATED_MAX_MODEL_LEN
EOF

python3 -m pip install -q lmcache

launch_vllm_server "$MODEL" "$PORT" config.yaml   --disable-log-requests   --trust-remote-code   --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

echo "LMCache vLLM server running (PID=$SERVER_PID, log=$SERVER_LOG)"
wait "$SERVER_PID"
