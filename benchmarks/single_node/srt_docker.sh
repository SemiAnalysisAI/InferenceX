#!/usr/bin/env bash

# Keep the pool's existing Docker lifecycle; commands come from native SRT YAML.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh" --validation-only
check_env_vars MODEL PORT RUN_EVAL EVAL_ONLY INFMAX_CONTAINER_WORKSPACE
for flag in RUN_EVAL EVAL_ONLY; do
    if [[ "${!flag}" != true && "${!flag}" != false ]]; then
        echo "$flag must be true or false" >&2
        exit 1
    fi
done
if [[ "$RUN_EVAL" == true || "$EVAL_ONLY" == true ]]; then
    check_env_vars MODEL_NAME
fi
SERVER_LOG="$INFMAX_CONTAINER_WORKSPACE/server.log"
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
fi
bash "$INFMAX_CONTAINER_WORKSPACE/srt-docker-server.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'rc=$?; kill "$SERVER_PID" 2>/dev/null || true; exit "$rc"' EXIT
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh"
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
export INFERENCEX_SERVER_PID INFERENCEX_SERVER_STATE
if [[ "$EVAL_ONLY" != true ]]; then
    bash "$INFMAX_CONTAINER_WORKSPACE/srt-docker-client.sh"
fi
if [[ "$RUN_EVAL" == true || "$EVAL_ONLY" == true ]]; then
    bash "$INFMAX_CONTAINER_WORKSPACE/benchmarks/single_node/srt_eval.sh" \
        "http://127.0.0.1:$PORT" "$INFMAX_CONTAINER_WORKSPACE/infx-eval-exit-code"
fi
