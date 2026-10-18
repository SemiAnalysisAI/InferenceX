#!/usr/bin/env bash
# GPU-owning H3 diffusion server for the video-generation serving-client recipe.
# srt-slurm starts this as a generic service on the job's GPUs; run_client.sh is
# the custom benchmark that addresses it by node IP after the /health readiness probe.
set -eo pipefail

INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export INFMAX_CONTAINER_WORKSPACE
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/check_env.sh"
check_env_vars H3_SPEC_PATH H3_PACKAGE_ROOT H3_RESULT_DIR H3_SERVER_PORT H3_SERVER_ROLE

export PYTHONPATH="${H3_PACKAGE_ROOT}:${INFMAX_CONTAINER_WORKSPACE}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${H3_SET:-}" ]]; then
    check_env_vars H3_PLAN_SHA256 H3_RUNTIME H3_RUNTIME_REVISION H3_MODEL_PATH GPU_COUNT
    exec python3 -m evaluator.cli srt-yaml-serve \
        --spec "$H3_SPEC_PATH" --set "$H3_SET" --plan-sha256 "$H3_PLAN_SHA256" \
        --runtime "$H3_RUNTIME" --runtime-revision "$H3_RUNTIME_REVISION" \
        --model-path "$H3_MODEL_PATH" --port "$H3_SERVER_PORT" \
        --gpus "$GPU_COUNT" --record "$H3_RESULT_DIR/server"
fi
exec python3 -m evaluator.cli srt-serve \
    --spec "$H3_SPEC_PATH" \
    --role "$H3_SERVER_ROLE" \
    --port "$H3_SERVER_PORT" \
    --host 0.0.0.0 \
    --record "$H3_RESULT_DIR/server"
