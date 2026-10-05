#!/usr/bin/env bash
# GPU-owning H3 diffusion server for the custom-benchmark serving-client recipe.
# srt-slurm starts this as a generic service; run_client.sh is the custom benchmark.
set -eo pipefail

if [[ ! -f "${INFMAX_CONTAINER_WORKSPACE:-}/benchmarks/benchmark_lib.sh" ]]; then
    INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export INFMAX_CONTAINER_WORKSPACE
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars H3_SPEC_PATH H3_PACKAGE_ROOT H3_SERVER_PORT H3_SERVER_ROLE

export PYTHONPATH="${H3_PACKAGE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m evaluator.cli srt-serve \
    --spec "$H3_SPEC_PATH" \
    --role "$H3_SERVER_ROLE" \
    --port "$H3_SERVER_PORT"
