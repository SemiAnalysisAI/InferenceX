#!/usr/bin/env bash
# Custom-benchmark H3 client. srt-slurm starts the diffusion server as a service;
# this script runs against SRT_SERVICE_<NAME>_IPS after readiness.
set -eo pipefail

if [[ ! -f "${INFMAX_CONTAINER_WORKSPACE:-}/benchmarks/benchmark_lib.sh" ]]; then
    INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export INFMAX_CONTAINER_WORKSPACE
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars H3_RESULT_DIR H3_SPEC_PATH H3_PACKAGE_ROOT H3_SERVER_SERVICE H3_SERVER_PORT

service_key="$(printf '%s' "$H3_SERVER_SERVICE" | tr '[:lower:]-' '[:upper:]_' | tr -c 'A-Z0-9_' '_')"
ips_var="SRT_SERVICE_${service_key}_IPS"
ips="${!ips_var:-}"
if [[ -z "$ips" ]]; then
    echo "ERROR: $ips_var is unset; declare services[] named $H3_SERVER_SERVICE" >&2
    exit 2
fi
host="${ips%%,*}"
endpoint="http://${host}:${H3_SERVER_PORT}"
echo "Using srt-slurm H3 service endpoint: $endpoint"
mkdir -p "$H3_RESULT_DIR"

export PYTHONPATH="${H3_PACKAGE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m evaluator.cli srt-client \
    --spec "$H3_SPEC_PATH" \
    --endpoint "$endpoint" \
    --output "$H3_RESULT_DIR"
