#!/usr/bin/env bash
# Custom-benchmark H3 client: every CONC_LIST concurrency against the one server
# srt-slurm started as a service, each cell publishing its measured window to the
# native power telemetry (SRT_MEASUREMENT_WINDOW_DIR).
set -eo pipefail

INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export INFMAX_CONTAINER_WORKSPACE
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/check_env.sh"
check_env_vars H3_RESULT_DIR H3_SPEC_PATH H3_PACKAGE_ROOT H3_SERVER_SERVICE H3_SERVER_PORT \
    CONC_LIST SRT_MEASUREMENT_WINDOW_DIR

service_key="$(printf '%s' "$H3_SERVER_SERVICE" | tr '[:lower:]-' '[:upper:]_' | tr -c 'A-Z0-9_' '_')"
ips_var="SRT_SERVICE_${service_key}_IPS"
ips="${!ips_var:-}"
if [[ -z "$ips" ]]; then
    echo "ERROR: $ips_var is unset; declare services[] named $H3_SERVER_SERVICE" >&2
    exit 2
fi
host="${ips%%,*}"
endpoint="http://${host}:${H3_SERVER_PORT}"
echo "Using srt-slurm H3 service endpoint: $endpoint (concurrencies: $CONC_LIST)"
mkdir -p "$H3_RESULT_DIR" "$SRT_MEASUREMENT_WINDOW_DIR"

# shellcheck disable=SC2086 # CONC_LIST is a space-separated list of integers.
export PYTHONPATH="${H3_PACKAGE_ROOT}:${INFMAX_CONTAINER_WORKSPACE}${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m evaluator.cli srt-client \
    --spec "$H3_SPEC_PATH" \
    --endpoint "$endpoint" \
    --output "$H3_RESULT_DIR" \
    --window-dir "$SRT_MEASUREMENT_WINDOW_DIR" \
    --concurrencies $CONC_LIST
