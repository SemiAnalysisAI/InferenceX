#!/usr/bin/env bash
# Terminal-service entry for H3 A/B (or full supervisor) workloads on srt-slurm.
# srt-slurm owns allocation, container lifecycle, logs and teardown; this script
# only runs the H3 supervisor against caller-supplied configuration.
set -eo pipefail

if [[ ! -f "${INFMAX_CONTAINER_WORKSPACE:-}/benchmarks/benchmark_lib.sh" ]]; then
    INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export INFMAX_CONTAINER_WORKSPACE
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars H3_RESULT_DIR H3_SPEC_PATH H3_PACKAGE_ROOT

echo "H3_SRT_WORKLOAD_STARTED"
mkdir -p "$H3_RESULT_DIR"

export PYTHONPATH="${H3_PACKAGE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m evaluator.cli srt-gpu-job \
    --spec "$H3_SPEC_PATH" \
    --output "$H3_RESULT_DIR"
