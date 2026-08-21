#!/usr/bin/env bash
set -euo pipefail

# This hook is enabled only by the dense B300 prefill experiment. The setup
# script itself is a child of the worker wrapper, so bind the parent shell;
# the subsequently exec'd TRT-LLM worker inherits that affinity.
if [[ "${TLLM_NUMA_AWARE_WORKER_AFFINITY:-}" != "0" ]]; then
    exit 0
fi

case "${CUDA_VISIBLE_DEVICES:-}" in
    0) cpuset=0-11,96-107 ;;
    1) cpuset=12-23,108-119 ;;
    2) cpuset=24-35,120-131 ;;
    3) cpuset=36-47,132-143 ;;
    4) cpuset=48-59,144-155 ;;
    5) cpuset=60-71,156-167 ;;
    6) cpuset=72-83,168-179 ;;
    7) cpuset=84-95,180-191 ;;
    *)
        echo "CPU_BIND_DIAGNOSTIC skipped: CVD=${CUDA_VISIBLE_DEVICES:-<unset>}"
        exit 0
        ;;
esac

taskset -pc "${cpuset}" "${PPID}" >/dev/null
echo "CPU_BIND_DIAGNOSTIC CVD=${CUDA_VISIBLE_DEVICES} parent_pid=${PPID} requested=${cpuset}"
taskset -pc "${PPID}"
