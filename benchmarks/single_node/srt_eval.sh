#!/usr/bin/env bash

# SRT owns readiness and lifecycle; InferenceX owns evaluation and its artifacts.
set -eo pipefail
if [[ $# != 2 || -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 endpoint status-file" >&2
    exit 1
fi
SRT_EVAL_STATUS_FILE="$2"
trap 'rc=$?; printf "%s\n" "$rc" > "$SRT_EVAL_STATUS_FILE"' EXIT

source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh"
check_env_vars MODEL MODEL_NAME CONC TP EP_SIZE DP_ATTENTION IS_MULTINODE MAX_MODEL_LEN
export PORT="${1##*:}"
if [[ ! "$PORT" =~ ^[1-9][0-9]*$ || "$IS_MULTINODE" != false ]]; then
    echo "ERROR: single-node eval requires a local endpoint and single-node metadata" >&2
    exit 1
fi
cd "$INFERENCEX_REPO_ROOT"
if [[ -d /model ]]; then
    export MODEL_PATH=/model
fi

eval_rc=0
run_eval --framework lm-eval --port "$PORT" || eval_rc=$?
append_lm_eval_summary || eval_rc=1
exit "$eval_rc"
