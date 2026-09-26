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
check_env_vars MODEL MODEL_NAME CONC TP EP_SIZE DP_ATTENTION IS_MULTINODE IS_AGENTIC
# AgentX evaluates at the native context with the workflow's eval framework.
eval_args=()
if [[ "$IS_AGENTIC" != 1 ]]; then
    check_env_vars MAX_MODEL_LEN
    eval_args=(--framework lm-eval)
fi
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
run_eval "${eval_args[@]}" --port "$PORT" || eval_rc=$?
# AgentX eval-only run_eval already staged and removed its results.
if [[ "$IS_AGENTIC" != 1 ]]; then
    append_lm_eval_summary || eval_rc=1
fi
exit "$eval_rc"
