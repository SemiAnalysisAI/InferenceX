#!/usr/bin/env bash

# SRT owns readiness and lifecycle; InferenceX owns evaluation and its artifacts.
set -eo pipefail
if [[ $# != 3 || -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 endpoint status-file diagnostic-mode" >&2
    exit 1
fi
SRT_EVAL_STATUS_FILE="$2"
SRT_DIAGNOSTIC_MODE="$3"
trap 'rc=$?; printf "%s\n" "$rc" > "$SRT_EVAL_STATUS_FILE"' EXIT

source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh"
check_env_vars MODEL MODEL_NAME CONC TP EP_SIZE DP_ATTENTION IS_MULTINODE IS_AGENTIC
case "$SRT_DIAGNOSTIC_MODE" in
    none) ;;
    kimi-metrics|native-cpu-restore)
        if [[ "$IS_AGENTIC" != 1 || "${EVAL_ONLY:-}" != true || "${EVAL_FRAMEWORK:-}" != bfcl ]]; then
            echo "ERROR: BFCL diagnostics require an AgentX eval-only job" >&2
            exit 1
        fi
        ;;
    *) echo "ERROR: unknown diagnostic mode: $SRT_DIAGNOSTIC_MODE" >&2; exit 1 ;;
esac
# AgentX evaluates at the native context with the workflow's eval framework.
eval_args=()
if [[ "$IS_AGENTIC" != 1 ]]; then
    check_env_vars MAX_MODEL_LEN
    eval_args=(--framework lm-eval)
elif [[ "${MODEL_PREFIX:-}" == glm5.2 ]]; then
    # GLM-5.2's template defaults to maximum reasoning effort without
    # chat_template_kwargs, which mini-swe-agent never passes; the heavy thinking
    # exhausts the shared step budget. The recipe env does not reach post-eval.
    export SWEBENCH_AGENT_STEP_LIMIT=150
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
# Put diagnostic evidence in the eval artifact root even when evaluation fails.
if [[ "$SRT_DIAGNOSTIC_MODE" == kimi-metrics ]]; then
    metrics_rc=0
    curl --fail --silent --show-error --max-time 30 "http://127.0.0.1:$PORT/metrics" \
        -o bfcl_diagnostic_metrics.txt || metrics_rc=$?
    tar -czf bfcl_diagnostic_metrics_artifacts.tar.gz bfcl_diagnostic_metrics.txt || metrics_rc=$?
    if [[ "$eval_rc" == 0 ]]; then eval_rc="$metrics_rc"; fi
elif [[ "$SRT_DIAGNOSTIC_MODE" == native-cpu-restore && "$eval_rc" == 0 ]]; then
    timeout 600 python3 "$INFERENCEX_REPO_ROOT/experimental/bfcl/verify_native_cpu_restore.py" \
        --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
        --output "$INFERENCEX_REPO_ROOT/native_cpu_restore_report.json" || eval_rc=$?
fi
exit "$eval_rc"
