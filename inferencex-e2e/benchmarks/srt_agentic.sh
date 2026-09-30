#!/usr/bin/env bash
set -eo pipefail
set -x

# Client-only AgentX trace replay for single- and multi-node srt-slurm jobs.
# srt-slurm owns server startup; this script runs as benchmark.type=custom
# against a fresh, already-ready frontend for exactly one concurrency.

# Jobs inherit the legacy scripts' /workspace, which srt-slurm does not mount;
# fall back to the repo mount this client runs from.
if [[ ! -f "${INFMAX_CONTAINER_WORKSPACE:-}/benchmarks/benchmark_lib.sh" ]]; then
    INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
: "${IS_MULTINODE:=false}" "${PORT:=8000}"
export INFMAX_CONTAINER_WORKSPACE IS_MULTINODE PORT
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars RESULT_DIR EVAL_ONLY CONC
validate_agentic_concurrency "$CONC"
if [[ "${CONC_LIST+x}" ]]; then
    validate_agentic_concurrency "$CONC_LIST"
    if [[ "$CONC_LIST" != "$CONC" ]]; then
        echo "ERROR: CONC must match the single CONC_LIST value" >&2
        exit 1
    fi
fi
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
    check_env_vars SRT_FRONTEND_PORT
    export AIPERF_SERVER_URL="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}"
fi

# benchmark_lib deliberately clears inherited MAX_MODEL_LEN for AgentX so a
# workflow default cannot silently truncate a model's native context. Native
# srt-slurm topologies may still expose a smaller, explicit service limit (for
# example when both P/D roles are configured identically below model-native
# context). Restore that limit only through this dedicated opt-in.
if [[ -n "${AIPERF_MAX_CONTEXT_LENGTH:-}" ]]; then
    if ! [[ "$AIPERF_MAX_CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: AIPERF_MAX_CONTEXT_LENGTH must be a positive integer" >&2
        exit 1
    fi
    export MAX_MODEL_LEN="$AIPERF_MAX_CONTEXT_LENGTH"
fi

check_env_vars \
    MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC \
    RESULT_FILENAME DURATION

if [[ -z "${AIPERF_SERVER_URL:-}" ]]; then
    if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
        export AIPERF_SERVER_URL="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}"
    else
        export AIPERF_SERVER_URL="http://localhost:${PORT}"
    fi
fi
echo "Using srt-slurm frontend endpoint: $AIPERF_SERVER_URL"

# A router frontend does not re-export engine metrics; read them from each worker.
if [[ -z "${AIPERF_SERVER_METRICS_URLS:-}" && "${SRTCTL_FRONTEND_TYPE:-}" != dynamo ]]; then
    endpoints="${SRT_AGG_ENDPOINTS:-${SRT_PREFILL_ENDPOINTS:+$SRT_PREFILL_ENDPOINTS,}${SRT_DECODE_ENDPOINTS:-}}"
    if [[ -n "${endpoints%,}" ]]; then
        AIPERF_SERVER_METRICS_URLS=$(sed -E 's#([^,]+)#http://\1/metrics#g' <<< "${endpoints%,}")
        export AIPERF_SERVER_METRICS_URLS
    fi
fi

resolve_trace_source
install_agentic_deps
if [[ "${EVAL_ONLY}" == "true" ]]; then
    _wait_for_openai_chat_route --port "$PORT"
fi

# Keep the multi-node artifact suffix; single-node collection uses the caller's name.
if [[ -n "${CONC_LIST:-}" ]]; then
    export RESULT_FILENAME="${RESULT_FILENAME}_conc${CONC}"
    RESULT_DIR="${RESULT_DIR}/conc_${CONC}"
fi
mkdir -p "$RESULT_DIR"

echo "Running agentic concurrency $CONC on this server deployment"
build_replay_cmd "$RESULT_DIR"
# Recipes whose legacy launch rendered prompts client-side opt in here.
if [[ "${AIPERF_APPLY_CHAT_TEMPLATE:-}" == true ]]; then
    REPLAY_CMD+=" --apply-chat-template"
fi
# Let this point's admitted responses finish before the deployment is torn down.
if [[ -n "${AIPERF_BENCHMARK_GRACE_PERIOD:-}" ]]; then
    REPLAY_CMD+=" --benchmark-grace-period $AIPERF_BENCHMARK_GRACE_PERIOD"
fi
run_agentic_replay_and_write_outputs "$RESULT_DIR"
