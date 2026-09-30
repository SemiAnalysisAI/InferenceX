#!/bin/bash
# Agentic trace replay for one concurrency on freshly started disaggregated servers.
# Usage: bash trace_replay.sh <model_dir> <model_name> <concurrency> <log_path>
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
check_env_vars ENGINE MODEL_PATH MODEL_NAME ROUTER_PORT
if [[ $# -ne 4 ]]; then
    echo "Error: trace_replay.sh requires 4 positional arguments" >&2
    exit 1
fi

model_path=$1
model_name=$2
concurrency=$3
log_path=$4
validate_agentic_concurrency "$concurrency"
if [[ -n "${CONC:-}" && "$CONC" != "$concurrency" ]]; then
    echo "ERROR: CONC must match the requested trace replay concurrency" >&2
    exit 1
fi
if [[ "${CONC_LIST+x}" ]]; then
    validate_agentic_concurrency "$CONC_LIST"
    if [[ "$CONC_LIST" != "$concurrency" ]]; then
        echo "ERROR: CONC_LIST must match the requested trace replay concurrency" >&2
        exit 1
    fi
fi

MODEL="${MODEL_PATH}"
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
RESULT_DIR="${RESULT_DIR:-${log_path}/agentic}"
source "$(dirname "$0")/../../benchmark_lib.sh"

PORT="${ROUTER_PORT}"
check_env_vars DURATION RESULT_FILENAME
export MODEL DURATION MAX_MODEL_LEN
export CONC="$concurrency" USERS="$concurrency"

if [[ "$PREFILL_ENABLE_DP" == true ]]; then
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
fi

resolve_trace_source
install_agentic_deps

# Preserve the workflow's per-concurrency artifact and raw-log paths.
RESULT_DIR="$RESULT_DIR/conc_${CONC}"
mkdir -p "$RESULT_DIR"
export RESULT_FILENAME="${RESULT_FILENAME}_conc${CONC}"
echo "Agentic trace replay: concurrency=$CONC on this server deployment"
build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
