#!/bin/bash
# Agentic trace-replay runner for the disaggregated servers.
#
# Usage: bash trace_replay.sh <model_dir> <model_name> <concurrency_list> <log_path>

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
check_env_vars ENGINE MODEL_PATH MODEL_NAME ROUTER_PORT
if [[ $# -ne 4 ]]; then
    echo "Error: trace_replay.sh requires 4 positional arguments" >&2
    exit 1
fi

model_path=$1
model_name=$2
concurrency_list=${3}
# vllm-disagg uses --served-model-name MODEL_NAME; sglang defaults to MODEL_PATH
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    MODEL="${MODEL_NAME}"
else
    MODEL="${MODEL_PATH}"
fi
log_path=${4}

IFS='x' read -r -a chosen_concurrencies <<< "${concurrency_list}"

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

RESULT_DIR="${RESULT_DIR:-${log_path}/agentic}"
mkdir -p "$RESULT_DIR"

source "$(dirname "$0")/../../benchmark_lib.sh"

# Wipe every KV cache tier on each backend worker before a concurrency point so it
# is measured cold. Hits each worker directly (the router does not fan /flush_cache
# out) using the URLs server_sglang.sh resolved into SERVER_FLUSH_URLS_CSV.
#   L1 (GPU radix) + L2 (host hicache): POST /flush_cache, a NO-OP while any request
#       is in flight, so drain-retry until "Cache flushed" or FLUSH_DRAIN_TIMEOUT.
#   L3 (umbp / mooncake store): POST /hicache/storage-backend/clear, non-200 when
#       L3 is off.
# Best-effort: never hard-fails the sweep.
clear_kv_caches() {
    local drain_tmo="${FLUSH_DRAIN_TIMEOUT}"
    local urls_csv="${SERVER_FLUSH_URLS_CSV:-}"
    if [[ -z "$urls_csv" ]]; then
        echo "[clear_caches] WARN: SERVER_FLUSH_URLS_CSV unset; skipping cache flush" >&2
        return 0
    fi
    local -a urls
    IFS=',' read -r -a urls <<< "$urls_csv"
    local url start ok resp code
    for url in "${urls[@]}"; do
        [[ -n "$url" ]] || continue
        # L1 + L2: drain-retry until flushed (no-op while requests in flight).
        start=$(date +%s); ok=0; resp=""
        while :; do
            resp=$(curl -sf -m 10 -X POST "${url}/flush_cache" 2>/dev/null || true)
            echo "$resp" | grep -qi "Cache flushed" && { ok=1; break; }
            (( $(date +%s) - start >= drain_tmo )) && break
            sleep 3
        done
        if [[ "$ok" == 1 ]]; then
            echo "[clear_caches] ${url}: L1+L2 flushed"
        else
            echo "[clear_caches] WARN ${url}: L1+L2 flush NOT confirmed after ${drain_tmo}s (resp='${resp:0:80}')" >&2
        fi
        # L3: storage-backend clear (umbp / mooncake). 200 when a backend is attached.
        code=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST "${url}/hicache/storage-backend/clear" 2>/dev/null || echo 000)
        if [[ "$code" == 200 ]]; then
            echo "[clear_caches] ${url}: L3 store cleared"
        else
            echo "[clear_caches] ${url}: L3 clear http=${code} (no storage backend / L3 off — ok)"
        fi
    done
}

PORT="${ROUTER_PORT}"
check_env_vars DURATION RESULT_FILENAME FLUSH_DRAIN_TIMEOUT CLEAR_CACHE_BETWEEN_CONC
export MODEL DURATION MAX_MODEL_LEN
# The workflow guard / upload steps expect one "${RESULT_FILENAME}_conc<N>.json" per
# concurrency, so each conc below is suffixed with _conc<N> (as agentic_srt.sh does).
RESULT_FILENAME_BASE="${RESULT_FILENAME}"

mkdir -p "$RESULT_DIR"

if [ "$PREFILL_ENABLE_DP" = "true" ]; then
    set -x
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    set +x
fi

resolve_trace_source
install_agentic_deps

ANY_FAILED=0
for max_concurrency in "${chosen_concurrencies[@]}"; do

    echo "=========================================="
    echo "Agentic trace replay: conc=$max_concurrency"
    echo "=========================================="

    # Measure each conc point cold (no prefix reuse from the previous conc).
    # CLEAR_CACHE_BETWEEN_CONC=0 disables; best-effort, never fails the run.
    if [[ "${CLEAR_CACHE_BETWEEN_CONC}" == "1" ]]; then
        echo "conc=$max_concurrency: clearing L1/L2/L3 on all backends (no server restart)"
        clear_kv_caches || echo "WARNING: cache clear had issues for conc=$max_concurrency" >&2
    fi

    # benchmark-multinode-tmpl.yml expects the per-conc nesting (LOGS/agentic/conc_*/...)
    # even though CI runs one concurrency per job; nesting also keeps local multi-conc
    # sweeps from overwriting each other (same layout as agentic_srt.sh).
    CONC_RESULT_DIR="$RESULT_DIR/conc_${max_concurrency}"
    mkdir -p "$CONC_RESULT_DIR"

    CONC="$max_concurrency"
    USERS="$max_concurrency"
    export CONC USERS
    build_replay_cmd "$CONC_RESULT_DIR"

    # Must match the workflow guard's "${RESULT_FILENAME}_conc*.json" glob and the
    # agg / checkpoint upload steps.
    export RESULT_FILENAME="${RESULT_FILENAME_BASE}_conc${max_concurrency}"
    if ! run_agentic_replay_and_write_outputs "$CONC_RESULT_DIR"; then
        echo "WARNING: agentic trace replay for conc=$max_concurrency failed (replay or validation) after writing available results" >&2
        ANY_FAILED=1
    fi

    echo "-----------------------------------------"

done

export RESULT_FILENAME="$RESULT_FILENAME_BASE"

if [ "$ANY_FAILED" -ne 0 ]; then
    echo "WARNING: at least one conc had a non-zero exit; per-conc result files were still written when possible." >&2
fi
