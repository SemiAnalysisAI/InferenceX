#!/bin/bash
# Dual-Engine Disaggregated Benchmark Runner
#
# ENGINE=sglang (default): SGLang benchmark
# ENGINE=vllm:             vLLM benchmark
#
# Produces JSON result files via benchmark_serving.py so that the CI pipeline
# can collect and process results.
#
# Usage: bash bench.sh <n_prefill> <n_decode> <prefill_gpus> <decode_gpus> \
#            <model_dir> <model_name> <log_path> <isl> <osl> \
#            <concurrency_list> <req_rate> <random_range_ratio> <num_prompts_multiplier>

ENGINE="${ENGINE:-sglang-disagg}"

model_path=$1
model_name=$2
concurrency_list=${3:-"1"}
MODEL_PATH="${MODEL_PATH:-${model_path}/${model_name}}"
# vllm-disagg uses --served-model-name MODEL_NAME; sglang defaults to MODEL_PATH
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    MODEL="${MODEL_NAME:-${MODEL_PATH}}"
else
    MODEL="${MODEL_PATH}"
fi
log_path=${4:-/run_logs}

# Split BENCH_MAX_CONCURRENCY (x-delimited, e.g. "8x16x32") into an array.
# Falls back to 1 if unset so the loop always runs at least once.
IFS='x' read -r -a chosen_concurrencies <<< "${concurrency_list}"


ROUTER_PORT="${ROUTER_PORT:-30000}"

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

RESULT_DIR="${RESULT_DIR:-${log_path}/agentic}"
mkdir -p "$RESULT_DIR"

source "$(dirname "$0")/../../benchmark_lib.sh"

# clear_kv_caches — wipe all KV cache tiers on every backend worker before a
# concurrency point, so each conc is measured cold (no prefix reuse bleeding in
# from the previous conc). Mirrors mori-scheduler/scripts/benchmark/lib/
# clear_caches.sh, but the worker base URLs are already resolved by
# server_sglang.sh (SERVER_FLUSH_URLS_CSV) so no SSH/IP lookup is needed.
#
# Tiers (SGLang server APIs), hit on EACH worker directly (the router does not
# fan /flush_cache out):
#   L1 (GPU radix) + L2 (host hicache): POST /flush_cache  — NO-OP while any
#       request is in flight, so we drain-retry until "Cache flushed" or
#       FLUSH_DRAIN_TIMEOUT (default 120s) elapses.
#   L3 (umbp / mooncake store):         POST /hicache/storage-backend/clear
#       — HTTP != 200 when L3 is off, tolerated.
# Best-effort: logs WARN, never hard-fails the sweep.
clear_kv_caches() {
    local drain_tmo="${FLUSH_DRAIN_TIMEOUT:-120}"
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

# REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

PORT="${ROUTER_PORT}"
MODEL="${MODEL:-${BENCH_MODEL}}"
DURATION="${DURATION:-1800}"
export MODEL DURATION MAX_MODEL_LEN
RESULT_DIR="${RESULT_DIR:-${profile_folder}}"
# Base name for the per-conc aggregate written by the existing
# utils.agentic.aggregation.process_agentic_result module.
# The workflow guard / upload steps expect a "${RESULT_FILENAME}_conc<N>.json"
# file per concurrency, so each concurrency below is always suffixed with
# _conc<N> (matching agentic_srt.sh on the gb200 path).
RESULT_FILENAME_BASE="${RESULT_FILENAME:-agentic_bench}"

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

    # Clear all KV cache tiers on every backend before this conc point so it is
    # measured cold (no prefix reuse from the previous conc). Default on; set
    # CLEAR_CACHE_BETWEEN_CONC=0 to disable. Best-effort — never fails the run.
    if [[ "${CLEAR_CACHE_BETWEEN_CONC:-1}" == "1" ]]; then
        echo "conc=$max_concurrency: clearing L1/L2/L3 on all backends (no server restart)"
        clear_kv_caches || echo "WARNING: cache clear had issues for conc=$max_concurrency" >&2
    fi

    # Mirror agentic_srt.sh (the srtctl/gb200 path): every concurrency writes
    # its artifacts into a conc_<N>/ subdir of RESULT_DIR. The CI matrix explodes
    # agentic runs to one concurrency per job, but benchmark-multinode-tmpl.yml
    # still expects the per-conc nesting (LOGS/agentic/conc_*/...) and the
    # _conc<N> result-file suffix, so we always nest to keep the layout identical
    # across runners and avoid overwriting earlier runs in local multi-conc sweeps.
    CONC_RESULT_DIR="$RESULT_DIR/conc_${max_concurrency}"
    mkdir -p "$CONC_RESULT_DIR"

    CONC="$max_concurrency"
    USERS="$max_concurrency"
    export CONC USERS
    build_replay_cmd "$CONC_RESULT_DIR"

    # Per-conc result name consumed by write_agentic_result_json. Always suffix
    # with _conc<N> so the file matches
    # the workflow guard's "${RESULT_FILENAME}_conc*.json" glob (and the agg /
    # checkpoint upload steps) for both single-conc CI runs and multi-conc sweeps.
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
