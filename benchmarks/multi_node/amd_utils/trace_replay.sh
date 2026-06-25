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

# REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

PORT="${ROUTER_PORT}"
MODEL="${MODEL:-${BENCH_MODEL}}"
DURATION="${DURATION:-1800}"
export MODEL DURATION MAX_MODEL_LEN
RESULT_DIR="${RESULT_DIR:-${profile_folder}}"
# Base name for the per-conc agg JSON written by process_agentic_result.py.
# CI runs a single concurrency per job and the workflow checks for exactly
# "${RESULT_FILENAME}.json", so the base is only suffixed for local multi-conc
# sweeps (below) to keep each concurrency's result file distinct.
RESULT_FILENAME_BASE="${RESULT_FILENAME:-agentic_bench}"

mkdir -p "$RESULT_DIR"

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_061526
resolve_trace_source
install_agentic_deps

ANY_FAILED=0
for max_concurrency in "${chosen_concurrencies[@]}"; do

    echo "=========================================="
    echo "Agentic trace replay: conc=$max_concurrency"
    echo "=========================================="

    # Mirror agentic_srt.sh (the srtctl/gb200 path): a single concurrency writes
    # its artifacts flat into RESULT_DIR so the agentic/ layout is identical
    # across runners and the CI upload paths (LOGS/agentic/...) line up without
    # runner-side flattening. The CI matrix explodes agentic runs to one
    # concurrency per job, so this is the normal path; only local multi-conc
    # sweeps keep the conc<N>/ nesting to avoid overwriting earlier runs.
    if [ "${#chosen_concurrencies[@]}" -gt 1 ]; then
        CONC_RESULT_DIR="$RESULT_DIR/conc${max_concurrency}"
    else
        CONC_RESULT_DIR="$RESULT_DIR"
    fi
    mkdir -p "$CONC_RESULT_DIR"

    CONC="$max_concurrency"
    USERS="$max_concurrency"
    export CONC USERS
    build_replay_cmd "$CONC_RESULT_DIR"

    # Per-conc result name consumed by write_agentic_result_json /
    # process_agentic_result.py. Keep the unsuffixed base for single-conc (CI)
    # runs so the workflow's "${RESULT_FILENAME}.json" check matches; suffix
    # with _conc<N> only for multi-conc sweeps so results don't overwrite.
    if [ "${#chosen_concurrencies[@]}" -gt 1 ]; then
        export RESULT_FILENAME="${RESULT_FILENAME_BASE}_conc${max_concurrency}"
    else
        export RESULT_FILENAME="$RESULT_FILENAME_BASE"
    fi
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
