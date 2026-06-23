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

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${MODEL_PATH:-${model_path}/${model_name}}"
# vllm-disagg uses --served-model-name MODEL_NAME; sglang defaults to MODEL_PATH
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    BENCH_MODEL="${MODEL_NAME:-${MODEL_PATH}}"
else
    BENCH_MODEL="${MODEL_PATH}"
fi
log_path=$7

chosen_isl=${8:-1024}
chosen_osl=${9:-1024}
concurrency_list=${10:-"512x1"}
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    chosen_req_rate=${11:-inf}
else
    chosen_req_rate=${11:-1}
fi
random_range_ratio=${12:-0.8}
num_prompts_multiplier=${13:-10}

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

ROUTER_PORT="${ROUTER_PORT:-30000}"

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

profile_folder="${log_path}/${ENGINE}_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p "$profile_folder"

source "$(dirname "$0")/../../benchmark_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
    export PORT="${ROUTER_PORT}"
    export MODEL="${MODEL:-${BENCH_MODEL}}"
    export DURATION="${DURATION:-1800}"
    export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
    export AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-/workspace}"

    RESULT_DIR="${RESULT_DIR:-/workspace/LOGS/agentic}"
    RESULT_FILENAME_BASE="${RESULT_FILENAME:-agentic_bench}"
    mkdir -p "$RESULT_DIR"

    resolve_trace_source
    install_agentic_deps

    ANY_FAILED=0
    FIRST_RESULT_FILE=""
    for max_concurrency in "${chosen_concurrencies[@]}"; do
        echo "=========================================="
        echo "Agentic trace replay: conc=$max_concurrency"
        echo "=========================================="

        CONC_RESULT_DIR="$RESULT_DIR/conc${max_concurrency}"
        mkdir -p "$CONC_RESULT_DIR"

        export CONC="$max_concurrency"
        export USERS="$max_concurrency"
        build_replay_cmd "$CONC_RESULT_DIR"
        echo "$REPLAY_CMD" > "$CONC_RESULT_DIR/benchmark_command.txt"

        set +e
        $REPLAY_CMD 2>&1 | tee "$CONC_RESULT_DIR/benchmark.log"
        REPLAY_RC=${PIPESTATUS[0]}
        set -e

        PER_CONC_RESULT_FILENAME="${RESULT_FILENAME_BASE}_conc${max_concurrency}"
        RESULT_DIR="$CONC_RESULT_DIR" \
            AGENTIC_OUTPUT_DIR="$AGENTIC_OUTPUT_DIR" \
            RESULT_FILENAME="$PER_CONC_RESULT_FILENAME" \
            USERS="$max_concurrency" \
            python3 "$INFMAX_CONTAINER_WORKSPACE/utils/process_agentic_result.py" || {
                echo "WARNING: process_agentic_result.py failed for conc=$max_concurrency" >&2
                ANY_FAILED=1
            }

        PER_CONC_RESULT_FILE="$AGENTIC_OUTPUT_DIR/${PER_CONC_RESULT_FILENAME}.json"
        if [[ -z "$FIRST_RESULT_FILE" && -f "$PER_CONC_RESULT_FILE" ]]; then
            FIRST_RESULT_FILE="$PER_CONC_RESULT_FILE"
        fi

        python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
            "$CONC_RESULT_DIR/aiperf_artifacts" -o "$CONC_RESULT_DIR" 2>&1 || true

        python3 "$INFMAX_CONTAINER_WORKSPACE/utils/generate_aiperf_plots.py" \
            "$CONC_RESULT_DIR" 2>&1 || true

        if [[ "$REPLAY_RC" -ne 0 ]]; then
            echo "WARNING: agentic trace replay for conc=$max_concurrency exited with code $REPLAY_RC after writing available results" >&2
            ANY_FAILED=1
        fi

        echo "-----------------------------------------"

        if [[ "$ENGINE" == "vllm-disagg" ]]; then
            echo "[BENCH] Cooldown: waiting 10s for idle KV block reaper..."
            sleep 10
        fi
    done

    # The multinode workflow checks for ${RESULT_FILENAME}.json. Keep the
    # per-concurrency artifacts and also provide the expected aggregate path.
    if [[ -n "$FIRST_RESULT_FILE" && ! -f "$AGENTIC_OUTPUT_DIR/${RESULT_FILENAME_BASE}.json" ]]; then
        cp "$FIRST_RESULT_FILE" "$AGENTIC_OUTPUT_DIR/${RESULT_FILENAME_BASE}.json"
    fi

    if [[ "$ANY_FAILED" -ne 0 ]]; then
        echo "WARNING: at least one conc had a non-zero exit; per-conc result files were still written when possible." >&2
    fi

    exit 0
fi

for max_concurrency in "${chosen_concurrencies[@]}"; do

    export_file="${profile_folder}/concurrency_${max_concurrency}_req_rate_${chosen_req_rate}_gpus_$((prefill_gpus+decode_gpus))_ctx_${prefill_gpus}_gen_${decode_gpus}"

    num_prompts=$(( max_concurrency * num_prompts_multiplier ))
    if [[ "$num_prompts" -lt 16 ]]; then
        num_prompts=16
    fi

    echo "profile_folder: $profile_folder"
    echo "max_concurrency: $max_concurrency"
    echo "chosen_req_rate: $chosen_req_rate"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "ROUTER_PORT: $ROUTER_PORT"
    echo "chosen_isl: $chosen_isl"
    echo "chosen_osl: $chosen_osl"
    echo "num_prompts: $num_prompts"
    echo "export_file: $export_file"

    # Engine-specific extra flags
    extra_flags=""
    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        extra_flags="--trust-remote-code --tokenizer $MODEL_PATH"
    else
        if [ "$IS_MTP" = "true" ]; then
            extra_flags="--use-chat-template"
        fi
    fi

    run_benchmark_serving \
        --bench-serving-dir "$REPO_ROOT" \
        --model "$BENCH_MODEL" \
        --port "$ROUTER_PORT" \
        --backend openai \
        --input-len "$chosen_isl" \
        --output-len "$chosen_osl" \
        --random-range-ratio "$random_range_ratio" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --result-filename "$export_file" \
        --result-dir /workspace/ \
        $extra_flags

    echo "-----------------------------------------"

    # vLLM: cooldown between rounds for idle KV block reaper
    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        echo "[BENCH] Cooldown: waiting 10s for idle KV block reaper..."
        sleep 10
    fi
done
