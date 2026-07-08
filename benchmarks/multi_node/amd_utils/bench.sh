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
    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        # vLLM disagg serves --served-model-name=$MODEL_NAME. The workflow's
        # MODEL env is the HF repo id (e.g. amd/Kimi-K2.5-MXFP4), which vLLM's
        # OpenAI endpoint rejects unless it matches the served model name. Keep
        # MODEL as result metadata and use AIPERF_MODEL only for the request body.
        export AIPERF_MODEL="${AIPERF_MODEL:-${SERVED_MODEL_NAME:-${MODEL:-${BENCH_MODEL}}}}"
    fi
    export DURATION="${DURATION:-1800}"
    export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
    export AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-/workspace}"
    export RESULT_FILENAME="${RESULT_FILENAME:-agentic_bench}"

    RESULT_DIR="${RESULT_DIR:-/run_logs/slurm_job-${SLURM_JOB_ID}/LOGS/agentic}"
    mkdir -p "$RESULT_DIR"

    resolve_trace_source
    install_agentic_deps

    # Multinode agentic matrix entries carry a single concurrency, but keep
    # the loop so local one-off runs can pass a small x-separated list.
    replay_failed=0
    for max_concurrency in "${chosen_concurrencies[@]}"; do
        export CONC="$max_concurrency"
        export USERS="$max_concurrency"
        build_replay_cmd "$RESULT_DIR"
        run_agentic_replay_and_write_outputs "$RESULT_DIR" || replay_failed=1

        if [[ "$ENGINE" == "vllm-disagg" ]]; then
            echo "[BENCH] Cooldown: waiting 10s for idle KV block reaper..."
            sleep 10
        fi
    done

    exit "$replay_failed"
fi

if (( chosen_isl <= 0 || chosen_osl <= 0 )); then
    echo "ERROR: fixed-seq benchmark received invalid ISL/OSL=${chosen_isl}/${chosen_osl}." >&2
    echo "ERROR: If this is an agentic-coding job, IS_AGENTIC was not propagated into the benchmark container." >&2
    exit 1
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
            if [[ "$MODEL_NAME" == "DeepSeek-V4-Pro" ]]; then
                extra_flags="--dsv4"
            else
                extra_flags="--use-chat-template"
            fi
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
