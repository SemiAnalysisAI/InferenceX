#!/bin/bash
# vLLM Disaggregated Benchmark Runner
#
# Produces JSON result files via benchmark_serving.py (same as SGLang bench.sh)
# so that the CI pipeline can collect and process results.
#
# Usage: bash bench.sh <n_prefill> <n_decode> <prefill_gpus> <decode_gpus> \
#            <model_dir> <model_name> <log_path> <isl> <osl> \
#            <concurrency_list> <req_rate> <random_range_ratio> <num_prompts_multiplier>

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${MODEL_PATH:-${model_path}/${model_name}}"
log_path=$7

chosen_isl=${8:-1024}
chosen_osl=${9:-1024}
concurrency_list=${10:-"512x1"}
chosen_req_rate=${11:-inf}
random_range_ratio=${12:-0.8}
num_prompts_multiplier=${13:-10}

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

ROUTER_PORT="${ROUTER_PORT:-30000}"

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

profile_folder="${log_path}/vllm_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p "$profile_folder"

source "$(dirname "$0")/../../benchmark_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

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

    run_benchmark_serving \
        --bench-serving-dir "$REPO_ROOT" \
        --model "$MODEL_PATH" \
        --port "$ROUTER_PORT" \
        --backend openai \
        --input-len "$chosen_isl" \
        --output-len "$chosen_osl" \
        --random-range-ratio "$random_range_ratio" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --result-filename "$export_file" \
        --result-dir /workspace/

    echo "-----------------------------------------"
done
