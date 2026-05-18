#!/bin/bash
#
# Client-side benchmark driver for ATOM disaggregated serving.
#
# Adapted from benchmarks/multi_node/amd_utils/bench.sh. The only meaningful
# differences are:
#   1. profile_folder uses an "atom_" prefix instead of "sglang_" so that
#      runners/launch_mi355x-amds.sh's collect_latest_results.py picks up
#      this framework's output via the "atom" prefix (NOTE: at the time of
#      writing, that helper only recognizes "sglang"/"vllm" prefixes; the
#      runner-side prefix list needs to be extended in a follow-up PR).
#   2. Uses the openai backend (atom ships an OpenAI-compatible server,
#      same as SGLang). No other client-side changes are required.

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${model_path}/${model_name}"
log_path=$7

chosen_isl=${8:-1024}
chosen_osl=${9:-1024}
concurrency_list=${10:-"1x1"}
chosen_req_rate=${11:-inf}
random_range_ratio=${12:-0.8}
num_prompts_multiplier=${13:-10}

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

head_node="localhost"
head_port="30000"

profile_folder="${log_path}/atom_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p "$profile_folder"

source "$(dirname "$0")/../../benchmark_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

for max_concurrency in "${chosen_concurrencies[@]}"; do

    export_file="${profile_folder}/concurrency_${max_concurrency}_req_rate_${chosen_req_rate}_gpus_$((prefill_gpus+decode_gpus))_ctx_${prefill_gpus}_gen_${decode_gpus}"

    echo "profile_folder: $profile_folder"
    echo "max_concurrency: $max_concurrency"
    echo "chosen_req_rate: $chosen_req_rate"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "head_port: $head_port"
    echo "chosen_isl: $chosen_isl"
    echo "chosen_osl: $chosen_osl"
    echo "export_file: $export_file"

    run_benchmark_serving \
        --bench-serving-dir "$REPO_ROOT" \
        --model "$MODEL_PATH" \
        --port "$head_port" \
        --backend openai \
        --input-len "$chosen_isl" \
        --output-len "$chosen_osl" \
        --random-range-ratio "$random_range_ratio" \
        --num-prompts $(( max_concurrency * num_prompts_multiplier )) \
        --max-concurrency "$max_concurrency" \
        --result-filename "$export_file" \
        --result-dir /workspace/

    echo "-----------------------------------------"
done
