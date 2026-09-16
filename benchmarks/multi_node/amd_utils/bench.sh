#!/bin/bash
# Disaggregated fixed-seq-len benchmark runner; writes JSON results via
# benchmark_serving.py for the CI pipeline.
#
# Usage: bash bench.sh <n_prefill> <n_decode> <prefill_gpus> <decode_gpus> \
#            <model_dir> <model_name> <log_path> <isl> <osl> \
#            <concurrency_list> <req_rate> <random_range_ratio> <num_prompts_multiplier>

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
check_env_vars ENGINE MODEL_PATH MODEL_NAME ROUTER_PORT
if [[ $# -ne 13 ]]; then
    echo "Error: bench.sh requires 13 positional arguments" >&2
    exit 1
fi

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
# vllm-disagg uses --served-model-name MODEL_NAME; sglang defaults to MODEL_PATH
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    BENCH_MODEL="${MODEL_NAME}"
else
    BENCH_MODEL="${MODEL_PATH}"
fi
log_path=$7

chosen_isl=${8}
chosen_osl=${9}
concurrency_list=${10}
chosen_req_rate=${11}
random_range_ratio=${12}
num_prompts_multiplier=${13}

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

profile_folder="${log_path}/${ENGINE}_isl_${chosen_isl}_osl_${chosen_osl}"
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

    extra_flags=""
    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        extra_flags="--trust-remote-code --tokenizer $MODEL_PATH"
    elif [[ "$ENGINE" == "atom-disagg" ]]; then
        extra_flags="--trust-remote-code --tokenizer $MODEL_PATH"
        if [ "$IS_MTP" = "true" ]; then
            # just override extra_flags as dsv3 use different tokenizer path
            if [[ "$MODEL_NAME" == DeepSeek-V4-Pro* ]]; then
                extra_flags="--dsv4"
            else
                extra_flags="--use-chat-template"
            fi
        fi
    else
        if [ "$IS_MTP" = "true" ]; then
            if [[ "$MODEL_NAME" == DeepSeek-V4-Pro* ]]; then
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

    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        echo "[BENCH] Cooldown: waiting 10s for idle KV block reaper..."
        sleep 10
    fi
done
