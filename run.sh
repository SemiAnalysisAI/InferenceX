export MODEL="/dev/shm/DeepSeek-V4-Pro/"
export TP=8
export CONC=32
has_arg() {
    local target="$1"
    shift
    for arg in "$@"; do
        if [[ "$arg" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

if has_arg d "$@"; then
    export DP_ATTENTION=true
    echo "DP_ATTENTION=true"
else
    export DP_ATTENTION=false
    echo "DP_ATTENTION=false"
fi
export ISL=8192
if has_arg p "$@"; then
    export PROFILE=1
    export ZCNT=1
    export OSL=40
    echo "PROFILE=$PROFILE ZCNT=$ZCNT OSL=$OSL"
else
    unset PROFILE
    export ZCNT=2
    export OSL=1024
    echo "PROFILE=$PROFILE ZCNT=$ZCNT OSL=${OSL}"
fi
export RANDOM_RANGE_RATIO=1.0
export RANDOM_RANGE_RATIO=0.8
export RESULT_FILENAME="dsv4_fp4_mi355x_sglang_tp16_conc16_dp_attention_isl8192_osl1024_random_range_ratio0.8"
export EP_SIZE=1
rm /workspace/profiles/*
#export EVAL_ONLY=true
#export RUN_EVAL=true

export SGLANG_DEBUG_DSV4_ATTN=1
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/profiles}"
export SGLANG_PROFILE_WITH_STACK=False
export TORCH_PROFILER_RECORD_SCOPES="${TORCH_PROFILER_RECORD_SCOPES:-USER_SCOPE}"
export MAX_MODEL_LEN=$((ISL+OSL))
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
pkill -9 tail
pkill -9 python
pkill -9 sglang

bash benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh
