#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Block size 128 is mandatory for MSA sparse attention. Use FP8 KV cache to
# reduce memory pressure and increase the available concurrency headroom.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

PROFILE_ARGS=()
benchmark_output_len="$OSL"
benchmark_random_range_ratio="$RANDOM_RANGE_RATIO"
benchmark_num_warmups="$((2 * CONC))"
if [ "${PROFILE:-0}" = "1" ]; then
    profile_token_budget="${M3_PROFILE_TOKEN_BUDGET:-8192}"
    profile_active_iterations=5
    profile_tail_margin=32
    case "$profile_token_budget" in
        8192|16384|32768)
            ;;
        *)
            echo "Invalid M3_PROFILE_TOKEN_BUDGET: $profile_token_budget" >&2
            exit 2
            ;;
    esac

    profile_phase="${M3_PROFILE_PHASE:-decode}"
    case "$profile_phase" in
        decode)
            profile_prefill_iterations=$(((ISL * CONC + profile_token_budget - 1) / profile_token_budget))
            profile_delay=$((profile_prefill_iterations + 16))
            # The first admitted request can decode while later requests are
            # still prefilling. Keep every request alive through the capture
            # window, then stop instead of generating the full benchmark OSL.
            profile_output_len=$((profile_delay + profile_active_iterations + profile_tail_margin))
            profile_description="a steady-state decode window after $profile_delay engine iterations"
            ;;
        prefill)
            profile_delay=0
            profile_output_len=64
            profile_description="the opening chunked-prefill window"
            ;;
        *)
            echo "Invalid M3_PROFILE_PHASE: $profile_phase" >&2
            exit 2
            ;;
    esac
    if [ "$profile_output_len" -gt "$OSL" ]; then
        echo "Profile output length $profile_output_len exceeds configured OSL $OSL; increase the token budget" >&2
        exit 2
    fi

    benchmark_num_prompts="$CONC"
    benchmark_num_warmups="$CONC"
    benchmark_output_len="$profile_output_len"
    benchmark_random_range_ratio="1.0"
    export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/tmp/inferencex-profile/${RESULT_FILENAME}}"
    rm -rf "$VLLM_TORCH_PROFILER_DIR"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"

    profiler_config="$(
        printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_with_flops":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":%d,"active_iterations":%d,"max_iterations":1}' \
            "$VLLM_TORCH_PROFILER_DIR" "$profile_delay" "$profile_active_iterations"
    )"
    PROFILE_ARGS=(
        --max-num-batched-tokens "$profile_token_budget"
        --profiler-config "$profiler_config"
        --compilation-config '{"cudagraph_mode":"NONE"}'
    )
    # ROCTracer does not expose every kernel launched inside a HIP graph.
    echo "Profiling $profile_description with a $profile_token_budget-token budget and exact output length $profile_output_len."
else
    benchmark_num_prompts="$((CONC * 10))"
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$benchmark_output_len" \
    --random-range-ratio "$benchmark_random_range_ratio" \
    --num-prompts "$benchmark_num_prompts" \
    --max-concurrency "$CONC" \
    --num-warmups "$benchmark_num_warmups" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
