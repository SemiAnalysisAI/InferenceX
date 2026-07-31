#!/usr/bin/env bash

# Qwen3.5-397B-A17B NVFP4 on four RTX PRO 6000 Blackwell GPUs, with the
# built-in MTP draft head driven through SGLang's EAGLE speculative path.
# SM120 has no trtllm-gen kernels, so the routed experts and the NVFP4 GEMMs
# run on FlashInfer CUTLASS and attention runs on FlashInfer. The node is
# PCIe-only, so collectives use plain NCCL rather than the custom all-reduce.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# copy. Either way, SERVE_MODEL is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    SERVE_MODEL="$MODEL_PATH"
else
    hf download "$MODEL"
    SERVE_MODEL="$MODEL"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

export SGLANG_ENABLE_JIT_DEEPGEMM=false
export PYTHONUNBUFFERED=1

SERVER_LOG=/workspace/server.log

# SGLang holds back total * (1 - mem-fraction-static) as slack and gives the
# rest to the KV and Mamba pools, so the MTP draft head (Qwen3_5ForCausalLMMTP,
# 4.05 GiB per rank on top of the 56.12 GiB target model) has to be paid for by
# raising the fraction, not lowering it. 0.65 and 0.75 both left the pools
# empty on this 96 GiB SKU once the draft's own Mamba state was accounted for.
# 0.85 came up but left only 4.56 GiB free per rank, too thin for a 16k prefill
# chunk, so 0.80 trades surplus KV (1.7M tokens, ~3x what concurrency 64 needs)
# for activation headroom.
MEM_FRAC_STATIC="${MEM_FRAC_STATIC:-0.80}"
CHUNKED_PREFILL_SIZE=$((ISL * 2))
MAX_PREFILL_TOKENS=$((ISL * 2))
# The client never opens more than CONC connections, so sizing the Mamba state
# pool for 128 requests (as the B300 recipe does) just strands memory that the
# draft head needs here.
MAX_RUNNING_REQUESTS="$CONC"
CONTEXT_LENGTH=$((ISL + OSL + 20))

# Default: recv every ~10 requests; if CONC >= 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
    SCHEDULER_RECV_INTERVAL=30
else
    SCHEDULER_RECV_INTERVAL=10
fi

if [[ "$EVAL_ONLY" == "true" ]]; then
    setup_eval_context
    CONTEXT_LENGTH="$EVAL_MAX_MODEL_LEN"
fi

echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
    --model-path "$SERVE_MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --data-parallel-size 1 \
    --ep-size "$EP_SIZE" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --moe-runner-backend flashinfer_cutlass \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 \
    --mamba-scheduler-strategy no_buffer \
    --disable-custom-all-reduce \
    --disable-radix-cache \
    --mem-fraction-static "$MEM_FRAC_STATIC" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --context-length "$CONTEXT_LENGTH" \
    --cuda-graph-max-bs-decode "$CONC" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL" \
    --stream-interval 20 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready \
    --port "$PORT" \
    --server-log "$SERVER_LOG" \
    --server-pid "$SERVER_PID"

# EAGLE-style spec decoding is trained against chat-formatted inputs, so the
# benchmark must send chat prompts or the acceptance rate silently collapses.
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --use-chat-template \
    --trust-remote-code

if [[ "$RUN_EVAL" == "true" ]]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
