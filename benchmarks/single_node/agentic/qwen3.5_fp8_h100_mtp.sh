#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.5 FP8 on H100 with SGLang EAGLE MTP. 80 GB HBM3 keeps
# mem-fraction-static at 0.75; attention is flashinfer (trtllm_mha is
# Blackwell-only).
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR \
    DURATION EP_SIZE
check_env_vars EVAL_ONLY

SCHEDULER_RECV_INTERVAL=10

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
nvidia-smi

# H100 max_model_len caps at 131k. The unfiltered corpus has requests up to
# ~1M tokens the server would reject; the 256k-capped variant (470 traces)
# keeps the rejection rate low.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
    # Hybrid GDN/Mamba allocates one KV and one Mamba host pool per rank.
    HICACHE_HOST_POOL_COUNT="2"
    HICACHE_WRITE_POLICY="write_through_selective"
    HICACHE_SIZE_GB=$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))
    if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
        echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB, TP=$TP, HICACHE_HOST_POOL_COUNT=$HICACHE_HOST_POOL_COUNT" >&2
        exit 1
    fi
    echo "HiCache CPU pool: ${HICACHE_SIZE_GB} GB per rank per host pool across TP=${TP}, host_pool_count=${HICACHE_HOST_POOL_COUNT}"
    CACHE_ARGS=(
        --page-size 64
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
    )
fi

echo "Starting SGLang server..."
export PYTHONNOUSERSITE=1
export SGLANG_ENABLE_SPEC_V2=1

SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

# Acceptance is pinned to the committed golden AL (golden_al_distribution/README.md):
# 3.39 is qwen3.5_mtp.yaml at num_speculative_tokens=3, thinking_on.
# SGLANG_SIMULATE_ACC_TOKEN_MODE exists from SGLang v0.5.16, which is why the
# image is pinned there. EVAL_ONLY leaves it off: simulated acceptance commits
# drafted tokens regardless of target logits and the eval would score ~0.
if [ "${EVAL_ONLY}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path="$MODEL_PATH" --served-model-name="$MODEL"
    --host=0.0.0.0
    --port="$PORT"
    --served-model-name "Qwen/Qwen3.5-397B-A17B-FP8"
    --trust-remote-code
    --tensor-parallel-size="$TP"
    --data-parallel-size=1
    --expert-parallel-size="$EP_SIZE"
    --quantization fp8
    --kv-cache-dtype fp8_e4m3
    --mamba-ssm-dtype bfloat16
    --attention-backend flashinfer
    --enable-flashinfer-allreduce-fusion
    --mem-fraction-static 0.75
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --tokenizer-worker-num 6
    --tokenizer-path "$MODEL"
    --enable-metrics
    "${SPEC_ARGS[@]}"
    "${CACHE_ARGS[@]}"
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
