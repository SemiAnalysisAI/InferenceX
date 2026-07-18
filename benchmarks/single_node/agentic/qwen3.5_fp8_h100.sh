#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on H100 using SGLang.
#
# H100 has 80 GB HBM3 (vs B300's 192 GB), so weights + KV fit tighter.
# Mem-fraction-static lowered to 0.75 and chunked-prefill-size halved to
# 8192 (mirrors fixed_seq_len/qwen3.5_fp8_h100.sh). Attention backend is
# flashinfer (sm_90); the trtllm_mha path is Blackwell-only.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-10}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
# H100 max_model_len caps at 131k (HBM-bound). The unfiltered with-subagents
# corpus has requests up to ~1M proxy tokens that the server would reject.
# Switch to the 256k-capped variant (470 traces, max in+out <= 256k); even
# at 131k context, the rejection rate is much lower than against the
# unfiltered corpus.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k

resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
    # Hybrid GDN/Mamba allocates one KV and one Mamba host pool per rank.
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    TOTAL_CPU_DRAM_GB="$REQUESTED_HICACHE_TOTAL_GB"
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
    MAX_HICACHE_SIZE_GB=$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB exceeds configured per-pool limit $MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
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

SGLANG_MULTI_TOKENIZER=/sgl-workspace/sglang/python/sglang/srt/managers/multi_tokenizer_mixin.py
if ! sed -n '/elif isinstance(output, BatchStrOutput):/,/input_token_logprobs_val=_extract_field_by_index/p' "$SGLANG_MULTI_TOKENIZER" \
    | grep -q 'cached_tokens_details=_extract_field_by_index'; then
    sed -i '/elif isinstance(output, BatchStrOutput):/,/input_token_logprobs_val=_extract_field_by_index/ {
        /cached_tokens=_extract_field_by_index(output, "cached_tokens", i),/a\
            cached_tokens_details=_extract_field_by_index(\
                output, "cached_tokens_details", i\
            ),
    }' "$SGLANG_MULTI_TOKENIZER"
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
    # --cuda-graph-max-bs "$CONC"
    # --max-running-requests "$CONC"
    # --max-prefill-tokens 8192
    # --chunked-prefill-size 8192
    --mem-fraction-static 0.75
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --tokenizer-worker-num 6
    --tokenizer-path "$MODEL"
    --enable-metrics
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
