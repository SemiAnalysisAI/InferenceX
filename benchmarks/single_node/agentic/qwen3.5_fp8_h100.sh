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
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - SGLang GPU KV only (RadixAttention prefix cache stays on —
#             agentic workloads rely on >95% theoretical hit rate).
#   hicache - SGLang HiCache with local CPU hierarchical cache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-10}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
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
case "$OFFLOADING" in
    none)
        # Leave SGLang's default RadixAttention prefix cache on — agentic
        # replay needs it; --disable-radix-cache would zero the hit rate.
        ;;
    hicache)
        # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
        # H100 nodes typically expose ~1.5-2 TB usable CPU DRAM; Qwen3.5's
        # hybrid GDN/Mamba path allocates two HiCache host pools per TP rank
        # (one KV, one Mamba). Workflow passes a generic TOTAL_CPU_DRAM_GB, so
        # keep the per-rank-per-pool conversion local to this script.
        TOTAL_CPU_DRAM_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-1500}"
        HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
        HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))}"
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
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

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
    --model-path="$MODEL"
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

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
