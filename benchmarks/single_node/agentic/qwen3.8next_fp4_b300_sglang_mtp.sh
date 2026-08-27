#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.8-Flash-Next NVFP4 on B300 with SGLang
# native NEXTN MTP. Day-zero recipe; SGLang is the plan-of-record engine for
# this model (MODELS.md). Throughput uses the golden synthetic AL; evals retain
# real target-model verification.
#
# The checkpoint is RadixArk/Qwen3.8-Flash-Next-NVFP4 (126 GiB,
# quantization_config.quant_method = modelopt), so --quantization modelopt_fp4
# matches the same flag the Qwen3.5 NVFP4 sibling uses. The model ships native
# MTP modules (kept unquantized by the checkpoint's ignore list), so NEXTN
# needs no external drafter.
#
# TP1: the cookbook's verified single-node command for this model is --tp 1 on
# both Blackwell parts. 126 GiB of NVFP4 weights fit on one B300, so the
# model is not sharded and every rank-crossing collective disappears.

source "$(dirname "$0")/../../benchmark_lib.sh"

# Use the lightweight GSM8K eval instead of the AgentX SWE-bench default.
export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-10}

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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    TOTAL_CPU_DRAM_GB="$REQUESTED_HICACHE_TOTAL_GB"
    # SGLang applies --hicache-size independently to Qwen's target KV and
    # Mamba pools. Native NEXTN also creates a draft KV pool with the same
    # slot count; its one attention layer adds 1/15 of the target KV bytes.
    # Reserve 1 GB/rank for page alignment and enforce H * 31/15 per rank.
    HICACHE_ALIGNMENT_RESERVE_GB=$TP
    HICACHE_USABLE_TOTAL_GB=$((TOTAL_CPU_DRAM_GB - HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$HICACHE_USABLE_TOTAL_GB" -lt 1 ]; then
        echo "Error: insufficient DRAM after HiCache alignment reserve" >&2
        exit 1
    fi
    MAX_HICACHE_SIZE_GB=$((HICACHE_USABLE_TOTAL_GB * 15 / TP / 31))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -lt 1 ] || [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB outside 1..$MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
    PROJECTED_HICACHE_TOTAL_GB=$(((HICACHE_SIZE_GB * TP * 31 + 14) / 15 + HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$PROJECTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: projected HiCache use ${PROJECTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    echo "HiCache CPU pools: ${HICACHE_SIZE_GB} GB target + Mamba + 1/15 draft per rank across TP=${TP}; projected node total ${PROJECTED_HICACHE_TOTAL_GB} GB <= ${TOTAL_CPU_DRAM_GB} GB"
    CACHE_ARGS=(
        --page-size 64
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
        --hicache-write-policy write_through_selective
    )
fi

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

# TP4 needs parallel tokenization to keep 256k AgentX warmups below the client
# request timeout. Keep TP2 on SGLang's single-worker default: multi-tokenizer
# startup races with the TP2 HiCache shared-memory initialization path.
TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

# AgentX concurrency counts live session trees rather than individual HTTP
# requests. Leave room for subagent fan-out and avoid spending HBM on graphs
# above the batch sizes that remain useful for this long-context workload.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
# Keep server-side connections alive beyond AIPerf's 300-second client pool
# timeout so bursty AgentX trajectories cannot reuse a closing idle socket.
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

if [ "${EVAL_ONLY:-false}" != "true" ]; then
    # golden_al_distribution/qwen3.8next_mtp.yaml:
    # qwen3.8-flash-next-fp8.thinking_on[3] = 2.32.
    # --speculative-num-steps 3 with 4 draft tokens is 3 speculative tokens
    # per verification step, i.e. the MTP=3 cell. AgentX replays run with
    # thinking on, so the thinking_on row is the right one.
    export SGLANG_SIMULATE_ACC_LEN=2.32
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    # Verified flags from the SGLang cookbook playground for this model on
    # B300 / NVFP4 / single node. Quantization is read from the
    # checkpoint, so no --quantization flag; the hybrid GDN linear-attention
    # layers take their own backends rather than --attention-backend.
    --linear-attn-prefill-backend flashinfer
    --linear-attn-decode-backend flashinfer
    # float32, not the cookbook's bfloat16. With NEXTN enabled the GDN linear
    # attention backend routes verification through flashinfer's
    # gated_delta_rule_mtp, which asserts initial_state.dtype == torch.float32
    # and aborts CUDA graph capture on a bf16 SSM state:
    #   AssertionError: initial_state must be float32, got torch.bfloat16
    #   flashinfer/gdn_decode.py:761, via gdn_backend.py target_verify
    # Confirmed on the H200 arm; same backend, same kernel, same NEXTN here.
    --mamba-ssm-dtype float32
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --reasoning-parser auto
    # NEXTN silently resets --max-running-requests to 48 when it is unset, so
    # this must stay explicit and sized to the AgentX concurrency.
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --mem-fraction-static 0.80
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

capture_cache_metrics() {
    {
        echo "=== SGLang cache metrics snapshot $(date --iso-8601=seconds) ==="
        curl -fsS "http://localhost:$PORT/metrics" 2>/dev/null \
            | grep -E '^(sglang:(cache_hit_rate|cached_tokens_total|prompt_tokens_total|hicache_host_used_tokens|hicache_host_total_tokens|token_usage|num_requests_running|num_requests_waiting))' \
            || true
        echo "============================================================"
    } >> "$SERVER_LOG"
}

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

capture_cache_metrics
trap capture_cache_metrics EXIT

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
