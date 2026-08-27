#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.8-Flash-Next FP8 on H200 using
# SGLang with MTP speculative decoding. Day-zero recipe; SGLang is the
# plan-of-record engine for this model (MODELS.md), and it is spec-decode only,
# per the AgentX policy that new agentic arms ship with speculative decoding
# enabled rather than as an STP/MTP A/B.
#
# H200 is Hopper, so this arm is FP8 (Qwen/Qwen3.8-Flash-Next-FP8, 172.8 GiB at
# TP8) rather than the NVFP4 checkpoint the Blackwell arms use: NVFP4 needs
# SM100 tensor cores. Attention backend stays flashinfer (sm_90).
#
# Structure follows the proven H100 MTP AgentX replay path
# (HiCache host-DRAM offload, the multi_tokenizer cached_tokens_details patch,
# aiperf-driven trace replay). H200's 141 GB HBM3e is roomier than H100's 80 GB,
# so --mem-fraction-static is 0.8 rather than 0.75, matching
# fixed_seq_len/qwen3.5_fp8_h200_mtp.sh. Attention backend stays flashinfer
# (sm_90); the trtllm_mha path is Blackwell-only.
#
# Speculative decoding mirrors fixed_seq_len/qwen3.5_fp8_h100_mtp.sh:
# SGLANG_ENABLE_SPEC_V2=1 with --speculative-algorithm EAGLE, 3 steps, eagle-topk
# 1 and 4 draft tokens, i.e. 3 speculative tokens per verification step.
#
# Throughput runs pin acceptance to the committed golden AL through SGLang's
# simulated-acceptance path; the EVAL_ONLY accuracy run leaves it off and keeps
# real verification. See the SGLANG_SIMULATE_ACC_* block.
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
# Keep the 256k-capped with-subagents corpus the H100 Qwen3.5 AgentX recipe
# uses (470 traces, max in+out <= 256k). The unfiltered corpus has requests up
# to ~1M proxy tokens that the server would reject; H200's extra HBM raises the
# context ceiling but not past 256k for this model at TP8.
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
export SGLANG_ENABLE_SPEC_V2=1

# 3 speculative tokens per step (num-steps 3, eagle-topk 1, 4 draft tokens),
# the same MTP shape as the fixed-seq-len Qwen3.5 recipes.
SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

# AgentX pins acceptance to the committed golden AL so submissions are compared
# on system performance at a fixed acceptance target rather than on draft-head
# quality (golden_al_distribution/README.md). 3.39 is the Qwen3.5 MTP curve at
# num_speculative_tokens=3, thinking_on (golden_al_distribution/qwen3.5_mtp.yaml)
# -- the same value the GB300 Qwen3.5 AgentX srt-slurm recipes pin.
# SGLANG_SIMULATE_ACC_TOKEN_MODE landed in SGLang v0.5.16, which is why this
# recipe pins that image rather than the non-MTP agentic sibling's v0.5.12.
#
# EVAL_ONLY leaves simulated acceptance off: it commits drafted tokens
# regardless of the target logits, so generated text is wrong and the eval would
# score ~0.
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
# AgentX concurrency counts live session trees rather than individual HTTP
# requests. Leave room for subagent fan-out, and do not spend HBM capturing
# graphs above the batch sizes that stay useful for this long-context workload.
# The Qwen3.5 H200 template left both flags commented out, so neither variable
# existed; NEXTN silently caps --max-running-requests at 48 when it is unset.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
if [ "$CUDA_GRAPH_MAX_BS" -gt 64 ]; then
    CUDA_GRAPH_MAX_BS=64
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    # Verified flags from the SGLang cookbook playground for this model on
    # H200 / FP8 / low latency / single node. NVFP4 is greyed out for Hopper,
    # so FP8 is the whole verified surface here. TP4 with EP4, not TP8: the
    # cookbook shards the 512-expert MoE across four ranks with expert
    # parallelism rather than sharding attention eight ways.
    --tp-size "$TP"
    --ep-size "$EP_SIZE"
    --dp-size 1
    --mem-fraction-static 0.85
    --chunked-prefill-size 8192
    --linear-attn-prefill-backend flashinfer
    --linear-attn-decode-backend flashinfer
    --mamba-ssm-dtype bfloat16
    "${SPEC_ARGS[@]}"
    --reasoning-parser auto
    # NEXTN silently resets --max-running-requests to 48 when it is unset, so
    # this must stay explicit and sized to the AgentX concurrency.
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
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
