#!/usr/bin/env bash
set -euo pipefail
set -x

# MiniMax-M3 MXFP8 MI300X (gfx942) AgentX (agentic-coding) recipe with EAGLE3
# speculative decoding — the spec-decoding=mtp variant of
# agentic/minimaxm3_fp8_mi300x.sh. Everything outside the speculative block
# mirrors the non-MTP agentic sibling (--block-size 128, --language-model-only, --kv-cache-dtype fp8, TRITON_ATTN,
# minimax_m3 parsers, VLLM_USE_BREAKABLE_CUDAGRAPH=0), so the spec-decode delta
# is readable at equal concurrency.
#
# One deliberate exception: --gpu-memory-utilization is 0.90, not the non-MTP
# sibling's 0.95. fixed_seq_len/minimaxm3_fp8_mi300x_mtp.sh passes no gmu flag at
# all, i.e. it runs vLLM's 0.90 default, and that is the proven MTP setting here.
# The H100 twin of this recipe died mid-warmup at 0.95 with torch.OutOfMemoryError
# on every rank once the EAGLE3 head and its KV were resident (run 30515793863).
#
# Speculative config: Inferact/MiniMax-M3-EAGLE3 draft head, 3 speculative
# tokens. Unlike the CUDA recipes the drafter needs no attention_backend
# override — the FlashInfer "page size 128 requires GQA/MQA" limitation that
# forces FLASH_ATTN for the MHA EAGLE3 head is FlashInfer/CUDA-specific, and
# here the whole server runs TRITON_ATTN, which serves the MHA draft fine
# (same reasoning as fixed_seq_len/minimaxm3_fp8_mi300x_mtp.sh).
#
# No in-place SupportsEagle3 patch: the pinned ROCm nightly is built from
# upstream 2026-06-23, after vllm-project/vllm#45546 (EAGLE3 support on the AMD
# MiniMax-M3 model) merged on 2026-06-14. The older vllm-openai-rocm:minimax-m3
# bring-up image still needs that runtime patch.
#
# Throughput runs pin synthetic acceptance to the committed golden AL; the
# EVAL_ONLY accuracy run keeps real target verification. See SYNTHETIC_ACCEPT_LEN.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

# The EAGLE3 draft is never pre-staged next to the target checkpoint; fetch it
# into the shared HF cache, retrying because that cache is a network FS where
# concurrent downloads hit huggingface_hub's WeakFileLock stale-handle race.
for attempt in 1 2 3 4 5; do
    hf download "$DRAFT_MODEL" && break
    if [ "$attempt" = 5 ]; then echo "hf download of $DRAFT_MODEL failed after $attempt attempts" >&2; exit 1; fi
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
done
rocm-smi || true
amd-smi || true

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTHONNOUSERSITE=1

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

# GPU-resident only: no Mooncake on AMD. The HIP-built Mooncake transfer engine's
# TCP transport collapses mid-run on gfx942 -- the store spews
# "ServerSession::readBody failed ... Connection reset by peer" until the replay
# aborts (run 30517360090, MI300X TP8 mtp dram c8). This recipe therefore carries
# no host-DRAM KV offload arm, and fails loudly rather than silently serving a
# GPU-resident run under a dram label.
if [[ "${KV_OFFLOADING}" != "none" ]]; then
    echo "Error: KV_OFFLOADING=$KV_OFFLOADING requested, but this recipe is GPU-resident only (Mooncake is not used on AMD)" >&2
    exit 1
fi
OFFLOAD_ARGS=()

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if (( EP_SIZE > 1 )); then
    EP_ARGS=(--enable-expert-parallel)
fi

VLLM_BACKEND_PORT="$PORT"
if [[ "$DP_ATTENTION" == "true" ]]; then
    VLLM_BACKEND_PORT=$((PORT + 1))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet 'vllm-router==0.1.14'
fi

# use 3 speculative tokens for all configs, matching the MiniMax-M3 MTP recipes
NUM_SPEC_TOKENS=3

# AgentX pins acceptance to the committed golden AL so submissions are compared
# on system performance at a fixed acceptance target rather than on draft-head
# quality (golden_al_distribution/README.md). 2.83 is the MiniMax-M3 EAGLE3
# curve at num_speculative_tokens=3, thinking_on
# (golden_al_distribution/minimaxm3_eagle3.yaml). The separate
# minimaxm3_eagle3_gqa.yaml curve belongs to the Inferact/MiniMax-M3-EAGLE3-GQA
# draft and is not mixed in here.
#
# EVAL_ONLY switches back to real verification: synthetic acceptance commits
# drafted tokens regardless of the target logits, so generated text is wrong and
# the eval would score ~0 (same split as dsv4_fp4_b*_vllm_mtp.sh).
SYNTHETIC_ACCEPT_LEN=2.83
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}"
else
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

# AgentX concurrency counts live session trees, not individual requests, so keep
# the non-MTP recipe's 2x scheduler headroom for subagent fan-out. Cudagraph
# capture is left at the sibling's default: the ROCm MiniMax-M3 MTP recipes run
# with VLLM_USE_BREAKABLE_CUDAGRAPH=0 and no explicit capture ceiling.
MAX_NUM_SEQS=$((2 * CONC))
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_BACKEND_PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --gpu-memory-utilization 0.90 \
    --block-size 128 \
    --language-model-only \
    --attention-backend TRITON_ATTN \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --speculative-config "$SPEC_CONFIG" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    "${OFFLOAD_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$DP_ATTENTION" == "true" ]]; then
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy consistent_hash \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$((PORT + 10000))" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
