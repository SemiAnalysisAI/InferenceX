#!/usr/bin/env bash
set -euo pipefail
set -x

# MiniMax-M3 MXFP8 H100 AgentX (agentic-coding) recipe with EAGLE3 speculative
# decoding — the spec-decoding=mtp variant of agentic/minimaxm3_fp8_h100.sh.
# Everything outside the speculative block mirrors the non-MTP agentic sibling
# (Mooncake host-DRAM KV offload, --block-size 128, --language-model-only,
# --kv-cache-dtype fp8, TRITON_ATTN, minimax_m3 parsers, vllm-router for
# DP-attention), so the spec-decode delta is readable at equal concurrency.
#
# Speculative config: Inferact/MiniMax-M3-EAGLE3 draft head, 3 speculative
# tokens — the same draft/level every merged MiniMax-M3 MTP recipe uses
# (fixed_seq_len/minimaxm3_fp8_{h100,h200,mi300x,mi325x}_mtp.sh).
#
# The drafter is pinned to FLASH_ATTN, as on every CUDA MiniMax-M3 MTP recipe:
# the EAGLE3 head is MHA and FlashInfer only serves page size 128 through its
# trtllm-gen kernel, which requires GQA/MQA. FLASH_ATTN accepts any
# multiple-of-16 block size, so the mandatory 128 is fine for the draft. (The
# ROCm recipes need no pin because their server runs TRITON_ATTN throughout.)
#
# Throughput runs pin synthetic acceptance to the committed golden AL; the
# EVAL_ONLY accuracy run keeps real target verification. See SYNTHETIC_ACCEPT_LEN.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"

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

# The EAGLE3 draft is never pre-staged next to the target checkpoint; fetch it
# into the shared HF cache. That cache is a network FS where concurrent
# day-zero downloads hit huggingface_hub's WeakFileLock "[Errno 116] Stale file
# handle" race, so retry (the download resumes) as the fixed-seq-len MTP
# recipes do.
for attempt in 1 2 3 4 5; do
    hf download "$DRAFT_MODEL" && break
    if [ "$attempt" = 5 ]; then echo "hf download of $DRAFT_MODEL failed after $attempt attempts" >&2; exit 1; fi
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
done
nvidia-smi

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONNOUSERSITE=1

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
MODEL_CPU_OFFLOAD_GB=26
MODEL_CHECKPOINT_PAGE_CACHE_GIB=414
MOONCAKE_LOCAL_BUFFER_GIB=4
if require_agentic_kv_offload_backend mooncake; then
        TOTAL_CPU_DRAM_GIB=$((TOTAL_CPU_DRAM_GB * 1000000000 / 1073741824))
        PER_RANK_GIB=$(((TOTAL_CPU_DRAM_GIB - MODEL_CHECKPOINT_PAGE_CACHE_GIB) / TP - MODEL_CPU_OFFLOAD_GB - MOONCAKE_LOCAL_BUFFER_GIB))
        if (( PER_RANK_GIB <= 0 )); then
            echo "Error: CPU DRAM budget is too small for checkpoint cache, model, and KV offload" >&2
            exit 1
        fi
        MOONCAKE_VERSION=0.3.11.post1
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            --force-reinstall "mooncake-transfer-engine-cuda13==$MOONCAKE_VERSION"
        python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GIB}GB",
  "local_buffer_size": "${MOONCAKE_LOCAL_BUFFER_GIB}GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
EOF
        export MOONCAKE_CONFIG_PATH PYTHONHASHSEED=0 MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio=0.80 \
            --eviction_ratio=0.10 > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        kill -0 "$MOONCAKE_MASTER_PID"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        )
fi

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
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))

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
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASH_ATTN\"}"
else
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASH_ATTN\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

# AgentX concurrency counts live session trees, not individual requests, so keep
# the non-MTP recipe's 2x scheduler headroom for subagent fan-out.
MAX_NUM_SEQS=$((2 * CONC))
# Cudagraph capture sizes are in TOKENS: a decode batch of S sequences verifies
# S*(1+NUM_SPEC_TOKENS) tokens, so cap capture at MAX_NUM_SEQS*(1+N) or the
# FULL_DECODE_ONLY ladder tops out at MAX_NUM_SEQS/(1+N) sequences and the
# largest decode batches fall back to eager.
MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * TOKENS_PER_SEQ))

vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_BACKEND_PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --gpu-memory-utilization 0.95 \
    --cpu-offload-gb "$MODEL_CPU_OFFLOAD_GB" \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --block-size 128 \
    --language-model-only \
    --enable-prefix-caching \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE" \
    --speculative-config "$SPEC_CONFIG" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
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
