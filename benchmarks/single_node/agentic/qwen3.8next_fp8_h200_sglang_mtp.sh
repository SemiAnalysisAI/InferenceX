#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-Flash-Next FP8 on H200 with SGLang NEXTN MTP. Hopper has no NVFP4
# tensor cores, so this arm serves Qwen/Qwen3.8-Flash-Next-FP8 (172.8 GiB);
# attention stays flashinfer (trtllm_mha is Blackwell-only).
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

# 256k-capped with-subagents corpus (470 traces): the unfiltered corpus has
# requests up to ~1M tokens the server would reject at this model's TP8 ceiling.
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
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

# Acceptance is pinned to the committed golden AL (golden_al_distribution/README.md).
# EVAL_ONLY leaves it off: simulated acceptance commits drafted tokens
# regardless of target logits and the eval would score ~0.
if [ "${EVAL_ONLY}" != "true" ]; then
    # golden_al_distribution/qwen3.8next_mtp.yaml: thinking_on[3] = 2.32
    # (3 speculative tokens per step; AgentX replays run with thinking on).
    export SGLANG_SIMULATE_ACC_LEN=2.32
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

{ set +x; } 2>/dev/null
# AgentX concurrency counts live session trees; leave room for subagent
# fan-out without spending HBM on graphs above useful batch sizes.
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
    # Cookbook H200 / FP8 low-latency flags: TP4 with EP4 shards the 512-expert
    # MoE with expert parallelism rather than sharding attention eight ways.
    --tp-size "$TP"
    --ep-size "$EP_SIZE"
    --dp-size 1
    --mem-fraction-static 0.85
    --chunked-prefill-size 8192
    --linear-attn-prefill-backend flashinfer
    --linear-attn-decode-backend flashinfer
    # float32, not the cookbook's bfloat16: with NEXTN the GDN backend verifies
    # through flashinfer's gated_delta_rule_mtp, which asserts
    # initial_state.dtype == torch.float32 and aborts CUDA graph capture
    # (flashinfer/gdn_decode.py:761 via gdn_backend.py target_verify).
    --mamba-ssm-dtype float32
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
