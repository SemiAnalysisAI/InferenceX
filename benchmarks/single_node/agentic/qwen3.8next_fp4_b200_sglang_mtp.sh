#!/usr/bin/env bash
set -eo pipefail
set -x

# Qwen3.8-Flash-Next NVFP4 on B200 with SGLang native NEXTN MTP. The
# RadixArk/Qwen3.8-Flash-Next-NVFP4 checkpoint (126 GiB, quant_method =
# modelopt) ships native MTP modules, so NEXTN needs no external drafter, and
# fits on one GPU, so the cookbook command is --tp 1.

source "$(dirname "$0")/../../benchmark_lib.sh"

# Use the lightweight GSM8K eval instead of the AgentX SWE-bench default.
export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY

SCHEDULER_RECV_INTERVAL=10

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

model_checkpoint_is_complete() {
    local directory="$1"
    local index="$directory/model.safetensors.index.json"

    [[ -f "$index" ]] || return 1
    [[ -z "$(find "$directory" -name '*.incomplete' -print -quit 2>/dev/null)" ]] || return 1
    python3 - "$directory" <<'PYEOF'
import json
import os
import sys

directory = sys.argv[1]
index = os.path.join(directory, "model.safetensors.index.json")
try:
    with open(index, encoding="utf-8") as index_file:
        shards = set(json.load(index_file)["weight_map"].values())
except (OSError, KeyError, TypeError, ValueError):
    sys.exit(1)
if not shards or any(not isinstance(shard, str) for shard in shards):
    sys.exit(1)
sys.exit(any(not os.path.isfile(os.path.join(directory, shard)) for shard in shards))
PYEOF
}

if [[ -z "${MODEL_PATH:-}" ]] || ! model_checkpoint_is_complete "$MODEL_PATH"; then
    echo "Error: complete staged Qwen3.8-Flash-Next NVFP4 checkpoint not found at ${MODEL_PATH:-<unset>}" >&2
    exit 1
fi
nvidia-smi

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
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
    HICACHE_SIZE_GB=$((HICACHE_USABLE_TOTAL_GB * 15 / TP / 31))
    if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
        echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB must be positive" >&2
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

# Parallel tokenization keeps 256k AgentX warmups below the client timeout (TP1 here).
TOKENIZER_ARGS=(--tokenizer-worker-num 6)

# AgentX concurrency counts live session trees; leave room for subagent
# fan-out without spending HBM on graphs above useful batch sizes.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$MAX_RUNNING_REQUESTS"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

MEM_FRACTION_STATIC=0.80
MAMBA_CACHE_ARGS=()
if [ "$CONC" -eq 16 ]; then
    MEM_FRACTION_STATIC=0.90
    MAMBA_CACHE_ARGS=(--max-mamba-cache-size 160)
fi

export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
# Keep server-side connections alive beyond AIPerf's 300-second client pool
# timeout so bursty AgentX trajectories cannot reuse a closing idle socket.
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

if [ "${EVAL_ONLY}" != "true" ]; then
    # golden_al_distribution/qwen3.8next_mtp.yaml: thinking_on[3] = 2.32
    # (3 speculative tokens per step; AgentX replays run with thinking on).
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
    # Quantization is read from the checkpoint, so no --quantization flag; the
    # hybrid GDN linear-attention layers take their own backends.
    --linear-attn-prefill-backend flashinfer
    --linear-attn-decode-backend flashinfer
    # SGLang rejects flashinfer linear-attn decode on SM100+ without
    # --mamba-ssm-dtype bfloat16; Hopper needs float32 instead (the
    # gated_delta_rule_mtp verify kernel asserts a float32 state).
    --mamba-ssm-dtype bfloat16
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --reasoning-parser auto
    # NEXTN silently resets --max-running-requests to 48 when it is unset, so
    # this must stay explicit and sized to the AgentX concurrency.
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    "${MAMBA_CACHE_ARGS[@]}"
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

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
