#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for MiniMax-M3 FP4 on MI355X using vLLM
# EAGLE3 speculative decoding.
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, CONC, KV_OFFLOADING,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

# Force the eval framework to lm-eval for this recipe. run_eval derives its
# default as swebench for agentic scenarios (scenario_default=swebench when
# IS_AGENTIC/SCENARIO_TYPE=agentic-coding), but EVAL_FRAMEWORK takes precedence
# over that default (benchmark_lib.sh: framework=${EVAL_FRAMEWORK:-...}), so
# setting it here makes the effective framework always lm-eval, never swebench.
export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
MODEL_REVISION="b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a"
DRAFT_MODEL_REVISION="96692486b5fd38ebf8fd2a5f6bb53427d30819a8"
NUM_SPEC_TOKENS=4
# golden_al_distribution/minimaxm3_eagle3_gqa.yaml:
# minimax-m3.thinking_on[4]
SYNTHETIC_ACCEPT_LEN=3.02

if [[ "$TP" != "4" ]]; then
    echo "This validated recipe supports TP4 only; got TP=$TP." >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ID+x}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [[ -n "${ROCR_VISIBLE_DEVICES+x}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

checkpoint_is_complete() {
    local dir="$1"
    [[ -d "$dir" && -f "$dir/config.json" ]] || return 1
    CKPT_DIR="$dir" python3 - <<'PYEOF'
import glob
import json
import os
import sys

directory = os.environ["CKPT_DIR"]
index = os.path.join(directory, "model.safetensors.index.json")
if os.path.isfile(index):
    with open(index) as handle:
        shards = sorted(set(json.load(handle)["weight_map"].values()))
    missing = [name for name in shards if not os.path.isfile(os.path.join(directory, name))]
    if missing:
        print(f"{len(missing)}/{len(shards)} shards missing, e.g. {missing[:3]}", file=sys.stderr)
        sys.exit(1)
elif not glob.glob(os.path.join(directory, "*.safetensors")):
    print("no shard index and no .safetensors present", file=sys.stderr)
    sys.exit(1)
PYEOF
}

MODEL_REVISION_ARGS=()
if [[ -n "${MODEL_REVISION:-}" ]]; then
    MODEL_REVISION_ARGS=(--revision "$MODEL_REVISION")
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if ! checkpoint_is_complete "$MODEL_PATH"; then
        hf download "$MODEL" "${MODEL_REVISION_ARGS[@]}" --local-dir "$MODEL_PATH"
    fi
    checkpoint_is_complete "$MODEL_PATH" || {
        echo "Error: $MODEL_PATH is incomplete after hf download $MODEL." >&2
        exit 1
    }
else
    hf download "$MODEL" "${MODEL_REVISION_ARGS[@]}"
    export MODEL_PATH="$MODEL"
fi

DRAFT_MODEL_REVISION_ARGS=(--revision "$DRAFT_MODEL_REVISION")
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$DRAFT_MODEL}"
if [[ "$DRAFT_MODEL_PATH" == "$DRAFT_MODEL" ]]; then
    hf download "$DRAFT_MODEL" "${DRAFT_MODEL_REVISION_ARGS[@]}"
elif ! checkpoint_is_complete "$DRAFT_MODEL_PATH"; then
    hf download "$DRAFT_MODEL" "${DRAFT_MODEL_REVISION_ARGS[@]}" --local-dir "$DRAFT_MODEL_PATH"
    checkpoint_is_complete "$DRAFT_MODEL_PATH" || {
        echo "Error: $DRAFT_MODEL_PATH is incomplete after hf download $DRAFT_MODEL." >&2
        exit 1
    }
fi

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

# Require the vLLM Prometheus stream in every official result. AIPerf
# deduplicates this endpoint against its automatic localhost discovery.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"

# Agentic sessions reuse one pooled HTTP connection across turns. Keep the
# server-side socket alive longer than the longest expected inter-turn gap and
# match AIPerf's TCP user timeout so a stale connection cannot turn an otherwise
# healthy long run into a ClientOSError.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-900}"
export AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT:-900000}"

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# AgentX replays growing multi-turn prefixes, so keep prefix caching enabled
# for both GPU-resident and native-offload configurations.
OFFLOAD_ARGS=()

case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_OFFLOAD_BYTES=$((TOTAL_CPU_DRAM_GB * 1024 * 1024 * 1024))
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_CONFIG=$(printf \
            '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%d,"lazy_offload":true}}' \
            "$CPU_OFFLOAD_BYTES")
        OFFLOAD_ARGS=(--kv-transfer-config "$OFFLOAD_CONFIG")
        ;;
    *)
        echo "Unsupported KV_OFFLOAD_BACKEND: ${KV_OFFLOAD_BACKEND:-}" >&2
        exit 1
        ;;
esac

# ---- LLM server config ----------------------------------------------------------
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

# Synthetic acceptance standardizes throughput against the committed golden
# EAGLE3-GQA curve. Accuracy evals use real target verification.
if [ "${EVAL_ONLY}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"ROCM_AITER_UNIFIED_ATTN\", \"draft_attention_window\": 32768}"
else
    SPEC_CONFIG="{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"ROCM_AITER_UNIFIED_ATTN\", \"draft_attention_window\": 32768, \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi
SPECULATIVE_ARGS=(--speculative-config "$SPEC_CONFIG")

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256
export VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048
export VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY=1
export VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT=1
export VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL=aiter
export VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER=aiter
export AITER_UNIFIED_ATTN_SLIDING_DECODE_3D=1
export VLLM_MINIMAX_M3_FUSED_CACHE_INSERT=1
export VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA=1
export VLLM_MINIMAX_M3_ROCM_FP32_ROUTER_GEMM=0
export VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP=1
export VLLM_MINIMAX_M3_ASM_SPARSE_ATTN=0
export AIPERF_WARMUP_REQUESTS_PER_LANE=10

GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_BATCHED_TOKENS=32768
MAX_NUM_SEQS=256
KV_CACHE_DTYPE=fp8
MAX_CUDAGRAPH_CAPTURE_SIZE=512
ATTENTION_CONFIG_ARGS=(--attention-config '{"indexer_kv_dtype":"fp8"}')

bash "$(dirname "$0")/apply_minimaxm3_agentx_patches.sh"
AITER_ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("aiter").origin)))')"
export AITER_FUSED_CACHE_INSERT_OVERLAY="$AITER_ROOT/aiter_meta"
export PYTHONPATH="$AITER_FUSED_CACHE_INSERT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
python3 "$(dirname "$0")/precompile_minimaxm3_aiter.py" --max-model-len 1048576

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    "${PARALLEL_ARGS[@]}"
    --trust-remote-code
    --block-size 128
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-chunked-prefill
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --language-model-only
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --attention-backend ROCM_AITER_UNIFIED_ATTN
    "${ATTENTION_CONFIG_ARGS[@]}"
    --moe-backend aiter
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --tool-call-parser minimax_m3
    --reasoning-parser minimax_m3
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --jit-monitor-verbose
    --stream-interval 20
    --hf-overrides '{"text_config": {"use_index_cache": true, "index_topk_freq": 4}}'
    "${SPECULATIVE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
