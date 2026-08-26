#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using ATOM with DSpark speculative decoding.
#
# Companion to kimik3_fp4_mi355x_mtp.sh, which runs the same checkpoint and the
# same concurrency points under vLLM, so the two arms are directly comparable.
#
# TP=8 ONLY, for the same reason as the vLLM arm: the MXFP4 checkpoint is
# 1.561 TB decimal (~195 GB/GPU across 8 GPUs of the 288 GB part), and TP=4
# would need ~390 GB/GPU and cannot load.
#
# The ATOM image is purpose-built for K3, so apply_k3_container_patches.sh is
# NOT sourced here -- that script reproduces a specific patched vLLM container
# byte-for-byte and does not apply to this stack.
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

if [[ -v SLURM_JOB_ID ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 is a 1.56 TB checkpoint and only fits at TP=8 on" >&2
    echo "       288 GB gfx950 parts (~195 GB/GPU). Got TP=$TP." >&2
    exit 1
fi

if [[ -v ROCR_VISIBLE_DEVICES ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

wait_for_amd_gpu_clean

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

# Require the ATOM Prometheus stream in every official result. AIPerf
# deduplicates this endpoint against its automatic localhost discovery.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound. Matches the vLLM K3 arm.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# VRAM space check
wait_for_amd_gpu_clean

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "ATOM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- KV offload -------------------------------------------------------------
# K3 is a hybrid: Kimi Delta Attention carries a per-request recurrent state
# alongside the paged KV. Both tiers are switched together here, because the
# state tier is what makes a resumed agentic turn cheap and the paged KV tier
# alone cannot restore one.
OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        # TOTAL_CPU_DRAM_GB is the AGGREGATE budget from the matrix generator.
        # LMCACHE_MAX_LOCAL_CPU_SIZE is per rank and every rank allocates its
        # own, so the aggregate is divided by TP as the agentic README
        # requires. Handing a rank the whole aggregate does not just overcommit
        # -- it never finishes pinning and hangs the launch partway through.
        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=True
        export LMCACHE_MAX_LOCAL_CPU_SIZE="$((TOTAL_CPU_DRAM_GB / TP))"
        # One chunk per hash block, so the KV grid and the state-checkpoint
        # grid coincide and the joint load aims both legs at one boundary.
        export LMCACHE_CHUNK_SIZE=256

        # OFFLOAD_PROFILE is deliberately left unset (default 0). The source
        # recipe sets it to 1, but that only turns on per-step offload
        # statistics in the connector, and the numbers behind this submission
        # were measured with it off. Noted here so the difference from the
        # recipe reads as a choice rather than an omission.

        # CPU state-offload tier for the KDA recurrent state.
        export OFFLOAD_STATE=1
        export OFFLOAD_STATE_STAGING_GROUPS=8
        export OFFLOAD_STATE_MIN_LOAD_TOKENS=0
        # Must be set: the staging buffer defaults to 2 chunks (8 MiB), one K3
        # state entry is 54.78 MiB, and a buffer too small to hold one entry
        # makes the tier decline to build -- one log line, then nothing
        # offloads, which reads exactly like a tier that is on and idle.
        export OFFLOAD_GPU_STAGING_CHUNKS=16

        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}"
        )
        ;;
    *)
        echo "Unsupported KV_OFFLOAD_BACKEND: $KV_OFFLOAD_BACKEND (expected empty or lmcache)" >&2
        exit 1
        ;;
esac

# ---- ATOM env ---------------------------------------------------------------
echo "Starting atom server..."
export PYTHONNOUSERSITE=1

# Required by ATOM: without it the aiter kernel logs flood the server log for
# the whole 3600 s replay.
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export AITER_SITUV2_A4W4=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_FLYDSL_STAGE2_FP8=1
export ATOM_MLA_MAX_SPLIT_PER_BATCH=256
# Anchor-only state checkpointing: the demand rung is 47% of checkpoint writes
# but reads back 2.8% of the time, against 85.2% for a prompt-end anchor, so it
# costs more in evictions than its reuse is worth on these traces.
export ATOM_STATE_CHECKPOINT_DEMAND=0

# ---- Per-concurrency knobs --------------------------------------------------
case "$CONC" in
    1|4)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=0
        STATE_CHECKPOINT_SLOTS=""
        ;;
    8)
        MAX_NUM_SEQS=16
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=1
        STATE_CHECKPOINT_SLOTS=16
        ;;
    10)
        MAX_NUM_SEQS=16
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.90
        ATOM_ENABLE_REPLAYSSM=0
        STATE_CHECKPOINT_SLOTS=16
        ;;
    *)
        echo "Unsupported CONC=$CONC" >&2
        exit 2
        ;;
esac
export ATOM_ENABLE_REPLAYSSM

# Extra in-GPU state checkpoint slots beyond the in-flight floor. Checkpoints
# and live requests share one pool, so without this the room to retain a
# checkpoint is whatever max-num-seqs happens to leave.
STATE_CKPT_ARGS=()
if [ -n "$STATE_CHECKPOINT_SLOTS" ]; then
    STATE_CKPT_ARGS=(--state-checkpoint-slots "$STATE_CHECKPOINT_SLOTS")
fi

# ---- Speculative ------------------------------------------------------------
# https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml
#  6 draft tokens -> AL 3.75
#  2 draft tokens -> AL 2.51
# https://github.com/ROCm/ATOM/pull/1948
if [ "$CONC" = 1 ]; then
    SPEC_DECODE_AL=3.75
    NUM_SPEC_TOKENS=6
else
    SPEC_DECODE_AL=2.51
    NUM_SPEC_TOKENS=2
fi
    
if [ "${EVAL_ONLY}" = "true" ]; then
    SPEC_ARGS=(
        --method dspark
        --draft-model Inferact/Kimi-K3-DSpark
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
    )
else
    SPEC_ARGS=(
        --method dspark
        --draft-model Inferact/Kimi-K3-DSpark
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
        --spec-decode-acceptance-length "$SPEC_DECODE_AL"
    )
fi
echo "SPEC_DECODE_AL=$SPEC_DECODE_AL NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS"

# ---- LLM server -------------------------------------------------------------
ATOM_CMD=(
    python -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --host 0.0.0.0
    --server-port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --kv_cache_dtype fp8
    --block-size 128
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enable_prefix_caching
    --state-checkpoint-interval-tokens -1
    "${STATE_CKPT_ARGS[@]}"
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
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
