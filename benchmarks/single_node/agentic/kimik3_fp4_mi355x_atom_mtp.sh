#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using ATOM with DSpark speculative decoding.
#
# Companion to kimik3_fp4_mi355x_mtp.sh, which runs the same checkpoint under
# vLLM, so the two arms are directly comparable.
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
#   MODEL, MODEL_PATH, TP, DCP_SIZE, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP DCP_SIZE=${DCP_SIZE:-1} CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

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

# ---- Per-concurrency knobs --------------------------------------------------
# Concurrency 1-4 is the latency floor: everything GPU-resident, no decode
# context parallelism, the deepest draft the golden curve publishes, and an
# 8192-token prefill step.
# From concurrency 8 up, decode is KV-bandwidth-bound over 100k+ token agentic
# contexts, so DCP8 shards the KV read across all 8 GPUs and the LMCache DRAM
# tier backs the paged KV.
#
# The LMCache paged-KV tier itself is not switched here: it follows
# kv-offloading in configs/amd-master.yaml, which is `none` for concurrency 1-4
# and `dram` from concurrency 8 up. What this block chooses is
# STATE_OFFLOAD_CPU_GIB, the per-rank slice of that CPU budget reserved for the
# Kimi Delta Attention recurrent state; 0 leaves the whole budget to the paged
# KV and keeps the state in GPU checkpoints instead.
case "$CONC" in
    # No KV offload; the working set fits in HBM.
    1|2|4)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=0
        STATE_CHECKPOINT_SLOTS=""
        NUM_SPEC_TOKENS=7
        SPEC_DECODE_AL=3.84
        STATE_OFFLOAD_CPU_GIB=0
        ;;
    # LMCache paged-KV tier on, whole per-rank budget; state stays in GPU
    # checkpoints, which is what ReplaySSM replays from.
    8)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=1
        STATE_CHECKPOINT_SLOTS=96
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=3.00
        STATE_OFFLOAD_CPU_GIB=0
        ;;
    12)
        MAX_NUM_SEQS=24
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=1
        STATE_CHECKPOINT_SLOTS=96
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=3.00
        STATE_OFFLOAD_CPU_GIB=0
        ;;
    # LMCache paged-KV tier plus ATOM's CPU state tier. Concurrency 16-40
    # reserves 32 GB/rank; the C56 causal control repartitions the same total
    # CPU budget to 64 GB/rank state plus 64 GB/rank paged KV.
    16)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        STATE_CHECKPOINT_SLOTS=""
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=3.00
        STATE_OFFLOAD_CPU_GIB=32
        ;;
    32)
        MAX_NUM_SEQS=64
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        STATE_CHECKPOINT_SLOTS=""
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        STATE_OFFLOAD_CPU_GIB=32
        ;;
    40)
        MAX_NUM_SEQS=80
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        STATE_CHECKPOINT_SLOTS=""
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        STATE_OFFLOAD_CPU_GIB=32
        ;;
    56)
        MAX_NUM_SEQS=72
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=0
        STATE_CHECKPOINT_SLOTS=""
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        STATE_OFFLOAD_CPU_GIB=64
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

# ---- KV offload -------------------------------------------------------------
# K3 is a hybrid: Kimi Delta Attention carries a per-request recurrent state
# alongside the paged KV. The paged KV rides this LMCache tier from
# concurrency 8 up; from concurrency 16 up the CPU state tier is switched on
# alongside it, because the state tier is what makes a resumed agentic turn
# cheap and the paged KV tier alone cannot restore one.
OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        # TOTAL_CPU_DRAM_GB is the AGGREGATE budget from the matrix generator.
        # LMCACHE_MAX_LOCAL_CPU_SIZE and OFFLOAD_STATE_CPU_SIZE are per rank and
        # every rank allocates its own, so the aggregate is divided by TP as the
        # agentic README requires. Handing a rank the whole aggregate does not
        # just overcommit -- it never finishes pinning and hangs the launch
        # partway through.
        PER_RANK_CPU_GB="$((TOTAL_CPU_DRAM_GB / TP))"
        LMCACHE_CPU_GB="$((PER_RANK_CPU_GB - STATE_OFFLOAD_CPU_GIB))"

        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=True
        export LMCACHE_MAX_LOCAL_CPU_SIZE="$LMCACHE_CPU_GB"
        # DCP-locked: the offload hash block is block-size(128) x dcp(8) = 1024,
        # so the KV grid and the state-checkpoint grid coincide and the joint
        # load aims both legs at one boundary. 512 or 2048 misaligns it.
        export LMCACHE_CHUNK_SIZE=1024
        export OFFLOAD_KV_FOR_HYBRID=1
        # Statistics only -- per-step offload counters in the connector. Kept on
        # because the submitted numbers were measured with it on.
        export OFFLOAD_PROFILE=1

        if [ "$STATE_OFFLOAD_CPU_GIB" -gt 0 ]; then
            # CPU state-offload tier for the KDA recurrent state.
            export OFFLOAD_STATE=1
            export OFFLOAD_STATE_CPU_SIZE="$STATE_OFFLOAD_CPU_GIB"
            export OFFLOAD_STATE_STAGING_GROUPS=8
            export OFFLOAD_STATE_MIN_LOAD_TOKENS=0
            # Must be set: the staging buffer defaults to 2 chunks (8 MiB), one
            # K3 state entry is 54.78 MiB, and a buffer too small to hold one
            # entry makes the tier decline to build -- one log line, then
            # nothing offloads, which reads exactly like a tier that is on and
            # idle.
            export OFFLOAD_GPU_STAGING_CHUNKS=32
        fi

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
# Anchor-only state checkpointing: the demand rung is 47% of checkpoint writes
# but reads back 2.8% of the time, against 85.2% for a prompt-end anchor, so it
# costs more in evictions than its reuse is worth on these traces.
export ATOM_STATE_CHECKPOINT_DEMAND=0

# ---- Speculative ------------------------------------------------------------
# https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml
#  7 draft tokens -> AL 3.84
#  3 draft tokens -> AL 3.00
# Concurrency 32 and up serve without a draft model: past the throughput knee
# the draft forward no longer pays for itself against the resident batch.
SPEC_ARGS=()
if [ "$NUM_SPEC_TOKENS" -gt 0 ]; then
    SPEC_ARGS=(
        --method dspark
        --draft-model Inferact/Kimi-K3-DSpark
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
    )
    if [ "${EVAL_ONLY}" != "true" ]; then
        SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
    fi
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
    --decode-context-parallel-size "${DCP_SIZE:-1}"
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
