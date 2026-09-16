#!/usr/bin/env bash
set -eo pipefail
set -x

# MiniMax-M3 MXFP4 on MI355X / MI350X (gfx950) with ATOM EAGLE3. Companion to
# minimaxm3_fp4_mi355x_mtp.sh (same checkpoint under vLLM). TP2/TP4 follow
# the official ATOM MXFP4 recipe; TP8 is accepted for larger-memory variants.
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, DCP_SIZE, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, RESULT_FILENAME, DURATION, EP_SIZE, DP_ATTENTION,
#   EVAL_ONLY, ENABLE_PREFIX_CACHING, AITER_LOG_LEVEL
# Eval-only runs also require EVAL_FRAMEWORK and, for lm-eval, EVAL_TASKS_DIR.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL MODEL_PATH TP DCP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR RESULT_FILENAME DURATION EP_SIZE DP_ATTENTION EVAL_ONLY ENABLE_PREFIX_CACHING AITER_LOG_LEVEL
if [[ "$KV_OFFLOADING" != "none" ]]; then
    check_env_vars KV_OFFLOAD_BACKEND
fi
if [[ "$EVAL_ONLY" == "true" ]]; then
    check_env_vars EVAL_FRAMEWORK
    if [[ "$EVAL_FRAMEWORK" == "lm-eval" || "$EVAL_FRAMEWORK" == "lm_eval" ]]; then
        check_env_vars EVAL_TASKS_DIR
    fi
fi

echo "MODEL=$MODEL TP=$TP DCP_SIZE=$DCP_SIZE CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

if [[ -n "${SLURM_JOB_ID+x}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [ "$TP" -ne 2 ] && [ "$TP" -ne 4 ] && [ "$TP" -ne 8 ]; then
    echo "Error: MiniMax-M3 MXFP4 supports TP2, TP4, or TP8 on 288 GB gfx950 parts." >&2
    exit 1
fi

if [[ -n "${ROCR_VISIBLE_DEVICES+x}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ "$MODEL_PATH" == "$MODEL" ]]; then
    hf download "$MODEL"
elif [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
fi

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
hf download "$DRAFT_MODEL"

wait_for_amd_gpu_clean

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps
# ATOM's server runs from the image's system venv, not the AIPerf venv from
# install_agentic_deps; MiniMax's tokenizer fallback needs these there.
ATOM_RUNTIME_DEPS=/tmp/inferencex-atom-runtime-deps
/opt/venv/bin/python -m pip install --quiet --target "$ATOM_RUNTIME_DEPS" --no-deps sentencepiece tiktoken

# Require the ATOM Prometheus stream in every official result.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

# Long agentic turns against a 1M context are prefill-bound on the server.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

wait_for_amd_gpu_clean

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

# Per-concurrency knobs. STATE_OFFLOAD_CPU_GIB is the per-rank slice of the
# CPU budget reserved for the MiniMax-M3 state; 0 leaves it all to the paged KV.
# MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS and GPU_MEM_UTIL are overridden below
# by the official ATOM launch settings.
case "$CONC" in
    1|2|4|5)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=2.78
        STATE_OFFLOAD_CPU_GIB=0
        ;;
    8|10|12|14|15|20|24|28)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=1
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=2.78
        STATE_OFFLOAD_CPU_GIB=0
        ;;
    # 32 GB/rank carved out for the attention state tier.
    16)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=2.78
        STATE_OFFLOAD_CPU_GIB=32
        ;;
    32)
        MAX_NUM_SEQS=64
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=2.78
        STATE_OFFLOAD_CPU_GIB=32
        ;;
    # No draft model past the throughput knee and no hybrid CPU state tier.
    40)
        MAX_NUM_SEQS=80
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        STATE_OFFLOAD_CPU_GIB=0
        ;;
    48)
        MAX_NUM_SEQS=96
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        STATE_OFFLOAD_CPU_GIB=0
        ;;
    56)
        MAX_NUM_SEQS=72
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        STATE_OFFLOAD_CPU_GIB=32
        ;;
    *)
        echo "Unsupported CONC=$CONC" >&2
        exit 2
        ;;
esac
# Official MiniMax-M3 ATOM launch settings override the per-band capacity knobs.
MAX_NUM_SEQS=$((2 * CONC))
MAX_NUM_BATCHED_TOKENS=32768
GPU_MEM_UTIL=0.9
export ATOM_ENABLE_REPLAYSSM

# MiniMax-M3 attention carries a per-request recurrent state alongside the
# paged KV. The CPU state tier is what makes a resumed agentic turn cheap; the
# paged KV tier alone cannot restore one.
OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=True

        case "$CONC" in
            40|48)
                # Validated high-concurrency tier: CPU only, chunk 256, no hybrid state offload.
                export LMCACHE_MAX_LOCAL_CPU_SIZE=256
                export LMCACHE_CHUNK_SIZE=256
                # ATOM_SLRU needs rocm/atom-dev:nightly_202609140645-lirzhang-triton-build or later.
                export ATOM_PREFIX_CACHE_POLICY=slru
                export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5
                export LMCACHE_CACHE_POLICY=ATOM_SLRU
                export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3
                ;;
            *)
                # TOTAL_CPU_DRAM_GB is the aggregate budget; these are per rank, so
                # divide by TP (agentic README). Handing a rank the whole aggregate
                # never finishes pinning and hangs the launch.
                PER_RANK_CPU_GB="$((TOTAL_CPU_DRAM_GB / TP))"
                LMCACHE_CPU_GB="$((PER_RANK_CPU_GB - STATE_OFFLOAD_CPU_GIB))"

                export LMCACHE_MAX_LOCAL_CPU_SIZE="$LMCACHE_CPU_GB"
                # DCP-locked: the offload hash block is block-size(128) x dcp(8) = 1024,
                # so the KV grid and the state-checkpoint grid coincide and the joint
                # load aims both legs at one boundary. 512 or 2048 misaligns it.
                export LMCACHE_CHUNK_SIZE=1024
                export OFFLOAD_KV_FOR_HYBRID=1
                # Statistics only; the submitted numbers were measured with it on.
                export OFFLOAD_PROFILE=1

                if [ "$STATE_OFFLOAD_CPU_GIB" -gt 0 ]; then
                    export OFFLOAD_STATE=1
                    export OFFLOAD_STATE_CPU_SIZE="$STATE_OFFLOAD_CPU_GIB"
                    export OFFLOAD_STATE_STAGING_GROUPS=8
                    export OFFLOAD_STATE_MIN_LOAD_TOKENS=0
                    # The staging buffer defaults to 2 chunks (8 MiB) and one state
                    # entry is 54.78 MiB; a buffer too small for one entry makes the
                    # tier decline to build, which reads like a tier that is on and idle.
                    export OFFLOAD_GPU_STAGING_CHUNKS=32
                fi
                ;;
        esac

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

echo "Starting atom server..."
export PYTHONNOUSERSITE=1

# Without it the aiter kernel logs flood the server log for the whole replay.
export AITER_LOG_LEVEL
export AITER_SITUV2_A4W4=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_FLYDSL_STAGE2_FP8=1
export ATOM_FORCE_ATTN_TRITON=1

# golden_al_distribution/minimaxm3_eagle3_gqa.yaml: minimax-m3.thinking_on[3] -> AL 2.78.
# Synthetic acceptance on throughput runs, real target verification on eval-only.
SPEC_ARGS=()
if [ "$NUM_SPEC_TOKENS" -gt 0 ]; then
    SPEC_ARGS=(
        --method eagle3
        --draft-model "$DRAFT_MODEL"
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
    )
    if [ "${EVAL_ONLY}" != "true" ]; then
        SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
    fi
fi
echo "SPEC_DECODE_AL=$SPEC_DECODE_AL NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS"

ATOM_CMD=(
    python -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --kv_cache_dtype fp8
    --block-size 128
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --index-cache-dtype fp8
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","vision_tower","multi_modal_projector","patch_merge_mlp","*block_sparse_moe"]}'
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
if [[ "$ENABLE_PREFIX_CACHING" != "true" ]]; then
    ATOM_CMD+=(--no-enable_prefix_caching)
fi
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
PYTHONPATH="$ATOM_RUNTIME_DEPS${PYTHONPATH:+:$PYTHONPATH}" \
    "${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
