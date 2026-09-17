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

# Official MiniMax-M3 ATOM launch settings.
MAX_NUM_SEQS=$((2 * CONC))
MAX_NUM_BATCHED_TOKENS=32768
GPU_MEM_UTIL=0.95

# One place per concurrency. A band that is not listed exits; a band that is
# listed but leaves a feature off simply does not get it -- both failures are
# visible, unlike enabling a feature on a point nobody measured.
NUM_SPEC_TOKENS=3
SPEC_DECODE_AL=2.78
INDEXER_CP=0
OFFLOAD_TIER=""
case "$CONC" in
    1|2|4|5|8|10|12|14|16) ;;
    15|24)    INDEXER_CP=1 ;;
    20)       INDEXER_CP=1; OFFLOAD_TIER=cpu256 ;;
    25|30)    OFFLOAD_TIER=cpu256 ;;
    28|32)    INDEXER_CP=1 ;;
    40|48)    INDEXER_CP=1; OFFLOAD_TIER=cpu256 ;;
    *) echo "Unsupported CONC=$CONC" >&2; exit 2 ;;
esac

# Sized for the running batch, which is 22-41% of CONC on this workload, not for
# CONC itself. ModelRunner trims this to min(2*CONC, 8192).
CUDAGRAPH_CAPTURE_SIZES="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]"

OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=True

        # GPUs 0-3 are on NUMA node 0, 4-7 on node 1. Ranks pinning host memory on
        # one node starve each other: 256 GB/rank takes 45 min (TP2) / 27 min (TP4)
        # all on node 0, and 21 s with TP2 split one per node.
        if [[ -z "${ROCR_VISIBLE_DEVICES+x}" ]]; then
            case "$TP" in
                2) NUMA_GPUS=0,4 ;;
                4) NUMA_GPUS=0,1,4,5 ;;
                *) NUMA_GPUS="" ;;
            esac
            if [[ -n "$NUMA_GPUS" ]]; then
                export ROCR_VISIBLE_DEVICES="$NUMA_GPUS"
                export HIP_VISIBLE_DEVICES="$NUMA_GPUS"
                echo "NUMA-spread GPUs for offload: $NUMA_GPUS"
            fi
        fi

        case "$OFFLOAD_TIER" in
            cpu256)
                export LMCACHE_MAX_LOCAL_CPU_SIZE=256
                export LMCACHE_CHUNK_SIZE=256
                # ATOM_SLRU needs rocm/atom-dev:nightly_202609140645-lirzhang-triton-build or later.
                export ATOM_PREFIX_CACHE_POLICY=slru
                export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5
                export LMCACHE_CACHE_POLICY=ATOM_SLRU
                # Must cover every rank; rank 0 alone halves the offload benefit.
                export LMCACHE_LOOKUP_SERVER_WORKER_IDS="$(seq -s, 0 $((TP - 1)))"
                ;;
            *)
                echo "CONC=$CONC has no measured offload tier" >&2
                exit 2
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

# TP4 only: ATOM requires tp_size == sparse_num_index_heads (4 for M3). CONC=20 is
# in both search-space rows, so this is what keeps it off the TP2 offload curve.
if [ "$INDEXER_CP" -eq 1 ] && [ "$TP" -eq 4 ]; then
    export ATOM_M3_INDEXER_CP=1
    echo "ATOM_M3_INDEXER_CP=1 (TP4, CONC=$CONC)"
fi

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
SPEC_ARGS=(
    --method eagle3
    --draft-model "$DRAFT_MODEL"
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
)
if [ "${EVAL_ONLY}" != "true" ]; then
    SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
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
    --cudagraph-capture-sizes "$CUDAGRAPH_CAPTURE_SIZES"
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
