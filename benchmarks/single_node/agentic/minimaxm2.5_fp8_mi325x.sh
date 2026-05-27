#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for MiniMax-M2.5 FP8 on MI325X using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
# MiniMax-M2.5 servers run at max_model_len ~256k; the unfiltered 052726
# corpus has requests up to ~1M proxy tokens that would be rejected.
# Switch to the 256k-capped variant (217 traces, max in+out <= 256k).
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k

resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=""
case "$OFFLOADING" in
    none) ;;
    cpu)
        # SimpleCPUOffloadConnector now works on ROCm with the
        # vllm/vllm-openai-rocm:nightly-51f22dcfd0... image (vllm-project/vllm@20cac26b).
        # Use the same offload path as NVIDIA so cross-vendor cpu-offload
        # numbers are apples-to-apples.
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $TOTAL_CPU_DRAM_GB --disable-hybrid-kv-cache-manager"
        ;;
    *) echo "Error: unsupported OFFLOADING value '$OFFLOADING'" >&2; exit 1 ;;
esac

if [ "$EP_SIZE" -gt 1 ]; then EP=" --enable-expert-parallel"; else EP=" "; fi

echo "Starting vllm server..."
export VLLM_ROCM_USE_AITER=1
export PYTHONNOUSERSITE=1

vllm serve $MODEL \
--host 0.0.0.0 \
--port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--kv-cache-dtype fp8 \
--block-size=32 \
--max-num-seqs $CONC \
--attention-backend "ROCM_AITER_UNIFIED_ATTN" \
--trust-remote-code \
$OFFLOAD_ARGS > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
