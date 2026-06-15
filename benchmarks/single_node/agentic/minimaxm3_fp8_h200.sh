#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace-replay benchmark for MiniMax-M3 MXFP8 on H200 using vLLM, with
# CPU KV-cache offloading. 427B/26B-active MoE with MSA sparse attention.
#
# M3-specific serve flags (vs the M2.5 H200 agentic sibling):
#   --block-size 128       mandatory for MSA (sparse_block_size is 128; the
#                          default 16 misaligns sparse indexing).
#   --language-model-only  text-only workload; skips the vision encoder, freeing
#                          VRAM for KV.
#   NO --kv-cache-dtype fp8: MiniMax-M3-MXFP8 ships without calibrated KV scales,
#                          so fp8 KV corrupts output (vllm-project/vllm#45381).
#                          Keep BF16 KV (matches the M3 H200 fixed-seq-len recipe).
#   Prefix caching is left ENABLED (vLLM default): agentic coding traces share
#   large prefixes across turns/subagents, so prefix-cache reuse — and offloading
#   that cache to the CPU tier — is exactly what this test exercises.
#
# CPU offloading (OFFLOADING=cpu) uses vLLM v1's native KV-offloading connector
# (--kv-offloading-backend native), the same path the M2.5 H200 agentic recipe
# uses, to extend usable concurrency past the on-HBM KV cliff.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
# M3 serves at max_model_len up to ~250k; use the 256k-capped trace corpus
# (470 traces, max in+out <= 256k) so requests aren't rejected at the cap.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k

# ~444 GB of MXFP8 weights off a shared network FS; concurrent day-zero
# downloads hit huggingface_hub's WeakFileLock "[Errno 116] Stale file handle"
# race. Retry the (resumable) download, then serve with HF_HUB_OFFLINE=1 so
# vllm's snapshot_download does a lock-free local-cache read.
SERVE_OFFLINE=()
if [[ "$MODEL" != /* ]]; then
  for attempt in 1 2 3 4 5; do
    hf download "$MODEL" && break
    if [ "$attempt" = 5 ]; then echo "hf download failed after $attempt attempts" >&2; exit 1; fi
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
  done
  SERVE_OFFLINE=(env HF_HUB_OFFLINE=1)
fi

resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=""
case "$OFFLOADING" in
    none) ;;
    cpu)
        # vLLM v1 native KV-offloading connector (CPU tier).
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $TOTAL_CPU_DRAM_GB --disable-hybrid-kv-cache-manager"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu)" >&2
        exit 1
        ;;
esac

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1
# ~444 GB of MXFP8 weights off shared FS; engine startup can exceed the
# default 600s readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

"${SERVE_OFFLINE[@]}" vllm serve $MODEL \
--host 0.0.0.0 \
--port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--language-model-only \
--max-num-seqs $CONC \
--stream-interval 20 \
--trust-remote-code \
$OFFLOAD_ARGS > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
