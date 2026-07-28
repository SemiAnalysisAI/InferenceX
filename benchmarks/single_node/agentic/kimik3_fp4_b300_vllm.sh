#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 (MXFP4) on B300 using vLLM.
#
# Day-zero single-node recipe. Serve flags follow the Kimi-K3 production recipe
# already exercised by the DSpark AL collector
# (benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm.sh) and the upstream
# recipes.vllm.ai/moonshotai/Kimi-K3 guidance, adapted to the agentic scenario.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
#
# TP8 is the only single-node layout. The MXFP4 checkpoint is ~1.5 TB on disk;
# at TP4 that is ~375 GB of weights per GPU against B300's 288 GB of HBM, so
# only the full 8-GPU shard (~188 GB/GPU) fits. Do not add TP4/TP2 arms.
#
# Weights are pre-staged node-local: `Kimi-K3` is in the b300-nv launcher's
# STAGED_MODELS, so MODEL_PATH resolves to the read-only /scratch/models/Kimi-K3
# mount and no download happens on the runner. The download block below only
# covers stand-alone runs.
#
# KV_OFFLOADING: `none` only. Kimi-K3 is a KDA/MLA hybrid — its linear-attention
# (KDA) layers carry a recurrent state rather than a paged KV cache, and
# SimpleCPUOffloadConnector has no hybrid-geometry handling (it offloads uniform
# paged blocks). The DRAM-offload arm is deferred until that interaction is
# validated on-node; see the PR description.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

# The 2.8T MXFP4 checkpoint only fits across all 8 B300s (see header).
if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on B300 requires TP=8 (a ~1.5 TB MXFP4 checkpoint does not fit at TP<8), got TP='$TP'" >&2
    exit 1
fi

# GPU-resident only for now; a dram arm needs the hybrid-KV work noted above.
require_agentic_kv_offload_none

if [[ -n "${EP_SIZE:-}" && "${EP_SIZE}" -gt 1 ]]; then
    echo "Error: this recipe ships the pure-TP8 profile; EP_SIZE='$EP_SIZE' is not wired yet" >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE.
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Kimi-K3 production serving environment ---------------------------------
export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
# Loading ~1.5 TB of MXFP4 shards off the staged mount takes well past the
# default readiness window even with fastsafetensors.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONNOUSERSITE=1

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# Agentic fan-out: keep the scheduler headroom convention shared by the other
# agentic recipes. Capture decode graphs only up to that batch size — a 93-layer
# 2.8T model makes capturing vLLM's full 2048-wide ladder prohibitively slow.
MAX_NUM_SEQS=$((2 * CONC))

echo "Starting vllm server..."

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization 0.90
    --max-num-seqs "$MAX_NUM_SEQS"
    # Agentic replays run at the model's native context limit.
    --max-model-len 1048576
    --trust-remote-code
    --load-format fastsafetensors
    --moe-backend auto
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    # FP8 KV cache requires the prefill query quantization flag; MLA prefill
    # runs on FlashInfer per the production recipe.
    --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}'
    --max-cudagraph-capture-size "$MAX_NUM_SEQS"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
