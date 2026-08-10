#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 (MXFP4) on MI355X (gfx950) using
# vLLM with DSpark speculative decoding at level 2 (num_speculative_tokens=2).
# AMD sister of benchmarks/single_node/agentic/kimik3_fp4_b300_vllm_mtp.sh; the
# harness scaffolding follows the AMD convention in dsv4_fp4_mi355x_vllm_mtp.sh
# (env-var contract, HF-id model resolution, ROCR->HIP mapping, server metrics,
# replay invocation). The target-server deltas are all ROCm/AITER:
#   - AITER MLA on the asm-padded persistent route
#     (VLLM_ROCM_AITER_MLA_ASM_PADDING=asm) instead of FLASHINFER_MLA, so the
#     speculative-config carries no attention_backend override.
#   - AITER MoE (VLLM_ROCM_USE_AITER_MOE=1, --moe-backend aiter).
#
# Validated bring-up on rocm/pytorch-private:hy-kk-08092026 (vLLM 0.26.1rc1.dev,
# 8x gfx950, TP8) at native 1M context: asm AITER MLA + AITER MoE, fp8 KV cache,
# --gpu-memory-utilization 0.95, --max-model-len 1048576, --max-num-batched-tokens
# 4096. The asm-MLA 1M warmup only converges at that util/chunk, and concurrent
# long-context prefills are bounded with --max-num-seqs (<=8) so the MoE/linear
# activations stay within the ~15 GB/GPU headroom left after fp8 KV. GPU-resident
# KV only -- no DRAM offload arm is wired for this recipe.
#
# Model resolution is by HF id: MODEL (moonshotai/Kimi-K3) and DRAFT_MODEL
# (Inferact/Kimi-K3-DSpark) are passed straight through and the server resolves
# them (the launcher leaves MODEL_PATH unset on single-node runs).
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION
#
# TP8 is the only single-node layout: the MXFP4 checkpoint does not fit below 8
# GPUs. This recipe ships the pure-TP8 profile (EP_SIZE=1, DP_ATTENTION=false).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on MI355X requires TP=8 (the MXFP4 checkpoint does not fit at TP<8), got TP='$TP'" >&2
    exit 1
fi

if [ "$EP_SIZE" -gt 1 ]; then
    echo "Error: this recipe ships the pure-TP8 profile; EP_SIZE='$EP_SIZE' is not wired yet" >&2
    exit 1
fi

if [ "$DP_ATTENTION" = "true" ]; then
    echo "Error: this recipe ships the pure-TP8 profile; DP_ATTENTION=true is not wired yet" >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"

# Resolve by HF id: pass the names straight to the server, which downloads /
# resolves the checkpoints. The mi355x launcher leaves MODEL_PATH unset on the
# single-node path, so this falls through to the HF-name branch.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="${WRITABLE_MODELS_DIR:-/data/models}/${DRAFT_MODEL##*/}"
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
    hf download "$DRAFT_MODEL"
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
fi

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
rocm-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps
# Nightly ROCm image may be missing runtime deps; ensure they are present.
agentic_pip_install --quiet Pillow fastapi uvicorn

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf scrapes the vLLM engine's own /metrics for the server-side throughput
# columns (pure TP: engine == public endpoint).
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
# Loading the MXFP4 shards past the default readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
# AIPerf pins one pooled keep-alive connection per agentic session and reuses it
# across turns; outlast the client pool so an inter-turn idle gap cannot race the
# server closing the socket (aiohttp ServerDisconnectedError -> warmup failure).
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-900}"

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# GPU-resident KV only. fp8 KV at 0.95 util already fills HBM after weights;
# no DRAM-offload arm is validated for this recipe.
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    *)
        echo "Error: kimik3 MI355X ships GPU-resident KV only; KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' is not wired" >&2
        exit 1
        ;;
esac

# ---- DSpark speculative decoding --------------------------------------------
# DSpark level 2 (num_speculative_tokens=2, the value adopted for this recipe)
# on the Inferact/Kimi-K3-DSpark draft head, probabilistic drafting with
# synthetic acceptance pinned to the committed golden AL, per the AgentX policy
# in golden_al_distribution/README.md (a submission chooses the draft length,
# not the acceptance target). Mirrors the B300 sibling; the only delta is the
# ROCm image serves MLA via AITER, so no attention_backend override is set.
# EVAL_ONLY switches to real block verification (synthetic acceptance commits
# drafts regardless of target logits and would zero the SWE-bench score).
NUM_SPEC_TOKENS=2
# Committed golden AL at K=2 on the probabilistic/block curve
# (golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml:
# thinking_on 2 -> 2.51).
SYNTHETIC_ACCEPT_LEN=2.51
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}"
else
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

# AgentX concurrency counts session trees, not requests; subagent fan-out can
# push instantaneous request concurrency above CONC, so leave 2x scheduler
# headroom -- but cap at 8. The asm-MLA 1M warmup and long-context prefill
# activations only stay within the per-GPU headroom at max-num-seqs <= 8 (higher
# OOMs the MoE/linear activations on 200k+ ISL bursts).
MAX_NUM_SEQS=$((2 * CONC))
if [ "$MAX_NUM_SEQS" -gt 8 ]; then
    MAX_NUM_SEQS=8
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)

echo "Starting vllm server..."
set -x
# AITER MLA + MoE, asm-padded persistent MLA route (the config validated at 1M).
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export SAFETENSORS_FAST_GPU=1

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --distributed-executor-backend mp
    --kv-cache-dtype fp8
    "${PARALLEL_ARGS[@]}"
    --gpu-memory-utilization 0.95
    --max-model-len 1048576
    --max-num-batched-tokens 4096
    --max-num-seqs "$MAX_NUM_SEQS"
    --moe-backend aiter
    --enable-prefix-caching
    --speculative-config "$SPEC_CONFIG"
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
