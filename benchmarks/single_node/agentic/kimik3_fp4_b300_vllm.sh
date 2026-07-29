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
# KV_OFFLOADING: `none` (GPU-resident) or `dram` with
# KV_OFFLOAD_BACKEND=vllm-simple (SimpleCPUOffloadConnector).
#
# Note on the DRAM arm: Kimi-K3 is a KDA/MLA hybrid — its linear-attention (KDA)
# layers carry a recurrent state rather than paged KV blocks, and
# SimpleCPUOffloadConnector offloads uniform paged blocks with no
# hybrid-geometry handling. The arm is expected to offload only the MLA layers'
# paged blocks; watch the bring-up sweep for KV-geometry errors at server init.

source "$(dirname "$0")/../../benchmark_lib.sh"

# Force the eval framework to lm-eval (gsm8k) for this recipe. run_eval derives
# its default as swebench for agentic scenarios (scenario_default=swebench when
# IS_AGENTIC/SCENARIO_TYPE=agentic-coding), but EVAL_FRAMEWORK takes precedence
# over that default (benchmark_lib.sh: framework=${EVAL_FRAMEWORK:-...}), so
# setting it here makes the effective framework always lm-eval, never swebench.
export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

# The 2.8T MXFP4 checkpoint only fits across all 8 B300s (see header).
if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on B300 requires TP=8 (a ~1.5 TB MXFP4 checkpoint does not fit at TP<8), got TP='$TP'" >&2
    exit 1
fi

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
# B300 NVLink / NCCL tuning. VLLM_USE_NCCL_SYMM_MEM=0: the container's
# CUDA-graph capture path enters vLLM's nccl_symm_mem_context without
# set_graph_pool_id, causing a capture assert; disabling it makes the context a
# no-op so capture proceeds (fixed in newer vLLM cuda_graph.py).
export VLLM_USE_NCCL_SYMM_MEM=0
export TORCH_SYMMMEM=NVSHMEM
export NCCL_CUMEM_ENABLE=1
export NCCL_MNNVL_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_DMABUF_ENABLE=0
# vLLM server & engine flags
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
# Loading ~1.5 TB of MXFP4 shards off the staged mount takes well past the
# default readiness window even with fastsafetensors.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024
export TILELANG_CLEANUP_TEMP_FILES=1
# UCX transport settings for DRAM offload
export UCX_MEMTYPE_CACHE=n
export UCX_MEMTYPE_REG_WHOLE=n
export UCX_RCACHE_MAX_UNRELEASED=1024
export UCX_TLS=cuda_copy,cuda_ipc,tcp
export PYTHONNOUSERSITE=1
# AIPerf pins one pooled keep-alive connection per agentic session and reuses it
# across turns, while the Rust frontend's default VLLM_HTTP_TIMEOUT_KEEP_ALIVE is
# 5s. An inter-turn idle gap longer than that lets the client reuse a socket at
# the moment the server closes it -> aiohttp ServerDisconnectedError -> AIPerf
# treats it as a terminal warmup failure and aborts the whole job. This killed
# the dram c4 arm ~15 min into run 30324907690 with a perfectly healthy server
# (it kept serving after the client gave up). Outlast the client pool so the
# race cannot occur. Same fix as glm5.2_fp4_b300_sglang.sh's
# SGLANG_TIMEOUT_KEEP_ALIVE=900.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
# Agentic warmup dispatches large prompts at once; allow up to 15 minutes of TCP
# progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- KV offloading ----------------------------------------------------------
GMU=0.90
OFFLOAD_ARGS=()
PREFIX_MATCH_ARGS=(--prefix-match-unit 64)
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    native)
        require_agentic_kv_offload_backend native
        # Identical prefixes must hash to identical block keys run-to-run.
        export PYTHONHASHSEED=42
        # VLLM_USE_SIMPLE_KV_OFFLOAD=1 tells vLLM to auto-wire the
        # SimpleCPUOffloadConnector when --kv-offloading-backend native is set.
        # kv-offloading-size is the total GiB pool across all TP ranks; use the
        # full TOTAL_CPU_DRAM_GB budget passed in by the launcher. On failure to
        # load a KV block from DRAM, recompute rather than fail the request.
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        PREFIX_MATCH_ARGS=()
        OFFLOAD_ARGS=(
            --kv-offloading-backend native
            --kv-offloading-size "$TOTAL_CPU_DRAM_GB"
            --kv-transfer-config '{"kv_load_failure_policy":"recompute"}'
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected empty or native)" >&2
        exit 1
        ;;
esac

MAX_NUM_SEQS=$((2 * CONC))

echo "Starting vllm server..."

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GMU"
    --max-num-seqs "$MAX_NUM_SEQS"
    # Agentic replays run at the model's native context limit.
    --max-model-len 1048576
    --trust-remote-code
    --tokenizer-mode kimi_k3
    --language-model-only
    --load-format fastsafetensors
    --moe-backend auto
    --enable-prefix-caching
    --enable-cumem-allocator
    --no-enable-flashinfer-autotune
    --kv-cache-dtype fp8
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    # FP8 KV cache requires the prefill query quantization flag; MLA prefill
    # runs on FlashInfer per the production recipe.
    --attention-backend FLASHINFER_MLA
    --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}'
    "${PREFIX_MATCH_ARGS[@]}"
    --max-cudagraph-capture-size "$MAX_NUM_SEQS"
    --disable-uvicorn-access-log
    "${OFFLOAD_ARGS[@]}"
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
