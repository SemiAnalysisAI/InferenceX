#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for GLM-5.2 NVFP4 on the 8x RTX PRO 6000 Blackwell
# (SM120) node using SGLang.
#
# Flags start from the merged B300 NVFP4 cookbook recipe
# (benchmarks/single_node/agentic/glm5.2_fp4_b300_sglang.sh, STP only) and
# then apply the SM120 deltas this GPU family needs:
#
#   * TP8 only. The checkpoint is 433 GB, so pure TP across all eight 96 GB
#     GPUs is the only layout that leaves room for the KV pool; TP4 does not
#     fit the weights at all.
#   * Shared-experts fusion is force-disabled. SGLang enables it for this
#     config (n_routed_experts=256, n_shared_experts=1, EP off), but
#     nvidia/GLM-5.2-NVFP4 stores the shared expert loose and unquantized
#     (BF16 [2048, 6144]) while the routed experts are packed NVFP4
#     ([2048, 3072]), so the fused loader aborts during weight load with
#     "The size of tensor a (3072) must match the size of tensor b (6144)".
#     This is the same trap the in-tree comment records for the
#     compressed-tensors Kimi-K2.5 checkpoint.
#   * Attention falls back to Triton MLA. SGLang would default this DSA model
#     to --attention-backend dsa, whose indexer metadata comes only from
#     DeepGEMM, and DeepGEMM has no SM120 kernel. Sparse attention is
#     therefore not exercised on this GPU family, and prefill cost grows
#     superlinearly with context.
#   * MoE runner backend is left at SGLang's own SM120 choice for
#     modelopt_fp4 (flashinfer_cutlass; trtllm-gen MoE is SM100-only) and is
#     overridable through MOE_RUNNER_BACKEND.
#   * HiCache is not supported on RTX PRO 6000, so this recipe is
#     GPU-resident KV only.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB \
    RESULT_DIR \
    DURATION \
    EP_SIZE \
    DP_ATTENTION

if [[ "$TP" != "8" ]]; then
    echo "GLM-5.2 SGLang on RTX PRO 6000 requires TP8: the 433 GB NVFP4 checkpoint does not fit in fewer than eight 96 GB GPUs" >&2
    exit 1
fi
if [[ "$KV_OFFLOADING" != "none" ]]; then
    echo "GLM-5.2 SGLang on RTX PRO 6000 supports GPU-resident KV cache only (HiCache is unsupported on this GPU family)" >&2
    exit 1
fi
if [[ "$DP_ATTENTION" == "true" ]]; then
    echo "GLM-5.2 SGLang on RTX PRO 6000 does not support DP attention: attention-DP replicates the KV pool per rank, which the 96 GB GPUs cannot hold alongside 54 GB of weights" >&2
    exit 1
fi
if [[ "$EP_SIZE" != "1" ]]; then
    echo "GLM-5.2 SGLang on RTX PRO 6000 supports EP1 only" >&2
    exit 1
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE.
# Either way, MODEL_PATH is what the server is launched with.
MODEL_REVISION="${GLM52_MODEL_REVISION:-aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa}"
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --revision "$MODEL_REVISION" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL" --revision "$MODEL_REVISION"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi
nvidia-smi
nvidia-smi topo -m || true

# The FP8 KV pool holds 464,768 tokens per rank (~51 KB/token across 78
# layers) at mem-fraction-static 0.85, so the model's 1M context does not fit
# and the server is capped at 256k. Replay the matching 256k-capped corpus
# instead of the 1M default this model prefix would otherwise select.
export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126_256k}"
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=12.0a
# All eight GPUs are SYS-connected (PCIe only, no NVLink) on this node, and
# NCCL 2.28.9 segfaults probing its bnxt_re devices; the runner already sets
# NCCL_IB_DISABLE=1. Keep collectives on the local PCIe/SHM transports.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# NOTE for whoever revisits the DSA path: SGLang carries a set of SM120
# kernel fixups (SGLANG_OPT_FP8_WO_A_GEMM / SGLANG_OPT_USE_TOPK_V2 /
# SGLANG_OPT_USE_TILELANG_MHC_PRE / SGLANG_OPT_DEEPGEMM_HC_PRENORM off,
# SGLANG_FP8_PAGED_MQA_LOGITS_TORCH on) that it applies only inside its
# DeepseekV4ForCausalLM branch, and GLM-5.2 (GlmMoeDsaForCausalLM) misses
# them. They were tested on-node and are NOT needed on this Triton MLA
# fallback path — boot and generation are identical with and without them —
# so they are deliberately not exported here. They become relevant again if
# the sparse DSA backend ever works on SM120.
#
# Do not simply switch --attention-backend back to dsa: enabling the sparse
# path on SM120 was attempted on-node (2026-07-26) and needs kernel work, not
# a flag. Patching SGLang to route the indexer's paged-MQA logits through its
# TileLang/torch implementations (both already used by the dsv4 indexer) and
# to stop building the DeepGEMM schedule plan does clear the "Unsupported
# architecture" abort, and the server then boots and serves — but four more
# walls follow:
#   1. TileLang's CUDA sparse-MLA kernel takes bf16 KV only (its fp8 variants
#      are ROCm-only), which halves the KV pool to 256,256 tokens/rank.
#   2. That kernel asks for 170,048 B of dynamic shared memory; SM120 allows
#      ~100 KB. Its tile size is not a knob — the barrier arrive_counts
#      (384/256/128) encode the BI=64 thread mapping, so block_I=32 compiles
#      and then reads out of bounds.
#   3. The plain (v1) kernel does fit at block_I=32 / num_stages=1 / 128
#      threads, but CUDA-graph capture fails because TileLang JIT-compiles
#      inside the capture (cudaErrorStreamCaptureUnsupported).
#   4. With graphs disabled it runs and is still numerically wrong: degenerate
#      repetition on a 26-token prompt and an empty answer on the 45k-token
#      needle, identically with the TileLang and the torch reference logits
#      kernels — so the fault is the sparse-MLA kernel itself, not the indexer.
# Fixing this means an SM120-shaped sparse-MLA kernel (split the d_v=512
# accumulation so KV tiles stay under ~32 KB, re-derive the barrier counts,
# validate against the dense path), not a config change. Full logs from the
# attempt: rtx6000pro-lat:/home/ubuntu/glm52-sglang-patch/out/server.dsa*.log.

# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session (client-side
# keep-alive 300s) while uvicorn's default SGLANG_TIMEOUT_KEEP_ALIVE is 5s;
# inter-turn idle gaps can reuse a socket exactly as the server closes it ->
# ECONNRESET -> terminal warmup failure. Outlast the client pool.
export SGLANG_TIMEOUT_KEEP_ALIVE=900
# Dense-attention prefill on this node measured 872 tok/s at 4.4k context
# falling to 418 tok/s at 244k (single stream), so a warmup snapshot of
# 100k-token histories needs several minutes per trajectory. Double the
# shared 1800s warmup grace so warmup drains instead of being declared
# failed; grace is a maximum wait, not a fixed sleep.
export AGENTIC_WARMUP_GRACE_PERIOD="${AGENTIC_WARMUP_GRACE_PERIOD:-3600}"

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"

MOE_ARGS=()
if [[ -n "${MOE_RUNNER_BACKEND:-}" ]]; then
    MOE_ARGS=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --quantization modelopt_fp4
    # nvidia/GLM-5.2-NVFP4 keeps the shared expert loose and in BF16, so the
    # fused-shared-expert loader cannot consume it (see the header note).
    --disable-shared-experts-fusion
    # SGLang would default this DSA model to --attention-backend dsa, whose
    # indexer metadata is built by deep_gemm.get_paged_mqa_logits_metadata
    # (the only CUDA option; 'cutedsl' is gated to SM100). DeepGEMM aborts
    # with "Assertion error (attention.hpp:227): Unsupported architecture" on
    # SM120 during warmup, so fall back to Triton MLA. Sparse attention is
    # consequently not exercised on this GPU family.
    --attention-backend "$ATTENTION_BACKEND"
    "${MOE_ARGS[@]}"
    # GLM-5.2 emits the GLM-4.7-style <tool_call>/<arg_key>/<arg_value>
    # format; glm47 is required for structured message.tool_calls, and the
    # reasoning parser keeps hybrid-thinking output in reasoning_content.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --context-length "$CONTEXT_LENGTH"
    --chunked-prefill-size 8192
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --watchdog-timeout 1800
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

echo "Starting SGLang server for RTX PRO 6000..."
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

cleanup_agentic_server() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
