#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 FP8 on MI355X using SGLang with
# EAGLE/MTP speculative decoding. First GLM-5.2 FP8 AgentX recipe on MI355X;
# spec-decode only, per the AgentX policy that agentic recipes are run and
# published with speculative decoding enabled (MODELS.md).
#
# Port of the validated agentic/glm5.2_fp4_mi355x_sglang_mtp.sh (amd/GLM-5.2-MXFP4).
# The FP8 deltas are the blocks marked "FP8:" below -- the checkpoint
# (zai-org/GLM-5.2-FP8, ~756 GB of block-quantized e4m3 weights, 141 shards,
# against ~380 GB for the MXFP4 checkpoint), the memory notes that follow from
# the larger resident weights, and the TP8-only arm selection. Serve flags are
# otherwise the MXFP4 script unchanged so the two precision curves on this SKU
# stay comparable. The ROCm GLM-5.2 FP8 sibling on MI325X
# (agentic/glm5.2_fp8_mi325x_mtp.sh) is the precedent for serving this
# checkpoint on gfx9 SGLang: quantization is auto-detected from the
# checkpoint's quantization_config (quant_method=fp8, 128x128 weight blocks),
# so no --quantization flag is passed.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility under slurm cgroups.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# FP8: runners/launch_mi355x-amds.sh mounts the NFS hf-hub cache for this
# checkpoint (like MiniMax-M3) so the ~756 GB pull happens once for the
# cluster instead of once per node-local NVMe cache. `hf download` resumes
# into a partially populated cache, so concurrent cells converge on one copy.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
rocm-smi || true
amd-smi || true

# A server killed on this node minutes earlier (previous job, crashed run)
# can still be draining its HBM: KFD reclaim takes minutes, and booting into a
# half-drained node fails RCCL init with HIP 'unhandled cuda error' /
# 'invalid argument'. Wait for the GPUs to come back before launching.
# Per-GPU threshold: idle nodes hold a small driver/firmware VRAM baseline
# (observed up to ~4%/GPU, node-dependent), while a draining or occupied
# GPU sits at 50-90%. Require every GPU <= 10%.
GPU_CLEAN=false
for i in $(seq 1 90); do
    VRAM_MAX=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if ($NF > m) m = $NF} END {print m+0}')
    if [ "${VRAM_MAX:-0}" -le 10 ]; then echo "GPUs clean (vram%max=$VRAM_MAX after $((i*10))s)"; GPU_CLEAN=true; break; fi
    echo "waiting for prior-job GPU memory reclaim: vram%max=$VRAM_MAX"; sleep 10
done
[ "$GPU_CLEAN" = "true" ] || { echo "Error: GPUs still draining prior job's memory after 15min" >&2; exit 1; }

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

export PYTHONNOUSERSITE=1
# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session (client-side
# keep-alive 300s) while uvicorn's default SGLANG_TIMEOUT_KEEP_ALIVE is 5s;
# inter-turn idle gaps can reuse a socket exactly as the server closes it.
# Outlast the client pool so the race cannot occur.
export SGLANG_TIMEOUT_KEEP_ALIVE=900
# The DSA indexer's top-k v2 kernel (default since v0.5.14) is JIT-compiled
# from CUDA-only source (cooperative_groups.h) and cannot build for gfx950;
# v1 dispatches to the precompiled HIP op in sgl-kernel (upstream MI355X CI
# runs DSA models the same way). Still honored by v0.5.19 (environ.py).
export SGLANG_OPT_USE_TOPK_V2=false

# HiCache L2 (host DRAM). KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.
#
# FP8: with the ~756 GB checkpoint resident at TP8 (~94.5 GB/rank) inside
# --mem-fraction-static 0.85 of 288 GB, the device KV pool is roughly 150 GB
# per rank (MXFP4 TP8: ~182.7 GB/rank). ratio 1.5 therefore pins about 1.8 TB
# of host DRAM across TP8, comfortably inside the ~3.0 TB available on
# cluster:mi355x-amds nodes (the MXFP4 TP arm's ratio 1.5 pins ~2.9 TB there).
# The agentic-coding corpus saturates any fixed DRAM pool at conc >= 10;
# ratio 2.5 yields more throughput at conc 10-16 but exceeds physical DRAM on
# these nodes and must be set via HICACHE_RATIO on nodes that can take it.
# The DP-attention arm keeps the MXFP4 script's 0.5.
CACHE_ARGS=()
if agentic_kv_offload_enabled; then
    if [ "$DP_ATTENTION" = "true" ]; then
        HICACHE_RATIO="${HICACHE_RATIO:-0.5}"
    else
        HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    fi
    # write_through_selective skips DRAM writes for non-reusable KV blocks,
    # reducing host-bus traffic without affecting the cache hit rate.
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    case "${KV_OFFLOAD_BACKEND:-}" in
        hicache)
            echo "HiCache (GPU+host DRAM only): ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
            CACHE_ARGS=(
                --enable-hierarchical-cache
                --hicache-ratio "$HICACHE_RATIO"
                --hicache-write-policy "$HICACHE_WRITE_POLICY"
                --hicache-io-backend "$HICACHE_IO_BACKEND"
                --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
            )
            ;;
        *)
            # The MXFP4 script also wires a Mooncake L3 tier; it is deliberately
            # not carried here (HiCache host DRAM is the supported AMD tier).
            echo "Error: unsupported KV_OFFLOAD_BACKEND '${KV_OFFLOAD_BACKEND:-}' (expected: hicache)" >&2
            exit 1
            ;;
    esac
fi

# Arm selection. FP8: TP8-only in the master config. The ~756 GB checkpoint
# does not fit below TP8 with room for KV (TP4 would leave ~56 GB/rank for the
# KV pool, DSA indexer, and EAGLE verification batches), which is also why the
# MI325X FP8 sibling is TP8-only. Two arms: GPU-resident KV at low
# concurrency (latency), HiCache host offload at conc >= 8 (throughput).
#
# NOTE: the DP-attention path below is DORMANT (no dp-attn arms in
# amd-master.yaml): DSA + dp-attention hangs a collective under long-context
# prefill on ROCm (reproduced on v0.5.14 with and without HiCache). Kept
# intact from the MXFP4 script so the arm can be enabled without re-deriving it.
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=(python3 -m sglang_router.launch_router)
    PARALLEL_ARGS+=(--dp "$TP" --enable-dp-attention)
    CHUNKED_PREFILL_SIZE=32768
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
    # Swap the DP gather collectives to gatherv/reduce-scatter on ROCm
    # (dsv4_fp4_mi355x_sglang.sh precedent): with the defaults the DSA DP path
    # hangs a collective under long-context prefill load.
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES=5
elif [ "$CONC" -le 16 ]; then
    # Chunked prefill 32k: smaller chunks let the scheduler interleave decode
    # steps between prefill chunks, reducing TPOT for concurrent sessions.
    # Per-chunk activation headroom is ~1.7 GiB/rank at 32k (vs ~7 GiB/rank
    # at 131k, which OOMed the MXFP4 recipe at 0.85), so 0.85 is safe here too:
    # the non-static headroom is a fraction of the 288 GB card and does not
    # depend on the checkpoint size.
    CHUNKED_PREFILL_SIZE=32768
else
    CHUNKED_PREFILL_SIZE=32768
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi
# 2xCONC in-flight slots: MTP draft+verify transiently batches more tokens
# than CONC sessions; headroom prevents scheduler stalls under burst.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
# SGLang interpolates a bs list [1..max_bs] automatically; cap at 64 to
# keep graph-capture memory bounded without giving up coverage.
CUDA_GRAPH_MAX_BS=$(( MAX_RUNNING_REQUESTS < 64 ? MAX_RUNNING_REQUESTS : 64 ))

# AgentX pins acceptance to the committed golden AL so submissions are compared
# on system performance at a fixed acceptance target rather than on draft-head
# quality (golden_al_distribution/README.md). Same MTP depth as the MXFP4
# MI355X sibling (num-steps 5, 6 draft tokens = 5 speculative tokens) so the
# two precision curves on this SKU share one acceptance target: 3.61 is the
# GLM-5.2 thinking_on curve at K=5 (golden_al_distribution/glm5.2_mtp.yaml,
# SPEED-Bench coding, run 28058352479). FP8: that curve was measured on
# glm-5.2-fp8, i.e. on this checkpoint.
#
# EVAL_ONLY leaves simulated acceptance off: it commits drafted tokens
# regardless of the target logits, so generated text is wrong and the eval
# would score ~0.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.61
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    # gfx950 has calibrated FP8 attention scales (the MXFP4 MI355X sibling and
    # the DSv4 MI355X recipe both run fp8_e4m3 KV); only gfx942 keeps bf16 KV.
    --kv-cache-dtype fp8_e4m3
    # DSA indexer kernels: the tilelang prefill/decode backends are the ones
    # the MXFP4 MI355X sibling validated on gfx950 and remain options in v0.5.19.
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    # GLM-5.2 emits the GLM-4.7-style tool-call format; glm47 is required for
    # structured message.tool_calls (SWE-bench agentic evals die without it).
    # The glm45 reasoning parser keeps hybrid thinking in reasoning_content.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --speculative-algorithm EAGLE
    --speculative-num-steps 5
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 6
    "${CACHE_ARGS[@]}"
    --watchdog-timeout 1800
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_SIMULATE_ACC_* env vars at launch (empty => real verification) ==="
    env | grep -E '^SGLANG_SIMULATE_ACC_' | sort || true
    echo "============================================================================"
} | tee "$SERVER_LOG"

echo "Starting SGLang server for MI355X..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
    echo "Starting SGLang router on port $PORT for $TP DP ranks..."
    "${SGLANG_ROUTER_CMD[@]}" \
        --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
        --policy consistent_hashing \
        --request-id-headers x-correlation-id \
        --dp-aware \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
        --connect-timeout-secs 900 \
        --request-timeout-secs 14400 \
        --disable-health-check \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    # GLM-5.2's chat template defaults to reasoning_effort=Max when the client
    # passes no chat_template_kwargs (mini-swe-agent doesn't); the heavy
    # thinking burns the shared 75-step budget. Double it, as the GLM-5.2 B200
    # and MI325X recipes do.
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
