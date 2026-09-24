#!/usr/bin/env bash
set -eo pipefail
set -x

# GLM-5.2 NVFP4 on B200 with SGLang EAGLE/MTP. Port of
# glm5.2_fp4_b300_sglang_mtp.sh; the B200 deltas are marked "B200:" below.
# DP_ATTENTION=false is the low-latency arm (TP8, fp8 KV, cutedsl bf16 GEMM);
# DP_ATTENTION=true is the DEP arm (TP8 + DP8 attention + --ep-size), kept
# intact but not wired into the master config.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION
check_env_vars EVAL_ONLY

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# B200: launch_b200-nscale-slurm.sh rewrites MODEL to a cluster-local path, so
# keep the HF repo id separately for the unstaged case.
HF_MODEL_ID="nvidia/GLM-5.2-NVFP4"

# A non-empty directory is not a staged checkpoint: an aborted pull leaves
# config.json and friends with no tokenizer or weights, and SGLang then dies
# in AutoTokenizer.from_pretrained. Require the tokenizer, the shard index,
# and every shard it names.
checkpoint_is_complete() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    [[ -f "$dir/tokenizer_config.json" ]] || return 1
    [[ -f "$dir/tokenizer.json" || -f "$dir/tokenizer.model" ]] || return 1
    [[ -f "$dir/model.safetensors.index.json" ]] || return 1
    CKPT_DIR="$dir" python3 - <<'PYEOF'
import json, os, sys
d = os.environ["CKPT_DIR"]
with open(os.path.join(d, "model.safetensors.index.json")) as fh:
    shards = sorted(set(json.load(fh)["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(d, s))]
if missing:
    print(f"{len(missing)}/{len(shards)} shards missing, e.g. {missing[:3]}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    if ! checkpoint_is_complete "$MODEL_PATH"; then
        # Every concurrency runs as its own allocation against the same Lustre
        # path; one cell pulls the ~433 GB checkpoint and the rest wait on the
        # lock. hf download resumes into a partially populated --local-dir.
        mkdir -p "$MODEL_PATH"
        MODEL_DOWNLOAD_LOCK="${MODEL_PATH%/}.download.lock"
        echo "Checkpoint at $MODEL_PATH is incomplete; acquiring $MODEL_DOWNLOAD_LOCK"
        exec 9>"$MODEL_DOWNLOAD_LOCK"
        check_env_vars MODEL_DOWNLOAD_LOCK_TIMEOUT
        flock -w "$MODEL_DOWNLOAD_LOCK_TIMEOUT" 9 || {
            echo "Error: timed out waiting for another cell to stage $MODEL_PATH" >&2
            exit 1
        }
        if checkpoint_is_complete "$MODEL_PATH"; then
            echo "Another cell staged $MODEL_PATH while we waited"
        else
            hf download "$HF_MODEL_ID" --local-dir "$MODEL_PATH"
        fi
        flock -u 9
        exec 9>&-
        checkpoint_is_complete "$MODEL_PATH" || {
            echo "Error: $MODEL_PATH is still incomplete after hf download $HF_MODEL_ID." >&2
            exit 1
        }
    fi
else
    hf download "$HF_MODEL_ID"
    export MODEL_PATH="$HF_MODEL_ID"
fi
nvidia-smi

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # HiCache spills evicted prefixes to a pinned host pool. On the 1M-context
    # corpus the working set outgrows HBM past conc 8 (TP8) / 64 (DP8) and the
    # radix hit rate collapses to <0.1, so every turn re-prefills its history.
    # GLM-5.2 is MLA-family: every TP rank holds complete per-token KV. Ratio
    # 0.75 gives only 1,257,728 host slots, so c12/c16 use a 169 GB/rank
    # absolute pool (with the coupled 38.73 GB/rank DSA indexer, ~1,662 GB
    # across TP8 of the 1,731 GB b200-nscale budget).
    DEFAULT_HICACHE_RATIO=0.75
    DEFAULT_HICACHE_SIZE=0
    case "$CONC" in
        12|16) DEFAULT_HICACHE_SIZE=169 ;;
    esac
    MAX_HICACHE_SIZE=270
    HICACHE_SIZE="$DEFAULT_HICACHE_SIZE"
    if ! [[ "$HICACHE_SIZE" =~ ^[0-9]+$ ]]; then
        echo "Error: HICACHE_SIZE must be a non-negative integer, got $HICACHE_SIZE" >&2
        exit 1
    fi
    if awk -v s="$HICACHE_SIZE" -v cap="$MAX_HICACHE_SIZE" 'BEGIN { exit !(s > cap) }'; then
        echo "Error: HICACHE_SIZE=$HICACHE_SIZE exceeds configured limit $MAX_HICACHE_SIZE" >&2
        exit 1
    fi
    HICACHE_RATIO="$DEFAULT_HICACHE_RATIO"
    HICACHE_WRITE_POLICY="write_back"
    HICACHE_IO_BACKEND="direct"
    HICACHE_MEM_LAYOUT="page_first_direct"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
    if awk -v s="$HICACHE_SIZE" 'BEGIN { exit !(s > 0) }'; then
        echo "HiCache CPU tier: target_size=$HICACHE_SIZE GB, total_capacity=${TOTAL_CPU_DRAM_GB} GB, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
        CACHE_ARGS+=(--hicache-size "$HICACHE_SIZE")
    else
        if awk -v r="$HICACHE_RATIO" -v cap="$DEFAULT_HICACHE_RATIO" 'BEGIN { exit !(r > cap) }'; then
            echo "Error: HICACHE_RATIO=$HICACHE_RATIO exceeds configured limit $DEFAULT_HICACHE_RATIO" >&2
            exit 1
        fi
        echo "HiCache CPU tier: ratio=$HICACHE_RATIO, total_capacity=${TOTAL_CPU_DRAM_GB} GB, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
        CACHE_ARGS+=(--hicache-ratio "$HICACHE_RATIO")
    fi
fi

# With attention-DP, front the DP ranks with sglang-router using consistent
# hashing on the AIPerf correlation id so multi-turn sessions stay on the DP
# rank that holds their radix-cache prefix.
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
fi

# GLM-5.2 ships its own nextn head, so EAGLE runs off the checkpoint. Three
# draft tokens per step is the draft length whose golden AL is pinned below.
SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
CHUNKED_PREFILL_SIZE=8192
if [ "$DP_ATTENTION" = "true" ]; then
    # chunked-prefill-size is a whole-engine budget split across DP ranks:
    # 8192 becomes 1,024 tokens/rank/step under dp8 and a conc-256 warmup
    # could not drain within AIPerf's 1800 s grace period. 32768 = ~4096/rank.
    CHUNKED_PREFILL_SIZE=32768
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --tokenizer-worker-num "$TP"
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    )
    # The nextn layer is unquantized (hf_quant_config excludes model.layers.78*),
    # so the draft MoE is bf16 and pinned to the triton runner; inheriting the
    # target's FlashInfer all-to-all dies at init with "Pre-permute function for
    # flashinfer to triton is not registered". SGLang only applies its fix on
    # is_hip(), so set the ROCm values explicitly. Only matters with EP a2a.
    SPEC_ARGS+=(
        --speculative-moe-a2a-backend none
        --speculative-moe-runner-backend triton
    )
else
    # Cookbook low-latency levers; the DP-attention cell omits them.
    PARALLEL_ARGS+=(
        --kv-cache-dtype fp8_e4m3
        --bf16-gemm-backend cutedsl
        --max-prefill-tokens 8192
    )
fi

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts.
MAX_RUNNING_REQUESTS=$((2 * CONC))
GRAPH_ARGS=()
if [ "$DP_ATTENTION" != "true" ]; then
    # --cuda-graph-max-bs counts requests, not verification tokens; SGLang's
    # spec-decode graph runner scales by --speculative-num-draft-tokens itself.
    CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
    [ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
    GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi

# B200: 180 GB HBM3e vs B300's 288 GB. B300's 0.85 leaves 43 GB headroom
# there but 27 GB here, and the EAGLE verification activations and 4-token
# graph capture come out of it on top of the DSA indexer temporaries. 0.83
# restores ~31 GB.
MEM_FRACTION_STATIC="0.83"

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Each concurrency is a separate Slurm allocation on a shared home directory;
# keep FlashInfer autotune, Triton, Inductor and CUDA JIT caches
# allocation-local so concurrent cells cannot overwrite the same files.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export SGLANG_CACHE_DIR="/tmp/sglang-cache-${SLURM_JOB_ID}"
fi
# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session while uvicorn's
# default keep-alive is 5 s; an inter-turn idle gap can reuse a socket as the
# server closes it (ECONNRESET, terminal warmup failure).
export SGLANG_TIMEOUT_KEEP_ALIVE=900

# Acceptance is pinned to the committed golden AL (golden_al_distribution/README.md):
# 2.99 is glm5.2_mtp.yaml at num_speculative_tokens=3, thinking_on. One curve
# per model: collected on FP8, and the NVFP4 checkpoint ships the same nextn head.
# SGLANG_SIMULATE_ACC_TOKEN_MODE exists from SGLang v0.5.16; an older image
# silently honors ACC_LEN/ACC_METHOD and ignores the token mode.
# EVAL_ONLY leaves it off: simulated acceptance commits drafted tokens
# regardless of target logits and the eval would score ~0.
if [ "${EVAL_ONLY}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.99
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
    --quantization modelopt_fp4
    # GLM-5.2 emits the GLM-4.7 tool-call format; glm45 leaves calls as raw
    # text and the SWE-bench eval dies with RepeatedFormatError. Neither parser
    # affects replay throughput.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    "${SPEC_ARGS[@]}"
    "${GRAPH_ARGS[@]}"
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

echo "Starting SGLang server for B200..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
    echo "Starting SGLang router on port $PORT for $TP DP ranks..."
    python3 -m sglang_router.launch_router \
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

if [ "${EVAL_ONLY}" = "true" ]; then
    # The chat template defaults to reasoning_effort=Max when no
    # chat_template_kwargs are passed (mini-swe-agent passes none), and the heavy
    # thinking burns the shared 75-step budget (12/23 exited LimitsExceeded).
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
