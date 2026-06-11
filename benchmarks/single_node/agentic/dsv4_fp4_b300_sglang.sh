#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on B300 using SGLang.
#
# OFFLOADING values:
#   none    - SGLang GPU KV cache with RadixAttention prefix caching.
#   hicache - SGLang HiCache local CPU tier with DSv4 UnifiedRadixCache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"

# The B200 DeepSeek-V4 Blackwell image installs SGLang editable under
# /workspace, so its launcher mounts InferenceX at /ix instead. Resolve the
# agentic tooling and results against the actual repository mount so the image
# can keep its /workspace install and GitHub Actions can collect the outputs.
if [[ ! -d "$INFMAX_CONTAINER_WORKSPACE/utils/aiperf" ]]; then
    export INFMAX_CONTAINER_WORKSPACE="$INFERENCEX_ROOT"
fi
if [[ "${RESULT_DIR:-}" == /workspace/* && "$INFMAX_CONTAINER_WORKSPACE" != /workspace ]]; then
    export RESULT_DIR="$INFMAX_CONTAINER_WORKSPACE/${RESULT_DIR#/workspace/}"
fi

source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=1000000
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
nvidia-smi

resolve_trace_source

# Keep AIPerf's Transformers-main dependency from replacing the older
# Transformers build pinned by the B200-specialized SGLang image. The server
# always launches with the image's original interpreter; AIPerf and result
# processing use the isolated environment when InferenceX is mounted at /ix.
SGLANG_PYTHON="$(command -v python3)"
if [[ "$INFMAX_CONTAINER_WORKSPACE" != /workspace ]]; then
    AGENTIC_VENV="${AGENTIC_VENV:-/tmp/inferencex-agentic-venv}"
    "$SGLANG_PYTHON" -m venv "$AGENTIC_VENV"
    export PATH="$AGENTIC_VENV/bin:$PATH"
fi
install_agentic_deps

# The 2026-06-09 image creates every local DP scheduler from the same HTTP
# port, so PortArgs derives the same RPC/metrics TCP range for every rank and
# rank 1 collides with rank 0 before startup completes. Give each scheduler a
# private derivation port until the upstream image includes the fix.
if [ "$DP_ATTENTION" = "true" ]; then
    "$SGLANG_PYTHON" - <<'PY'
from pathlib import Path

controller_path = Path("/sgl-workspace/sglang/python/sglang/srt/managers/data_parallel_controller.py")
source = controller_path.read_text()
old = "            tmp_port_args = PortArgs.init_new(server_args)\n"
new = (
    "            rank_server_args = dataclasses.replace(\n"
    "                server_args, port=server_args.port + (dp_rank + 1) * 100\n"
    "            )\n"
    "            tmp_port_args = PortArgs.init_new(rank_server_args)\n"
)
if old in source:
    source = source.replace("import faulthandler\n", "import dataclasses\nimport faulthandler\n", 1)
    source = source.replace(old, new, 1)
    controller_path.write_text(source)
elif new not in source:
    raise RuntimeError(f"Unexpected SGLang DP controller source: {controller_path}")
if new not in controller_path.read_text():
    raise RuntimeError(f"Failed to patch SGLang DP controller: {controller_path}")
server_args_path = Path("/sgl-workspace/sglang/python/sglang/srt/server_args.py")
source = server_args_path.read_text()
old = (
    '                    wait_port_available(rpc_port, "rpc_port")\n'
    '                    wait_port_available(metrics_port, "metrics_port")\n'
)
if old in source:
    source = source.replace(old, "", 1)
    server_args_path.write_text(source)
elif old in server_args_path.read_text():
    raise RuntimeError(f"Failed to patch SGLang DP port checks: {server_args_path}")
PY
fi

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
case "$OFFLOADING" in
    none)
        ;;
    hicache)
        # DeepSeek V4 HiCache currently rejects --hicache-size and supports
        # capacity control only through a host/device token-capacity ratio.
        # DSv4 allocates several physical host sub-pools for each logical host
        # token. On B300 TP8, ratio=4 consumes about 237 GB/rank (1.9 TB total)
        # while model loading/page cache is still resident and the OS kills a
        # rank, so leave transient startup headroom with ratio=2. B200 has a
        # smaller device KV pool and 3.8 TiB of host RAM, so ratio=8 provides a
        # substantially larger useful CPU tier while staying within its node
        # budget.
        # TP4 ratio=4 works at C32 but fills its roughly 500 GB host tier at
        # C48/C64. Ratio=8 still cannot retain the C64 session working set long
        # enough to produce host hits. Ratio=16 provides roughly 21M logical
        # host tokens while remaining below the B300 node's host budget.
        if [ "$TP" -ge 8 ]; then
            DEFAULT_HICACHE_RATIO=2
        else
            DEFAULT_HICACHE_RATIO=16
        fi
        HICACHE_RATIO="${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
        export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
        CACHE_ARGS=(
            --enable-hierarchical-cache
            --hicache-ratio "$HICACHE_RATIO"
            --hicache-write-policy "$HICACHE_WRITE_POLICY"
            --hicache-io-backend "$HICACHE_IO_BACKEND"
            --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
        )
        echo "HiCache DSv4 CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
fi

PARALLEL_ARGS=(--tp "$TP")
METRICS_ARGS=(--enable-metrics)
MEM_FRACTION_STATIC=0.88
CHUNKED_PREFILL_SIZE=8192
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
        --ep-size "$EP_SIZE"
        --moe-runner-backend flashinfer_mxfp4
        --disable-flashinfer-autotune
        --enable-prefill-delayer
    )
    METRICS_ARGS=()
    MEM_FRACTION_STATIC=0.84
    CHUNKED_PREFILL_SIZE=16384
else
    PARALLEL_ARGS+=(
        --moe-runner-backend flashinfer_mxfp4
        --disable-flashinfer-autotune
    )
fi

MODEL_ARGS=(
    --attention-backend compressed
    --page-size 256
    --disable-shared-experts-fusion
)

MAX_RUNNING_REQUESTS=$CONC
CUDA_GRAPH_MAX_BS=$CONC
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Agentic warmup dispatches hundreds of large prompts at once. SGLang's
# tokenizer process can leave request bytes unacknowledged for longer than
# AIPerf's 30-second TCP_USER_TIMEOUT while it admits that initial burst,
# causing Linux to abort otherwise-live localhost connections. Keep the
# six-hour request timeout unchanged, but allow up to 15 minutes for TCP
# progress before declaring the connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
SGLANG_CMD=(
    "$SGLANG_PYTHON" -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --swa-full-tokens-ratio 0.1
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --context-length "$MAX_MODEL_LEN"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chat-template "$SCRIPT_DIR/../chat_templates/deepseek_v4_thinking.jinja"
    --watchdog-timeout 1800
    "${MODEL_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for B300..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

capture_cache_metrics() {
    {
        echo "=== SGLang cache metrics snapshot $(date --iso-8601=seconds) ==="
        curl -fsS "http://localhost:$SGLANG_BACKEND_PORT/metrics" 2>/dev/null \
            | grep -E '^(sglang:(cache_hit_rate|cached_tokens_total|prompt_tokens_total|hicache_host_used_tokens|hicache_host_total_tokens|token_usage|num_requests_running|num_requests_waiting))' \
            || true
        echo "============================================================"
    } >> "$SERVER_LOG"
}

wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
    echo "Starting SGLang router on port $PORT for $TP DP ranks..."
    "$SGLANG_PYTHON" -m sglang_router.launch_router \
        --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
        --policy manual \
        --assignment-mode min_load \
        --request-id-headers x-correlation-id \
        --dp-aware \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
        --request-timeout-secs 3600 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${#METRICS_ARGS[@]}" -gt 0 ]; then
    capture_cache_metrics
    trap capture_cache_metrics EXIT
fi

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
