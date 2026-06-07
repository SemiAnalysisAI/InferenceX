#!/usr/bin/env bash
set -euo pipefail
set -x

# Multi-turn trace-replay benchmark for Kimi-K2.5 NVFP4 on B300 using vLLM.
#
# Based on the repo's kimik2.5_fp4_b300.sh server recipe, but swaps the
# aiperf/weka agentic path (resolve_trace_source / build_replay_cmd /
# run_agentic_replay_and_write_outputs) for our self-contained replay sweep
# (utils/custom_replay/sweep_pareto.py over a *.replay.jsonl), sweeping
# "number of concurrent sessions" to trace the latency/throughput pareto.
# No weka, no aiperf.
#
# Required env vars:
#   MODEL       NVFP4 Kimi-K2.5 checkpoint (path or HF id)
#   TP          tensor-parallel size
#   DATASET     path to *.replay.jsonl (from make_replay_dataset.py)
#   RESULT_DIR  output dir for pareto.csv / pareto.png / conc*.json
# Optional:
#   OFFLOADING        none|cpu|lmcache       (default none)
#   TOTAL_CPU_DRAM_GB CPU KV pool for offload (default 2500)
#   CONCURRENCIES     comma list to sweep    (default 1,2,4,8,16,32,64,128)
#   DURATION          window per point s     (default 120)
#   WARMUP            warmup per point s      (default 20)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP DATASET RESULT_DIR

OFFLOADING="${OFFLOADING:-none}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32,64,128}"
DURATION="${DURATION:-120}"
WARMUP="${WARMUP:-20}"
# server max batch must cover the largest concurrency we sweep
MAX_NUM_SEQS=$(echo "$CONCURRENCIES" | tr ',' '\n' | sort -n | tail -1)

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- Server config (mirrors kimik2.5_fp4_b300.sh) ---------------------------
OFFLOAD_ARGS=()
case "$OFFLOADING" in
    none) ;;
    cpu)
        TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2500}"
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    lmcache)
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        agentic_pip_install --quiet --no-cache-dir lmcache
        python3 -c "import lmcache.integration.vllm.vllm_v1_adapter" >/dev/null
        TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2500}"
        export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        export LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR="${LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR:-true}"
        export LMCACHE_LAZY_MEMORY_INITIAL_RATIO="${LMCACHE_LAZY_MEMORY_INITIAL_RATIO:-0.01}"
        export LMCACHE_LAZY_MEMORY_STEP_RATIO="${LMCACHE_LAZY_MEMORY_STEP_RATIO:-0.02}"
        OFFLOAD_ARGS=(
            --kv-offloading-backend lmcache
            --kv-offloading-size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    *) echo "Error: unsupported OFFLOADING '$OFFLOADING' (none|cpu|lmcache)" >&2; exit 1 ;;
esac

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
VLLM_CMD=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size="$TP"
    --gpu-memory-utilization 0.90
    --max-num-seqs "$MAX_NUM_SEQS"
    --reasoning-parser kimi_k2
    --tool-call-parser kimi_k2
    --compilation_config.pass_config.fuse_allreduce_rms true
    --kv-cache-dtype fp8
    --max-cudagraph-capture-size 2048
    --stream-interval 20
    --trust-remote-code
    --enable-prefix-caching
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"; printf '\n' >> "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID (max-num-seqs=$MAX_NUM_SEQS)"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Replay sweep (our path, replaces aiperf/weka) --------------------------
REPLAY_DIR="$(cd "$(dirname "$0")/../../../utils/custom_replay" && pwd)"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
REPLAY_VENV="$RESULT_DIR/.replay-venv"
uv venv "$REPLAY_VENV" --python 3.12
uv pip install --python "$REPLAY_VENV/bin/python" -r "$REPLAY_DIR/requirements.txt"
REPLAY_PY="$REPLAY_VENV/bin/python"

start_gpu_monitor --output "$RESULT_DIR/gpu_metrics.csv" --interval 1 || true

"$REPLAY_PY" "$REPLAY_DIR/sweep_pareto.py" \
    --dataset "$DATASET" \
    --base-url "http://0.0.0.0:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --concurrencies "$CONCURRENCIES" \
    --duration "$DURATION" \
    --warmup "$WARMUP" \
    --result-dir "$RESULT_DIR" \
    --title "Kimi-K2.5 NVFP4 vLLM TP$TP — $(basename "$DATASET")"

stop_gpu_monitor || true

echo "Pareto results in $RESULT_DIR (pareto.csv, pareto.png, conc*.json)"
kill "$SERVER_PID" 2>/dev/null || true
