#!/usr/bin/env bash
set -euo pipefail
set -x

# Custom multi-turn agentic-trace replay for Kimi-K2.6 FP4 on vLLM, single node.
#
# Sibling of dsv4_fp4_vllm_replay.sh: same self-contained replayer
# (utils/custom_replay/) against a standardized *.replay.jsonl of Claude Code
# traces, sweeping "number of concurrent sessions" for the latency/throughput
# pareto. The *.replay.jsonl is model-agnostic, so the SAME dataset used for
# DeepSeek-V4 is replayed here. No weka, no aiperf, no disagg.
#
# Required env vars:
#   MODEL          FP4 Kimi-K2.6 checkpoint (path or HF id)
#   TP             tensor-parallel size (use 4, matching the dsv4 run)
#   DATASET        path to *.replay.jsonl (from make_replay_dataset.py)
#   RESULT_DIR     output dir for pareto.csv / pareto.png / conc*.json
# Optional:
#   CONCURRENCIES  comma list to sweep            (default 1,2,4,8,16,32,64,128)
#   DURATION       measurement window per point s (default 120)
#   WARMUP         warmup ramp per point s        (default 20)
#   MAX_MODEL_LEN  server context cap             (default 1000000)
#   MAX_NUM_SEQS   server max running seqs        (default = max(CONCURRENCIES))
#   USE_THINK_TIME 1 to replay recorded idle gaps (default 0 = saturate)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP DATASET RESULT_DIR

CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32,64,128}"
DURATION="${DURATION:-120}"
WARMUP="${WARMUP:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
# max-num-seqs must be >= the largest concurrency we sweep
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    MAX_NUM_SEQS=$(echo "$CONCURRENCIES" | tr ',' '\n' | sort -n | tail -1)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi || true

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# ---- Launch Kimi-K2.6 FP4 vLLM server (single engine, pure TP) --------------
# FP4 weights + fp8 KV + prefix caching ON (our agentic traces are ~93%
# prefix-cache reuse). Server flags mirror the repo's kimik2.5_fp4_*_vllm recipe
# (kimi_k2 reasoning/tool parsers, fused allreduce+rms, stream-interval 20),
# applied to the K2.6 checkpoint.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export PYTHONNOUSERSITE=1

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --data-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8 \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --max-cudagraph-capture-size 2048 \
    --stream-interval 20 \
    --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Replayer deps via uv (isolated from the container's vLLM python) --------
REPLAY_DIR="$(cd "$(dirname "$0")/../../../utils/custom_replay" && pwd)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
REPLAY_VENV="$RESULT_DIR/.replay-venv"
uv venv "$REPLAY_VENV" --python 3.12
uv pip install --python "$REPLAY_VENV/bin/python" -r "$REPLAY_DIR/requirements.txt"
REPLAY_PY="$REPLAY_VENV/bin/python"

# ---- Sweep concurrency -> pareto --------------------------------------------
THINK_FLAG=()
[ "${USE_THINK_TIME:-0}" = "1" ] && THINK_FLAG=(--use-think-time)

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
    --title "Kimi-K2.6 FP4 vLLM TP$TP — $(basename "$DATASET")" \
    "${THINK_FLAG[@]}"

stop_gpu_monitor || true

echo "Pareto results in $RESULT_DIR (pareto.csv, pareto.png, conc*.json)"
kill "$SERVER_PID" 2>/dev/null || true
