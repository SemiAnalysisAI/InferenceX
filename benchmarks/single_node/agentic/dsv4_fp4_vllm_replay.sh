#!/usr/bin/env bash
set -euo pipefail
set -x

# Custom multi-turn agentic-trace replay for DeepSeek-V4 FP4 on vLLM, single node.
#
# Unlike the aiperf/weka agentic launchers in this directory, this one uses our
# self-contained replayer (utils/custom_replay/) against a standardized
# *.replay.jsonl of Claude Code traces, and sweeps "number of concurrent
# sessions" to trace the latency/throughput pareto. No weka, no aiperf, no
# disagg.
#
# Required env vars:
#   MODEL          FP4 DeepSeek-V4 checkpoint (path or HF id), e.g. deepseek-ai/DeepSeek-V4-Pro
#   TP             tensor-parallel size (GPUs in the single engine)
#   DATASET        path to *.replay.jsonl (from make_replay_dataset.py)
#   RESULT_DIR     output dir for pareto.csv / pareto.png / conc*.json
# Optional:
#   CONCURRENCIES  comma list to sweep            (default 1,2,4,8,16,32,64,128)
#   REPEATS        replay dataset N times; repeat>0 gets a varied prefix =
#                  realistic cache miss (default 1). Completion-based: every
#                  turn runs to completion, no time window.
#   MAX_MODEL_LEN  server context cap             (default 1000000)
#   MAX_NUM_SEQS   server max running seqs        (default = max(CONCURRENCIES))
#   USE_THINK_TIME 1 to replay recorded idle gaps (default 0 = saturate)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP DATASET RESULT_DIR

CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32,64,128}"
REPEATS="${REPEATS:-1}"
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

# ---- Launch DeepSeek-V4 FP4 vLLM server (single engine, pure TP) ------------
# FP4 weights + fp8 KV + the FP4 indexer cache, mirroring the repo's
# dsv4_fp4_*_vllm recipe. Prefix caching ON — our agentic traces are ~93%
# prefix-cache reuse, so this is the path under test.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export PYTHONNOUSERSITE=1
export VLLM_FLOAT32_MATMUL_PRECISION=high

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --tensor-parallel-size "$TP" \
    --data-parallel-size 1 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --attention_config.use_fp4_indexer_cache=True \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Replayer deps via uv (isolated from the container's vLLM python) --------
REPLAY_DIR="$(cd "$(dirname "$0")/../../../utils/custom_replay" && pwd)"
# Use the container's own (aarch64) python for the replayer deps. Do NOT use uv:
# a mounted x86_64 uv on PATH causes "cannot execute binary file: Exec format error".
REPLAY_VENV="$RESULT_DIR/.rvenv"
if python3 -m venv --system-site-packages "$REPLAY_VENV" 2>/dev/null; then
    "$REPLAY_VENV/bin/pip" install -q -r "$REPLAY_DIR/requirements.txt"
    REPLAY_PY="$REPLAY_VENV/bin/python"
else
    echo "venv unavailable; installing replayer deps into container python"
    python3 -m pip install --break-system-packages -q -r "$REPLAY_DIR/requirements.txt"
    REPLAY_PY=python3
fi

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
    --repeats "$REPEATS" \
    --result-dir "$RESULT_DIR" \
    --title "DeepSeek-V4 FP4 vLLM TP$TP — $(basename "$DATASET")" \
    "${THINK_FLAG[@]}"

stop_gpu_monitor || true

echo "Pareto results in $RESULT_DIR (pareto.csv, pareto.png, conc*.json)"
kill "$SERVER_PID" 2>/dev/null || true
