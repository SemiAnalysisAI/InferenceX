#!/usr/bin/env bash
set -euo pipefail
set -x

# Custom multi-turn agentic-trace replay for Kimi-K2.5 NVFP4 on SGLang, single
# GB300 node. SGLang counterpart of kimik2.5_fp4_b300_replay.sh (vLLM): serves
# the model with sglang.launch_server, then runs the SAME self-contained replay
# sweep (utils/custom_replay/sweep_pareto.py) over a *.replay.jsonl, sweeping
# "number of concurrent sessions" for the latency/throughput pareto.
#
# NOTE: the replayer sends messages and measures token timing with ignore_eos +
# fixed max_tokens, then DISCARDS the output. It does not parse tool calls or
# reasoning, so we omit --tool-call-parser / --reasoning-parser here (their
# sglang names for kimi_k25 are unverified and would only add startup-failure
# surface). The model's bundled chat_template.jinja and hf_quant_config.json
# (NVFP4 / modelopt) are picked up automatically.
#
# Required env vars:
#   MODEL          NVFP4 Kimi-K2.5 checkpoint (path or HF id)
#   TP             tensor-parallel size
#   DATASET        path to *.replay.jsonl (from make_replay_dataset.py)
#   RESULT_DIR     output dir for pareto.csv / pareto.png / conc*.json
# Optional:
#   CONCURRENCIES  comma list to sweep            (default 1,2,4,8,16,32,64,128)
#   WARMUP         1 = untimed prefix-cache prime before each point (default 1)
#   MAX_MODEL_LEN  server context cap             (default 262144)
#   MAX_NUM_SEQS   server max running seqs        (default = max(CONCURRENCIES))
#   USE_THINK_TIME 1 to replay recorded idle gaps (default 0 = saturate)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP DATASET RESULT_DIR

CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32,64,128}"
WARMUP="${WARMUP:-1}"          # 1 = prime prefix cache before each point (default on)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
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

# ---- Launch Kimi-K2.5 NVFP4 SGLang server (single engine, pure TP) ----------
# RadixAttention prefix caching is ON by default (do NOT --disable-radix-cache);
# our agentic traces are ~85-95% prefix reuse, the path under test.
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --trust-remote-code
    --page-size 256
    --context-length "$MAX_MODEL_LEN"
    --max-running-requests "$MAX_NUM_SEQS"
    --cuda-graph-max-bs "$MAX_NUM_SEQS"
    --enable-metrics
    --watchdog-timeout 1800
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"; printf '\n' >> "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID (max-running-requests=$MAX_NUM_SEQS)"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Replayer deps via the container's own (aarch64) python ------------------
REPLAY_DIR="$(cd "$(dirname "$0")/../../../utils/custom_replay" && pwd)"
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
WARMUP_FLAG=()
[ "$WARMUP" = "1" ] && WARMUP_FLAG=(--warmup)

start_gpu_monitor --output "$RESULT_DIR/gpu_metrics.csv" --interval 1 || true

"$REPLAY_PY" "$REPLAY_DIR/sweep_pareto.py" \
    --dataset "$DATASET" \
    --base-url "http://0.0.0.0:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --concurrencies "$CONCURRENCIES" \
    --result-dir "$RESULT_DIR" \
    --title "Kimi-K2.5 NVFP4 SGLang TP$TP — $(basename "$DATASET")" \
    "${WARMUP_FLAG[@]}" \
    "${THINK_FLAG[@]}"

stop_gpu_monitor || true

echo "Pareto results in $RESULT_DIR (pareto.csv, pareto.png, conc*.json)"
kill "$SERVER_PID" 2>/dev/null || true
