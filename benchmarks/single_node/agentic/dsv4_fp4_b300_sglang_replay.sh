#!/usr/bin/env bash
set -euo pipefail
set -x

# Custom multi-turn agentic-trace replay for DeepSeek-V4 FP4 on SGLang, single
# GB300 node. SGLang counterpart of dsv4_fp4_vllm_replay.sh: it serves the model
# with sglang.launch_server instead of vllm, then runs the SAME self-contained
# replay sweep (utils/custom_replay/sweep_pareto.py) over a *.replay.jsonl of
# Claude Code traces, sweeping "number of concurrent sessions" to trace the
# latency/throughput pareto. No weka, no aiperf, no disagg.
#
# Required env vars:
#   MODEL          FP4 DeepSeek-V4 checkpoint (path or HF id)
#   TP             tensor-parallel size (GPUs in the single engine)
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
# max running requests must be >= the largest concurrency we sweep
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

# ---- Launch DeepSeek-V4 FP4 SGLang server (single engine, pure TP) ----------
# This uses the purpose-built deepseek-v4-grace-blackwell SGLang image, whose
# transformers/sglang recognize model_type=deepseek_v4 natively (so no
# config.json model_type patch is needed — and tom.chen's checkpoint dir is
# read-only to us anyway). Prefix caching (RadixAttention) is ON by default; do
# NOT pass --disable-radix-cache — our agentic traces are ~85-95% prefix reuse,
# which is the path under test.
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
# Aligned to the proven single-node B300 dsv4 sglang recipe
# (benchmarks/single_node/fixed_seq_len/dsv4_fp4_b300_sglang.sh, TP-only profile).
# CRITICAL: the FP4 experts REQUIRE the flashinfer_mxfp4 MoE runner. Without it
# sglang falls back to the Triton fp8 fused-MoE kernel, which asserts
# "Hidden size mismatch" on FP4-packed weights (observed crash, job 6719). We
# still pass --chat-template because our replay drives /v1/chat/completions
# (the fixed-seq-len recipe uses completions, so it didn't need one).
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1

CHAT_TEMPLATE="$(cd "$(dirname "$0")/.." && pwd)/chat_templates/deepseek_v4_thinking.jinja"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --trust-remote-code
    --moe-runner-backend flashinfer_mxfp4
    --chunked-prefill-size 8192
    --disable-flashinfer-autotune
    --mem-fraction-static 0.90
    --swa-full-tokens-ratio 0.1
    --max-running-requests "$MAX_NUM_SEQS"
    --chat-template "$CHAT_TEMPLATE"
    --enable-metrics
    --watchdog-timeout 1800
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"; printf '\n' >> "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID (max-running-requests=$MAX_NUM_SEQS)"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Replayer deps via the container's own (aarch64) python ------------------
# Do NOT use a mounted x86_64 uv on PATH (Exec format error on aarch64).
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
    --title "DeepSeek-V4 FP4 SGLang TP$TP — $(basename "$DATASET")" \
    "${WARMUP_FLAG[@]}" \
    "${THINK_FLAG[@]}"

stop_gpu_monitor || true

echo "Pareto results in $RESULT_DIR (pareto.csv, pareto.png, conc*.json)"
kill "$SERVER_PID" 2>/dev/null || true
