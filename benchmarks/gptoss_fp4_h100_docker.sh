#!/usr/bin/env bash

# === Required Env Vars ===
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME
set -euo pipefail

# Create a basic vLLM config
cat > config.yaml << EOF
compilation-config: '{"cudagraph_mode":"PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
export TORCH_CUDA_ARCH_LIST="9.0"
PORT=${PORT:-8888}

# Start server in the background, shld be openai/gpt-oss-120b
set -x
PYTHONNOUSERSITE=1 vllm serve "$MODEL" --host=0.0.0.0 --port="$PORT" \
  --config config.yaml \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size="$TP" \
  --max-num-seqs="$CONC" \
  --served-model-name "gpt-oss-120b" \
  --disable-log-requests > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

cleanup() {
  set +e
  if ps -p "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait for server readiness via log message
echo "Waiting for vLLM server to become ready on port $PORT ..."
READY=0
for i in $(seq 1 600); do
  if grep -q "Application startup complete" "$SERVER_LOG"; then
    READY=1
    break
  fi
  if ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM server exited prematurely. Last 200 lines of log:" >&2
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$READY" -ne 1 ]]; then
  echo "Timed out waiting for vLLM server readiness." >&2
  tail -n 200 "$SERVER_LOG" >&2 || true
  exit 1
fi
echo "vLLM server up"
RUN_MODE=${RUN_MODE:-benchmark}

curl -s "http://0.0.0.0:${PORT}/v1/models"

if [[ "$RUN_MODE" == "eval" ]]; then
  EVAL_RESULT_DIR=${EVAL_RESULT_DIR:-eval_out}
  OPENAI_SERVER_BASE="http://0.0.0.0:${PORT}"
  OPENAI_CHAT_BASE="$OPENAI_SERVER_BASE/v1/chat/completions"
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

  # --- Install LightEval (no lm-eval; no sitecustomize patch needed) ---
  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir "lighteval" || true
  python3 -m pip install -q --no-cache-dir "litellm[caching]>=1.66.0" || true

  echo "Using model: ${MODEL}"
  rm -rf "/workspace/${EVAL_RESULT_DIR}"/* 2>/dev/null || true

  # Task spec: encode few-shot as gsm8k|k (default k=5)
  TASK_BASE="${EVAL_TASK:-gsm8k}"
  FEWSHOT="${NUM_FEWSHOT:-5}"
  TASK_SPEC="${TASK_BASE}|${FEWSHOT}"
  MODEL_ID="${OPENAI_MODEL_NAME:-$MODEL}"
  MODEL_ARGS="provider=openai,model_name=${MODEL_ID},base_url=http://0.0.0.0:${PORT}/v1,api_key=${OPENAI_API_KEY}"

  set -x
  lighteval endpoint litellm \
    "${MODEL_ARGS}" \
    "${TASK_SPEC}" \
    --output-dir "/workspace/${EVAL_RESULT_DIR}" \
    --max-samples "${LIMIT:-0}" \
    --remove-reasoning-tags
  set +x
else

  # Default values for optional vars used by the benchmark
  ISL=${ISL:-256}
  OSL=${OSL:-256}
  RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1.0}
  RESULT_FILENAME=${RESULT_FILENAME:-results}

  # Install deps and run benchmark in the same container
  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir datasets pandas || true
  if [[ ! -d bench_serving ]]; then
    git clone https://github.com/kimbochen/bench_serving.git
  fi

  set -x
  python3 bench_serving/benchmark_serving.py \
    --model="$MODEL" \
    --backend=vllm \
    --base-url="http://0.0.0.0:$PORT" \
    --dataset-name=random \
    --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio="$RANDOM_RANGE_RATIO" \
    --num-prompts=$(( CONC * 10 )) --max-concurrency="$CONC" \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics='ttft,tpot,itl,e2el' \
    --result-dir=/workspace/ \
    --result-filename="$RESULT_FILENAME.json"
  set +x
fi
