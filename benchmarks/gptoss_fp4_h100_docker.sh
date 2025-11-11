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

if [[ "$RUN_MODE" == "eval" ]]; then
  # Run evaluation in-container using lm-evaluation-harness
  EVAL_RESULT_DIR=${EVAL_RESULT_DIR:-eval_out}
  OPENAI_SERVER_BASE="http://0.0.0.0:${PORT}"
  OPENAI_CHAT_BASE="$OPENAI_SERVER_BASE/v1/chat/completions"
  export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

  python3 -m pip install -q --upgrade pip || true
  python3 -m pip install -q --no-cache-dir "lm-eval[api]" || true
  # Temporary: workaround known harness issue
  python3 -m pip install -q --no-cache-dir --no-deps "git+https://github.com/EleutherAI/lm-evaluation-harness.git@main" || true
  fi

  echo "Using model: $MODEL"

  # Clean up previous eval results if any
  rm -f /workspace/"${EVAL_RESULT_DIR}"*.json

  set -x
  python3 -m lm_eval --model local-chat-completions \
    --tasks ${EVAL_TASK:-gsm8k} \
    --apply_chat_template \
    --num_fewshot ${NUM_FEWSHOT:-5} \
    --batch_size 4 \
    --output_path "/workspace/${EVAL_RESULT_DIR}" \
    --model_args "model=$MODEL,base_url=$OPENAI_CHAT_BASE,api_key=$OPENAI_API_KEY,eos_string=</s>,max_retries=3,num_concurrent=4,tokenized_requests=False" \
    --gen_kwargs "max_tokens=8192,temperature=0,top_p=1"
  set +x

  # Append a Markdown table to the GitHub Actions job summary
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    RES_DIR="${EVAL_RESULT_DIR:-eval_out}"
    # Find the most recent JSON anywhere under RES_DIR (handles nested outputs)
    RES_FILE="$(find "$RES_DIR" -type f -name '*.json' -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -n1)"

    {
      echo "### ${EVAL_TASK:-gsm8k} Evaluation"
      echo ""
      if [ -z "$RES_FILE" ]; then
        echo "> No result JSON found in \`$RES_DIR\`."
      else
        # Prefer Python (usually available on self-hosted); fall back to jq if desired
        if command -v python3 >/dev/null 2>&1; then
python3 - "$FRAMEWORK" "$PRECISION" "$TP" "$EP_SIZE" "$DP_ATTENTION" "${EVAL_TASK:-gsm8k}" "$RES_FILE" <<'PY'
import sys, json, re, os
framework, precision, tp, ep, dp, task, path = sys.argv[1:8]
with open(path, 'r') as f:
    data = json.load(f)

pe = data.get("pretty_env_info","")
gpu_lines = [l for l in pe.splitlines() if l.startswith("GPU ")]
names = [re.sub(r"GPU \d+:\s*", "", l).strip() for l in gpu_lines]
from collections import Counter
c = Counter(names)
gpu_summary = " + ".join([f"{n}\u00D7 {name}" for name, n in c.items()]) if c else "Unknown GPU"
cpu_line = next((l.split(":",1)[1].strip() for l in pe.splitlines() if l.startswith("Model name:")), None)
hardware = gpu_summary + (f" ({cpu_line})" if cpu_line else "")

task_key = task
# Fallback: if provided task missing, try first available key
res_all = data.get("results", {}) or {}
res = res_all.get(task_key) if isinstance(res_all, dict) else {}
if not res and isinstance(res_all, dict) and res_all:
    task_key = next(iter(res_all.keys()))
    res = res_all.get(task_key, {})
strict = res.get("exact_match,strict-match")
flex   = res.get("exact_match,flexible-extract")
strict_se = res.get("exact_match_stderr,strict-match")
flex_se   = res.get("exact_match_stderr,flexible-extract")
n_eff = data.get("n-samples",{}).get("gsm8k",{}).get("effective")

def pct(x): return f"{x*100:.2f}%" if isinstance(x,(int,float)) else "N/A"
def se(x):  return f" \u00B1{(x*100):.2f}%" if isinstance(x,(int,float)) else ""

print("| Hardware | Framework | Precision | TP | EP | DP Attention | EM Strict | EM Flexible | N (eff) |")
print("|---|---|---:|--:|--:|:--:|--:|--:|--:|")
print(f"| {hardware} | {framework} | {precision} | {tp} | {ep} | {str(dp).lower()} | {pct(strict)}{se(strict_se)} | {pct(flex)}{se(flex_se)} | {n_eff or ''} |")

model = data.get("model_name") or data.get("configs",{}).get(task_key,{}).get("metadata",{}).get("model")
limit = data.get("config",{}).get("limit")
fewshot = data.get("n-shot",{}).get(task_key)
lim_str = str(int(limit)) if isinstance(limit,(int,float)) else str(limit)
print(f"\n_Model_: `{model}` &nbsp;&nbsp; _k-shot_: **{fewshot}** &nbsp;&nbsp; _limit_: **{lim_str}**  \n_Source_: `{os.path.basename(path)}`")
PY
        else
          # Minimal jq fallback (prints only metrics without hardware/CPU inference)
          jq -r --arg fw "$FRAMEWORK" --arg prec "$PRECISION" --arg tp "$TP" --arg ep "$EP_SIZE" --arg dp "$DP_ATTENTION" '
            def pct: (. * 100 | tostring) + "%";
            . as $root
            | "| Hardware | Framework | Precision | TP | EP | DP Attention | EM Strict | EM Flexible | N (eff) |",
              "|---|---|---:|--:|--:|:--:|--:|--:|--:|",
              ("| Unknown GPU | \($fw) | \($prec) | \($tp) | \($ep) | \($dp) | "
               + (.results.gsm8k["exact_match,strict-match"]    | pct) + " | "
               + (.results.gsm8k["exact_match,flexible-extract"]| pct) + " | "
               + (.["n-samples"].gsm8k.effective|tostring) + " |")
          ' "$RES_FILE"
        fi
      fi
      echo ""
    } >> "$GITHUB_STEP_SUMMARY" || true
  fi

  echo "Evaluation completed. Results in /workspace/${EVAL_RESULT_DIR}"
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
