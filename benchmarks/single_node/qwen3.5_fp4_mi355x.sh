#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

export SGLANG_USE_AITER=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--attention-backend aiter \
--mem-fraction-static=0.9 \
--model-loader-extra-config '{"enable_multithread_load": true}' \
--watchdog-timeout 1200 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# === DEBUG: environment + version dump ===
echo "=== ENVIRONMENT DIAGNOSTICS ==="
echo "host=$(hostname) procid=${SLURM_PROCID:-unset} ntasks=${SLURM_NTASKS:-unset} cpt=${SLURM_CPUS_PER_TASK:-unset}"
for fd in 0 1 2; do echo "fd$fd -> $(readlink /proc/self/fd/$fd)"; done
grep Cpus_allowed_list /proc/self/status || true
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"
echo "HSA_OVERRIDE_CPU_AFFINITY_DEBUG=${HSA_OVERRIDE_CPU_AFFINITY_DEBUG:-unset}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python3 - <<'PY'
import os, sys
import torch
print("python", sys.version)
print("torch", torch.__version__)
try:
    import aiter
    print("aiter", getattr(aiter, "__file__", None), getattr(aiter, "__version__", None))
except Exception as e:
    print("aiter import failed", e)
try:
    import sglang
    print("sglang", getattr(sglang, "__file__", None), getattr(sglang, "__version__", None))
except Exception as e:
    print("sglang import failed", e)
PY

pip show torch triton sglang aiter 2>/dev/null || true
rocminfo 2>/dev/null | sed -n '1,80p' || true
echo "=== END ENVIRONMENT DIAGNOSTICS ==="

# === DEBUG: test prompts to check model output quality ===
echo "=== SMOKE TEST PROMPTS ==="
python3 - <<'PY'
import hashlib, json, os, urllib.request

port = os.environ.get("PORT", "8888")
model = os.environ["MODEL"]
url = f"http://0.0.0.0:{port}/v1/chat/completions"

tests = [
    ("mul", "What is 17 * 19? Respond with only the number."),
    ("bags", "Tom has 3 bags with 4 apples each and buys 2 more apples. End with #### <number>."),
]

for name, prompt in tests:
    rows = []
    for _ in range(3):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 256,
        }).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        out = json.load(urllib.request.urlopen(req, timeout=300))
        msg = out["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        rows.append({
            "len": len(text),
            "sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text": text[:200],
        })
    print(name, json.dumps(rows, ensure_ascii=False))
PY
echo "=== END SMOKE TEST ==="

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
