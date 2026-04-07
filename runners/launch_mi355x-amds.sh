#!/usr/bin/env bash

SQUASH_FILE="/var/lib/squash/rocm_sgl-dev_v0.5.10rc0-rocm720-mi35x-20260402.sqsh"

set -x
salloc --partition="compute" --gres=gpu:4 --exclusive \
    --cpus-per-task=128 --time=180 --no-shell --job-name="$RUNNER_NAME"
JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)

srun --jobid=$JOB_ID \
  --container-image=$SQUASH_FILE \
  --container-mounts=$(pwd):/workspace/,/var/lib/hf-hub-cache/:/mnt/hf_hub_cache/ \
  --container-mount-home \
  --container-writable \
  --container-workdir=/workspace/ \
  --no-container-entrypoint \
  --export=ALL \
  bash -lc '
set -ex

export SGLANG_USE_AITER=1
export HF_HUB_CACHE=/mnt/hf_hub_cache/

# Disable buggy fused GDN projection Triton kernel from sglang commit 5bdc07d
python3 <<PATCH_EOF
for path in [
    "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py",
    "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_next.py",
]:
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "fused_qkvzba_split_reshape_cat" in line and "(" in line and "import" not in line and "def " not in line:
            for j in range(i - 1, max(i - 5, -1), -1):
                if "if _is_cuda" in lines[j]:
                    lines[j] = lines[j].replace("if _is_cuda", "if False  # disabled: 5bdc07d")
                    break
            break
    with open(path, "w") as f:
        f.writelines(lines)
PATCH_EOF

python3 -m sglang.launch_server \
    --model-path amd/Qwen3.5-397B-A17B-MXFP4 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 9000 \
    --tensor-parallel-size 4 \
    --attention-backend aiter \
    --mem-fraction-static 0.9 \
    --model-loader-extra-config '"'"'{"enable_multithread_load": true}'"'"' \
    --watchdog-timeout 1200 \
    --context-length 9416 \
    > /tmp/server.log 2>&1 &
SERVER_PID=$!

pip install -q --no-cache-dir "lm-eval[api]" || true
pip install -q --no-cache-dir --no-deps --force-reinstall \
    "git+https://github.com/EleutherAI/lm-evaluation-harness.git@b315ef3b05176acc9732bb7fdec116abe1ecc476" || true

# Wait for server ready
echo "Waiting for server on port 9000 ..."
for i in $(seq 1 120); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died. Last 50 lines:"
        tail -50 /tmp/server.log
        exit 1
    fi
    if curl -sf http://0.0.0.0:9000/health > /dev/null 2>&1; then
        echo "Server ready after ${i}x5s"
        break
    fi
    sleep 5
done

tail /tmp/server.log

# Smoke test
python3 - <<'"'"'PY'"'"'
import hashlib, json, os, urllib.request
port = "9000"
model = "amd/Qwen3.5-397B-A17B-MXFP4"
url = f"http://0.0.0.0:{port}/v1/chat/completions"
tests = [
    ("mul", "What is 17 * 19? Respond with only the number."),
    ("bags", "Tom has 3 bags with 4 apples each and buys 2 more apples. End with #### <number>."),
]
for name, prompt in tests:
    rows = []
    for _ in range(3):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1, "max_tokens": 256}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        out = json.load(urllib.request.urlopen(req, timeout=300))
        text = out["choices"][0]["message"].get("content") or out["choices"][0]["message"].get("reasoning_content") or ""
        rows.append({"len": len(text), "sha16": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text[:200]})
    print(name, json.dumps(rows, ensure_ascii=False))
PY

# Run eval
export OPENAI_API_KEY=EMPTY
python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --tasks utils/evals/gsm8k.yaml \
    --output_path /tmp/eval_results \
    --log_samples --limit 200 \
    --model_args "model=amd/Qwen3.5-397B-A17B-MXFP4,base_url=http://0.0.0.0:9000/v1/chat/completions,api_key=EMPTY,eos_string=,max_retries=5,num_concurrent=64,timeout=1800,tokenized_requests=False,max_length=9416" \
    --gen_kwargs "max_tokens=5320,temperature=0,top_p=1"

# Copy results for artifact upload
find /tmp/eval_results -name "results*.json" -exec cp {} /workspace/ \;
find /tmp/eval_results -name "samples*.jsonl" -exec cp {} /workspace/ \;
'

scancel $JOB_ID
