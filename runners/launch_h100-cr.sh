#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
PORT=8888

server_name="bmk-server"
client_name="bmk-client"
RUN_MODE="${RUN_MODE:-benchmark}"

set -x
docker run --rm -d --network=host --name=$server_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e PORT=$PORT \
-e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_h100_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

if ! docker ps --format "{{.Names}}" | grep -q "$server_name"; then
    echo "Server container launch failed."
    exit 1
fi

if [[ "$RUN_MODE" == "eval" ]]; then
  mkdir -p "${EVAL_RESULT_DIR:-eval_out}"

  OPENAI_SERVER_BASE="http://localhost:${PORT:-8888}"
  OPENAI_CHAT_BASE="$OPENAI_SERVER_BASE/v1/chat/completions"

  LM_EVAL_IMAGE="${LM_EVAL_IMAGE:-$IMAGE}"

  set -x
  docker run --rm --network=host --name="$client_name" \
    -v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/ \
    -e OPENAI_API_KEY=EMPTY \
    -e OPENAI_SERVER_BASE="$OPENAI_SERVER_BASE" \
    -e OPENAI_CHAT_BASE="$OPENAI_CHAT_BASE" \
    -e OPENAI_MODEL_NAME_COMPUTED="${OPENAI_MODEL_NAME:-}" \
    --entrypoint=/bin/bash \
    "$LM_EVAL_IMAGE" \
    -lc 'python3 -m pip install -q --upgrade pip || true; \
         python3 -m pip install -q --no-cache-dir "lm-eval[api]"; \
         # Health check (GET); avoids 405s on POST-only routes
         curl -fsS "$OPENAI_SERVER_BASE/health" >/dev/null || { echo "Health check failed"; exit 1; }; \
         # Resolve served model id if not provided
         if [ -z "$OPENAI_MODEL_NAME_COMPUTED" ]; then \
           OPENAI_MODEL_NAME_COMPUTED=$(curl -fsS "$OPENAI_SERVER_BASE/v1/models" \
             | python3 - <<PY
import sys, json
d=json.load(sys.stdin)
print((d.get("data") or [{}])[0].get("id",""))
PY
); fi; \
         echo "Using model: $OPENAI_MODEL_NAME_COMPUTED"; \
         # One POST sanity to ensure chat endpoint + model both work
         curl -fsS -X POST "$OPENAI_CHAT_BASE" -H "Content-Type: application/json" \
           -d "{\"model\":\"$OPENAI_MODEL_NAME_COMPUTED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"tool_choice\":\"none\",\"response_format\":{\"type\":\"text\"}}" >/dev/null || { echo "Chat POST failed"; exit 1; }; \
         # Run lm-eval with strong API-side guards
         python3 -m lm_eval --model local-chat-completions \
           --tasks ${EVAL_TASK:-gsm8k} \
           --apply_chat_template \
           --num_fewshot ${NUM_FEWSHOT:-5} \
           --limit ${LIMIT:-200} \
           --batch_size 1 \
           --output_path /workspace/${EVAL_RESULT_DIR:-eval_out} \
           --model_args "model=$OPENAI_MODEL_NAME_COMPUTED,base_url=$OPENAI_CHAT_BASE,api_key=$OPENAI_API_KEY,temperature=0.0,eos_string=</s>,max_retries=12,timeout=120,num_concurrent=1,stop=Question:,stop=</s>,stop=<|im_end|>,extra_body={\"tool_choice\":\"none\",\"response_format\":{\"type\":\"text\"}}"' 
else
    # Benchmark mode: original throughput client
    git clone https://github.com/kimbochen/bench_serving.git

    set -x
    docker run --rm --network=host --name=$client_name \
    -v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
    -e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
    --entrypoint=/bin/bash \
    $IMAGE \
    -lc "pip install -q datasets pandas && \
    python3 bench_serving/benchmark_serving.py \
    --model=$MODEL \
    --backend=vllm \
    --base-url=\"http://localhost:$PORT\" \
    --dataset-name=random \
    --random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
    --num-prompts=$(( $CONC * 10 )) --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics='ttft,tpot,itl,e2el' \
    --result-dir=/workspace/ \
    --result-filename=$RESULT_FILENAME.json"
fi

docker stop $server_name
