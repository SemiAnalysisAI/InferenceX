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
    # Eval mode: run lm-eval against the OpenAI-compatible vLLM endpoint
    mkdir -p "${EVAL_RESULT_DIR:-eval_out}"
    # Allow overriding the OpenAI model name if the served name differs from HF repo id.
    # Defaults to the trailing component (e.g., 'openai/gpt-oss-120b' -> 'gpt-oss-120b').
    OPENAI_MODEL_NAME_COMPUTED="${OPENAI_MODEL_NAME:-${MODEL##*/}}"
    # Allow overriding the client image used for lm-eval. Default to reuse server image to avoid pulling extras.
    LM_EVAL_IMAGE="${LM_EVAL_IMAGE:-$IMAGE}"
    set -x
    docker run --rm --network=host --name=$client_name \
    -v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
    --entrypoint=/bin/bash \
    $LM_EVAL_IMAGE \
    -lc "export PIP_NO_CACHE_DIR=1; \
         (python3 -m pip install -q --upgrade pip || true); \
         python3 -m pip install -q --no-cache-dir 'lm-eval[api]' && \
         python3 -m lm_eval --model openai-chat-completions \
                --tasks ${EVAL_TASK:-gsm8k} \
                --num_fewshot ${NUM_FEWSHOT:-5} \
                --limit ${LIMIT:-200} \
                --batch_size auto \
                --output_path /workspace/${EVAL_RESULT_DIR:-eval_out} \
                --model_args model=${OPENAI_MODEL_NAME_COMPUTED},api_base=http://localhost:${PORT:-8888},api_key=EMPTY,temperature=0.0"
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
