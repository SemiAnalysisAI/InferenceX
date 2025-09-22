#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache/"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
PORT=8888

server_name="bmk-server"
client_name="bmk-client"

set -x
docker run --rm -d --network host --name $server_name \
--runtime nvidia --gpus all --ipc host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e PORT=$PORT \
-e TORCH_CUDA_ARCH_LIST="10.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
$(echo "$IMAGE" | sed 's/#/\//') \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_b200${FRAMEWORK_SUFFIX}_docker.sh"

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ Application\ startup\ complete ]]; then
        break
    fi
done < <(docker logs -f --tail=0 $server_name 2>&1)

git clone https://github.com/kimbochen/bench_serving.git

# warmup for JIT kernels - only for DeepSeek-R1-0528-FP4 model
if [[ "$MODEL" == "nvidia/DeepSeek-R1-0528-FP4" ]]; then
    echo "Running warmup for DeepSeek-R1-0528-FP4 model..."
    WARMUP_PROMPTS=$(( $CONC * 10 ))
    echo "Warmup prompts: $WARMUP_PROMPTS"
    docker run --rm --network host --name warmup-client \
    -v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
    -e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
    --entrypoint=/bin/bash \
    $(echo "$IMAGE" | sed 's/#/\//') \
    -lc "pip install -q datasets pandas && \
    python3 bench_serving/benchmark_serving.py \
    --model $MODEL --backend vllm --base-url http://localhost:$PORT \
    --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
    --num-prompts $WARMUP_PROMPTS --max-concurrency $CONC \
    --request-rate inf --ignore-eos"
fi

set -x
docker run --rm --network host --name $client_name \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=/bin/bash \
$(echo "$IMAGE" | sed 's/#/\//') \
-lc "pip install -q datasets pandas && \
python3 bench_serving/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://localhost:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json"

while [ -n "$(docker ps -aq)" ]; do
    docker stop $server_name
    sleep 5
done
