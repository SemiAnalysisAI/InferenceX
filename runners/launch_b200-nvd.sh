#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache/"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
PORT=8888

# Create unique cache directory based on model parameters
MODEL_NAME=$(basename "$MODEL")

server_name="bmk-server"
client_name="bmk-client"

nvidia-smi

# GPUs must be idle
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "[ERROR] GPU busy from previous run"; nvidia-smi; exit 1
fi

set -x
# Use --init flag to run an init process (PID 1) inside container for better signal handling and zombie process cleanup
# Ref: https://www.paolomainardi.com/posts/docker-run-init/

# NCCL_GRAPH_REGISTER tries to automatically enable user buffer registration with CUDA Graphs. 
# Disabling it can reduce perf but will improve CI stability. i.e. we won't see vLLM/Sglang crashes.
# Ref: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-register


docker run --rm -d --init --network host --name $server_name \
--runtime nvidia --gpus all --ipc host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e PORT=$PORT \
-e NCCL_GRAPH_REGISTER=0 \
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


if [[ "$MODEL" == "nvidia/DeepSeek-R1-0528-FP4" || "$MODEL" == "deepseek-ai/DeepSeek-R1-0528" ]]; then
  if [[ "$OSL" == "8192" ]]; then
    NUM_PROMPTS=$(( CONC * 20 ))
  else
    NUM_PROMPTS=$(( CONC * 50 ))
  fi
else
  NUM_PROMPTS=$(( CONC * 10 ))
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
--num-prompts $NUM_PROMPTS \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json"

# Try graceful first
docker stop -t 90 "$server_name" || true
# Wait until it's really dead
docker wait "$server_name" >/dev/null 2>&1 || true
# Force remove if anything lingers
docker rm -f "$server_name" >/dev/null 2>&1 || true

# Give a moment for GPU processes to fully terminate
sleep 2
# Verify GPUs are now idle; if not, print diag and (optionally) reset
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "[WARN] After stop, GPU still busy:"; nvidia-smi
  # Last resort if driver allows and GPUs appear idle otherwise:
  #nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7 2>/dev/null || true
fi

nvidia-smi
