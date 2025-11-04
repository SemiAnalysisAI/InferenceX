#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/mnt/hf_hub_cache/"
PORT=8000

container_name="gsm8k-eval"

echo "Starting GSM8k evaluation container..."

set -x
docker run --rm --network=host --name=$container_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e NUM_FEWSHOT -e LIMIT -e PORT=$PORT -e RESULT_FILENAME \
-e TORCH_CUDA_ARCH_LIST="10.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
$IMAGE \
-lc "pip install -q lm-eval[vllm] && bash /workspace/benchmarks/gsm8k_${FRAMEWORK}_docker.sh"

set +x

echo "GSM8k evaluation completed successfully"
