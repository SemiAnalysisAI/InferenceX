#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
PORT=8888

server_name="bmk-server"

# Route spec-decoding=mtp configs to the _mtp benchmark script (parity with
# the h200 launchers, which have carried SPEC_SUFFIX since #392).
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

set -x
docker run --rm --network=host --name=$server_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e RUN_EVAL -e EVAL_ONLY -e RUNNER_TYPE -e RESULT_FILENAME -e RANDOM_RANGE_RATIO -e PORT=$PORT \
-e PROFILE -e SGLANG_TORCH_PROFILER_DIR -e VLLM_TORCH_PROFILER_DIR -e VLLM_RPC_TIMEOUT \
-e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/single_node/${SCENARIO_SUBDIR}"${EXP_NAME%%_*}_${PRECISION}_h100${SPEC_SUFFIX}.sh"
