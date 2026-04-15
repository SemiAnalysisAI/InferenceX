#!/usr/bin/bash

source "$(dirname "$0")/lib_single_node_script.sh"

HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
PORT=8888
SCRIPT_PATH=$(resolve_single_node_benchmark_script "${EXP_NAME%%_*}" "$PRECISION" "h100" "$FRAMEWORK" "${SPEC_DECODING:-none}") || exit 1

server_name="bmk-server"

set -x
docker run --rm --network=host --name=$server_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e EP_SIZE -e DP_ATTENTION -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e RUN_EVAL -e EVAL_ONLY -e RUNNER_TYPE -e RESULT_FILENAME -e RANDOM_RANGE_RATIO -e PORT=$PORT \
-e SPEC_DECODING -e DISAGG \
-e BENCHMARK_TYPE -e EXPORT_FILE -e RUNTIME_STACK_ID -e HARDWARE_PROFILE_ID -e CANONICAL_MODEL_ID -e REQUEST_MODE -e MAX_CONCURRENCY \
-e SUPPORT_STATUS -e VLLM_CPU_OFFLOAD_GB -e VLLM_SWAP_SPACE_GB -e SGLANG_MEM_FRACTION_OVERRIDE -e SGLANG_CHUNKED_PREFILL_OVERRIDE \
-e MAX_SESSIONS -e MAX_TURNS_PER_SESSION -e MAX_OUTPUT_LEN -e NUM_WARMUP_SESSIONS -e IGNORE_WAITS -e IGNORE_EOS \
-e PROFILE -e SGLANG_TORCH_PROFILER_DIR -e VLLM_TORCH_PROFILER_DIR -e VLLM_RPC_TIMEOUT \
-e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
$IMAGE \
"$SCRIPT_PATH"
