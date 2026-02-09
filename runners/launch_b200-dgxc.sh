#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache/"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
PORT=8888

# Create unique cache directory based on model parameters
MODEL_NAME=$(basename "$MODEL")

server_name="bmk-server"

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

docker run --rm --init --network host --name $server_name \
--runtime nvidia --gpus all --ipc host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e PORT=$PORT -e EP_SIZE -e DP_ATTENTION \
-e NCCL_GRAPH_REGISTER=0 \
-e TORCH_CUDA_ARCH_LIST="10.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
-e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e RESULT_FILENAME -e RANDOM_RANGE_RATIO -e RUN_EVAL -e RUNNER_TYPE \
--entrypoint=/bin/bash \
$(echo "$IMAGE" | sed 's/#/\//') \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_b200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"

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
