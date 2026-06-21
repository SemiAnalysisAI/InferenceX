#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
PORT=8888

server_name="bmk-server"

# Route spec-decoding=mtp configs to the _mtp benchmark script (parity with
# the h200 launchers, which have carried SPEC_SUFFIX since #392).
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

DOCKER_ENV_VARS=(
    HF_TOKEN
    HF_HUB_CACHE
    EXP_NAME
    IMAGE
    MODEL
    MODEL_PREFIX
    TP
    EP_SIZE
    DP_ATTENTION
    CONC
    MAX_MODEL_LEN
    ISL
    OSL
    FRAMEWORK
    PRECISION
    SPEC_DECODING
    DISAGG
    RUN_EVAL
    EVAL_ONLY
    EVAL_FRAMEWORK
    EVAL_TASKS_DIR
    EVAL_MAX_MODEL_LEN
    OPENAI_API_KEY
    RUNNER_TYPE
    RESULT_FILENAME
    RANDOM_RANGE_RATIO
    SCENARIO_TYPE
    SCENARIO_SUBDIR
    IS_AGENTIC
    OFFLOADING
    TOTAL_CPU_DRAM_GB
    DURATION
    RESULT_DIR
    PYTHONDONTWRITEBYTECODE
    PYTHONPYCACHEPREFIX
    PROFILE
    SGLANG_TORCH_PROFILER_DIR
    VLLM_TORCH_PROFILER_DIR
    VLLM_RPC_TIMEOUT
)
DOCKER_ENV_ARGS=()
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache/}"
for env_name in "${DOCKER_ENV_VARS[@]}"; do
    DOCKER_ENV_ARGS+=(--env "$env_name")
done

set -x
docker run --rm --network=host --name="$server_name" \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
-v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/ \
"${DOCKER_ENV_ARGS[@]}" \
--env "PORT=$PORT" \
--env TORCH_CUDA_ARCH_LIST="9.0" \
--env CUDA_DEVICE_ORDER=PCI_BUS_ID \
--env CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
--entrypoint=/bin/bash \
"$IMAGE" \
"benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h100${SPEC_SUFFIX}.sh"
