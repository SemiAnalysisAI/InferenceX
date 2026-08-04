#!/usr/bin/bash
set -euo pipefail

# mi325x-tw is a NON-SLURM cluster: the runner executes on the GPU node itself
# (docker + ROCm, no salloc/srun). So run the container directly on the node,
# like launch_h100-cr.sh but AMD/ROCm (--device=/dev/kfd,/dev/dri instead of
# --runtime=nvidia --gpus). dsv4 MI325X is single-node (TP8), so one node's
# 8 GPUs suffice. Runs the same _mi325x.sh benchmark as the amds launcher.

HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/home/gharunner/hf_hub_cache/}"
mkdir -p "$HF_HUB_CACHE_MOUNT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/hf_hub_cache}"
PORT=8888
server_name="bmk-server-${RUNNER_NAME:-mi325x-tw}"

# Route spec-decoding=mtp configs to the _mtp benchmark script.
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

set -x
docker rm -f "$server_name" >/dev/null 2>&1 || true
docker run --rm --network=host --name="$server_name" \
--device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
--ipc=host --privileged --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
--security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
-v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
-v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e PP_SIZE -e DCP_SIZE -e PCP_SIZE -e GPU_COUNT -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e RUN_EVAL -e EVAL_ONLY -e RUNNER_TYPE -e RESULT_FILENAME -e RANDOM_RANGE_RATIO -e PORT="$PORT" \
-e DP_ATTENTION -e EP_SIZE -e DP_SIZE -e EVAL_MAX_MODEL_LEN -e SPEC_DECODING -e NUM_SPEC_TOKENS \
-e PROFILE -e SGLANG_TORCH_PROFILER_DIR -e VLLM_TORCH_PROFILER_DIR -e VLLM_RPC_TIMEOUT \
-e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
--entrypoint=/bin/bash \
"$IMAGE" \
benchmarks/single_node/${SCENARIO_SUBDIR}"${EXP_NAME%%_*}_${PRECISION}_mi325x${SPEC_SUFFIX}.sh"
