#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
PORT=8888

server_name="bmk-server"

set -x
## Propagate GitHub summary file into the container when available
GH_SUM_ENV=""
GH_SUM_MOUNT=""
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  GH_SUM_ENV="-e GITHUB_STEP_SUMMARY=${GITHUB_STEP_SUMMARY}"
  GH_SUM_DIR="$(dirname "${GITHUB_STEP_SUMMARY}")"
  if [ -d "${GH_SUM_DIR}" ]; then
    GH_SUM_MOUNT="-v ${GH_SUM_DIR}:${GH_SUM_DIR}"
  fi
fi
docker run --rm --network=host --name=$server_name \
--runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e RUN_EVAL -e RESULT_FILENAME -e RANDOM_RANGE_RATIO -e PORT=$PORT \
-e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
 ${GH_SUM_ENV} ${GH_SUM_MOUNT} \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_h100_docker.sh"
