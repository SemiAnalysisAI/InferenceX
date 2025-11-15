#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="$HOME/hf_hub_cache/"
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
docker run --rm --ipc=host --shm-size=16g --network=host --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e PORT=$PORT \
-e ISL -e OSL -e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e RANDOM_RANGE_RATIO -e RESULT_FILENAME \
 ${GH_SUM_ENV} ${GH_SUM_MOUNT} \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/"${EXP_NAME%%_*}_${PRECISION}_mi300x_docker.sh"
