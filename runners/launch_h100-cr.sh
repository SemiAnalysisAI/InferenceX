#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/home/ubuntu/hf_hub_cache/"
PORT=${PORT:-8888}

set -x
SUMMARY_MOUNT=""
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  SUM_DIR="$(dirname "$GITHUB_STEP_SUMMARY")"
  if [ -d "$SUM_DIR" ]; then
    SUMMARY_MOUNT="-v $SUM_DIR:$SUM_DIR"
  fi
fi

docker run --rm --network=host \
  --runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/ \
  $SUMMARY_MOUNT \
  -e USER=runner -e LOGNAME=runner -e HOME=/workspace \
  -e XDG_CACHE_HOME=/workspace/.cache \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch/inductor \
  -e TRITON_CACHE_DIR=/workspace/.cache/torch/triton \
  -e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL \
  -e RANDOM_RANGE_RATIO -e RESULT_FILENAME -e PORT="$PORT" -e GITHUB_STEP_SUMMARY \
  -e RUN_MODE -e EVAL_TASK -e NUM_FEWSHOT -e LIMIT -e EVAL_RESULT_DIR \
  -e FRAMEWORK -e PRECISION -e EP_SIZE -e DP_ATTENTION -e OPENAI_MODEL_NAME \
  -e TORCH_CUDA_ARCH_LIST="9.0" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  --entrypoint=/bin/bash \
  "$IMAGE" \
  benchmarks/"${EXP_NAME%%_*}_${PRECISION}_h100_docker.sh"
set +x
