#!/usr/bin/env bash
# Launch script for GreenNode RTX 5090 runners (plain Docker, no Slurm/enroot).
# Selected by benchmark-tmpl.yml via: bash ./runners/launch_${RUNNER_NAME%%_*}.sh
# where RUNNER_NAME like "rtx5090-greennode_00" → this script.

set -x

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
export PORT="${PORT:-8888}"

docker pull "$IMAGE"

RUN_ENV=(
  HF_TOKEN HF_HUB_CACHE PORT RANDOM_RANGE_RATIO
  EXP_NAME MODEL MODEL_PREFIX ISL OSL MAX_MODEL_LEN
  IMAGE FRAMEWORK PRECISION TP EP_SIZE DP_ATTENTION
  CONC SPEC_DECODING DISAGG
  RUN_EVAL EVAL_ONLY SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
  OFFLOADING TOTAL_CPU_DRAM_GB DURATION
  RESULT_DIR RESULT_FILENAME
  PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX
  RUNNER_NAME RUNNER_TYPE
)
ENV_ARGS=()
for v in "${RUN_ENV[@]}"; do
  ENV_ARGS+=(-e "$v")
done

docker run --rm \
  --gpus all \
  --ipc=host \
  --network host \
  --shm-size=16g \
  -v "$GITHUB_WORKSPACE:/workspace" \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -w /workspace \
  "${ENV_ARGS[@]}" \
  --entrypoint bash \
  "$IMAGE" \
  "benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_rtx5090.sh"
