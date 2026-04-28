#!/bin/bash

export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"

export MODEL_NAME="DeepSeek-R1-0528"   # key from models_vllm.yaml
export MODEL_DIR="$HOME/.cache/huggingface/hub"
export MODEL_PATH="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
# export NODE0_ADDR="10.21.9.8"          # prefill (rank-0) node's IP
# export IPADDRS="10.21.9.8,10.21.9.29"  # all nodes: prefill IPs, then decode IPs
export NODE0_ADDR="10.21.9.47"          # this node's IP
export IPADDRS="10.21.9.47,10.21.9.29"  # all nodes: prefill IPs, then decode IPs
export xP=1 yD=1
export NNODES=2
export GPUS_PER_NODE=8

export NODE_RANK=1
export DRY_RUN=0

export BENCH_INPUT_LEN=1024
export BENCH_OUTPUT_LEN=1024
export BENCH_MAX_CONCURRENCY="32x64x128x256x512"

# Repo root (3 levels up from this script's directory)
export DI_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Mount point inside the container (must match WS_PATH computation below)
export DOCKER_MOUNT_PATH="/workspace"
# Container-side path to the scripts directory
export WS_PATH="${DOCKER_MOUNT_PATH}/benchmarks/multi_node/amd_utils"
# Remap host MODEL_PATH into the container's /models mount
export DOCKER_MODEL_PATH="${MODEL_PATH/#$MODEL_DIR//models}"

export SLURM_JOB_ID=1
mkdir -p "/tmp/slurm_job-${SLURM_JOB_ID}"

CONTAINER_NAME="vllm-disagg-decode"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run --rm \
    --name "$CONTAINER_NAME" \
    --init \
    --stop-timeout 10 \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --device=/dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --device=/dev/infiniband/uverbs4 \
    --device=/dev/infiniband/uverbs5 \
    --device=/dev/infiniband/uverbs6 \
    --device=/dev/infiniband/uverbs7 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v /sys:/sys \
    $(command -v nicctl >/dev/null 2>&1 && echo "-v $(which nicctl):/usr/sbin/nicctl") \
    -v ${MODEL_DIR}:/models \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 128G \
    -v /tmp:/run_logs \
    -v ${DI_REPO_DIR}:${DOCKER_MOUNT_PATH} \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e NNODES=$NNODES \
    -e NODE0_ADDR=$NODE0_ADDR \
    -e IPADDRS=$IPADDRS \
    -e MODEL_DIR=/models \
    -e MODEL_NAME=$MODEL_NAME \
    -e MODEL_PATH=$DOCKER_MODEL_PATH \
    -e WS_PATH=${WS_PATH} \
    -e GPUS_PER_NODE=$GPUS_PER_NODE \
    -e NODE_RANK=$NODE_RANK \
    -e xP=$xP \
    -e yD=$yD \
    -e IBDEVICES=$IBDEVICES \
    -e DRY_RUN=$DRY_RUN \
    -e ENGINE=vllm-disagg \
    -e HF_HUB_CACHE=/models \
    -e UCX_TLS=tcp,self,shm,rocm_ipc,rocm_copy,cma \
    -e UCX_SOCKADDR_TLS_PRIORITY=tcp \
    -e UCX_MEMTYPE_CACHE=y \
    -e UCX_RNDV_SCHEME=get_zcopy \
    -e UCX_RNDV_THRESH=4k \
    -e UCX_ROCM_IPC_MIN_ZCOPY=0 \
    -e UCX_LOG_LEVEL=warn \
    -e HSA_ENABLE_SDMA=1 \
    -e PROXY_STREAM_IDLE_TIMEOUT=${PROXY_STREAM_IDLE_TIMEOUT:-300} \
    -e VLLM_MORIIO_CONNECTOR_READ_MODE=${VLLM_MORIIO_CONNECTOR_READ_MODE:-1} \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \
    -e PREFILL_ENABLE_EP=${PREFILL_ENABLE_EP:-false} \
    -e PREFILL_ENABLE_DP=${PREFILL_ENABLE_DP:-false} \
    -e DECODE_ENABLE_EP=${DECODE_ENABLE_EP:-false} \
    -e DECODE_ENABLE_DP=${DECODE_ENABLE_DP:-false} \
    -e PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-8} \
    -e DECODE_TP_SIZE=${DECODE_TP_SIZE:-8} \
    --entrypoint /bin/bash \
    vllm-router-rocm:0.1.0 \
    -lc "mkdir -p /run_logs/slurm_job-${SLURM_JOB_ID} && ${WS_PATH}/server.sh 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/server_\$(hostname).log"
