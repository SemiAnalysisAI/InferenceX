#!/bin/bash

export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"

export MODEL_NAME="DeepSeek-R1-0528"   # key from models.yaml
export MODEL_DIR="/root/.cache/huggingface/hub"
export MODEL_PATH="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
export NODE0_ADDR="10.21.9.8"          # prefill (rank-0) node's IP
export IPADDRS="10.21.9.8,10.21.9.29"  # all nodes: prefill IPs, then decode IPs
export xP=1 yD=1

export NODE_RANK=1
export DRY_RUN=0

# Point VLLM_WS_PATH to the directory containing server.sh, env.sh, etc.
export VLLM_WS_PATH="${VLLM_WS_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

export SLURM_JOB_ID=1
mkdir -p "/run_logs/slurm_job-${SLURM_JOB_ID}"

bash "${VLLM_WS_PATH}/server.sh"
