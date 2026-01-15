#!/usr/bin/bash

set -x

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars CONC_LIST ISL OSL IMAGE CONFIG_FILE MODEL_PATH

if ! command -v srtctl &> /dev/null; then
    echo "Error: srtctl not found. Please install srt-slurm first."
    exit 1
fi

# Use the srt-slurm-trtllm repo cloned by launch script
TRTLLM_REPO_DIR="srt-slurm-trtllm"
if [ ! -d "$TRTLLM_REPO_DIR" ]; then
    echo "Error: srt-slurm-trtllm not found. It should have been cloned by launch script."
    exit 1
fi

BASE_CONFIG="$TRTLLM_REPO_DIR/$CONFIG_FILE"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config not found at $BASE_CONFIG"
    exit 1
fi

echo "Using base config: $BASE_CONFIG"
echo "Config contents:"
cat "$BASE_CONFIG"

echo "Submitting job with srtctl..."
srtctl apply -f "$BASE_CONFIG" --tags "h200,dsr1,fp8,${ISL}x${OSL},infmax-$(date +%Y%m%d)"

echo "Job submitted via srtctl"

