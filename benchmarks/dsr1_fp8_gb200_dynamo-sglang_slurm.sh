
#!/bin/bash

set -x

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars CONC_LIST ISL OSL IMAGE SPEC_DECODING MODEL_PATH \
    PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN \
    PREFILL_NODES DECODE_NODES N_ADDITIONAL_FRONTENDS SGL_SLURM_JOBS_PATH # SGL_SLURM_JOBS_PATH FIXME

# Always clone and setup Dynamo
echo "Cloning Dynamo repository..."
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    git clone --branch ishan/sa-1.1-sgl-dsr1-fp8 https://github.com/ai-dynamo/dynamo.git
else
    git clone --branch update-result-file-name https://github.com/Elnifio/dynamo.git
fi

cd "$SGL_SLURM_JOBS_PATH"

# Set up SGL launch script-specific environment variables
export TIME_LIMIT="04:00:00"
export MODEL_PATH=$MODEL_PATH
export CONFIG_DIR=$CONFIG_DIR
export CONTAINER_IMAGE=$IMAGE

# Launch jobs based on ISL/OSL
# Replace ' ' in CONC_LIST with 'x' such that the concurrency list is represented
# by a list of numbers delimted by 'x'. This is because of how the underlying launch script
# expects the concurrencies.
bash ./submit_disagg.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $N_ADDITIONAL_FRONTENDS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    $SCRIPT_MODE