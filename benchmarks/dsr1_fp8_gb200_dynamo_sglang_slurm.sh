#!/usr/bin/bash

set -x

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars 

SGL_SLURM_JOBS_PATH="components/backends/sglang/slurm_jobs"

# Always clone and setup Dynamo
echo "Cloning Dynamo repository..."
rm -rf "$DYNAMO_PATH"
git clone --branch update-result-file-name https://github.com/Elnifio/dynamo.git $DYNAMO_PATH
cd dynamo
cd "$SGL_SLURM_JOBS_PATH"

