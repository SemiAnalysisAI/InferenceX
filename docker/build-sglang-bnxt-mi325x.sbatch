#!/bin/bash
#SBATCH --job-name=build-sgl-bnxt
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=360
#SBATCH --output=%u-build-mi325x-%j.log
#SBATCH --error=%u-build-mi325x-%j.log
#SBATCH --chdir=/tmp

set -euo pipefail
bash "${SLURM_SUBMIT_DIR}/build-sglang-bnxt-mi325x.sh"
