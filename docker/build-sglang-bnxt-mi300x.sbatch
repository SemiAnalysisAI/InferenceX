#!/bin/bash
#SBATCH --job-name=sgl-build-mi300x
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/%u/build-mi300x-%j.log
#SBATCH --error=/home/%u/build-mi300x-%j.log

set -euo pipefail
bash "${SLURM_SUBMIT_DIR}/build-sglang-bnxt-mi300x.sh"
