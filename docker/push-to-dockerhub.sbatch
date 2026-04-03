#!/bin/bash
#SBATCH --job-name=push-cdna3-dockerhub
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/push-cdna3-dockerhub-%j.log
#SBATCH --error=/home/%u/push-cdna3-dockerhub-%j.log

set -euo pipefail

ENV_FILE="/nfsdata/sa/j9s/.env.local"
if [ -f "${ENV_FILE}" ]; then
  set -a; source "${ENV_FILE}"; set +a
fi

python3 "${SLURM_SUBMIT_DIR}/push-to-dockerhub.py"
