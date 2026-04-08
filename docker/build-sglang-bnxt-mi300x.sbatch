#!/bin/bash
#SBATCH --job-name=sgl-build-cdna3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/build-cdna3-%j.log
#SBATCH --error=/home/%u/build-cdna3-%j.log

set -euo pipefail

# Ensure crane is available (download if not present)
if ! command -v crane &>/dev/null && [ ! -x /tmp/crane ]; then
  echo "Downloading crane..."
  curl -sL "https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz" \
    -o /tmp/crane.tar.gz
  tar -xz -C /tmp crane -f /tmp/crane.tar.gz
  chmod +x /tmp/crane
fi
export CRANE_BIN="${CRANE_BIN:-/tmp/crane}"

bash "${SLURM_SUBMIT_DIR}/build-sglang-bnxt-mi300x.sh"
