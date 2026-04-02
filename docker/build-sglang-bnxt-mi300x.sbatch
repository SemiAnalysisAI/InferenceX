#!/bin/bash
#SBATCH --job-name=sgl-build-mi300x
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/j9s/build-mi300x-%j.log
#SBATCH --error=/home/j9s/build-mi300x-%j.log
#SBATCH --chdir=/tmp

set -euo pipefail
bash /nfsdata/sa/j9s/inferencex/docker/build-sglang-bnxt-mi300x.sh
