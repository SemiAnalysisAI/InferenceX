#!/usr/bin/env bash

# B300 uses the same low-concurrency TRTLLM bring-up recipe as B200. The B300
# runner may rewrite MODEL to the pre-staged /data/models/dsv4-pro path before
# this script is invoked. The job itself is already launched under srun/pyxis;
# scrub Slurm's PMI environment, then use mpirun to give TRTLLM a valid OpenMPI
# runtime instead of direct-launching under srun.

export TRTLLM_DSV4_USE_MPIRUN="${TRTLLM_DSV4_USE_MPIRUN:-1}"
export TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV="${TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV:-1}"
export TRTLLM_DSV4_BOOTSTRAP="${TRTLLM_DSV4_BOOTSTRAP:-0}"

bash "$(dirname "$0")/dsv4_fp4_b200_trt.sh"
