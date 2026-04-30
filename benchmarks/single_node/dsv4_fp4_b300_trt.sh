#!/usr/bin/env bash

# B300 uses the same low-concurrency TRTLLM bring-up recipe as B200. The B300
# runner may rewrite MODEL to the pre-staged /data/models/dsv4-pro path before
# this script is invoked. The job itself is already launched under srun; keep
# mpirun local so OpenMPI does not try to use Slurm PMI/PMIx from inside pyxis.

export TRTLLM_DSV4_USE_MPIRUN="${TRTLLM_DSV4_USE_MPIRUN:-1}"
export OMPI_MCA_plm="${OMPI_MCA_plm:-isolated}"
export OMPI_MCA_ras="${OMPI_MCA_ras:-^slurm}"

bash "$(dirname "$0")/dsv4_fp4_b200_trt.sh"
