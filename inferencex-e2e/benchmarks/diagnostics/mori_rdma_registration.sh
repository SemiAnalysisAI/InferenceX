#!/usr/bin/env bash
set -eo pipefail

source benchmarks/check_env.sh
check_env_vars IMAGE SLURM_JOB_ID MORI_RDMA_DEVICES MORI_SHMEM_MODE MORI_RDMA_TC MORI_IO_TC \
  MORI_IO_QP_MAX_SEND_WR MORI_IO_QP_MAX_CQE MORI_IO_QP_MAX_SGE \
  MORI_IO_SQ_BACKOFF_TIMEOUT_US SGLANG_MORI_QP_PER_TRANSFER \
  SGLANG_MORI_NUM_WORKERS

mkdir -p speedbench_results
ulimit -c 0
export MORI_DISABLE_AUTO_XGMI=1

{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'image=%s\n' "$IMAGE"
  printf 'memlock=%s\n' "$(ulimit -l)"
  printf 'visible_devices=%s\n' "$ROCR_VISIBLE_DEVICES"
  if command -v rocm-smi >/dev/null 2>&1; then rocm-smi --showuniqueid --json; fi
  if command -v ibv_devinfo >/dev/null 2>&1; then ibv_devinfo -l; fi
  for device in rdma0 rdma1 rdma2 rdma3 rdma4 rdma5 rdma6 rdma7; do
    path="/sys/class/infiniband/$device/device/driver"
    if [ -e "$path" ]; then printf '%s=%s\n' "$path" "$(readlink -f "$path")"; fi
    for field in vendor device; do
      path="/sys/class/infiniband/$device/device/$field"
      if [ -r "$path" ]; then printf '%s=%s\n' "$path" "$(cat "$path")"; fi
    done
  done
} > speedbench_results/host-runtime.txt 2>&1

timeout --signal=TERM --kill-after=15s 7m \
  python3 -u benchmarks/diagnostics/mori_rdma_registration.py \
  > speedbench_results/mori-rdma-preflight.log 2>&1
