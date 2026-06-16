#!/usr/bin/env bash
# First-stage command inside the disagg Docker image (see _disagg_ssh_remote_inner.sh).
# Optional: rebuild **libbnxt** (userspace libibverbs) from a tarball before server.sh — not DKMS.

set -euo pipefail

: "${SLURM_JOB_ID:?}"

if [[ "${REBUILD_LIBBNXT_IN_CONTAINER:-0}" == "1" ]]; then
  export REBUILD_BNXT=1
  export PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:?Set PATH_TO_BNXT_TAR_PACKAGE to a path visible in-container (e.g. /workspace/driver/libbnxt_re-*.tar.gz)}"
  bash /workspace/benchmarks/multi_node/amd_utils/rebuild_bnxt.sh
  # Prefer freshly built libbnxt_re in /usr/local/lib over inbox/shipped providers in /usr/lib.
  export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ "${INSTALL_MORI_IN_CONTAINER:-0}" == "1" ]]; then
  bash /workspace/scripts/install_mori_in_container.sh
  if [[ -f /tmp/mori_pythonpath_prefix ]]; then
    _mori_pyp="$(cat /tmp/mori_pythonpath_prefix)"
    export PYTHONPATH="${_mori_pyp}${PYTHONPATH:+:${PYTHONPATH}}"
    unset _mori_pyp
  fi
fi

mkdir -p "/run_logs/slurm_job-${SLURM_JOB_ID}"
exec bash /workspace/benchmarks/multi_node/amd_utils/server.sh 2>&1 | tee "/run_logs/slurm_job-${SLURM_JOB_ID}/server_$(hostname).log"
