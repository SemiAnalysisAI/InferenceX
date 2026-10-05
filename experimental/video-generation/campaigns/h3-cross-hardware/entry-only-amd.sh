#!/usr/bin/env bash
# Enter only the existing task-owned runtime. This never allocates or installs.
set -eo pipefail
h3_helper="$(dirname -- "${BASH_SOURCE[0]}")/benchmark_lib.sh"
h3_helper_hash=$(sha256sum "$h3_helper") && [[ "${h3_helper_hash%% *}" == '@BENCHMARK_LIB_SHA256@' ]] || {
    echo 'Prepared validation helper missing or changed' >&2
    exit 2
}
source "$h3_helper" --validation-only
check_env_vars SLURM_JOB_ID SLURM_STEP_ID SLURM_STEP_GPUS H3_AMD_ALLOCATION_UUIDS ROCR_VISIBLE_DEVICES
runtime_root=/it-share/data/wenyao-minimax-h3
workspace=$runtime_root/work
container_name=wenyao-minimax-h3-rocm
[[ -d "$runtime_root/enroot-data/$container_name" && -f "$workspace/campaigns/h3-cross-hardware/runtime-inspected.json" && $# -gt 0 ]]
exec 9>"$runtime_root/.session.lock"
flock -n 9 || { echo "Prepared runtime has an active session" >&2; exit 2; }
export ENROOT_DATA_PATH="$runtime_root/enroot-data"
export ENROOT_CACHE_PATH="$runtime_root/cache"
export ENROOT_RUNTIME_PATH="/tmp/inferencex-h3-${SLURM_JOB_ID}-${SLURM_STEP_ID}/runtime"
export ENROOT_TEMP_PATH="/tmp/inferencex-h3-${SLURM_JOB_ID}-${SLURM_STEP_ID}/tmp"
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin
# ROCR supplies the physical mask; the supervisor sets logical HIP/CUDA ordinals.
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
h3_env=(ROCR_VISIBLE_DEVICES H3_AMD_ALLOCATION_UUIDS H3_AMD_MONITOR_RECEIPT PYTHONDONTWRITEBYTECODE
        SLURM_JOB_ID SLURM_STEP_ID SLURMD_NODENAME SLURM_PROCID SLURM_NTASKS
        SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_CPU_BIND SLURM_CPUS_PER_TASK)
h3_args=()
for name in "${h3_env[@]}"; do
    if [[ -v "$name" ]]; then h3_args+=(--env "$name"); fi
done
exec /usr/local/bin/enroot start --rw --mount "$workspace:/work" \
    --mount /dev/kfd:/dev/kfd --mount /dev/dri:/dev/dri \
    "${h3_args[@]}" --env SGLANG_USE_AITER=1 --env HF_HOME=/work/.cache/huggingface \
    -- "$container_name" @ENTRY@
