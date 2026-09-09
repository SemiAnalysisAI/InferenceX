#!/usr/bin/env bash
# Enter only the existing task-owned runtime. This never allocates or installs.
set -euo pipefail
runtime_root=/it-share/data/wenyao-minimax-h3
workspace=$runtime_root/work
container_name=wenyao-minimax-h3-rocm
: "${SLURM_JOB_ID:?must enter through srun}"
: "${SLURM_STEP_ID:?must enter through an allocated step}"
: "${SLURM_STEP_GPUS:?Slurm must bind the full AMD node}"
: "${H3_AMD_ALLOCATION_UUIDS:?physical allocation proof required}"
: "${ROCR_VISIBLE_DEVICES:?HIP binding required}"
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
h3_env=(ROCR_VISIBLE_DEVICES H3_AMD_ALLOCATION_UUIDS PYTHONDONTWRITEBYTECODE
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
