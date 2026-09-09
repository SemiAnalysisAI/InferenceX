#!/usr/bin/env bash
# Copy beside the PREPARED runtime, fill the three paths, then pin its SHA256
# in the site config. This entry never allocates, imports, installs, or downloads.
set -euo pipefail
runtime_root=/REPLACE_WITH_EXISTING_RUNTIME_PARENT
workspace=/REPLACE_WITH_PERSISTENT_WORKSPACE
container_name=REPLACE_WITH_EXISTING_ENROOT_NAME

: "${SLURM_JOB_ID:?must enter through srun}"
: "${SLURM_STEP_ID:?must enter through an allocated step}"
: "${SLURM_STEP_GPUS:?Slurm must assign GPUs}"
[[ -d "$runtime_root/enroot-data/$container_name" && -f "$runtime_root/.rootfs-ready" ]]
[[ -d "$workspace" && $# -gt 0 ]]
exec 9>"$runtime_root/.session.lock"
flock -n 9 || { echo "Prepared runtime has an active session" >&2; exit 2; }
export ENROOT_DATA_PATH="$runtime_root/enroot-data"
export ENROOT_RUNTIME_PATH="/tmp/inferencex-h3-${SLURM_JOB_ID}-${SLURM_STEP_ID}/runtime"
export ENROOT_TEMP_PATH="/tmp/inferencex-h3-${SLURM_JOB_ID}-${SLURM_STEP_ID}/tmp"
export ENROOT_CACHE_PATH="$runtime_root/cache"

# This site uses AutoDetect=nvidia and /dev/nvidia minor-number ordering.
# NVML query indices use PCI ordering and differ under partial allocations.
h3_gpu_uuids=$(python3 - <<'PY'
import os, re
from pathlib import Path
value = os.environ['SLURM_STEP_GPUS']
if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*', value):
    raise SystemExit('Unsupported Slurm GPU assignment; no index fallback')
ids = []
for part in value.split(','):
    bounds = list(map(int, part.split('-')))
    if len(bounds) == 1:
        ids.append(bounds[0])
    elif 0 <= bounds[0] <= bounds[1] < 8:
        ids.extend(range(bounds[0], bounds[1] + 1))
    else:
        raise SystemExit('GPU range outside the single eight-GPU node')
if len(ids) != len(set(ids)) or not ids or any(i >= 8 for i in ids):
    raise SystemExit('Invalid global GPU assignment')
devices = {}
for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
    fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
    minor, identity = fields.get('Device Minor', '').strip(), fields.get('GPU UUID', '').strip()
    if minor.isdigit() and re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', identity):
        if int(minor) in devices:
            raise SystemExit('Duplicate NVIDIA device minor')
        devices[int(minor)] = identity
if any(index not in devices for index in ids):
    raise SystemExit('Assigned Slurm device files lack NVIDIA UUIDs')
print(','.join(devices[index] for index in ids))
PY
)
h3_gpu_rows=$(nvidia-smi --id="$h3_gpu_uuids" --query-gpu=uuid,name --format=csv,noheader)
H3_ASSIGNED_GPU_UUIDS=$(python3 - "$h3_gpu_rows" "$h3_gpu_uuids" <<'PY'
import csv, os, re, sys
rows = list(csv.reader(sys.argv[1].splitlines()))
expected = os.environ.get('H3_EXPECTED_GPU_MODEL', 'H200')
if expected not in {'H100', 'H200', 'B200'}:
    raise SystemExit('Unsupported expected NVIDIA GPU model')
if not rows or any(len(row) != 2 or not re.search(r'\b' + expected + r'\b', row[1]) or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', row[0].strip()) for row in rows):
    raise SystemExit('Assigned hardware does not match expected physical ' + expected + ' GPUs')
observed = [row[0].strip() for row in rows]
if sorted(observed) != sorted(sys.argv[2].split(',')):
    raise SystemExit('NVIDIA query differs from assigned physical UUIDs')
print(','.join(observed))
PY
)
export H3_ASSIGNED_GPU_UUIDS
export H3_ORIGINAL_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
export NVIDIA_VISIBLE_DEVICES=$H3_ASSIGNED_GPU_UUIDS
export CUDA_VISIBLE_DEVICES=$H3_ASSIGNED_GPU_UUIDS
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
h3_env=(NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES CUDA_VISIBLE_DEVICES
        H3_ASSIGNED_GPU_UUIDS H3_ORIGINAL_CUDA_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE
        SLURM_JOB_ID SLURM_STEP_ID SLURMD_NODENAME SLURM_PROCID SLURM_NTASKS
        SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_CPU_BIND SLURM_CPUS_PER_TASK)
h3_args=()
for name in "${h3_env[@]}"; do
    if [[ -v "$name" ]]; then h3_args+=(--env "$name"); fi
done
# Keep the existing cache/mount layout; add other already-prepared mounts here.
# This saved image has /etc/rc -> exec bash "$@", so it expects -c. Adapt
# the tail to the verified entrypoint when reusing a different prepared image.
exec enroot start --root --rw --mount "$workspace:/work" \
    "${h3_args[@]}" --env HF_HOME=/work/.cache/huggingface \
    -- "$container_name" -c 'cd /work && exec "$@"' bash "$@"
