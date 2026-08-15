#!/usr/bin/env bash
# Report per-GPU free VRAM inside the ROCm image (catches leaked allocations left
# behind by a wedged container, which amd-smi may still report as in-use).
#
#   SLURM_REUSE_JOBID=16758 NODE=mia1-p01-g06 bash experimental/kimik3-v4/probe_free_vram.sh
#
set -uo pipefail

SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
NODE="${NODE:-mia1-p01-g06}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263}"

[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }

srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE" -N1 -n1 bash -lc "
docker run --rm --device /dev/dri --device /dev/kfd --group-add video --privileged \
  --entrypoint python3 ${IMAGE} -c '
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(\"gpu\", i, \"free_GiB\", round(free / 2**30, 1), \"of\", round(total / 2**30, 1))
'
"
