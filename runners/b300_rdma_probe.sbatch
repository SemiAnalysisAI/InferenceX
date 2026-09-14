#!/usr/bin/env bash
#SBATCH --partition=batch_1
#SBATCH --account=benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10
set -euo pipefail
out="$RUNNER_TEMP/pr3088-rdma"
mkdir -p "$out"
echo "node=$(hostname) job=$SLURM_JOB_ID $(date -u --iso-8601=seconds)"
scontrol show job "$SLURM_JOB_ID"
echo "host_cpu_affinity=$(taskset -pc $$)"
enroot list | tee "$out/enroot-list.txt"
if grep -Ei '(^|[_-])(2528|2646|2659|2660|2663|2665|2666|2710|pr3088)([_-]|$)' "$out/enroot-list.txt"; then
  echo 'Possible task runtime exists: inspect compatibility before another container.'
  exit 75
fi
test -r "$SQUASH_FILE"
srun --mpi=pmix --exact --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --cpu-bind=cores   --container-image="$SQUASH_FILE"   --container-mounts="$GITHUB_WORKSPACE:/workspace,$out:/evidence"   --no-container-mount-home --container-remap-root --container-workdir=/tmp   --no-container-entrypoint --export=ALL   bash /workspace/runners/b300_runtime_diagnostic.sh
