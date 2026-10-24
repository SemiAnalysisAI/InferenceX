#!/bin/bash
#SBATCH --job-name=k3-load-rank4-diag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --nodelist=mia1-p01-g18
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root="$PWD"
mkdir -p "$root/logs"
cp "$root/perf-analysis.md" "$root/logs/perf-analysis.md"
export PYTHONNOUSERSITE=1
srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-node=8 --mem=0 \
  --container-image=registry-1.docker.io#yukiozzz/vllm-openai-rocm:sha256:7b6ed1d61b1237c616acbd111ee5f76cb747f7cf27f5f53fa3a7010727e5a8fc \
  --container-remap-root --container-writable \
  --container-mounts="$root:/probe:ro,$root/logs:/logs,/it-share/data/Kimi-K3:/model:ro,/it-share/data/Inferact-Kimi-K3-DSpark:/models/Inferact-Kimi-K3-DSpark:ro,/dev/infiniband:/dev/infiniband,/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro" \
  --container-workdir=/probe python3 /probe/supervisor.py
