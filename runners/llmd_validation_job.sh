#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "${SLURM_SUBMIT_DIR:?}/../../.." && pwd)"
receipt_dir="$repo_root/LOGS/pr3052-validation"
mkdir -p "$receipt_dir"
[[ "${SLURM_JOB_NUM_NODES:?}" == 4 && "${GPUS_PER_NODE:?}" == 4 ]]
scontrol show job "$SLURM_JOB_ID" > "$receipt_dir/allocated-job.txt"
grep -Eq '(^|[[:space:]])TimeLimit=00:39:00([[:space:]]|$)' "$receipt_dir/allocated-job.txt"
grep -Eq '(^|[[:space:]])Requeue=0([[:space:]]|$)' "$receipt_dir/allocated-job.txt"
grep -Eq '(^|[[:space:]])NumNodes=4([[:space:]]|$)' "$receipt_dir/allocated-job.txt"

# Keep preparation inside the granted job so it counts against the budget.
timeout 120s srun --jobid="$SLURM_JOB_ID" --nodes=4 --ntasks=4 \
    --ntasks-per-node=1 --kill-on-bad-exit=1 bash -s <<'NODE'
set -euo pipefail
repo_root="$(cd "${SLURM_SUBMIT_DIR:?}/../../.." && pwd)"
receipt="$repo_root/LOGS/pr3052-validation/node-${SLURM_PROCID:?}.txt"
exec > "$receipt" 2>&1
date -u +%FT%TZ
hostname
printf 'job=%s rank=%s model=%s\n' "$SLURM_JOB_ID" "$SLURM_PROCID" "$MODEL_DIR"
awk '/Cpus_allowed_list|Mems_allowed_list/ { print }' /proc/self/status
nvidia-smi --query-gpu=uuid,name --format=csv,noheader
gpu_uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ "$(printf '%s\n' "$gpu_uuids" | wc -l)" -eq 4 ]]
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_processes" ]]

# No task-owned llm-d rootfs/entry exists in the recovered receipts. Old task
# jobs 24949/25156 used a different SGLang runtime; do not adopt others' roots.
enroot list
printf 'ENROOT_DATA_PATH=%s ENROOT_RUNTIME_PATH=%s\n' "${ENROOT_DATA_PATH:-default}" "${ENROOT_RUNTIME_PATH:-default}"
printf '%s\n' 'No saved compatible task-owned llm-d rootfs; reuse the validated cached squash. Retain the new named rootfs for recovery.'
test -r "$LLMD_SQUASH_FILE"
test -d "$MODEL_DIR"
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
p = Path(os.environ['MODEL_DIR'])
shared = Path('/mnt/lustre01/models/DeepSeek-V4-Pro')
for name in ('config.json', 'model.safetensors.index.json'):
    print(name, hashlib.sha256((p / name).read_bytes()).hexdigest())
    assert (p / name).read_bytes() == (shared / name).read_bytes(), name
index = json.loads((p / 'model.safetensors.index.json').read_text())
shards = sorted(set(index['weight_map'].values()))
assert shards and all((p / name).is_file() and (p / name).stat().st_size > 0 for name in shards)
revision = (shared / '.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]
assert revision == 'b5968e9190ef611bbf34a7229255be88a0e937c1', revision
assert all((p / name).stat().st_size == (shared / name).stat().st_size for name in shards)
print(json.dumps({'shared_cache_revision': revision, 'local_matches': 'config/index bytes and shard sizes', 'shards': len(shards),
                  'model_bytes': sum((p / name).stat().st_size for name in shards)}))
PY
NODE
exec bash "$repo_root/benchmarks/multi_node/llm-d/job.slurm"
