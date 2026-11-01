#!/usr/bin/env bash
set -eo pipefail

source /infmax-workspace/benchmarks/check_env.sh
check_env_vars HF_HUB_CACHE HF_CHECKPOINT_REPO HF_CHECKPOINT_REVISION HF_CHECKPOINT_BUDGET_BYTES

if [[ ! "$HF_CHECKPOINT_REVISION" =~ ^[0-9a-f]{40}$ ]] ||
   [[ ! "$HF_CHECKPOINT_BUDGET_BYTES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Checkpoint staging requires a commit SHA and a positive byte budget" >&2
    exit 1
fi
mkdir -p "$HF_HUB_CACHE"
cache_name="models--${HF_CHECKPOINT_REPO//\//--}"
exec 9>"$HF_HUB_CACHE/.${cache_name}.download.lock"
flock -x 9
# Only staging goes online; the parent worker's serving environment is unchanged.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache

repo = os.environ["HF_CHECKPOINT_REPO"]
revision = os.environ["HF_CHECKPOINT_REVISION"]
cache = os.environ["HF_HUB_CACHE"]
budget = int(os.environ["HF_CHECKPOINT_BUDGET_BYTES"])
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
files = info.siblings
if info.sha != revision or not files or any(
    type(file.size) is not int or file.size < 0 for file in files
):
    raise SystemExit("Checkpoint metadata must identify the pinned revision and every file size")


def cached(file):
    path = try_to_load_from_cache(repo, file.rfilename, revision=revision, cache_dir=cache)
    return isinstance(path, str) and Path(path).is_file() and Path(path).stat().st_size == file.size


total = sum(file.size for file in files)
missing = [file for file in files if not cached(file)]
remaining = sum(file.size for file in missing)
# Do not credit old revisions or incomplete blobs. A complete snapshot needs no reserve.
required = remaining + max(0, budget - total) if missing else 0
available = shutil.disk_usage(cache).free
print(f"Checkpoint disk: available={available} remaining={remaining} required={required}", flush=True)
if available < required:
    raise SystemExit("Insufficient shared-cache disk for the pinned checkpoint")
if missing:
    snapshot_download(repo, revision=revision, cache_dir=cache)
    if any(not cached(file) for file in files):
        raise SystemExit("Pinned checkpoint is incomplete after download")
PY
