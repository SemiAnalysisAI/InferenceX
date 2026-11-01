#!/usr/bin/env bash
set -eo pipefail

source /infmax-workspace/benchmarks/check_env.sh
check_env_vars HF_HUB_CACHE HF_CHECKPOINT_REPO HF_CHECKPOINT_REVISION HF_CHECKPOINT_BUDGET_BYTES

if [[ ! "$HF_CHECKPOINT_REVISION" =~ ^[0-9a-f]{40}$ ]] ||
   [[ ! "$HF_CHECKPOINT_BUDGET_BYTES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Checkpoint staging requires a commit SHA and a positive byte budget" >&2
    exit 1
fi
if [[ "$HF_HUB_OFFLINE" == "1" || "$TRANSFORMERS_OFFLINE" == "1" ]]; then
    echo "Checkpoint staging requires online Hugging Face access" >&2
    exit 1
fi

mkdir -p "$HF_HUB_CACHE"
cache_name="models--${HF_CHECKPOINT_REPO//\//--}"
exec 9>"$HF_HUB_CACHE/.${cache_name}.download.lock"
flock -x 9
available=$(df -PB1 "$HF_HUB_CACHE" | awk 'NR == 2 {print $4}')
cached=0
if [[ -d "$HF_HUB_CACHE/$cache_name" ]]; then
    cached=$(du -sB1 "$HF_HUB_CACHE/$cache_name" | awk '{print $1}')
fi
if (( available + cached < HF_CHECKPOINT_BUDGET_BYTES )); then
    echo "Insufficient shared-cache disk: available=$available cached=$cached required=$HF_CHECKPOINT_BUDGET_BYTES" >&2
    exit 1
fi
hf download "$HF_CHECKPOINT_REPO" \
    --revision "$HF_CHECKPOINT_REVISION" \
    --cache-dir "$HF_HUB_CACHE"
