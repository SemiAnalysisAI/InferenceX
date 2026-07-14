#!/bin/bash
# Submit every gen-only config in configs/ as a separate SLURM job.
# configs/ holds the per-case trtllm_config.yaml (the Pareto MTP points copied from the cmh
# reference run, with EPLB stripped / perfect-router + ratio-1 applied).
#
# Usage:
#   bash submit_all.sh              # submit all configs in configs/
#   bash submit_all.sh --dry-run    # print sbatch commands without submitting
#   bash submit_all.sh configs/ctx1_gen1_dep32_concurrency128_mtp3.yaml   # submit a single config
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$HERE/scripts/submit.py"
CONFIG_DIR="$HERE/configs"

DRY=""
TARGET="$CONFIG_DIR"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY="--dry-run" ;;
    *.yaml)    TARGET="$arg" ;;
  esac
done

if [[ ! -d "$CONFIG_DIR" ]] || ! ls "$CONFIG_DIR"/*.yaml >/dev/null 2>&1; then
  echo "ERROR: no configs in $CONFIG_DIR. Copy the per-case trtllm_config.yaml there first." >&2
  exit 1
fi

if [[ "$TARGET" == *.yaml ]]; then
  echo "Submitting single config: $TARGET"
  python3 "$SUBMIT" -c "$TARGET" $DRY
else
  echo "Submitting all configs in: $CONFIG_DIR"
  python3 "$SUBMIT" -d "$CONFIG_DIR" $DRY
fi
