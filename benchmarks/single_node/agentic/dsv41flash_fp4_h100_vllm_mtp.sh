#!/usr/bin/env bash
set -eo pipefail

# THROWAWAY ANALYSIS BRANCH -- not a benchmark.
#
# This entry point is hijacked so the existing h100-dgxc launcher can drive the
# Engram gate scan on real hardware without new workflow plumbing. CONC is
# reused as the shard index (conc-list [1,2,3] -> three single-node jobs, one
# per shard, 8 GPUs each). Do not merge this branch.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC RESULT_DIR
export GPU_COUNT="$TP"

if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
mkdir -p "$RESULT_DIR"
export PYTHONUNBUFFERED=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# The probe recomputes gates in torch; keep the engine single-stream and plain.
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m pip install -q --no-input datasets 2>&1 | tail -2 || true

# Two nodes for this pass: conc-list [1,2] -> shards 0 and 1.
NUM_SHARDS=${ENGRAM_NUM_SHARDS:-2}
# Part files from the previous 3-shard run sit in the same NFS output dir under
# colliding names, and they hold copy-averaged gates. Rescan unless asked not to.
RESUME_FLAG=""
if [[ -z "${ENGRAM_RESUME:-}" ]]; then
    RESUME_FLAG="--no-resume"
fi
SHARD=$((CONC - 1))
echo "=== Engram gate scan: shard ${SHARD} of ${NUM_SHARDS}, TP=${TP} ==="

cd "$INFERENCEX_REPO_ROOT"
python3 analysis/engram/scan.py \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --max-chunks-per-domain "${ENGRAM_MAX_CHUNKS:-96}" \
    $RESUME_FLAG \
    --out "$RESULT_DIR/engram_scan"

echo "=== scan complete for shard ${SHARD} ==="
