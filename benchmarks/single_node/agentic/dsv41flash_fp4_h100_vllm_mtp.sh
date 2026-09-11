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
# conc 9 is the sentinel for the ablation eval rather than a scan shard: it
# runs baseline and gate-zeroed evals back to back in one allocation.
if [[ "$CONC" == 9 ]]; then
    exec bash "$INFERENCEX_REPO_ROOT/analysis/engram/ablation_driver.sh"
fi
# conc 10: likelihood ablation -- one measurement per token instead of one per
# eval question, which is the only way an 0.4pp-scale effect is measurable.
if [[ "$CONC" == 10 ]]; then
    cd "$INFERENCEX_REPO_ROOT"
    exec python3 analysis/engram/nll_ablation.py --tp "$TP" \
        --chunks-per-domain "${ENGRAM_NLL_CHUNKS:-200}" \
        --out "$RESULT_DIR/engram_nll"
fi
# conc 11: Terminal-Bench 4.0 feasibility probe (no GPU work of its own).
# conc 11 is intercepted by the launcher before salloc (login-node container
# probe, added on this branch by another session). conc 12 is the Modal
# egress / tunnel-ingress probe, which must run ON a compute node because
# that is where a vLLM server would live.
if [[ "$CONC" == 12 ]]; then
    exec bash "$INFERENCEX_REPO_ROOT/analysis/engram/tbench_driver.sh"
fi

SHARD=$((CONC - 1))
echo "=== Engram gate scan: shard ${SHARD} of ${NUM_SHARDS}, TP=${TP} ==="

cd "$INFERENCEX_REPO_ROOT"
python3 analysis/engram/scan.py \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --max-chunks-per-domain "${ENGRAM_MAX_CHUNKS:-5000}" \
    $RESUME_FLAG \
    --out "$RESULT_DIR/engram_scan"

echo "=== scan complete for shard ${SHARD} ==="
