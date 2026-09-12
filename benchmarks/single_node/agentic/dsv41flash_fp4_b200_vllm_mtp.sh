#!/usr/bin/env bash
set -eo pipefail

# THROWAWAY ANALYSIS BRANCH -- not a benchmark.
#
# B200 entry point for the Terminal-Bench 4.0 run. Same sentinel scheme as the
# H100 script: CONC is reused as a mode selector rather than a concurrency.
# Do not merge this branch.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC RESULT_DIR

# conc 13 = Engram active, conc 14 = Engram removed. Paired Terminal-Bench.
if [[ "$CONC" == 13 ]]; then
    exec bash "$INFERENCEX_REPO_ROOT/analysis/engram/tbench_run.sh"
fi
if [[ "$CONC" == 14 ]]; then
    export TBENCH_ABLATE=1
    exec bash "$INFERENCEX_REPO_ROOT/analysis/engram/tbench_run.sh"
fi

# conc 17 = Terminal-Bench validation slice. One node, 90 minutes, baseline
# Engram. Its only job is to show the zero-output-budget rejections are gone
# before another 15 node-hours go into a paired run: the first pair scored
# 0.045/0.030 against a published 31.2 because 164 and 32 turns respectively
# were rejected with "max_tokens must be at least 1, got 0".
if [[ "$CONC" == 17 ]]; then
    export TBENCH_TIMEOUT_S=5400
    # Default agent timeout is 8h, so 0.1 caps a task at 48 min and most
    # finish inside the 90-minute window rather than all timing out.
    export TBENCH_TIMEOUT_MULT=0.1
    exec bash "$INFERENCEX_REPO_ROOT/analysis/engram/tbench_run.sh"
fi

# conc 15/16 are the TP4 single-node analysis tasks, one node each:
#   15 = CRUXEval-O output prediction, baseline vs Engram removed, nothing executed
#   16 = likelihood ablation with the phase arms (prefill-only / decode-only)
if [[ "$CONC" == 15 || "$CONC" == 16 ]]; then
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
    export VLLM_USE_V2_MODEL_RUNNER=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python3 -m pip install -q --no-input datasets 2>&1 | tail -2 || true
    cd "$INFERENCEX_REPO_ROOT"
    if [[ "$CONC" == 15 ]]; then
        exec python3 analysis/engram/cruxeval_ablation.py \
            --model "$MODEL_PATH" --tp "$TP" \
            --limit "${ENGRAM_CRUX_LIMIT:-0}" \
            --out "$RESULT_DIR/engram_cruxeval"
    fi
    exec python3 analysis/engram/nll_ablation.py \
        --model "$MODEL_PATH" --tp "$TP" \
        --arms "${ENGRAM_NLL_ARMS:-baseline,ablated,prefill_only,decode_only}" \
        --chunks-per-domain "${ENGRAM_NLL_CHUNKS:-100}" \
        --out "$RESULT_DIR/engram_nll"
fi

echo "dsv41flash b200: no mode for CONC=$CONC on this analysis branch" >&2
exit 1
