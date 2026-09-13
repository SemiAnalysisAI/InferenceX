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

# conc 19 = one-off SSD KV-offload feasibility probe. Nothing in the repo
# offloads KV to disk, so this finds out whether it works at all on a B200
# node before any schema or recipe work: LMCache with a disk tier and a 1 GB
# CPU tier, then check a prompt comes back after being evicted from both.
if [[ "$CONC" == 19 ]]; then
    exec bash "$INFERENCEX_REPO_ROOT/analysis/ssd_offload/probe.sh"
fi

# conc 15/16 are the TP4 single-node analysis tasks, one node each:
#   15 = CRUXEval-O output prediction, baseline vs Engram removed, nothing executed
#   16 = likelihood ablation with the phase arms (prefill-only / decode-only)
# conc 18 is the likelihood run again with a boundary sweep: does decode_only's
# recovery depend on how much context was built with the gate shut? Fewer chunks
# per domain because it pays the whole measurement once per boundary.
if [[ "$CONC" == 18 ]]; then
    export ENGRAM_NLL_BOUNDARIES="${ENGRAM_NLL_BOUNDARIES:-512,1792,3072,3456}"
    export ENGRAM_NLL_CHUNKS="${ENGRAM_NLL_CHUNKS:-25}"
    CONC=16
fi

# conc 20 = CRUXEval-O again, served the way gsm8k was: the DeepSeek-V4.1 chat
# encoding with thinking on and reasoning effort high (vLLM's chat defaults),
# instead of the raw two-shot completion conc 15 ran. Same items, same arms;
# only the prompt regime changes. The budget is raised because the model now
# reasons before the [ANSWER] block, and a truncated reasoning trace would be
# graded wrong for a reason that has nothing to do with Engram.
if [[ "$CONC" == 20 ]]; then
    export ENGRAM_CRUX_PROMPT=chat
    export ENGRAM_CRUX_MAX_TOKENS="${ENGRAM_CRUX_MAX_TOKENS:-12288}"
    export ENGRAM_CRUX_MAX_MODEL_LEN="${ENGRAM_CRUX_MAX_MODEL_LEN:-16384}"
    CONC=15
fi

# conc 21 = routing-pinned ablation. Teacher-forced CRUXEval answers scored
# with Engram on/off while the MoE expert choice is recorded from one arm and
# forced onto another, to split the ablation penalty into lost features vs
# the downstream routing shift they cause. TP4, one node.
if [[ "$CONC" == 21 ]]; then
    export ENGRAM_MODE_ROUTING=1
    CONC=15
fi

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
    if [[ -n "${ENGRAM_MODE_ROUTING:-}" ]]; then
        exec python3 analysis/engram/routing_ablation.py \
            --model "$MODEL_PATH" --tp "$TP" \
            --limit "${ENGRAM_CRUX_LIMIT:-0}" \
            --arms "${ENGRAM_ROUTE_ARMS:-baseline,ablated,baseline_selfpin,ablated_selfpin,ablated_pinned,baseline_routeabl}" \
            --out "$RESULT_DIR/engram_routing"
    fi
    if [[ "$CONC" == 15 ]]; then
        exec python3 analysis/engram/cruxeval_ablation.py \
            --model "$MODEL_PATH" --tp "$TP" \
            --limit "${ENGRAM_CRUX_LIMIT:-0}" \
            --prompt-style "${ENGRAM_CRUX_PROMPT:-auto}" \
            --max-tokens "${ENGRAM_CRUX_MAX_TOKENS:-2048}" \
            --max-model-len "${ENGRAM_CRUX_MAX_MODEL_LEN:-8192}" \
            --arms "${ENGRAM_CRUX_ARMS:-baseline,ablated,prefill_only,decode_only}" \
            --out "$RESULT_DIR/engram_cruxeval"
    fi
    exec python3 analysis/engram/nll_ablation.py \
        --model "$MODEL_PATH" --tp "$TP" \
        --arms "${ENGRAM_NLL_ARMS:-baseline,ablated,prefill_only,decode_only}" \
        --chunks-per-domain "${ENGRAM_NLL_CHUNKS:-100}" \
        ${ENGRAM_NLL_BOUNDARIES:+--boundaries "$ENGRAM_NLL_BOUNDARIES"} \
        --out "$RESULT_DIR/engram_nll"
fi

echo "dsv41flash b200: no mode for CONC=$CONC on this analysis branch" >&2
exit 1
