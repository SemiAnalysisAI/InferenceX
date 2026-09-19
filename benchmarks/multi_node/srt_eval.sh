#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Accuracy evaluation using InferenceX benchmark_lib.
# Requires: endpoint infmax_workspace and explicit workflow metadata.

set -e

if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 endpoint infmax_workspace" >&2
    exit 1
fi

ENDPOINT=$1
INFMAX_WORKSPACE=$2

HOST=$(echo "$ENDPOINT" | sed -E 's|https?://||; s|:.*||')
PORT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')

echo "Eval Config: endpoint=${ENDPOINT}; host=${HOST}; port=${PORT}; workspace=${INFMAX_WORKSPACE}"

# cd to workspace so that relative paths (e.g., utils/evals/*.yaml) resolve
cd "${INFMAX_WORKSPACE}"

source "${INFMAX_WORKSPACE}/benchmarks/benchmark_lib.sh"

# The workflow supplies topology and concurrency; srt-slurm supplies MODEL_NAME
# from the recipe's served model name. Missing inputs are configuration errors.
check_env_vars \
    IS_MULTINODE MODEL_NAME EVAL_CONC PREFILL_TP PREFILL_EP \
    PREFILL_DP_ATTN DECODE_DP_ATTN

# Translate the explicit workflow names to benchmark_lib's metadata names.
export EVAL_CONCURRENT_REQUESTS="$EVAL_CONC"
export TP="$PREFILL_TP"
export CONC="$EVAL_CONC"
export EP_SIZE="$PREFILL_EP"
export DP_ATTENTION="$PREFILL_DP_ATTN"
export PREFILL_DP_ATTENTION="$PREFILL_DP_ATTN"
export DECODE_DP_ATTENTION="$DECODE_DP_ATTN"

echo "Running evaluation for ${MODEL_NAME} with concurrent-requests=${EVAL_CONCURRENT_REQUESTS}..."
eval_rc=0
run_eval --port "$PORT" || eval_rc=$?

echo "Generating lm-eval summary..."
append_lm_eval_summary || true

mkdir -p /logs/eval_results
echo "Copying eval artifacts to /logs/eval_results/..."
cp -v meta_env.json /logs/eval_results/ 2>/dev/null || true
stage_eval_artifacts /logs/eval_results "$PWD" || true

if [[ "$eval_rc" -ne 0 ]]; then
    echo "Evaluation failed with exit code ${eval_rc}"
    exit "$eval_rc"
fi

echo "Evaluation complete"
