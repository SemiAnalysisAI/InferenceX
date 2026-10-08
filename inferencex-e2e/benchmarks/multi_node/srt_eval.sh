#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# srt-slurm multi-node post_eval: srt_eval.sh ENDPOINT INFMAX_WORKSPACE.
# The eval artifacts land in /logs/eval_results, which the launcher collects.
set -eo pipefail
if [[ $# -ne 2 || -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 endpoint infmax_workspace" >&2
    exit 1
fi
source "$2/benchmarks/check_env.sh"
# The workflow supplies the topology; srt-slurm supplies MODEL_NAME (served) and EVAL_CONC.
check_env_vars \
    IS_MULTINODE MODEL_NAME EVAL_CONC PREFILL_TP PREFILL_EP PREFILL_DP_ATTN DECODE_DP_ATTN
if [[ "$IS_MULTINODE" != true ]]; then
    echo "ERROR: multi-node eval requires IS_MULTINODE=true" >&2
    exit 1
fi
cd "$2"
PYTHONSAFEPATH=1 PYTHONPATH="$2${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m infx.bench eval \
    --endpoint "$1" --concurrency "$EVAL_CONC" --stage-to /logs/eval_results
