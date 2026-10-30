#!/usr/bin/env bash

# Block-verification arm of the MiniMax-M3 EAGLE3 GQA AL collection:
# minimaxm3_fp4_b300_vllm.sh with the GQA draft and rejection_sample_method=block.
# Dispatch with model-prefix=minimaxm3 collector-variant=gqa_block_rejection_sample_method.

exec env DRAFT_MODEL=Inferact/MiniMax-M3-EAGLE3-GQA \
    SPEC_EXTRA=', "rejection_sample_method": "block"' \
    COLLECTOR=benchmarks/single_node/speedbench/minimaxm3_fp4_b300_vllm_gqa_block_rejection_sample_method.sh \
    bash "$(dirname "$0")/minimaxm3_fp4_b300_vllm.sh" "$@"
