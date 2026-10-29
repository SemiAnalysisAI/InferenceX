#!/usr/bin/env bash

# Probabilistic-drafting, block-verification arm of the DSV4-Pro DSpark AL collection:
# dsv4dspark_fp4_b300_vllm.sh with draft_sample_method=probabilistic and
# rejection_sample_method=block. Dispatch with model-prefix=dsv4dspark
# collector-variant=probabilistic_sample_method_block_rejection_sample_method.

exec env DRAFT_SAMPLE_METHOD=probabilistic REJECTION_SAMPLE_METHOD=block \
    bash "$(dirname "$0")/dsv4dspark_fp4_b300_vllm.sh" "$@"
