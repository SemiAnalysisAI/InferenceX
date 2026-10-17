#!/usr/bin/env bash

# Probabilistic-drafting arm of the DSV4-Pro DSpark AL collection: identical to
# dsv4dspark_fp4_b300_vllm.sh except draft_sample_method=probabilistic (the recipe
# uses greedy). A separate file only because speedbench-al.yml resolves the collector
# as ${model-prefix}_fp4_b300_vllm.sh; it delegates so the two arms cannot drift.
# Dispatch with model-prefix=dsv4dsparkprob.

exec env DRAFT_SAMPLE_METHOD=probabilistic \
    bash "$(dirname "$0")/dsv4dspark_fp4_b300_vllm.sh" "$@"
