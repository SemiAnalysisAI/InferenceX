#!/usr/bin/env bash

# Probabilistic-drafting arm of the Kimi-K3 DSpark AL collection with the engine's default
# (standard) verifier: kimik3_fp4_b300_vllm.sh plus draft_sample_method=probabilistic.
# Dispatch with model-prefix=kimik3 collector-variant=probabilistic_sample_method.

exec env SPEC_EXTRA=', "draft_sample_method": "probabilistic"' \
    COLLECTOR=benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm_probabilistic_sample_method.sh \
    bash "$(dirname "$0")/kimik3_fp4_b300_vllm.sh" "$@"
