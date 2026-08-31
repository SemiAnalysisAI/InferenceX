#!/usr/bin/env bash

# Probabilistic-drafting arm of the DSV4-Pro DSpark AL collection.
#
# Identical to dsv4dspark_fp4_b300_vllm.sh in every respect except
# draft_sample_method, which becomes "probabilistic" instead of the published
# recipe's "greedy". Running both arms is what answers the open question on
# DSV4-Pro (probabilistic beat greedy at every level on Kimi-K3, see
# golden_al_distribution/kimik3_dspark*.yaml, but that has not been shown here).
#
# This exists as a separate file only because speedbench-al.yml resolves the
# collector purely as ${model-prefix}_fp4_b300_vllm.sh, so a second dispatchable
# entry point is the only way to launch the second arm. It delegates instead of
# duplicating the collector so the two arms cannot drift apart.
#
# Dispatch with model-prefix=dsv4dsparkprob.

exec env DRAFT_SAMPLE_METHOD=probabilistic \
    bash "$(dirname "$0")/dsv4dspark_fp4_b300_vllm.sh" "$@"
