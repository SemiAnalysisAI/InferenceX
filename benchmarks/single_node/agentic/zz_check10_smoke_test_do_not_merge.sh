#!/usr/bin/env bash
# [DO NOT MERGE] Throwaway smoke test for codeowner-signoff-verify Check 10:
# agentic spec-decode configs must simulate acceptance at the committed golden
# AL from golden_al_distribution/. This script DELIBERATELY omits synthetic
# acceptance (real rejection sampling) so the verifier should FAIL Check 10 and
# name this file. The PR adding it will be closed without merging.
set -euo pipefail

vllm serve deepseek-ai/DeepSeek-V4-Pro-NVFP4 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
