#!/usr/bin/env bash

exec env DRAFT_SAMPLE_METHOD=probabilistic REJECTION_SAMPLE_METHOD=block \
    bash "$(dirname "$0")/dsv4dspark_fp4_b300_vllm.sh" "$@"
