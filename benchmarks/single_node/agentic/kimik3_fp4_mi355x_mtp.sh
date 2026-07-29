#!/usr/bin/env bash
set -euo pipefail

# DSpark variant of kimik3_fp4_mi355x.sh. The MI355X launcher routes
# spec-decoding=mtp rows to this suffix, while the shared base recipe owns the
# model, KV-offload, AgentX replay, and eval plumbing.
#
# These overrides are the DSpark + SimpleCPUOffloadConnector combination
# validated on gfx950 in PR #2367. Keep them together: widening the batch or
# re-enabling CUDA graphs reproduced an eight-rank GPU memory access fault.
export SPEC_DECODE=true
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"
export SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-true}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-true}"

exec "$(dirname "$0")/kimik3_fp4_mi355x.sh" "$@"
