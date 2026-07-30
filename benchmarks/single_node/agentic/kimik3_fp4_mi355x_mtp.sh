#!/usr/bin/env bash
set -euo pipefail

# DSpark variant of kimik3_fp4_mi355x.sh. The MI355X launcher routes
# spec-decoding=mtp rows to this suffix, while the shared base recipe owns the
# model, KV-offload, AgentX replay, and eval plumbing.
#
# Keep this wrapper aligned with the upstream AMD Kimi-K3 DSpark reproducer.
# The first AgentX validation is deliberately GPU-only at c1 so a server or
# kernel failure cannot be attributed to a KV-offload connector.
export SPEC_DECODE=true
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
export PREFIX_CACHING="${PREFIX_CACHING:-auto}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
# Diagnostic-only synchronization: surface the exact ROCm kernel that raises
# the asynchronous HSA 0x1016 / hipErrorLaunchFailure seen in run 30519967494.
export AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}"

exec "$(dirname "$0")/kimik3_fp4_mi355x.sh" "$@"
