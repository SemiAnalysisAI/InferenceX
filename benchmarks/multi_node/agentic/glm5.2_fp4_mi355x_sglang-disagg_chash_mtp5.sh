#!/usr/bin/env bash

# Agentic trace-replay recipe: 1P1D GLM-5.2 MXFP4 disagg with chash decode routing.
#
# Topology (from amd-agentx-trace-glm.yaml):
#   prefill: TP8 / EP1 / no DP-attn, HiCache L2 ratio 2
#   decode:  TP8 / EP8 / DP8 / DPA, consistent_hashing router
#
# Tunables aligned with the passing chash bring-up + upstream single-node MTP5 (#2570):
#   MTP5 (steps=5 / draft=6), SGLANG_SIMULATE_ACC_LEN=3.61,
#   write_through_selective HiCache; decode cuda-graph-bs follows models.yaml (1–128).
#
# Default conc sweep: 16, 20, 24, 28 (override via CONC_LIST or test-config --conc).
# Default profiling duration: 3600s (60 min; override via DURATION).
#
# Dispatch example:
#   test-config --config-files configs/amd-agentx-trace-glm.yaml \
#     --config-keys glm5.2-fp4-mi355x-sglang-disagg-agentic-p-tp8-d-tp8ep8dp8dpa-hicache-mtp5-chash-c16-28-trace \
#     --conc 16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HICACHE_RATIO="${HICACHE_RATIO:-2}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
export HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-1}"
export HICACHE_PREFETCH_POLICY="${HICACHE_PREFETCH_POLICY:-best_effort}"

export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-5}"
export SGLANG_SIMULATE_ACC_LEN="${SGLANG_SIMULATE_ACC_LEN:-3.61}"
export SGLANG_SIMULATE_ACC_METHOD="${SGLANG_SIMULATE_ACC_METHOD:-match-expected}"
export SGLANG_SIMULATE_ACC_TOKEN_MODE="${SGLANG_SIMULATE_ACC_TOKEN_MODE:-real-draft-token}"

export ROUTER_DECODE_POLICY="${ROUTER_DECODE_POLICY:-consistent_hashing}"
export DECODE_ROUTER_POLICY="${DECODE_ROUTER_POLICY:-consistent_hashing}"

export CONC_LIST="${CONC_LIST:-16 20 24 28}"
export DURATION="${DURATION:-3600}"

source "$SCRIPT_DIR/glm5.2_fp4_mi355x_sglang-disagg.sh"
