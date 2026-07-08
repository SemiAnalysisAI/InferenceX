#!/usr/bin/env bash
export SPEC_DECODING=none
export DECODE_MTP_SIZE=0
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export BLOCK_SIZE="${BLOCK_SIZE:-128}"
export MEM_FRAC_STATIC="${MEM_FRAC_STATIC:-0.8}"
export MAX_MODEL_LEN=32768
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/disaggregated_recipe.sh"
