#!/usr/bin/env bash
# DSR1-0528 FP8 MI300X vLLM PD-disaggregation recipe.
#
# Generic, fully parameterized disagg driver: every topology in the target
# matrix is expressed purely through the env vars the InferenceX workflow
# exports from .github/configs/amd-master.yaml (per-scenario prefill/decode
# num-worker, tp, ep, dp-attn, plus PREFILL_NODES/DECODE_NODES/DECODE_MTP_SIZE
# carried in additional-settings). No per-scenario branching lives here.
#
# Supported topologies (1k1k and 8k1k): 1P1D, 1P2D, 1P3D, 2P1D, 2P2D, including
# EP8 (prefill.ep=8 and/or decode.ep=8) and mixed TP8-prefill | EP8-decode,
# with MTP speculative-decoding sizes 0/1/2/3 via DECODE_MTP_SIZE.
#
# Model-specific vLLM flags live in amd_utils/models_vllm.yaml under the
# DeepSeek-R1-0528 entry; EP/DP/TP/MTP are layered on by amd_utils/submit.sh
# and amd_utils/server_vllm.sh.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    DECODE_TP \
    DECODE_EP \
    DECODE_DP_ATTN \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO \
    FRAMEWORK

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

export TIME_LIMIT="08:00:00"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# EP/DP booleans (same derivation as the other vLLM/SGLang disagg recipes).
# PREFILL_EP/DECODE_EP are integers from the YAML: 1 => TP-only (EP off),
# 8 => EP8 (--enable-expert-parallel). server_vllm.sh adds the flag when the
# boolean is true and sizes EP to the per-worker TP size.
if [[ "${PREFILL_EP:-1}" -eq 1 ]]; then
    export PREFILL_ENABLE_EP=false
else
    export PREFILL_ENABLE_EP=true
fi

if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
    export PREFILL_ENABLE_DP=true
else
    export PREFILL_ENABLE_DP=false
fi

if [[ "${DECODE_EP:-1}" -eq 1 ]]; then
    export DECODE_ENABLE_EP=false
else
    export DECODE_ENABLE_EP=true
fi

if [[ "$DECODE_DP_ATTN" == "true" ]]; then
    export DECODE_ENABLE_DP=true
else
    export DECODE_ENABLE_DP=false
fi

# MTP speculative-decoding depth (0/1/2/3). Rides in via decode.additional-settings;
# default to 0 (disabled) and re-export so it propagates through submit.sh ->
# sbatch (--export=ALL) -> job.slurm -> server_vllm.sh, which derives IS_MTP and
# the speculative config from it.
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# DP8EP mode (1k1k EP8 rows + mixed 8k1k EP8-decode rows). Distinct from dp-attn:
# vLLM does data-parallel attention via --data-parallel-size, not the SGLang
# --enable-dp-attention. These booleans ride in via additional-settings; default
# false and re-export so they reach server_vllm.sh (via submit.sh --export=ALL +
# job.slurm -e). When true, server_vllm.sh emits
# --tensor-parallel-size 1 --data-parallel-size <tp_size> --enable-expert-parallel
# --all2all-backend <backend> and suppresses the plain --tensor-parallel-size injection.
export PREFILL_DP8EP="${PREFILL_DP8EP:-false}"
export DECODE_DP8EP="${DECODE_DP8EP:-false}"

# DP8EP is authoritative over the legacy EP/DP-attention booleans: server_vllm.sh's
# DP8EP branch emits --data-parallel-size + --enable-expert-parallel itself, so force
# the per-role ENABLE_EP/ENABLE_DP off here (the YAML keeps an honest ep:8). This
# guarantees no duplicate --enable-expert-parallel regardless of the ep integer.
if [[ "$PREFILL_DP8EP" == "true" ]]; then
    export PREFILL_ENABLE_EP=false
    export PREFILL_ENABLE_DP=false
fi
if [[ "$DECODE_DP8EP" == "true" ]]; then
    export DECODE_ENABLE_EP=false
    export DECODE_ENABLE_DP=false
fi

# Parameter order matches SGLang disagg submit.sh; arg 16 is optional NODELIST.
JOB_ID=$(bash ./submit.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    ${PREFILL_ENABLE_EP} ${PREFILL_ENABLE_DP} \
    ${DECODE_ENABLE_EP} ${DECODE_ENABLE_DP} \
    ${PREFILL_TP} ${DECODE_TP} \
    ${RANDOM_RANGE_RATIO} \
    "${NODELIST:-}")

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
