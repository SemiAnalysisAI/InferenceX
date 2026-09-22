#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    MODEL_NAME \
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
    FRAMEWORK \
    PREFILL_IMAGE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

export TIME_LIMIT=08:00:00
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE
export PREFILL_IMAGE

if [[ "$PREFILL_NODES" -ne 1 || "$DECODE_NODES" -ne 1 || \
      "$PREFILL_NUM_WORKERS" -ne 1 || "$DECODE_NUM_WORKERS" -ne 1 ]]; then
    echo "Error: tilert supports exactly 1 prefill node/worker + 1 decode node/worker" \
         "(got PREFILL_NODES=$PREFILL_NODES x$PREFILL_NUM_WORKERS, DECODE_NODES=$DECODE_NODES x$DECODE_NUM_WORKERS)" >&2
    exit 1
fi

# TileRT configuration. Every value is explicit here: server_tilert.sh
# validates each one with check_env_vars and supplies no defaults of its own.
export TILERT_VERSION=0.1.6.post1
export TILERT_PROFILE=glm5_2          # decode_server --model (TileRT model profile)
export TILERT_MODEL_TYPE=glm-5        # weight_converter --model_type (fallback converter)
export TILERT_MODEL_PKG=glm_5_2_rocm  # per-model converter package, preferred when importable
export SERVED_MODEL_NAME=glm5_2
# GLM-5.3's full context window (config.json max_position_embeddings), as every
# in-tree GLM-5.2 recipe serves. (202752 was GLM-5.1's, inherited from the B200
# TileRT recipe this mirrors.)
#
# Memory at this context, per rank, bf16 wire layout (verified against the
# tilert 0.1.6 and vLLM 0.24.0 sources and the MI355X logs, 287.98 GiB cards):
#   decode : weights 90.72 GiB + engine cache window 93.25 GiB
#            + PD receive buffer 99.06 GiB (receive_server.py, dense in max_seq_len)
#   prefill: weights 90.45 GiB + profiling/non-torch 40.3 GiB + vLLM KV 91.71 GiB
#            + PD staging buffer 99.06 GiB (prefill_connector.py, TP rank 0,
#              allocated OUTSIDE vLLM's gpu-memory-utilization budget)
# Undivided, neither side starts: the decode rank is node-marginal (~283 of
# 288 GiB) and the prefill rank cannot fit at any utilization (~321 GiB).
# tilert 0.1.6.post1 keeps both buffers on the GPU but shards them by layer
# across the eight devices (layer lid on device lid % 8, TILERT_PD_SHARDS,
# default on), so each card holds 12.54 GiB instead of 99.06 GiB on one.
# convert() dequantises each layer on the device that received it, which spreads
# its transients too (108.6 KiB/token, 82.9 GiB at 800k tokens) instead of
# leaving them on cuda:0. Measured on 2x8 MI350X at this context with bf16 KV:
# decode peaks at 202.1 GiB per card, prefill at 269.2 GiB of 287.69 GiB, and
# the KV path stays device-to-device at 108 GB/s (81 GB in 751 ms, 54% of the
# 4x400 GbE line rate). No host hop, so the DRAM patch and
# TILERT_PD_BUFFER_DEVICE are not used here.
export TILERT_MAX_MODEL_LEN=1048576
export TILERT_PD_BUFFER_DEVICE=cuda
export TILERT_TRANSPORT=mooncake
export TILERT_PARSER=none
export TILERT_RDMA_STRICT=0
export TILERT_CONVERT_LOCK_WAIT=21600
export TILERT_SIMULATE_ACC_METHOD=match-expected
export TILERT_WEIGHTS_DIR="/models/${MODEL_NAME}-tilert-tp${DECODE_TP}"
# bf16 MLA KV on both roles. This is the only layout TileRT 0.1.6 can consume
# from vLLM on ROCm: MlaNsaProfile.classify_layers infers the layout from the
# cache tensor stride and accepts exactly 1152 B/token (bf16) or 656 B/token
# (fp8_ds_mla). vLLM's ROCM_AITER_MLA_SPARSE backend has no fp8_ds_mla; its
# plain "fp8" writes a flat 576 B/token row, which the connector rejects at
# register_kv_caches. Explicit bfloat16 rather than auto so the stride does not
# depend on the model dtype. Never float16: it passes the 1152 B check and is
# then read as bf16.
export PREFILL_KV_DTYPE=bfloat16
# The ROCm backend supports block sizes [1, 64] and vLLM picks 1, which makes
# the connector's KI plane copy fail and MLA address the wrong rows.
export PREFILL_BLOCK_SIZE=64
export DECODE_KV_DTYPE=bf16
# With the PD staging buffer in host memory, vLLM needs 90.45 (weights) + 40.3
# (profiling) + 91.71 GiB (KV for one 1048576-token request) = 222.5 GiB inside
# its budget: 0.85 x 287.98 = 244.8 GiB leaves 22 GiB of KV margin and 43 GiB
# outside the budget for the ~6.3 GiB non-torch baseline measured on the decode
# OOM node (287.98 - 95.94 free - 184.17 - 1.58 reserved). 0.75 (216 GiB) refuses
# with "91.71 GiB KV cache is needed ... available 85.25 GiB".
export GPU_MEM_UTIL=0.85
export SKIP_CONTAINER_BARRIER=0
# Two images, one per rank, ~32 GB each. On a node that has neither cached the
# pull alone outlasts the SGLang path's 300s default and the 1800s this script
# used to hardcode, and the rank that comes up first waits out the whole
# timeout while its peer is still pulling.
export CONTAINER_BARRIER_TIMEOUT=5400
export ROUTER_PORT=30000
export PREFILL_PORT=8000
export DECODE_CTRL_PORT=5556
export DECODE_HTTP_PORT=5557
export DECODE_WAIT=7200               # prefill waits for the decode ctrl port
export PREFILL_WAIT=3600              # prefill waits for its own vLLM port
export ROUTER_WAIT=10800              # decode waits for the router port to open

if [[ "$SPEC_DECODING" == "mtp" ]]; then
    # TileRT decode drafts at depth 3 (the only depth the ROCm GLM profile
    # builds) and the golden acceptance curve is keyed on it. The vLLM prefill
    # rank only has to materialise the MTP layer's KV, so it runs at 1.
    export DECODE_MTP_SIZE=3
    export PREFILL_SPEC_TOKENS=1
else
    export DECODE_MTP_SIZE=0
    export PREFILL_SPEC_TOKENS=0
fi
export TILERT_QUEUE_TIMEOUT=0
# submit.sh and job.slurm validate KV_OFFLOADING, and the fixed-seq-len matrix
# emits it only for agentic scenarios. TileRT has no KV offload backend.
export KV_OFFLOADING=none

if [[ "$PREFILL_EP" -ne 1 || "$DECODE_EP" -ne 1 || \
      "$PREFILL_DP_ATTN" == "true" || "$DECODE_DP_ATTN" == "true" ]]; then
    echo "Error: tilert runs pure TP8 on both roles; ep must be 1 and dp-attn false" >&2
    exit 1
fi
export PREFILL_ENABLE_EP=false
export PREFILL_ENABLE_DP=false
export DECODE_ENABLE_EP=false
export DECODE_ENABLE_DP=false

JOB_ID=$(bash ./submit.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    ${PREFILL_ENABLE_EP} ${PREFILL_ENABLE_DP} \
    ${DECODE_ENABLE_EP} ${DECODE_ENABLE_DP} \
    ${PREFILL_TP} ${DECODE_TP} \
    ${RANDOM_RANGE_RATIO})

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
