#!/usr/bin/env bash
set -eo pipefail

export GPUS_PER_NODE=4 TIME_LIMIT="${TIME_LIMIT:-08:00:00}" CONTAINER_IMAGE="$IMAGE"
export PREFILL_WORKERS="${PREFILL_WORKERS:-${PREFILL_NUM_WORKERS:-1}}"
export DECODE_WORKERS="${DECODE_WORKERS:-${DECODE_NUM_WORKERS:-1}}"

cd "$(dirname "$0")/llm-d"
exec bash ./submit.sh "$PREFILL_NODES" "$DECODE_NODES" \
    "$ISL" "$OSL" "${CONC_LIST// /x}" inf "$RANDOM_RANGE_RATIO"
