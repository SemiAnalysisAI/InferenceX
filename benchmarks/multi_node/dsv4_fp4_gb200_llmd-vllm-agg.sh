#!/usr/bin/env bash
set -eo pipefail

export GPUS_PER_NODE=4 TIME_LIMIT=08:00:00 CONTAINER_IMAGE="$IMAGE"
export PREFILL_WORKERS=1 DECODE_WORKERS=1

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/llm-d"
exec bash ./submit.sh "$PREFILL_NODES" "$DECODE_NODES" \
    "$ISL" "$OSL" "${CONC_LIST// /x}" inf "$RANDOM_RANGE_RATIO"
