#!/usr/bin/env bash
#
# Submit a multi-node llmd-vllm wide-EP P/D disagg benchmark job to SLURM.
# Modeled after benchmarks/multi_node/amd_utils/submit.sh; prints JOB_ID on
# stdout so the runner can poll for completion.
#
# Topology (matches the llm-d wide-EP guide reference):
#   1 prefill instance with DP=PREFILL_NODES * GPUS_PER_NODE
#   1 decode  instance with DP=DECODE_NODES  * GPUS_PER_NODE
#   each instance spans PREFILL_NODES / DECODE_NODES nodes via vLLM
#   --data-parallel-hybrid-lb. Total nodes = PREFILL_NODES + DECODE_NODES.

set -euo pipefail

# Repo root resolved from this script's location, so paths below are
# independent of the caller's $PWD (the wrapper cd's into llm-d/ before
# invoking this script).
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

check_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Error: ${name} not set" >&2
        exit 1
    fi
}

check_env SLURM_ACCOUNT
check_env SLURM_PARTITION
check_env TIME_LIMIT
check_env MODEL_PATH
check_env MODEL_NAME
check_env CONTAINER_IMAGE
check_env RUNNER_NAME
check_env BENCHMARK_LOGS_DIR

PREFILL_NODES=$1
DECODE_NODES=$2
ISL=$3
OSL=$4
CONCURRENCIES=$5
REQUEST_RATE=${6:-inf}
RANDOM_RANGE_RATIO=${7:-0.8}

NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

export DOCKER_IMAGE_NAME=$CONTAINER_IMAGE
export MODEL_DIR=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export NUM_NODES=$NUM_NODES
export PREFILL_NODES=$PREFILL_NODES
export DECODE_NODES=$DECODE_NODES
export GPUS_PER_NODE=$GPUS_PER_NODE
# Worker count per role (Option B): the role's nodes are split into this many
# INDEPENDENT DP/EP engines (default 1 = one engine over all role nodes). Each
# engine spans role_nodes/workers nodes, so DP_SIZE is PER-ENGINE. Matches how
# dynamo/AMD and upstream oci-high-tpt run 2P high-tpt (2 prefill : 1 decode).
export PREFILL_WORKERS="${PREFILL_WORKERS:-1}"
export DECODE_WORKERS="${DECODE_WORKERS:-1}"
if (( PREFILL_NODES % PREFILL_WORKERS != 0 )); then
    echo "Error: PREFILL_NODES ($PREFILL_NODES) not divisible by PREFILL_WORKERS ($PREFILL_WORKERS)" >&2
    exit 1
fi
if (( DECODE_NODES % DECODE_WORKERS != 0 )); then
    echo "Error: DECODE_NODES ($DECODE_NODES) not divisible by DECODE_WORKERS ($DECODE_WORKERS)" >&2
    exit 1
fi
export PREFILL_DP_SIZE=$(( PREFILL_NODES / PREFILL_WORKERS * GPUS_PER_NODE ))
export DECODE_DP_SIZE=$((  DECODE_NODES  / DECODE_WORKERS  * GPUS_PER_NODE ))
export BENCH_INPUT_LEN=$ISL
export BENCH_OUTPUT_LEN=$OSL
export BENCH_MAX_CONCURRENCY=$CONCURRENCIES
export BENCH_REQUEST_RATE=$REQUEST_RATE
export BENCH_RANDOM_RANGE_RATIO=$RANDOM_RANGE_RATIO
# Match the AMD multinode default.
export BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"

export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_CONC="${EVAL_CONC:-}"
export FRAMEWORK="${FRAMEWORK:-llmd-vllm}"
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"
export SPEC_DECODING="${SPEC_DECODING:-none}"
export IS_MULTINODE="${IS_MULTINODE:-true}"
export CONFIG_FILE="${CONFIG_FILE:-}"

# Recipe may override SLURM time limit (longer topologies need more wall time).
if [[ -n "$CONFIG_FILE" ]]; then
    RECIPE_PATH="${REPO_ROOT}/benchmarks/multi_node/llm-d-recipes/${CONFIG_FILE}"
    if [[ -f "$RECIPE_PATH" ]]; then
        RECIPE_TIME=$(python3 -c "
import yaml, sys
r = yaml.safe_load(open('$RECIPE_PATH'))
t = r.get('slurm', {}).get('time_limit', '')
print(t)
" 2>/dev/null || true)
        [[ -n "$RECIPE_TIME" ]] && TIME_LIMIT="$RECIPE_TIME"
    fi
fi

mkdir -p "$BENCHMARK_LOGS_DIR"

JOB_ID=$(sbatch \
    --parsable \
    --exclusive \
    -N "$NUM_NODES" \
    -n "$NUM_NODES" \
    --ntasks-per-node=1 \
    --gres=gpu:"$GPUS_PER_NODE" \
    --time "$TIME_LIMIT" \
    --partition "$SLURM_PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --job-name "$RUNNER_NAME" \
    --output "${BENCHMARK_LOGS_DIR}/slurm_job-%j.out" \
    --error  "${BENCHMARK_LOGS_DIR}/slurm_job-%j.err" \
    "$(dirname "$0")/job.slurm")

if [[ -z "$JOB_ID" ]]; then
    echo "Error: sbatch failed" >&2
    exit 1
fi

echo "$JOB_ID"
