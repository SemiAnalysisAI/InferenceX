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
export EVAL_FRAMEWORK="${EVAL_FRAMEWORK:-lm-eval}"
export EVAL_LIMIT="${EVAL_LIMIT:-}"
export EVAL_SUITE="${EVAL_SUITE:-}"
export SWEBENCH_GEN_MODE="${SWEBENCH_GEN_MODE:-}"
export SWEBENCH_USE_MODAL="${SWEBENCH_USE_MODAL:-false}"
export MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-}"
export MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-}"
export IS_AGENTIC="${IS_AGENTIC:-0}"
export SCENARIO_TYPE="${SCENARIO_TYPE:-}"
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

# The recipe's eight-hour default exceeds this one-off validation budget.
[[ -z "${SLURM_JOB_ID:-}" && "$PREFILL_NODES" == 2 && "$DECODE_NODES" == 2 &&
   "$GPUS_PER_NODE" == 4 && "$PREFILL_WORKERS" == 1 && "$DECODE_WORKERS" == 1 &&
   "$ISL" == 8192 && "$OSL" == 1024 && "$CONCURRENCIES" == 1 &&
   "$BENCH_NUM_PROMPTS_MULTIPLIER" == 10 && "$RUN_EVAL" == false && "$EVAL_ONLY" == false &&
   "$IS_AGENTIC" == 0 && "$CONFIG_FILE" == dsv4-fp4-gb200-low-latency.yaml &&
   "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ && "${GITHUB_RUN_ATTEMPT:-}" == 1 &&
   "${TASK_JOB_NAME:-}" == "powerx3052-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" &&
   "${TASK_JOB_USER:-}" == "$(id -un)" ]] || { echo "Invalid bounded validation request" >&2; exit 1; }
[[ ! -e "${VALIDATION_JOB_RECEIPT:?}" && ! -e "${VALIDATION_LOGS:?}/request.txt" ]] || { echo "Refusing duplicate submission" >&2; exit 1; }
existing_job=$(squeue -h --name="$TASK_JOB_NAME" --user="$TASK_JOB_USER" -o '%i') || exit 1
[[ -z "$existing_job" ]] || {
    echo "Existing owned job; reconcile before submitting" >&2; exit 1;
}
# Refuse inherited sbatch environment options that could add an array or
# otherwise request resources outside the single explicit command below.
while IFS= read -r option; do
    [[ "$option" == SBATCH_PARTITION ]] || { echo "Unexpected sbatch option: $option" >&2; exit 1; }
done < <(compgen -A variable SBATCH_ || true)
TIME_LIMIT=00:39:00
mkdir -p "${VALIDATION_LOGS:?}"
printf '%s\n' "$TASK_JOB_NAME|$TASK_JOB_USER|4|4|$TIME_LIMIT|C1|8192|1024|16|no-eval" \
    > "$VALIDATION_LOGS/request.txt"

mkdir -p "$BENCHMARK_LOGS_DIR"

submit_rc=0
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
    --job-name "$TASK_JOB_NAME" --comment "$TASK_JOB_NAME" --no-requeue --export=ALL \
    --chdir "$REPO_ROOT/benchmarks/multi_node/llm-d" \
    --output "${BENCHMARK_LOGS_DIR}/slurm_job-%j.out" \
    --error  "${BENCHMARK_LOGS_DIR}/slurm_job-%j.err" \
    "$REPO_ROOT/runners/llmd_validation_job.sh") || submit_rc=$?
printf '%s\n' "$JOB_ID" > "$VALIDATION_LOGS/sbatch-output.txt"
JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Ambiguous sbatch response; do not retry" >&2; exit 1; }
printf '%s\n' "$JOB_ID" > "$VALIDATION_JOB_RECEIPT"
cp "$VALIDATION_JOB_RECEIPT" "$VALIDATION_LOGS/job-id.txt"
submission_record=$(scontrol show job "$JOB_ID" --oneliner) || exit 1
printf '%s\n' "$submission_record" > "$VALIDATION_LOGS/submission-job.txt"
for field in "JobId=$JOB_ID" "JobName=$TASK_JOB_NAME" "TimeLimit=00:39:00" "Requeue=0"; do
    [[ " $submission_record " == *" $field "* ]] || { echo "Unexpected submission metadata: $field" >&2; exit 1; }
done
node_count=$(printf '%s\n' "$submission_record" | tr ' ' '\n' | sed -n 's/^NumNodes=//p')
# Pending jobs may show equal minimum/maximum bounds instead of a scalar.
[[ "$node_count" == 4 || "$node_count" == 4-4 ]] || {
    echo "Unexpected submission node count: $node_count" >&2; exit 1;
}
request_tres=$(printf '%s\n' "$submission_record" | tr ' ' '\n' | sed -n 's/^ReqTRES=//p')
[[ ",$request_tres," == *,gres/gpu=16,* && " $submission_record " == *" UserId=$TASK_JOB_USER("* ]] || {
    echo "Unexpected submission GPU request or owner" >&2; exit 1;
}
[[ "$submit_rc" == 0 ]] || { echo "sbatch failed; do not retry" >&2; exit "$submit_rc"; }

echo "$JOB_ID"
