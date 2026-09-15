#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
#
# Submit a multi-node disaggregated benchmark job to SLURM. The router is co-located
# with the first prefill node, so NUM_NODES = PREFILL_NODES + DECODE_NODES.

usage() {
    cat << 'USAGE'
Usage:
  bash submit.sh <PREFILL_NODES> <PREFILL_WORKERS> <DECODE_NODES> <DECODE_WORKERS> \
                 <ISL> <OSL> <CONCURRENCIES> <REQUEST_RATE> \
                 <PREFILL_ENABLE_EP> <PREFILL_ENABLE_DP> \
                 <DECODE_ENABLE_EP> <DECODE_ENABLE_DP> \
                 <PREFILL_TP> <DECODE_TP> \
                 <RANDOM_RANGE_RATIO> [NODE_LIST]

Arguments:
  PREFILL_NODES        Number of prefill nodes
  PREFILL_WORKERS      Number of prefill workers (usually 1)
  DECODE_NODES         Number of decode nodes
  DECODE_WORKERS       Number of decode workers (usually 1)
  ISL                  Input sequence length
  OSL                  Output sequence length
  CONCURRENCIES        Concurrency levels, delimited by 'x' (e.g., "8x16x32")
  REQUEST_RATE         Request rate ("inf" for max throughput)
  PREFILL_ENABLE_EP    true/false or 1/0 (expert parallelism on prefill)
  PREFILL_ENABLE_DP    true/false or 1/0 (data-parallel attention on prefill)
  DECODE_ENABLE_EP     true/false or 1/0 (expert parallelism on decode)
  DECODE_ENABLE_DP     true/false or 1/0 (data-parallel attention on decode)
  PREFILL_TP           Tensor parallel size per prefill node
  DECODE_TP            Tensor parallel size per decode node
  RANDOM_RANGE_RATIO   Random range ratio for benchmark client
  NODE_LIST            Optional: comma-separated hostnames (must match NUM_NODES)

Required environment variables:
  SLURM_ACCOUNT    SLURM account name
  SLURM_PARTITION  SLURM partition
  TIME_LIMIT       Job time limit (e.g., "08:00:00")
  MODEL_PATH       Path to model directory (e.g., /nfsdata)
  MODEL_NAME       Model name directory
  CONTAINER_IMAGE  Docker image name (e.g., vllm_disagg_pd:latest)
  RUNNER_NAME      Runner identifier (for job name)

Required environment variables (continued):
  DRY_RUN          1 = echo composed server/router launch commands instead of
                   running them (preview a recipe against a real allocation).
USAGE
}

check_env_vars \
    SLURM_ACCOUNT SLURM_PARTITION TIME_LIMIT MODEL_PATH MODEL_NAME \
    CONTAINER_IMAGE RUNNER_NAME FRAMEWORK GPUS_PER_NODE PREFILL_EP \
    PREFILL_DP_ATTN PREFILL_NUM_WORKERS PREFILL_PP_SIZE PREFILL_DCP_SIZE PREFILL_PCP_SIZE \
    DECODE_EP DECODE_DP_ATTN DECODE_NUM_WORKERS DECODE_PP_SIZE DECODE_DCP_SIZE \
    DECODE_PCP_SIZE DECODE_MTP_SIZE BENCH_NUM_PROMPTS_MULTIPLIER DRY_RUN RUN_EVAL \
    EVAL_ONLY EVAL_FRAMEWORK IS_MULTINODE SWEBENCH_USE_MODAL BENCHMARK_LOGS_DIR \
    KEEP_CONTAINERS ROUTER_TYPE ROUTER_PORT PROXY_PING_PORT HEADNODE_PORT \
    SERVER_PORT IS_AGENTIC KV_OFFLOADING

if [[ $# -lt 15 || $# -gt 16 ]]; then
    usage >&2
    exit 1
fi

PREFILL_NODES=$1
PREFILL_WORKERS=${2}
DECODE_NODES=$3
DECODE_WORKERS=${4}
ISL=$5
OSL=$6
CONCURRENCIES=$7
REQUEST_RATE=$8
PREFILL_ENABLE_EP=${9}
PREFILL_ENABLE_DP=${10}
DECODE_ENABLE_EP=${11}
DECODE_ENABLE_DP=${12}
PREFILL_TP=${13}
DECODE_TP=${14}
RANDOM_RANGE_RATIO=${15}
NODE_LIST=${16:-}

NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

export ENGINE="${FRAMEWORK}"
export MODEL_DIR=$MODEL_PATH
export DOCKER_IMAGE_NAME=$CONTAINER_IMAGE
export PROFILER_ARGS=$profiler_args

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    check_env_vars PROXY_STREAM_IDLE_TIMEOUT
    export PROXY_STREAM_IDLE_TIMEOUT
fi
# xP = prefill workers, yD = decode workers (may span multiple nodes)
export xP=$PREFILL_WORKERS
export yD=$DECODE_WORKERS
export PREFILL_TP_SIZE=$(( $PREFILL_NODES * $PREFILL_TP / $PREFILL_WORKERS ))
export PREFILL_ENABLE_EP
export PREFILL_ENABLE_DP
export PREFILL_TP
export PREFILL_EP
export PREFILL_DP_ATTN
export PREFILL_NUM_WORKERS
export PREFILL_PP_SIZE
export PREFILL_DCP_SIZE
export PREFILL_PCP_SIZE
export DECODE_TP_SIZE=$(( $DECODE_NODES * $DECODE_TP / $DECODE_WORKERS ))
export DECODE_ENABLE_EP
export DECODE_ENABLE_DP
export DECODE_TP
export DECODE_EP
export DECODE_DP_ATTN
export DECODE_NUM_WORKERS
export DECODE_PP_SIZE
export DECODE_DCP_SIZE
export DECODE_PCP_SIZE
export DECODE_MTP_SIZE

export NUM_NODES=$NUM_NODES
export GPUS_PER_NODE=$GPUS_PER_NODE
export MODEL_NAME=$MODEL_NAME
export BENCH_INPUT_LEN=${ISL}
export BENCH_OUTPUT_LEN=${OSL}
export BENCH_NUM_PROMPTS_MULTIPLIER
export BENCH_MAX_CONCURRENCY=${CONCURRENCIES}
export BENCH_REQUEST_RATE=${REQUEST_RATE}
export BENCH_RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}

# sbatch defaults to --export=ALL, so exporting DRY_RUN carries it through job.slurm
# and Docker (-e) into server_sglang.sh.
export DRY_RUN

export RUN_EVAL
export EVAL_ONLY
export EVAL_CONC="${EVAL_CONC:-}"
export EVAL_FRAMEWORK
export EVAL_SUITE="${EVAL_SUITE:-}"
export SWEBENCH_GEN_MODE="${SWEBENCH_GEN_MODE:-}"
export FRAMEWORK="${FRAMEWORK:-}"
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"
export SPEC_DECODING="${SPEC_DECODING:-}"
export IS_MULTINODE
export SWEBENCH_USE_MODAL
export MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-}"
export MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export SCENARIO_TYPE="${SCENARIO_TYPE:-}"
export EVAL_LIMIT="${EVAL_LIMIT:-}"

# Log directory: must be on NFS (shared filesystem) so the submit host can read SLURM output.
export BENCHMARK_LOGS_DIR
mkdir -p "$BENCHMARK_LOGS_DIR"

NODELIST_OPT=()
if [[ -n "${NODE_LIST//[[:space:]]/}" ]]; then
    IFS=',' read -r -a NODE_ARR <<< "$NODE_LIST"
    if [[ "${#NODE_ARR[@]}" -ne "$NUM_NODES" ]]; then
        echo "Error: NODE_LIST has ${#NODE_ARR[@]} nodes but NUM_NODES=${NUM_NODES}" >&2
        echo "Error: NODE_LIST='${NODE_LIST}'" >&2
        exit 1
    fi
    NODELIST_CSV="$(IFS=,; echo "${NODE_ARR[*]}")"
    NODELIST_OPT=(--nodelist "$NODELIST_CSV")
fi

# Exclude known-bad nodes for this (FRAMEWORK, MODEL_NAME) combo (e.g. broken Docker
# sockets) from node_excludes.yaml; SLURM_EXCLUDE_NODES overrides with an explicit
# list. Resolution must fail loudly rather than silently yield an empty exclude list,
# or a missing python3/PyYAML would quietly reintroduce the known-bad nodes.
EXCLUDE_OPT=()
NODE_EXCLUDES_YAML="$(dirname "$0")/node_excludes.yaml"
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
    RESOLVED_EXCLUDE_NODES="$SLURM_EXCLUDE_NODES"
elif [[ -f "$NODE_EXCLUDES_YAML" ]]; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "import yaml" >/dev/null 2>&1; then
        RESOLVED_EXCLUDE_NODES=$(python3 -c "
import yaml

with open('${NODE_EXCLUDES_YAML}') as f:
    cfg = yaml.safe_load(f) or {}

framework = '${FRAMEWORK}'
model = '${MODEL_NAME}'
for rule in cfg.get('rules', []):
    if rule.get('framework') == framework and model in (rule.get('models') or []):
        print(rule.get('exclude_nodes', ''))
        break
")
        PYTHON_EXCLUDE_RC=$?
        if [[ $PYTHON_EXCLUDE_RC -ne 0 ]]; then
            echo "Error: python3 failed (exit ${PYTHON_EXCLUDE_RC}) parsing ${NODE_EXCLUDES_YAML}" >&2
            echo "Error: fix the YAML, or set SLURM_EXCLUDE_NODES to bypass this lookup." >&2
            exit 1
        fi
    else
        # awk fallback matched to node_excludes.yaml's fixed rule/models/exclude_nodes
        # shape, for submit hosts without python3 or PyYAML.
        echo "Warning: python3/PyYAML unavailable on submit host; falling back to awk parsing of ${NODE_EXCLUDES_YAML}" >&2
        RESOLVED_EXCLUDE_NODES=$(awk -v fw="$FRAMEWORK" -v model="$MODEL_NAME" '
            /^  - framework:/ {
                line = $0
                sub(/^  - framework: */, "", line)
                fw_match = (line == fw)
                model_match = 0
                next
            }
            fw_match && /^      - / {
                m = $0
                sub(/^      - */, "", m)
                gsub(/^"|"$/, "", m)
                if (m == model) model_match = 1
                next
            }
            fw_match && model_match && /^    exclude_nodes:/ {
                val = $0
                sub(/^ *exclude_nodes: */, "", val)
                gsub(/^"|"$/, "", val)
                print val
                exit
            }
        ' "$NODE_EXCLUDES_YAML")
        AWK_EXCLUDE_RC=$?
        if [[ $AWK_EXCLUDE_RC -ne 0 ]]; then
            echo "Error: awk fallback failed (exit ${AWK_EXCLUDE_RC}) parsing ${NODE_EXCLUDES_YAML}" >&2
            echo "Error: fix the YAML/parser, or set SLURM_EXCLUDE_NODES to bypass this lookup." >&2
            exit 1
        fi
    fi
else
    RESOLVED_EXCLUDE_NODES=""
fi
if [[ -n "$RESOLVED_EXCLUDE_NODES" ]]; then
    EXCLUDE_OPT=(--exclude "$RESOLVED_EXCLUDE_NODES")
fi

# SLURM_REUSE_JOBID: run job.slurm directly in the current shell against the existing
# allocation. Inner srun calls pick it up via SLURM_JOB_ID; SLURM_OVERLAP=1 lets them
# share task slots with the interactive shell already holding the allocation.
if [[ -n "${SLURM_REUSE_JOBID:-}" ]]; then
    REUSE_JID="$SLURM_REUSE_JOBID"
    echo "Reusing existing Slurm allocation ${REUSE_JID} (skipping sbatch)" >&2

    ALLOC_NODELIST="${SLURM_JOB_NODELIST:-$(squeue -h -j "$REUSE_JID" -o '%N' 2>/dev/null)}"
    if [[ -z "$ALLOC_NODELIST" ]]; then
        echo "Error: could not resolve nodelist for job ${REUSE_JID}" >&2
        exit 1
    fi
    ALLOC_NNODES=$(scontrol show hostnames "$ALLOC_NODELIST" | wc -l)
    if [[ "$ALLOC_NNODES" -lt "$NUM_NODES" ]]; then
        echo "Error: allocation ${REUSE_JID} has ${ALLOC_NNODES} nodes, need ${NUM_NODES}" >&2
        exit 1
    fi

    export SLURM_JOB_ID="$REUSE_JID"
    export SLURM_JOBID="$REUSE_JID"
    export SLURM_JOB_NODELIST="$ALLOC_NODELIST"
    export SLURM_NODELIST="$ALLOC_NODELIST"
    export SLURM_NNODES="$ALLOC_NNODES"
    export SLURM_JOB_NUM_NODES="$ALLOC_NNODES"
    export SLURM_NTASKS="$ALLOC_NNODES"
    export SLURM_NPROCS="$ALLOC_NNODES"
    export SLURM_NTASKS_PER_NODE=1
    export SLURM_TASKS_PER_NODE="1(x${ALLOC_NNODES})"
    export SLURM_OVERLAP=1
    export SLURM_SUBMIT_DIR="$(pwd)"

    STDOUT_LOG="${BENCHMARK_LOGS_DIR}/slurm_job-${REUSE_JID}.out"
    STDERR_LOG="${BENCHMARK_LOGS_DIR}/slurm_job-${REUSE_JID}.err"
    rm -f "$STDOUT_LOG" "$STDERR_LOG"

    nohup bash "$(dirname "$0")/job.slurm" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
    INLINE_PID=$!
    echo "$INLINE_PID" > "${BENCHMARK_LOGS_DIR}/slurm_job-${REUSE_JID}.pid"
    echo "Started job.slurm (pid=${INLINE_PID}); logs: ${STDOUT_LOG}" >&2

    echo "$REUSE_JID"
    exit 0
fi

sbatch_cmd=(
    sbatch
    --parsable
    --exclusive
    -N "$NUM_NODES"
    -n "$NUM_NODES"
    "${NODELIST_OPT[@]}"
    "${EXCLUDE_OPT[@]}"
    --time "$TIME_LIMIT"
    --partition "$SLURM_PARTITION"
    --account "$SLURM_ACCOUNT"
    --job-name "$RUNNER_NAME"
    --output "${BENCHMARK_LOGS_DIR}/slurm_job-%j.out"
    --error "${BENCHMARK_LOGS_DIR}/slurm_job-%j.err"
    "$(dirname "$0")/job.slurm"
)

JOB_ID=$("${sbatch_cmd[@]}")
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to submit job with sbatch" >&2
    exit 1
fi
echo "$JOB_ID"
