#!/bin/bash
#SBATCH --job-name=llm-d-bench
#SBATCH --ntasks-per-node=1
# --output, --error, -N, -n, --time set by submit.sh
#
# Allocates PREFILL_NODES + DECODE_NODES nodes, derives per-node IPs, then
# srun-runs server.sh inside one Docker container per node. NODE_RANK
# (= SLURM_PROCID) drives role selection inside server.sh.

set -euo pipefail

echo "=== llm-d job start ==="
echo "UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"

# Repo root. $(pwd) = the sbatch submit dir, which the wrapper sets to
# benchmarks/multi_node/llm-d/ before invoking submit.sh, so 3 up =
# repo root. Using $(dirname "$0") would resolve to a SLURM staging
# copy under /var/spool/... and miss the checkout entirely.
DI_REPO_DIR=$(cd "$(pwd)/../../.." && pwd)
export DI_REPO_DIR
echo "REPO DIR: ${DI_REPO_DIR}"

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)
echo "Allocated nodes ($TOTAL_NODES): $(echo "$ALL_NODES" | tr '\n' ' ')"

if [[ "$TOTAL_NODES" -ne "$NUM_NODES" ]]; then
    echo "Error: SLURM allocated $TOTAL_NODES nodes, expected $NUM_NODES" >&2
    exit 1
fi

# Per-node IPs in rank order.
IPS=()
for NODE in $ALL_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
         bash -c 'ip route get 1.1.1.1 | awk "/src/ {print \$7}"')
    IPS+=("$IP")
done
echo "Node IPs: ${IPS[*]}"

if ! srun --nodes="$NUM_NODES" --ntasks-per-node=1 test -d "$MODEL_DIR"; then
    echo "FATAL: model not found on all $NUM_NODES nodes: $MODEL_DIR" >&2
    exit 1
fi

# Rank slicing:
#   prefill leader = rank 0
#   prefill workers = ranks 1 .. PREFILL_NODES-1
#   decode  leader = rank PREFILL_NODES (also coordinator: EPP + Envoy + bench)
#   decode  workers = ranks PREFILL_NODES+1 .. NUM_NODES-1
PREFILL_LEADER_IP="${IPS[0]}"
DECODE_LEADER_IP="${IPS[$PREFILL_NODES]}"

# DP leader addresses for vLLM --data-parallel-address (rank 0 of each instance).
PREFILL_DP_ADDR="$PREFILL_LEADER_IP"
DECODE_DP_ADDR="$DECODE_LEADER_IP"

ALL_IP_LIST=$(IFS=,; echo "${IPS[*]}")

SANITIZED_USER=$(echo "$USER" | tr -c 'a-zA-Z0-9_.-' '_')
DOCKER_CONT_NAME="llmd_bench_${SANITIZED_USER}_${SLURM_JOB_ID}"
export DOCKER_CONT_NAME
: "${BENCHMARK_LOGS_DIR:?BENCHMARK_LOGS_DIR not set}"
DOCKER_MOUNT_PATH="/workspace"

cleanup() {
    echo "[${SLURM_JOB_ID}] cleanup on $(hostname)"
    [[ -n "${WATCHER_PID:-}" ]] && kill "$WATCHER_PID" 2>/dev/null || true
}
trap cleanup INT TERM HUP EXIT

# Coordinator-done watcher. server.sh on the decode coordinator writes
# this marker after the bench finishes; we then scancel the allocation
# from outside the container (the image has no SLURM client tools).
# Without this, workers `wait` on local vLLM forever and the job runs
# to TIME_LIMIT.
BENCH_DONE_MARKER="$BENCHMARK_LOGS_DIR/.bench_done.$SLURM_JOB_ID"
rm -f "$BENCH_DONE_MARKER"
(
    while [[ ! -f "$BENCH_DONE_MARKER" ]]; do sleep 5; done
    echo "[${SLURM_JOB_ID}] coordinator finished; scancel'ing job"
    scancel "$SLURM_JOB_ID" 2>/dev/null || true
) &
WATCHER_PID=$!

# Container engine: 'docker' (default) for clusters where the SLURM
# user can talk to /var/run/docker.sock (e.g. h200-dgxc-slurm); 'pyxis'
# for clusters that require enroot+pyxis srun (e.g. gb200-nv, where the
# SLURM user is not in the docker group). Same env contract on both
# paths; only the per-node container launch differs.
LLMD_CONTAINER_ENGINE="${LLMD_CONTAINER_ENGINE:-docker}"
echo "LLMD_CONTAINER_ENGINE=$LLMD_CONTAINER_ENGINE"

if [[ "$LLMD_CONTAINER_ENGINE" == "docker" ]]; then
    # One docker run per node, one task per node. server.sh dispatches by NODE_RANK.
    srun \
        --kill-on-bad-exit=1 \
        --signal=TERM@30 \
        --unbuffered \
        bash -lc "
set -euo pipefail
echo \"Rank \$SLURM_PROCID on \$(hostname)\"

docker ps -aq --filter name=\"^${DOCKER_CONT_NAME}_\" | xargs -r docker rm -f || true

exec docker run --rm \
    --init \
    --entrypoint bash \
    --stop-timeout 10 \
    --user 0:0 \
    --network host \
    --ipc host \
    --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size 32G \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --cap-add SYS_RAWIO \
    --device /dev/infiniband \
    --security-opt seccomp=unconfined \
    --privileged \
    -v ${MODEL_DIR}:/models:ro \
    -v ${BENCHMARK_LOGS_DIR}:/benchmark_logs \
    -v ${DI_REPO_DIR}:${DOCKER_MOUNT_PATH} \
    -v ${DI_REPO_DIR}/benchmarks/multi_node/llm-d-recipes:/etc/llmd-recipes:ro \
    -v ${DI_REPO_DIR}/benchmarks/llm-d/epp-config.yaml:/etc/epp/config.yaml:ro \
    -v ${DI_REPO_DIR}/benchmarks/llm-d/envoy.yaml:/etc/envoy/envoy.yaml:ro \
    -e SLURM_JOB_ID=\$SLURM_JOB_ID \
    -e NODE_RANK=\$SLURM_PROCID \
    -e NUM_NODES=$NUM_NODES \
    -e PREFILL_NODES=$PREFILL_NODES \
    -e DECODE_NODES=$DECODE_NODES \
    -e PREFILL_WORKERS=$PREFILL_WORKERS \
    -e DECODE_WORKERS=$DECODE_WORKERS \
    -e ALL_IPS=$ALL_IP_LIST \
    -e PREFILL_LEADER_IP=$PREFILL_LEADER_IP \
    -e DECODE_LEADER_IP=$DECODE_LEADER_IP \
    -e PREFILL_DP_ADDR=$PREFILL_DP_ADDR \
    -e DECODE_DP_ADDR=$DECODE_DP_ADDR \
    -e MODEL_DIR=/models \
    -e MODEL_NAME=$MODEL_NAME \
    -e GPUS_PER_NODE=$GPUS_PER_NODE \
    -e PREFILL_DP_SIZE=$PREFILL_DP_SIZE \
    -e DECODE_DP_SIZE=$DECODE_DP_SIZE \
    -e BENCH_INPUT_LEN=$BENCH_INPUT_LEN \
    -e BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN \
    -e BENCH_MAX_CONCURRENCY=$BENCH_MAX_CONCURRENCY \
    -e BENCH_REQUEST_RATE=$BENCH_REQUEST_RATE \
    -e BENCH_RANDOM_RANGE_RATIO=$BENCH_RANDOM_RANGE_RATIO \
    -e BENCH_NUM_PROMPTS_MULTIPLIER=$BENCH_NUM_PROMPTS_MULTIPLIER \
    -e BENCHMARK_LOGS_DIR=/benchmark_logs \
    -e RUN_EVAL=$RUN_EVAL \
    -e EVAL_ONLY=$EVAL_ONLY \
    -e EVAL_CONC=$EVAL_CONC \
    -e FRAMEWORK=$FRAMEWORK \
    -e PRECISION=$PRECISION \
    -e MODEL_PREFIX=$MODEL_PREFIX \
    -e RUNNER_TYPE=$RUNNER_TYPE \
    -e RESULT_FILENAME=$RESULT_FILENAME \
    -e SPEC_DECODING=$SPEC_DECODING \
    -e IS_MULTINODE=$IS_MULTINODE \
    -e CONFIG_FILE=$CONFIG_FILE \
    --name \"${DOCKER_CONT_NAME}_\$SLURM_PROCID\" \
    \"\$DOCKER_IMAGE_NAME\" -lc '
        set -o pipefail
        ${DOCKER_MOUNT_PATH}/benchmarks/multi_node/llm-d/server.sh \
            2>&1 | tee /benchmark_logs/slurm_job-'\"\$SLURM_JOB_ID\"'_rank_'\"\$SLURM_PROCID\"'.log
    '
"

    srun bash -c "docker ps -aq --filter name=\"^${DOCKER_CONT_NAME}_\" | xargs -r docker rm -f" || true

elif [[ "$LLMD_CONTAINER_ENGINE" == "pyxis" ]]; then
    : "${LLMD_SQUASH_FILE:?LLMD_SQUASH_FILE must be set when LLMD_CONTAINER_ENGINE=pyxis}"
    if [[ ! -s "$LLMD_SQUASH_FILE" ]]; then
        echo "Error: LLMD_SQUASH_FILE does not exist or is empty: $LLMD_SQUASH_FILE" >&2
        exit 1
    fi

    # Pre-export non-SLURM static vars so pyxis --container-env=
    # whitelisting can pick them up. Vars that already come from submit.sh
    # via export are no-ops here; we re-state them for clarity. ALL_IPS
    # is computed locally above as $ALL_IP_LIST.
    export ALL_IPS="$ALL_IP_LIST"
    export NUM_NODES PREFILL_NODES DECODE_NODES PREFILL_LEADER_IP DECODE_LEADER_IP
    export PREFILL_WORKERS DECODE_WORKERS
    export PREFILL_DP_ADDR DECODE_DP_ADDR MODEL_NAME GPUS_PER_NODE
    export PREFILL_DP_SIZE DECODE_DP_SIZE
    export BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_MAX_CONCURRENCY
    export BENCH_REQUEST_RATE BENCH_RANDOM_RANGE_RATIO BENCH_NUM_PROMPTS_MULTIPLIER
    export RUN_EVAL EVAL_ONLY EVAL_CONC FRAMEWORK PRECISION MODEL_PREFIX
    export RUNNER_TYPE RESULT_FILENAME SPEC_DECODING IS_MULTINODE CONFIG_FILE

    PYXIS_ENV_LIST="NUM_NODES,PREFILL_NODES,DECODE_NODES,ALL_IPS,PREFILL_LEADER_IP,DECODE_LEADER_IP"
    PYXIS_ENV_LIST+=",PREFILL_WORKERS,DECODE_WORKERS"
    PYXIS_ENV_LIST+=",PREFILL_DP_ADDR,DECODE_DP_ADDR,MODEL_NAME,GPUS_PER_NODE"
    PYXIS_ENV_LIST+=",PREFILL_DP_SIZE,DECODE_DP_SIZE"
    PYXIS_ENV_LIST+=",BENCH_INPUT_LEN,BENCH_OUTPUT_LEN,BENCH_MAX_CONCURRENCY"
    PYXIS_ENV_LIST+=",BENCH_REQUEST_RATE,BENCH_RANDOM_RANGE_RATIO,BENCH_NUM_PROMPTS_MULTIPLIER"
    PYXIS_ENV_LIST+=",RUN_EVAL,EVAL_ONLY,EVAL_CONC,FRAMEWORK,PRECISION,MODEL_PREFIX"
    PYXIS_ENV_LIST+=",RUNNER_TYPE,RESULT_FILENAME,SPEC_DECODING,IS_MULTINODE,CONFIG_FILE"

    PYXIS_MOUNTS="${MODEL_DIR}:/models:ro"
    PYXIS_MOUNTS+=",${BENCHMARK_LOGS_DIR}:/benchmark_logs"
    PYXIS_MOUNTS+=",${DI_REPO_DIR}:${DOCKER_MOUNT_PATH}"
    PYXIS_MOUNTS+=",${DI_REPO_DIR}/benchmarks/multi_node/llm-d-recipes:/etc/llmd-recipes:ro"
    PYXIS_MOUNTS+=",${DI_REPO_DIR}/benchmarks/llm-d/epp-config.yaml:/etc/epp/config.yaml:ro"
    PYXIS_MOUNTS+=",${DI_REPO_DIR}/benchmarks/llm-d/envoy.yaml:/etc/envoy/envoy.yaml:ro"

    # Optional: mount the epp / pd-sidecar / envoy binaries from a shared
    # filesystem instead of relying on them being baked into the image.
    # This lets a STOCK vllm/vllm-openai image be used directly (no
    # combined-image rebuild per vLLM version bump) - see
    # benchmarks/llm-d/binaries.env + extract-binaries.sh. Each mount is
    # gated on the file existing, so this is a no-op when the binaries
    # have not been extracted (the baked-image path keeps working), and
    # harmless when they have (mounting a binary over the identical one).
    # shellcheck source=/dev/null
    [[ -f "${DI_REPO_DIR}/benchmarks/llm-d/binaries.env" ]] && \
        source "${DI_REPO_DIR}/benchmarks/llm-d/binaries.env"
    for _bin in epp pd-sidecar envoy; do
        if [[ -n "${LLMD_BIN_DIR:-}" && -x "${LLMD_BIN_DIR}/${_bin}" ]]; then
            PYXIS_MOUNTS+=",${LLMD_BIN_DIR}/${_bin}:/usr/local/bin/${_bin}:ro"
            echo "Mounting ${LLMD_BIN_DIR}/${_bin} -> /usr/local/bin/${_bin}"
        fi
    done

    # MODEL_DIR / BENCHMARK_LOGS_DIR / NODE_RANK are translated to their
    # in-container values inside bash -lc (host MODEL_DIR is the source
    # path of the bind mount, but server.sh expects /models inside).
    srun \
        --kill-on-bad-exit=1 \
        --signal=TERM@30 \
        --unbuffered \
        --container-image="$LLMD_SQUASH_FILE" \
        --container-name="${DOCKER_CONT_NAME}" \
        --container-mounts="$PYXIS_MOUNTS" \
        --container-remap-root \
        --container-writable \
        --container-env="$PYXIS_ENV_LIST" \
        bash -lc '
set -o pipefail
echo "Rank $SLURM_PROCID on $(hostname)"
export NODE_RANK="$SLURM_PROCID"
export MODEL_DIR=/models
export BENCHMARK_LOGS_DIR=/benchmark_logs
'"$DOCKER_MOUNT_PATH"'/benchmarks/multi_node/llm-d/server.sh \
    2>&1 | tee /benchmark_logs/slurm_job-${SLURM_JOB_ID}_rank_${SLURM_PROCID}.log
'

else
    echo "Unsupported LLMD_CONTAINER_ENGINE: $LLMD_CONTAINER_ENGINE (expected docker|pyxis)" >&2
    exit 1
fi
