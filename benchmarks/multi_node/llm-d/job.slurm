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

# Repo root (benchmarks/multi_node/llm-d/job.slurm -> ../../..)
DI_REPO_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
export DI_REPO_DIR

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

SANITIZED_USER=$(echo "${USER:-runner}" | tr -c 'a-zA-Z0-9_.-' '_')
DOCKER_CONT_NAME="llmd_bench_${SANITIZED_USER}_${SLURM_JOB_ID}"
export DOCKER_CONT_NAME
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$(pwd)/benchmark_logs}"
DOCKER_MOUNT_PATH="/workspace"

cleanup() {
    echo "[${SLURM_JOB_ID}] cleanup on $(hostname)"
}
trap cleanup INT TERM HUP EXIT

# One docker run per node, one task per node. server.sh dispatches by NODE_RANK.
srun \
    --kill-on-bad-exit=1 \
    --signal=TERM@30 \
    --unbuffered \
    bash -lc "
set -euo pipefail
echo \"Rank \$SLURM_PROCID on \$(hostname)\"

sudo docker ps -aq --filter name=\"^${DOCKER_CONT_NAME}_\" | xargs -r sudo docker rm -f || true

exec sudo docker run --rm \
    --init \
    --stop-timeout 10 \
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
    \"\$DOCKER_IMAGE_NAME\" bash -lc '
        set -o pipefail
        ${DOCKER_MOUNT_PATH}/benchmarks/multi_node/llm-d/server.sh \
            2>&1 | tee /benchmark_logs/slurm_job-'\"\$SLURM_JOB_ID\"'_rank_'\"\$SLURM_PROCID\"'.log
    '
"

srun bash -c "sudo docker ps -aq --filter name=\"^${DOCKER_CONT_NAME}_\" | xargs -r sudo docker rm -f" || true
