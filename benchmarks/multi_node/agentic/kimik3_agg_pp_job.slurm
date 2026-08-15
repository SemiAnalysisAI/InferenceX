#!/bin/bash
#SBATCH --job-name=kimik3-agg-pp
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#
# Aggregated vLLM PP job for MI355X (TP×PP across nodes). Rank 0 runs the
# OpenAI frontend + agentic replay; remaining ranks run headless PP workers.
# Invoked via kimik3_fp4_mi355x_vllm.sh (sbatch) from launch_mi355x-amds.sh.

set -euo pipefail

echo "=== Job Start (agg vLLM PP) ==="
echo "UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} NODES=${SLURM_JOB_NODELIST:-}"
echo "TP=${TP:-?} PP=${PP:-?} IMAGE=${DOCKER_IMAGE_NAME:-${CONTAINER_IMAGE:-}}"
echo "==============================="

: "${DOCKER_IMAGE_NAME:=${CONTAINER_IMAGE:?CONTAINER_IMAGE/DOCKER_IMAGE_NAME required}}"
: "${TP:?TP required}"
: "${PP:?PP required}"
: "${MODEL:?MODEL required}"
: "${MODEL_PREFIX:?MODEL_PREFIX required}"
: "${CONC_LIST:?CONC_LIST required}"
: "${RESULT_FILENAME:?RESULT_FILENAME required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE required}"
: "${BENCHMARK_LOGS_DIR:?BENCHMARK_LOGS_DIR required}"

NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:?}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/it-share/hf_cache/Kimi-K3}"
IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
SERVE_SCRIPT="benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm_tp8pp2.sh"
CONT_PREFIX="kimik3_agg_pp_${SLURM_JOB_ID}"

JOB_LOGS_DIR="${BENCHMARK_LOGS_DIR}/logs/slurm_job-${SLURM_JOB_ID}"
mkdir -p "$JOB_LOGS_DIR/agentic"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -lt "$NUM_NODES" ]]; then
    echo "Error: allocation has ${#NODES[@]} nodes, need ${NUM_NODES}" >&2
    exit 1
fi
NODES=("${NODES[@]:0:$NUM_NODES}")

MASTER_ADDR="$(srun -w "${NODES[0]}" -N1 -n1 hostname -I | awk '{print $1}')"
echo "MASTER_ADDR=${MASTER_ADDR} nodes=${NODES[*]}"

DOCKER_CMD_DETECT='if docker ps &>/dev/null; then DOCKER_CMD=docker; else DOCKER_CMD="sudo docker"; fi'

cleanup_containers() {
    set +e
    for node in "${NODES[@]}"; do
        srun --overlap -w "$node" -N1 -n1 bash -lc "
            ${DOCKER_CMD_DETECT}
            \$DOCKER_CMD ps -a --format '{{.Names}}' | grep -E '^${CONT_PREFIX}_r' | xargs -r \$DOCKER_CMD rm -f
        " || true
    done
}
trap cleanup_containers EXIT

# Pull once per node (mute progress to avoid SIGPIPE into srun pipes).
for node in "${NODES[@]}"; do
    srun --overlap -w "$node" -N1 -n1 bash -lc "
        ${DOCKER_CMD_DETECT}
        \$DOCKER_CMD pull '${DOCKER_IMAGE_NAME}' >/dev/null 2>&1 || true
    " || true
done

launch_rank() {
    local rank="$1"
    local node="$2"
    local cont="${CONT_PREFIX}_r${rank}"
    local result_host="${JOB_LOGS_DIR}/rank${rank}"
    mkdir -p "$result_host"

    # Rank 0 stays attached (agentic + exit status); workers are detached.
    local run_mode="--rm --init"
    if [[ "$rank" -ne 0 ]]; then
        run_mode="-d --rm"
    fi

    # shellcheck disable=SC2086
    srun --overlap -w "$node" -N1 -n1 --kill-on-bad-exit=1 bash -lc "
set -euo pipefail
${DOCKER_CMD_DETECT}
\$DOCKER_CMD rm -f '${cont}' 2>/dev/null || true
\$DOCKER_CMD run ${run_mode} --name '${cont}' \\
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  --network host --ipc host --group-add video \\
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \\
  -v '${GITHUB_WORKSPACE}:/workspace' \\
  -v '${HOST_MODEL_PATH}:/model:ro' \\
  -v '${result_host}:/results' \\
  -v '${JOB_LOGS_DIR}/agentic:/logs/agentic' \\
  -e GITHUB_WORKSPACE=/workspace \\
  -e INFMAX_CONTAINER_WORKSPACE=/workspace \\
  -e MODEL='${MODEL}' \\
  -e MODEL_PREFIX='${MODEL_PREFIX}' \\
  -e MODEL_PATH=/model \\
  -e NODE_RANK=${rank} \\
  -e NNODES=${NUM_NODES} \\
  -e MASTER_ADDR='${MASTER_ADDR}' \\
  -e MASTER_PORT='${MASTER_PORT}' \\
  -e TP='${TP}' \\
  -e PP='${PP}' \\
  -e PORT='${PORT}' \\
  -e CONC='${CONC:-}' \\
  -e CONC_LIST='${CONC_LIST}' \\
  -e KV_OFFLOADING='${KV_OFFLOADING:-dram}' \\
  -e KV_OFFLOAD_BACKEND='${KV_OFFLOAD_BACKEND:-vllm-simple}' \\
  -e TOTAL_CPU_DRAM_GB='${TOTAL_CPU_DRAM_GB}' \\
  -e DURATION='${DURATION:-3600}' \\
  -e RESULT_FILENAME='${RESULT_FILENAME}' \\
  -e RESULT_DIR=/results \\
  -e AGENTIC_OUTPUT_DIR=/workspace \\
  -e IS_AGENTIC=1 \\
  -e IS_MULTINODE=true \\
  -e DISAGG=false \\
  -e SCENARIO_TYPE=agentic-coding \\
  -e FRAMEWORK='${FRAMEWORK:-vllm}' \\
  -e PRECISION='${PRECISION:-fp4}' \\
  -e SPEC_DECODING='${SPEC_DECODING:-none}' \\
  -e PREFILL_TP='${TP}' \\
  -e PREFILL_PP_SIZE='${PP}' \\
  -e PREFILL_NUM_WORKERS='${PREFILL_NUM_WORKERS:-1}' \\
  -e DECODE_TP='${DECODE_TP:-$TP}' \\
  -e DECODE_PP_SIZE='${DECODE_PP_SIZE:-$PP}' \\
  -e DECODE_NUM_WORKERS='${DECODE_NUM_WORKERS:-0}' \\
  -e EVAL_ONLY='${EVAL_ONLY:-false}' \\
  -e RUN_EVAL='${RUN_EVAL:-false}' \\
  -e DISABLE_CUSTOM_ALL_REDUCE='${DISABLE_CUSTOM_ALL_REDUCE:-1}' \\
  -e ASYNC_SCHEDULING='${ASYNC_SCHEDULING:-auto}' \\
  -e AITER_N6288_CHUNK_PATCH='${AITER_N6288_CHUNK_PATCH:-1}' \\
  -e AITER_CA_FLUSH_SYNC_PATCH='${AITER_CA_FLUSH_SYNC_PATCH:-1}' \\
  -e AITER_GEMM_EXTRA_CSV='${AITER_GEMM_EXTRA_CSV:-/workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.combined.csv}' \\
  -e IBDEVICES='${IBDEVICES}' \\
  -e AIPERF_EXPERIMENTAL_FAST='${AIPERF_EXPERIMENTAL_FAST:-0}' \\
  -w /workspace \\
  '${DOCKER_IMAGE_NAME}' \\
  bash '${SERVE_SCRIPT}'
"
}

# Start PP workers first (headless), then rank 0 (blocks until agentic finishes).
for ((rank = NUM_NODES - 1; rank >= 1; rank--)); do
    echo "Launching rank-${rank} headless on ${NODES[$rank]}..."
    launch_rank "$rank" "${NODES[$rank]}"
    sleep 5
done

echo "Launching rank-0 serve + agentic on ${NODES[0]}..."
launch_rank 0 "${NODES[0]}"
RC=$?

# Stage agentic aiperf trees for launch_mi355x-amds.sh (expects conc_* under agentic/).
if [[ -d "${JOB_LOGS_DIR}/rank0" ]]; then
    for d in "${JOB_LOGS_DIR}/rank0"/conc_*; do
        [[ -d "$d" ]] || continue
        base="$(basename "$d")"
        mkdir -p "${JOB_LOGS_DIR}/agentic/${base}"
        cp -a "$d"/. "${JOB_LOGS_DIR}/agentic/${base}/" 2>/dev/null || true
    done
    # Also accept RESULT_DIR nested conc dirs from the serve script.
    if [[ -d "${JOB_LOGS_DIR}/rank0/aiperf_artifacts" ]]; then
        conc="${CONC:-${CONC_LIST%% *}}"
        mkdir -p "${JOB_LOGS_DIR}/agentic/conc_${conc}"
        cp -a "${JOB_LOGS_DIR}/rank0"/. "${JOB_LOGS_DIR}/agentic/conc_${conc}/" 2>/dev/null || true
    fi
fi

echo "=== Job End rc=${RC} ==="
exit "$RC"
