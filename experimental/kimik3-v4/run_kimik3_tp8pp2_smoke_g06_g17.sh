#!/usr/bin/env bash
# Smoke launcher: Kimi-K3 TP×PP2 agentic on 2× MI355X (g06 + g17 hold job).
# Serving profile aligned with kimik3-fp4-mi355x-vllm-agentic-mtp (no MTP).
#
# Usage (on ln, with a 2-node hold covering g06 + g17):
#   SLURM_REUSE_JOBID=16758 bash experimental/kimik3-v4/run_kimik3_tp8pp2_smoke_g06_g17.sh
#
set -euo pipefail

REPO="${REPO:-$HOME/InferenceX}"
SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
NODE0="${NODE0:-mia1-p01-g06}"
NODE1="${NODE1:-mia1-p02-g17}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263}"
HF_CACHE="${HF_CACHE:-/it-share/hf_cache}"
MODEL_PATH="${MODEL_PATH:-${HF_CACHE}/Kimi-K3}"
TP="${TP:-8}"
PP="${PP:-2}"
GPU_DEVICES="${GPU_DEVICES:-}"
if [[ -z "$GPU_DEVICES" ]]; then
  if [[ "$TP" -eq 4 ]]; then
    GPU_DEVICES="0,1,2,3"
  else
    GPU_DEVICES="0,1,2,3,4,5,6,7"
  fi
fi
TOPO_TAG="tp${TP}pp${PP}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_${TOPO_TAG}_smoke_logs}"
TS="$(date +%Y%m%d_%H%M%S)"
CONC="${CONC:-4}"
SPEC_DECODE="${SPEC_DECODE:-false}"
case "${SPEC_DECODE}" in
true|TRUE|1|yes|YES|on|ON|mtp|dspark) SPEC_TAG="_dspark" ;;
*) SPEC_TAG="" ;;
esac
RESULT_HOST="${LOG_ROOT}/conc${CONC}${SPEC_TAG}_${TS}"
MAIN_LOG="${LOG_ROOT}/kimik3_${TOPO_TAG}_c${CONC}${SPEC_TAG}_${TS}.log"
CONT0="kimik3_${TOPO_TAG}_r0_${TS}"
CONT1="kimik3_${TOPO_TAG}_r1_${TS}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
DURATION="${DURATION:-360}"
# kimik3-fp4-mi355x-vllm-agentic-mtp: dram 0.50 on cluster:mi355x-amds → ~1500 GB/node
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1500}"
KV_OFFLOADING="${KV_OFFLOADING:-dram}"
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-vllm-simple}"
# process_agentic_result.py requires KV_OFFLOAD_BACKEND_METADATA (JSON with a
# `name` matching KV_OFFLOAD_BACKEND) whenever offloading is enabled; without it
# the post-profiling aggregation SystemExits and (under set -e) skips lm-eval.
if [[ "${KV_OFFLOADING}" == "none" ]]; then
  KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-}"
else
  KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"${KV_OFFLOAD_BACKEND}\"}}"
fi
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
AITER_GEMM_MERGE="${AITER_GEMM_MERGE:-auto}"
AITER_GEMM_EXTRA_CSV="${AITER_GEMM_EXTRA_CSV:-${REPO}/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.combined.csv}"
AITER_GEMM_EXTRA_BASENAME="$(basename "${AITER_GEMM_EXTRA_CSV}")"
IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
RANK1_RESULT_HOST="${RANK1_RESULT_HOST:-${LOG_ROOT}/rank1_${TS}}"
COREDUMP_DIR="${COREDUMP_DIR:-/coredumps}"
# ROCm debug tier (set DEBUG_ROCM=0 to disable):
#   minimal   — HSA coredump only (cheap; use for most reruns)
#   serialize — minimal + AMD_SERIALIZE_KERNEL=3 + HIP_LAUNCH_BLOCKING (no log spam;
#               keeps full cudagraph capture list — use to pin runtime segfaults)
#   capture   — minimal + narrow cudagraph to MAX_CUDAGRAPH_CAPTURE_SIZE (default 6)
#   fault     — capture tier + serialize (still slow at capture; shrinks capture list)
#   full      — everything on (AMD_LOG_LEVEL=3, VLLM DEBUG; very slow, huge logs)
DEBUG_ROCM="${DEBUG_ROCM:-minimal}"
# Used when DEBUG_ROCM=capture|fault; 092823 fault was at PIECEWISE 39/44 ≈ batch M=5
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
# Skip the small capture sizes whose aiter MXFP4 fused_moe variants fault on
# gfx950 (M=2 hit an illegal device access at PIECEWISE 42/44).
MIN_CUDAGRAPH_CAPTURE_SIZE="${MIN_CUDAGRAPH_CAPTURE_SIZE:-1}"

mkdir -p "$RESULT_HOST" "$LOG_ROOT"
[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }

echo "=== MI355X ${TOPO_TAG} smoke (mtp-aligned, no spec-decode) ===" | tee "$MAIN_LOG"
echo "hold=${SLURM_REUSE_JOBID} nodes=${NODE0}+${NODE1} tp=${TP} pp=${PP} gpus=${GPU_DEVICES}" | tee -a "$MAIN_LOG"
echo "conc=${CONC} dur=${DURATION}s run_eval=${RUN_EVAL:-false} spec_decode=${SPEC_DECODE:-false} kv=${KV_OFFLOADING}/${KV_OFFLOAD_BACKEND} dram_gb=${TOTAL_CPU_DRAM_GB} enforce_eager=${ENFORCE_EAGER} aiter_extra=${AITER_GEMM_EXTRA_CSV} debug_rocm=${DEBUG_ROCM} max_cudagraph=${MAX_CUDAGRAPH_CAPTURE_SIZE:-auto} min_cudagraph=${MIN_CUDAGRAPH_CAPTURE_SIZE}" | tee -a "$MAIN_LOG"
echo "patches: n6288=${AITER_N6288_CHUNK_PATCH:-1} ca_flush_sync=${AITER_CA_FLUSH_SYNC_PATCH:-1} disable_custom_ar=${DISABLE_CUSTOM_ALL_REDUCE:-0} async_sched=${ASYNC_SCHEDULING:-auto} draft=${DRAFT_MODEL_PATH:-auto}" | tee -a "$MAIN_LOG"
mkdir -p "$RANK1_RESULT_HOST"

if [[ "$DEBUG_ROCM" != "0" ]]; then
  echo "Ensuring GPU coredump dir ${COREDUMP_DIR} on ${NODE0} and ${NODE1}..." | tee -a "$MAIN_LOG"
  for item in "${NODE0}" "${NODE1}"; do
    srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$item" -N1 -n1 bash -lc \
      "mkdir -p ${COREDUMP_DIR} 2>/dev/null || sudo mkdir -p ${COREDUMP_DIR}; chmod 1777 ${COREDUMP_DIR} 2>/dev/null || sudo chmod 1777 ${COREDUMP_DIR}" || true
  done
fi

MASTER_IP="$(srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE0" -N1 -n1 hostname -I | awk '{print $1}')"
echo "MASTER_IP=${MASTER_IP}" | tee -a "$MAIN_LOG"

# Shared docker invocation builder (runs ON the compute node via srun bash -s).
launch_docker_rank() {
  local rank="$1" cont="$2" result_mount="${3:-}"
  local result_vol=()
  [[ -n "$result_mount" ]] && result_vol=(-v "$result_mount:/results")
  cat <<EOF
set -euo pipefail
if docker ps &>/dev/null 2>&1; then D=docker; else D="sudo docker"; fi
# Avoid pipefail+SIGPIPE from awk exiting early while ip still writes.
set +o pipefail
HOST_IP=\$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print \$7; exit}')
HOST_IP="\${HOST_IP:-\$(hostname -I | awk '{print \$1}')}"
NET_IF=\$(ip route 2>/dev/null | awk '/^default/ {print \$5; exit}')
set -o pipefail
\$D rm -f ${cont} 2>/dev/null || true
# Mute pull progress: writing docker bars into an srun|tee pipe often yields SIGPIPE (141).
\$D pull ${IMAGE} >/dev/null 2>&1 || true
\$D run \$([ "${rank}" = "0" ] && echo --rm --init || echo -d) --name ${cont} \\
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \\
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=-1 \\
  --network host --ipc host --group-add video \\
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \\
  --add-host "\$(hostname):\${HOST_IP}" \\
  -v /sys:/sys --shm-size 128G \\
  -v ${REPO}:/workspace \\
  -v ${HF_CACHE}:/hf_cache \\
  -v ${MODEL_PATH}:/model:ro \\
  ${result_vol[@]+"${result_vol[@]}"} \\
  -v /tmp:/tmp \\
  \$([ "${DEBUG_ROCM}" != "0" ] && echo -v ${COREDUMP_DIR}:${COREDUMP_DIR}) \\
  -e GLOO_SOCKET_IFNAME="\${NET_IF}" \\
  -e NCCL_SOCKET_IFNAME="\${NET_IF}" \\
  -e NCCL_IB_HCA=${IBDEVICES} \\
  -e ROCR_VISIBLE_DEVICES=${GPU_DEVICES} \\
  -e HIP_VISIBLE_DEVICES=${GPU_DEVICES} \\
  -e KV_OFFLOAD_BACKEND_METADATA='${KV_OFFLOAD_BACKEND_METADATA}' \\
  --entrypoint bash ${IMAGE} -lc '
    export GITHUB_WORKSPACE=/workspace INFMAX_CONTAINER_WORKSPACE=/workspace
    export PYTHONPATH=/workspace/experimental/kimik3-v4/aiter/aiter_site:\${PYTHONPATH:-}
    export AITER_N6288_CHUNK_PATCH=${AITER_N6288_CHUNK_PATCH:-1}
    export AITER_CA_FLUSH_SYNC_PATCH=${AITER_CA_FLUSH_SYNC_PATCH:-1}
    export DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-0}
    export ASYNC_SCHEDULING=${ASYNC_SCHEDULING:-auto}
    export MODEL=moonshotai/Kimi-K3 MODEL_PATH=/model MODEL_PREFIX=kimik3
    export TP=${TP} PP=${PP} PP_SIZE=${PP} CONC=${CONC}
    # process_agentic_result.py reads PP_SIZE (not PP) for per-GPU denominator.
    # Keep IS_MULTINODE unset/false: aggregated TP×PP co-located path uses tp*pp.
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-20}
    export REJECTION_SAMPLE_METHOD=${REJECTION_SAMPLE_METHOD:-}
    export KV_OFFLOADING=${KV_OFFLOADING} KV_OFFLOAD_BACKEND=${KV_OFFLOAD_BACKEND}
    export TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB}
    export DURATION=${DURATION}
    export AIPERF_UNSAFE_OVERRIDE=true IS_AGENTIC=1 EVAL_ONLY=false
    export AIPERF_EXPERIMENTAL_FAST=${AIPERF_EXPERIMENTAL_FAST:-0}
    export RUN_EVAL=${RUN_EVAL:-false} EVAL_FRAMEWORK=${EVAL_FRAMEWORK:-lm-eval}
    export SPEC_DECODE=${SPEC_DECODE:-false}
    export SPEC_NUM_TOKENS=${SPEC_NUM_TOKENS:-2}
    export DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/hf_cache/Kimi-K3-DSpark}
    export SYNTHETIC_ACCEPT_LEN=${SYNTHETIC_ACCEPT_LEN:-2.51}
    export SCENARIO_TYPE=agentic-coding SCENARIO_SUBDIR=agentic/
    export FRAMEWORK=vllm PRECISION=fp4
    # Matrix label: mtp when DSpark on (matches kimik3-fp4-mi355x-vllm-agentic-mtp).
    if [[ "${SPEC_DECODE}" == "true" || "${SPEC_DECODE}" == "1" || "${SPEC_DECODE}" == "mtp" || "${SPEC_DECODE}" == "dspark" ]]; then
      export SPEC_DECODING=mtp
    else
      export SPEC_DECODING=none
    fi
    export RESULT_DIR=/results RESULT_FILENAME=kimik3_${TOPO_TAG}_smoke_c${CONC}${SPEC_TAG:-}
    export HF_HOME=/hf_cache HUGGINGFACE_HUB_CACHE=/hf_cache
    export NODE_RANK=${rank} NNODES=2 MASTER_ADDR=${MASTER_IP} MASTER_PORT=${MASTER_PORT} PORT=${PORT}
    export ENFORCE_EAGER=${ENFORCE_EAGER}
    export MIN_CUDAGRAPH_CAPTURE_SIZE=${MIN_CUDAGRAPH_CAPTURE_SIZE:-1}
    export AITER_GEMM_MERGE=${AITER_GEMM_MERGE}
    export AITER_GEMM_EXTRA_CSV=/workspace/experimental/kimik3-v4/aiter/${AITER_GEMM_EXTRA_BASENAME}
    export AITER_CONFIG_GEMM_BF16=/tmp/aiter_configs/bf16_tuned_gemm.csv
    export PYTHONNOUSERSITE=1
    # Host already points core_pattern at ${COREDUMP_DIR}; force unlimited
    # core size inside the container so PP worker SIGSEGV leaves a usable dump.
    ulimit -c unlimited 2>/dev/null || true
    case "${DEBUG_ROCM}" in
    0) ;;
    minimal)
      export HSA_ENABLE_DEBUG=1
      ;;
    serialize)
      # Runtime-fault repro without AMD_LOG_LEVEL spam and without shrinking
      # the cudagraph capture list (unlike capture|fault).
      export HSA_ENABLE_DEBUG=1
      export AMD_SERIALIZE_KERNEL=3
      export HIP_LAUNCH_BLOCKING=1
      export GPU_MAX_HW_QUEUES=1
      export PYTHONFAULTHANDLER=1
      ;;
    capture|fault)
      export HSA_ENABLE_DEBUG=1
      export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-6}
      if [[ "${DEBUG_ROCM}" == "fault" ]]; then
        export AMD_SERIALIZE_KERNEL=3
        export HIP_LAUNCH_BLOCKING=1
        export GPU_MAX_HW_QUEUES=1
      fi
      ;;
    full|1|true)
      export HSA_ENABLE_DEBUG=1
      export AMD_LOG_LEVEL=3
      export AMD_SERIALIZE_KERNEL=3
      export HIP_LAUNCH_BLOCKING=1
      export GPU_MAX_HW_QUEUES=1
      export VLLM_LOGGING_LEVEL=DEBUG
      export PYTHONFAULTHANDLER=1
      ;;
    *)
      echo "Unknown DEBUG_ROCM=${DEBUG_ROCM} (use 0|minimal|serialize|capture|fault|full)" >&2
      exit 1
      ;;
    esac
    python -m pip install -q --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ triton==3.7.0 tabulate || true
    cd /workspace
    bash benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm_tp8pp2.sh
  '$([[ "${rank}" != "0" ]] && echo " >/dev/null")
$([[ "${rank}" != "0" ]] && echo "\$D inspect -f '{{.State.Running}}' ${cont} | grep -q true")
$([[ "${rank}" != "0" ]] && echo "echo rank-1 container ${cont} up")
EOF
}

cleanup() {
  set +e
  for item in "${NODE1}:${CONT1}" "${NODE0}:${CONT0}"; do
    node="${item%%:*}"
    cont="${item#*:}"
    srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$node" -N1 -n1 bash -lc \
      "docker rm -f $cont 2>/dev/null || sudo docker rm -f $cont 2>/dev/null" || true
  done
}
trap cleanup EXIT INT TERM

# Drop stale containers from prior smoke runs.
for item in "${NODE1}" "${NODE0}"; do
  srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$item" -N1 -n1 bash -lc \
    'docker ps -a --format "{{.Names}}" 2>/dev/null | grep -E "^kimik3_tp[48]pp2_r[01]_" | xargs -r docker rm -f 2>/dev/null; \
     docker ps -a --format "{{.Names}}" 2>/dev/null | grep -E "^kimik3_tp[48]pp2_r[01]_" | xargs -r sudo docker rm -f 2>/dev/null' \
    || true
done

# Write rank scripts to NFS home so srun does not re-expand $D via nested
# unquoted heredocs (that previously turned "docker run" into bare "run").
RANK1_SCRIPT="${LOG_ROOT}/rank1_launch_${TS}.sh"
RANK0_SCRIPT="${LOG_ROOT}/rank0_launch_${TS}.sh"
launch_docker_rank 1 "$CONT1" "$RANK1_RESULT_HOST" >"$RANK1_SCRIPT"
launch_docker_rank 0 "$CONT0" "$RESULT_HOST" >"$RANK0_SCRIPT"
chmod +x "$RANK1_SCRIPT" "$RANK0_SCRIPT"

echo "rank-1 headless on ${NODE1}..." | tee -a "$MAIN_LOG"
# Do NOT pipe srun→tee for rank-1: docker -d printing the container id into a
# closing pipe yields SIGPIPE (141) even when the container started fine.
RANK1_SRUN_LOG="${LOG_ROOT}/rank1_srun_${TS}.log"
set +e
srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE1" -N1 -n1 \
  bash "$RANK1_SCRIPT" >"$RANK1_SRUN_LOG" 2>&1
R1_RC=$?
set -e
tee -a "$MAIN_LOG" <"$RANK1_SRUN_LOG" >/dev/null || true
if [[ "$R1_RC" -ne 0 ]]; then
  if srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE1" -N1 -n1 \
       bash -lc "docker ps --format '{{.Names}}' | grep -qx '${CONT1}'"; then
    echo "WARN: rank-1 srun rc=${R1_RC} but container ${CONT1} is up; continuing" | tee -a "$MAIN_LOG"
  else
    echo "ERROR: rank-1 launch failed rc=${R1_RC}" | tee -a "$MAIN_LOG" >&2
    exit "$R1_RC"
  fi
fi
sleep 5

echo "rank-0 serve + agentic on ${NODE0}..." | tee -a "$MAIN_LOG"
set +e
srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE0" -N1 -n1 bash "$RANK0_SCRIPT" 2>&1 | tee -a "$MAIN_LOG"
RC=${PIPESTATUS[0]}
set -e
echo "Done rc=${RC}. Log=${MAIN_LOG} artifacts=${RESULT_HOST}" | tee -a "$MAIN_LOG"
exit "$RC"
