#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# 1P+2D SGLang PD disaggregation (MI300X / MI325X) — SSH + Docker, no Slurm.
# Style follows run_1p2d.sh; uses InferenceX
#   scripts/_disagg_ssh_remote_inner.sh + benchmarks/.../server.sh
#
# Topology: NODE_RANK=0 = prefill + router + benchmark; ranks 1 and 2 = decode.
# IPADDRS order must be: prefill_ip,decode1_ip,decode2_ip (decode indices must match server.sh).
# NNODES=3, xP=1, yD=2.
#
# Prerequisites:
#   - Passwordless SSH from this machine to all three nodes
#   - IMAGE pulled on each node; REMOTE_REPO (InferenceX) path consistent
#   - Weights: prefill at PREFILL_MODEL_HOST_DIR/MODEL_NAME; both decodes at
#     DECODE_MODEL_HOST_DIR/MODEL_NAME (same path by default; decode2 can use DECODE_MODEL_HOST_DIR_2)
#   - Docker: DOCKER_BIN + USE_SUDO_FOR_DOCKER (do not use DOCKER="sudo docker"; it breaks ssh env)
#   - TCP open: BARRIER_SYNC_PORT (5000), SGLANG_PD_PORT (8000), ROUTER_PORT (30000)
#   - IBDEVICES: auto from detect_ibdevices_bnxt.sh when possible; otherwise export explicitly
#   - Optional in-container libbnxt: REBUILD_LIBBNXT_IN_CONTAINER=1 and
#       PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-….tar.gz (tar under InferenceX/driver on host)
#   - Optional in-container MoRI: INSTALL_MORI_IN_CONTAINER=1, INSTALL_MORI_MODE=git|path (see run_1p1d_sglang_mi300_mi325x.sh header)
#
# Usage:
#   cd /path/to/InferenceX && bash run_1p2d_sglang_mi300_mi325x.sh
#   ISL=1024 OSL=1024 CONC_LIST="8 16" bash run_1p2d_sglang_mi300_mi325x.sh
#   DRY_RUN=1 bash run_1p2d_sglang_mi300_mi325x.sh
# If this file is not inside the repo, set INFERENCEX_DIR to the InferenceX root on the launcher.
# ──────────────────────────────────────────────────────────────

# --- Configurable parameters (override via environment) ---
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.9-rocm700-mi30x}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528}"
# Do not set DOCKER to "sudo docker" here: a multi-word value breaks unquoted ssh env arrays.
MODEL_DIR="${MODEL_DIR:-/dev/shm}"
PREFILL_MODEL_HOST_DIR="${PREFILL_MODEL_HOST_DIR:-${MODEL_DIR}}"
DECODE_MODEL_HOST_DIR="${DECODE_MODEL_HOST_DIR:-${MODEL_DIR}}"
DECODE_MODEL_HOST_DIR_2="${DECODE_MODEL_HOST_DIR_2:-${DECODE_MODEL_HOST_DIR}}"

ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-4}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

PREFILL_TP="${PREFILL_TP:-8}"
PREFILL_EP="${PREFILL_EP:-1}"
PREFILL_DP_ATTN="${PREFILL_DP_ATTN:-false}"
DECODE_TP="${DECODE_TP:-8}"
DECODE_EP="${DECODE_EP:-1}"
DECODE_DP_ATTN="${DECODE_DP_ATTN:-false}"
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
xP="${xP:-1}"
yD="${yD:-2}"

# Default: repo root when this script lives at <InferenceX>/run_1p2d_sglang_mi300_mi325x.sh
INFERENCEX_DIR="${INFERENCEX_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REMOTE_REPO="${REMOTE_REPO:-${INFERENCEX_DIR}}"
SCRIPT_DIR="${INFERENCEX_DIR}/scripts"
INNER_LOCAL="${SCRIPT_DIR}/_disagg_ssh_remote_inner.sh"
INNER_REMOTE="${REMOTE_REPO}/scripts/_disagg_ssh_remote_inner.sh"

# Three nodes (first two defaults match run1p1d; override the third with DECODE_NODE_2)
PREFILL_NODE="${PREFILL_NODE:-137.220.56.211}"
DECODE_NODE_1="${DECODE_NODE_1:-149.28.121.18}"
DECODE_NODE_2="${DECODE_NODE_2:-207.148.10.255}"

SSH_USER="${SSH_USER:-$(whoami)}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ${HOME}/.ssh/id_ed25519 -o IdentitiesOnly=yes"
DRY_RUN="${DRY_RUN:-0}"
SKIP_LOCAL_MODEL_CHECK="${SKIP_LOCAL_MODEL_CHECK:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${MANUAL_JOB_ID:-local_sglang_1p2d_${TIMESTAMP}}"

LOG_DIR="${HOME}/logs/sglang_disagg"
mkdir -p "$LOG_DIR"
BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-${LOG_DIR}/benchmark_logs_${TIMESTAMP}}"
mkdir -p "${BENCHMARK_LOGS_DIR}"

# --- EP/DP booleans (same as run_dsr1_fp8_mi325x_sglang_disagg_ssh.sh) ---
if [[ "${PREFILL_EP}" -eq 1 ]]; then PREFILL_ENABLE_EP=false; else PREFILL_ENABLE_EP=true; fi
if [[ "${PREFILL_DP_ATTN}" == "true" ]]; then PREFILL_ENABLE_DP=true; else PREFILL_ENABLE_DP=false; fi
if [[ "${DECODE_EP}" -eq 1 ]]; then DECODE_ENABLE_EP=false; else DECODE_ENABLE_EP=true; fi
if [[ "${DECODE_DP_ATTN}" == "true" ]]; then DECODE_ENABLE_DP=true; else DECODE_ENABLE_DP=false; fi

_IBDETECT="${INFERENCEX_DIR}/benchmarks/multi_node/amd_utils/detect_ibdevices_bnxt.sh"
if [[ -z "${IBDEVICES:-}" && -f "${_IBDETECT}" ]]; then
  _b="$("${_IBDETECT}")"
  [[ -n "${_b}" ]] && IBDEVICES="${_b}"
fi

PREFILL_SSH="${SSH_USER}@${PREFILL_NODE}"
DECODE_SSH_1="${SSH_USER}@${DECODE_NODE_1}"
DECODE_SSH_2="${SSH_USER}@${DECODE_NODE_2}"

echo "============================================"
echo " SGLang PD Disagg 1P+2D (MI300X / MI325X)"
echo "============================================"
echo " Image:           ${IMAGE}"
echo " Prefill model:   ${PREFILL_MODEL_HOST_DIR}/${MODEL_NAME}"
echo " Decode1 model:   ${DECODE_MODEL_HOST_DIR}/${MODEL_NAME}"
echo " Decode2 model:   ${DECODE_MODEL_HOST_DIR_2}/${MODEL_NAME}"
echo " ISL/OSL:         ${ISL}/${OSL}"
echo " Concurrency:     ${CONC_LIST}"
echo " Prefill (rank0): ${PREFILL_SSH} TP=${PREFILL_TP}"
echo " Decode 1 (rk1):  ${DECODE_SSH_1} TP=${DECODE_TP}"
echo " Decode 2 (rk2):  ${DECODE_SSH_2} TP=${DECODE_TP}"
echo " xP/yD:           ${xP}/${yD}  GPUS_PER_NODE=${GPUS_PER_NODE}"
echo " Repo (remote):   ${REMOTE_REPO}"
echo " Logs (remote):   ${BENCHMARK_LOGS_DIR}"
echo " JOB_ID:          ${JOB_ID}"
echo " IBDEVICES:       ${IBDEVICES:-<unset>}"
echo "============================================"

if [[ ! -f "${INNER_LOCAL}" ]]; then
  echo "Error: missing ${INNER_LOCAL} (set INFERENCEX_DIR)." >&2
  exit 1
fi
if [[ -z "${IBDEVICES:-}" ]]; then
  echo "Error: set IBDEVICES or install the RDMA toolchain to run detect_ibdevices_bnxt.sh." >&2
  exit 1
fi
if [[ "${REBUILD_LIBBNXT_IN_CONTAINER:-0}" == "1" && -n "${PATH_TO_BNXT_TAR_PACKAGE:-}" ]]; then
  _bnxt_tar="${PATH_TO_BNXT_TAR_PACKAGE}"
  if [[ "${_bnxt_tar}" == /workspace/* ]]; then
    _bnxt_tar_host="${REMOTE_REPO}${_bnxt_tar#/workspace}"
  else
    _bnxt_tar_host="${_bnxt_tar}"
  fi
  for _bnxt_node in "${PREFILL_SSH}" "${DECODE_SSH_1}" "${DECODE_SSH_2}"; do
    if ! ssh ${SSH_OPTS} "${_bnxt_node}" test -f "${_bnxt_tar_host}"; then
      echo "Error: REBUILD_LIBBNXT_IN_CONTAINER=1 but no file on ${_bnxt_node}: ${_bnxt_tar_host}" >&2
      echo "  Docker uses -v REMOTE_REPO:/workspace → set REMOTE_REPO to InferenceX root on GPU nodes and copy driver/ there." >&2
      exit 1
    fi
  done
  unset _bnxt_tar _bnxt_tar_host _bnxt_node
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: no SSH. Set PREFILL_IP, DECODE1_IP, DECODE2_IP to skip discovery."
  exit 0
fi
if [[ "${SKIP_LOCAL_MODEL_CHECK}" != "1" ]]; then
  _miss=0
  [[ -d "${PREFILL_MODEL_HOST_DIR}/${MODEL_NAME}" ]] || _miss=1
  [[ -d "${DECODE_MODEL_HOST_DIR}/${MODEL_NAME}" ]] || _miss=1
  [[ -d "${DECODE_MODEL_HOST_DIR_2}/${MODEL_NAME}" ]] || _miss=1
  if [[ "${_miss}" -ne 0 ]]; then
    echo "Error: missing weights under prefill/decode host dirs (local check). Use SKIP_LOCAL_MODEL_CHECK=1 to skip." >&2
    exit 1
  fi
fi

resolve_ip() {
  local target=$1
  ssh ${SSH_OPTS} "${target}" "ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print \$7; exit}'"
}

if [[ -z "${PREFILL_IP:-}" || -z "${DECODE1_IP:-}" || -z "${DECODE2_IP:-}" ]]; then
  PREFILL_IP="$(resolve_ip "${PREFILL_SSH}")"
  DECODE1_IP="$(resolve_ip "${DECODE_SSH_1}")"
  DECODE2_IP="$(resolve_ip "${DECODE_SSH_2}")"
fi
if [[ -z "${PREFILL_IP}" || -z "${DECODE1_IP}" || -z "${DECODE2_IP}" ]]; then
  echo "Error: cannot resolve data-plane IPs; set PREFILL_IP, DECODE1_IP, DECODE2_IP." >&2
  exit 1
fi

IPADDRS="${PREFILL_IP},${DECODE1_IP},${DECODE2_IP}"
NODE0_ADDR="${PREFILL_IP}"
NNODES=3

run_remote() {
  local ssh_target=$1
  local rank=$2
  local log_file=$3
  local model_host_dir=$4
  local use_tee="${5:-0}"
  # shellcheck disable=SC2206
  local -a _ssh=( ssh ${SSH_OPTS} "${ssh_target}" env
    JOB_ID="${JOB_ID}"
    NODE_RANK="${rank}"
    NODE0_ADDR="${NODE0_ADDR}"
    IPADDRS="${IPADDRS}"
    NNODES="${NNODES}"
    "HOST_MODEL_DIR=${model_host_dir}"
    "HOST_REPO=${REMOTE_REPO}"
    "HOST_LOG_ROOT=${BENCHMARK_LOGS_DIR}"
    "IMAGE=${IMAGE}"
    "MODEL_NAME=${MODEL_NAME}"
    xP="${xP}"
    yD="${yD}"
    GPUS_PER_NODE="${GPUS_PER_NODE}"
    PREFILL_TP_SIZE="${PREFILL_TP}"
    PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP}"
    PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP}"
    DECODE_TP_SIZE="${DECODE_TP}"
    DECODE_ENABLE_EP="${DECODE_ENABLE_EP}"
    DECODE_ENABLE_DP="${DECODE_ENABLE_DP}"
    DECODE_MTP_SIZE="${DECODE_MTP_SIZE}"
    BENCH_INPUT_LEN="${ISL}"
    BENCH_OUTPUT_LEN="${OSL}"
    "BENCH_MAX_CONCURRENCY=${CONC_LIST}"
    BENCH_RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO}"
    BENCH_REQUEST_RATE=inf
    BENCH_NUM_PROMPTS_MULTIPLIER=10
    "IBDEVICES=${IBDEVICES}"
    "DOCKER_BIN=${DOCKER_BIN:-docker}"
    "USE_SUDO_FOR_DOCKER=${USE_SUDO_FOR_DOCKER:-1}"
    "DOCKER_SHM_SIZE=${DOCKER_SHM_SIZE:-}"
    "EXTRA_DOCKER_ARGS=${EXTRA_DOCKER_ARGS:-}"
    "MORI_RDMA_TC=${MORI_RDMA_TC:-}"
    DRY_RUN=0
    "HOST_IP=${HOST_IP:-}"
    "BARRIER_LOCAL_IP=${BARRIER_LOCAL_IP:-}"
    "BARRIER_SYNC_PORT=${BARRIER_SYNC_PORT:-}"
    "SGLANG_PD_PORT=${SGLANG_PD_PORT:-}"
    "ROUTER_PORT=${ROUTER_PORT:-}"
    "REBUILD_LIBBNXT_IN_CONTAINER=${REBUILD_LIBBNXT_IN_CONTAINER:-0}"
    "PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE:-}"
    "INSTALL_MORI_IN_CONTAINER=${INSTALL_MORI_IN_CONTAINER:-0}"
    "INSTALL_MORI_MODE=${INSTALL_MORI_MODE:-git}"
    "MORI_GIT_URL=${MORI_GIT_URL:-}"
    "MORI_GIT_REF=${MORI_GIT_REF:-}"
    "MORI_GIT_CLONE_DIR=${MORI_GIT_CLONE_DIR:-}"
    "MORI_SOURCE_PATH=${MORI_SOURCE_PATH:-}"
    "INSTALL_MORI_PYTHON_BIN=${INSTALL_MORI_PYTHON_BIN:-}"
    "INSTALL_MORI_NO_BUILD_ISOLATION=${INSTALL_MORI_NO_BUILD_ISOLATION:-0}"
    bash "${INNER_REMOTE}"
  )

  if [[ "${use_tee}" == "1" ]]; then
    "${_ssh[@]}" 2>&1 | tee "${log_file}"
  else
    "${_ssh[@]}" >"${log_file}" 2>&1
  fi
}

echo ""
echo "IPADDRS=${IPADDRS}"
echo "Launching decode-1 (NODE_RANK=1)..."
run_remote "${DECODE_SSH_1}" 1 "${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_1}.log" "${DECODE_MODEL_HOST_DIR}" 0 &
DECODE1_PID=$!

echo "Launching decode-2 (NODE_RANK=2)..."
run_remote "${DECODE_SSH_2}" 2 "${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_2}.log" "${DECODE_MODEL_HOST_DIR_2}" 0 &
DECODE2_PID=$!

sleep 2

echo "Launching prefill + router + benchmark on ${PREFILL_NODE} (NODE_RANK=0)..."
set +e
run_remote "${PREFILL_SSH}" 0 "${BENCHMARK_LOGS_DIR}/ssh_prefill_${PREFILL_NODE}.log" "${PREFILL_MODEL_HOST_DIR}" 1
PREFILL_RC=$?
set -e

echo ""
echo "Waiting for decode SSH sessions..."
sleep 5
DECODE1_RC=0
DECODE2_RC=0
set +e
wait "${DECODE1_PID}" || DECODE1_RC=$?
wait "${DECODE2_PID}" || DECODE2_RC=$?
set -e

echo ""
echo "──────────────────────────────────────────────"
echo " Prefill exit code: ${PREFILL_RC}"
echo " Decode1 exit code: ${DECODE1_RC}"
echo " Decode2 exit code: ${DECODE2_RC}"
echo " Prefill log: ${BENCHMARK_LOGS_DIR}/ssh_prefill_${PREFILL_NODE}.log"
echo " Decode1 log: ${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_1}.log"
echo " Decode2 log: ${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_2}.log"
echo " Remote /benchmark_logs and /tmp/run_logs_${JOB_ID} on each host."
echo "──────────────────────────────────────────────"

if [[ "${PREFILL_RC}" -ne 0 ]]; then
  exit "${PREFILL_RC}"
fi
if [[ "${DECODE1_RC}" -ne 0 ]]; then
  echo "Decode node 1 exited ${DECODE1_RC}" >&2
  exit "${DECODE1_RC}"
fi
if [[ "${DECODE2_RC}" -ne 0 ]]; then
  echo "Decode node 2 exited ${DECODE2_RC}" >&2
  exit "${DECODE2_RC}"
fi
exit 0
