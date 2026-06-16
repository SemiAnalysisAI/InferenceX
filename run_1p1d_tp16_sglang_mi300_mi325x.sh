#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# 1P+1D SGLang PD disaggregation — DECODE TP16 over 2 nodes (MI300X / MI325X).
# SSH + Docker, no Slurm. TP16 FORK of run_1p1d_sglang_mi300_mi325x.sh:
# hardcodes exactly TWO decode nodes that together form ONE TP16 decode worker.
# Engine logic unchanged: scripts/_disagg_ssh_remote_inner.sh + benchmarks/.../server.sh
# (server_sglang.sh already handles DECODE_NODES_PER_WORKER>1 via --dist-init-addr/
#  --nnodes/--node-rank; this fork only feeds it the right IPADDRS/NODE_RANK layout).
#
# Topology: NODE_RANK=0 = prefill + router + benchmark; NODE_RANK=1,2 = the two
# decode nodes of the single TP16 decode worker (decode head = NODE_RANK 1).
# IPADDRS order: prefill_ip,decode1_ip,decode2_ip. NNODES=3, xP=1, yD=1.
#
# Prerequisites:
#   - Passwordless SSH from this machine to both nodes
#   - IMAGE pulled on each node; REMOTE_REPO (InferenceX) path matches on both
#   - Weights: PREFILL_MODEL_HOST_DIR/MODEL_NAME and DECODE_MODEL_HOST_DIR/MODEL_NAME
#   - TCP open between nodes: 5000 / 8000 / 30000 (or your BARRIER_SYNC_PORT, etc.)
#   - IBDEVICES (run detect_ibdevices_bnxt.sh locally if needed)
#   - InferenceX on each node must include scripts/_disagg_ssh_remote_inner.sh (rsync/scp to decode if needed)
#   - Docker: default DOCKER_BIN=docker and USE_SUDO_FOR_DOCKER=1 (sudo docker run). If you are in the docker group:
#       export USE_SUDO_FOR_DOCKER=0
#   - Optional in-container libbnxt rebuild (helps bnxt_re ABI / MoRI availDevices): put tarball under driver/ on both nodes, then:
#       export REBUILD_LIBBNXT_IN_CONTAINER=1
#       export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-231.0.162.0.tar.gz
#   - Optional in-container MoRI reinstall (align with /sgl-workspace/mori or try a newer commit):
#       export INSTALL_MORI_IN_CONTAINER=1
#       # Default: git shallow clone (container needs GitHub); or path mode: place mori under REMOTE_REPO/mori on each node
#       export INSTALL_MORI_MODE=git   # or path
#       export MORI_GIT_REF=main
#       # If venv build fails, try: export INSTALL_MORI_NO_BUILD_ISOLATION=1
#
# Usage:
#   cd /path/to/InferenceX && bash run_1p1d_sglang_mi300_mi325x.sh
#   ISL=1024 OSL=1024 CONC_LIST="8 16" bash run_1p1d_sglang_mi300_mi325x.sh
# If this file is not inside the repo, set INFERENCEX_DIR to the InferenceX root on the launcher.
# ──────────────────────────────────────────────────────────────

# --- Configurable parameters (override via env) ---
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.9-rocm700-mi30x}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528}"
# Do not set DOCKER to "sudo docker" here: a multi-word value breaks unquoted env arrays
# (bash splits into DOCKER=sudo + bare "docker", so `env` runs docker instead of bash inner).

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
yD="${yD:-1}"

# Default: repo root when this script lives at <InferenceX>/run_1p1d_sglang_mi300_mi325x.sh
INFERENCEX_DIR="${INFERENCEX_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REMOTE_REPO="${REMOTE_REPO:-${INFERENCEX_DIR}}"
SCRIPT_DIR="${INFERENCEX_DIR}/scripts"
INNER_LOCAL="${SCRIPT_DIR}/_disagg_ssh_remote_inner.sh"
INNER_REMOTE="${REMOTE_REPO}/scripts/_disagg_ssh_remote_inner.sh"

# Three nodes: 1 prefill + 2 decode (the 2 decode nodes form one TP16 worker).
PREFILL_NODE="${PREFILL_NODE:-137.220.56.211}"
DECODE_NODE_1="${DECODE_NODE_1:-149.28.121.18}"
DECODE_NODE_2="${DECODE_NODE_2:-207.148.10.255}"

# Host-side parent dirs for weights (mounted in container as /models/<MODEL_NAME>). Default /dev/shm.
# If prefill only has weights under /mnt: export PREFILL_MODEL_HOST_DIR=/mnt
PREFILL_MODEL_HOST_DIR="${PREFILL_MODEL_HOST_DIR:-/dev/shm}"
DECODE_MODEL_HOST_DIR="${DECODE_MODEL_HOST_DIR:-/dev/shm}"

SSH_USER="${SSH_USER:-$(whoami)}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ${HOME}/.ssh/id_ed25519 -o IdentitiesOnly=yes"
DRY_RUN="${DRY_RUN:-0}"
SKIP_LOCAL_MODEL_CHECK="${SKIP_LOCAL_MODEL_CHECK:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${MANUAL_JOB_ID:-local_sglang_1p1d_${TIMESTAMP}}"

LOG_DIR="${HOME}/logs/sglang_disagg"
mkdir -p "$LOG_DIR"
BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-${LOG_DIR}/benchmark_logs_${TIMESTAMP}}"
mkdir -p "$BENCHMARK_LOGS_DIR"

# --- EP/DP flags (same as run_dsr1_fp8_mi325x_sglang_disagg_ssh.sh) ---
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
echo " SGLang PD Disagg 1P+1D (MI300X / MI325X)"
echo "============================================"
echo " Image:        ${IMAGE}"
echo " Model:        ${MODEL_NAME}"
echo " ISL/OSL:      ${ISL}/${OSL}"
echo " Concurrency:  ${CONC_LIST}"
echo " Prefill node: ${PREFILL_NODE} (TP=${PREFILL_TP})"
echo "   model:      ${PREFILL_MODEL_HOST_DIR}/${MODEL_NAME}"
echo " Decode nodes: ${DECODE_NODE_1}, ${DECODE_NODE_2} (one TP=${DECODE_TP} worker over 2 nodes)"
echo "   model:      ${DECODE_MODEL_HOST_DIR}/${MODEL_NAME}"
echo " Repo:         ${REMOTE_REPO}"
echo " Logs:         ${BENCHMARK_LOGS_DIR}"
echo " JOB_ID:       ${JOB_ID}"
echo " IBDEVICES:    ${IBDEVICES:-<unset>}"
echo "============================================"

if [[ ! -f "${INNER_LOCAL}" ]]; then
  echo "Error: missing ${INNER_LOCAL} (set INFERENCEX_DIR)." >&2
  exit 1
fi
if [[ -z "${IBDEVICES:-}" ]]; then
  echo "Error: set IBDEVICES or install rdma tools for detect_ibdevices_bnxt.sh." >&2
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
  echo "DRY_RUN=1: no SSH. Set PREFILL_IP / DECODE_IP_1 / DECODE_IP_2 to skip discovery in a future run."
  exit 0
fi
if [[ "${SKIP_LOCAL_MODEL_CHECK}" != "1" ]]; then
  if [[ ! -d "${PREFILL_MODEL_HOST_DIR}/${MODEL_NAME}" || ! -d "${DECODE_MODEL_HOST_DIR}/${MODEL_NAME}" ]]; then
    echo "Error: missing weights under ${PREFILL_MODEL_HOST_DIR}/${MODEL_NAME} and/or ${DECODE_MODEL_HOST_DIR}/${MODEL_NAME}. Use SKIP_LOCAL_MODEL_CHECK=1 to skip." >&2
    exit 1
  fi
fi

resolve_ip() {
  local target=$1
  ssh ${SSH_OPTS} "${target}" "ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print \$7; exit}'"
}

if [[ -z "${PREFILL_IP:-}" ]]; then PREFILL_IP="$(resolve_ip "${PREFILL_SSH}")"; fi
if [[ -z "${DECODE_IP_1:-}" ]]; then DECODE_IP_1="$(resolve_ip "${DECODE_SSH_1}")"; fi
if [[ -z "${DECODE_IP_2:-}" ]]; then DECODE_IP_2="$(resolve_ip "${DECODE_SSH_2}")"; fi
if [[ -z "${PREFILL_IP}" || -z "${DECODE_IP_1}" || -z "${DECODE_IP_2}" ]]; then
  echo "Error: set PREFILL_IP/DECODE_IP_1/DECODE_IP_2 or fix SSH 'ip route get 1.1.1.1' on all hosts." >&2
  exit 1
fi

# IPADDRS in NODE_RANK order: prefill (rank 0), decode head (rank 1), decode 2 (rank 2).
# server_sglang.sh derives the single TP16 decode worker from this + DECODE_TP_SIZE=16.
IPADDRS="${PREFILL_IP},${DECODE_IP_1},${DECODE_IP_2}"
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
echo "Launching decode head on ${DECODE_NODE_1} (NODE_RANK=1)..."
run_remote "${DECODE_SSH_1}" 1 "${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_1}.log" "${DECODE_MODEL_HOST_DIR}" 0 &
DECODE_SSH_PID_1=$!

echo "Launching decode node-2 on ${DECODE_NODE_2} (NODE_RANK=2)..."
run_remote "${DECODE_SSH_2}" 2 "${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_2}.log" "${DECODE_MODEL_HOST_DIR}" 0 &
DECODE_SSH_PID_2=$!

sleep 2

echo "Launching prefill + router + benchmark on ${PREFILL_NODE} (NODE_RANK=0)..."
set +e
run_remote "${PREFILL_SSH}" 0 "${BENCHMARK_LOGS_DIR}/ssh_prefill_${PREFILL_NODE}.log" "${PREFILL_MODEL_HOST_DIR}" 1
PREFILL_RC=$?
set -e

echo ""
echo "Waiting for decode SSH sessions..."
sleep 5
set +e
wait "${DECODE_SSH_PID_1}"
DECODE_RC_1=$?
wait "${DECODE_SSH_PID_2}"
DECODE_RC_2=$?
set -e

echo ""
echo "──────────────────────────────────────────────"
echo " Benchmark complete (prefill: ${PREFILL_RC}, decode1: ${DECODE_RC_1}, decode2: ${DECODE_RC_2})"
echo "  Prefill log:  ${BENCHMARK_LOGS_DIR}/ssh_prefill_${PREFILL_NODE}.log"
echo "  Decode1 log:  ${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_1}.log"
echo "  Decode2 log:  ${BENCHMARK_LOGS_DIR}/ssh_decode_${DECODE_NODE_2}.log"
echo " Logs: ${BENCHMARK_LOGS_DIR}/"
echo "──────────────────────────────────────────────"

if [[ "${PREFILL_RC}" -ne 0 ]]; then
  exit "${PREFILL_RC}"
fi
if [[ "${DECODE_RC_1}" -ne 0 ]]; then
  echo "Decode node 1 SSH exited ${DECODE_RC_1}" >&2
  exit "${DECODE_RC_1}"
fi
if [[ "${DECODE_RC_2}" -ne 0 ]]; then
  echo "Decode node 2 SSH exited ${DECODE_RC_2}" >&2
  exit "${DECODE_RC_2}"
fi
exit 0
