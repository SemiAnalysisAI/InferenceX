# shellcheck shell=bash
# Slurm allocation lifecycle, rank identity, and accelerator health checks. Sourced by common.sh.

collx_allocation_nodes_csv() {
  local job_id="$1" nodelist node output=""
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  nodelist="$(squeue -h -j "$job_id" -o %N 2>/dev/null)" || return 1
  [[ "$nodelist" =~ ^[][A-Za-z0-9._,-]+$ ]] || return 1
  while IFS= read -r node; do
    [ -n "$node" ] || continue
    [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 1
    [ -z "$output" ] || output+=,
    output+="$node"
  done < <(scontrol show hostnames "$nodelist" 2>/dev/null)
  [ -n "$output" ] || return 1
  printf '%s' "$output"
}

collx_resolve_slurm_rendezvous() {
  local job_id="$1" master_addr master_port socket_ifname="${COLLX_SOCKET_IFNAME:-}"
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || collx_die "invalid rendezvous allocation"
  # Relative node zero hosts global rank 0. Prefer the address on the validated socket interface:
  # a short hostname may resolve onto a management network that ranks cannot use.
  if [[ "$socket_ifname" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]]; then
    master_addr="$(srun --jobid="$job_id" --nodes=1 --ntasks=1 --relative=0 \
      --chdir=/tmp --export="$(collx_host_exports)" bash -s -- "$socket_ifname" \
      2>/dev/null <<'BASH' | head -n1
set -eo pipefail
ip -o -4 address show dev "$1" scope global \
  | awk 'NR == 1 {split($4, address, "/"); print address[1]}'
BASH
)"
    [[ "$master_addr" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] \
      || collx_die "could not resolve the allocated primary interface"
  else
    master_addr="$(srun --jobid="$job_id" --nodes=1 --ntasks=1 --relative=0 \
      --chdir=/tmp --export="$(collx_host_exports)" hostname -s 2>/dev/null | head -n1)"
    [[ "$master_addr" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
      || collx_die "could not resolve the allocated primary node"
  fi
  master_port="${COLLX_MASTER_PORT:-29551}"
  [[ "$master_port" =~ ^[1-9][0-9]*$ ]] && [ "$master_port" -le 65535 ] \
    || collx_die "invalid distributed rendezvous port"
  export MASTER_ADDR="$master_addr" MASTER_PORT="$master_port"
}

# Printed into `bash -c` ahead of the rank wrapper; sources the per-node env written by
# prepare_backend.sh write_rank_env.
collx_source_backend_env() {
  cat <<'BASH'
case "${SLURM_NODEID:-}" in ""|*[!0-9]*) exit 66;; esac
. "/ix/experimental/CollectiveX/.collx_backend/env/node-${SLURM_NODEID}.sh" || exit 66
BASH
}

# Printed into `bash -c` for one Slurm task per GPU; rank identity comes from Slurm, never from
# caller-supplied values.
collx_slurm_rank_wrapper() {
  cat <<'BASH'
case "${SLURM_PROCID:-}:${SLURM_NTASKS:-}:${SLURM_LOCALID:-}:${SLURM_NODEID:-}" in
  *[!0-9:]*|:*|*::*|*:) exit 67 ;;
esac
[ "$SLURM_NTASKS" = "$COLLX_NGPUS" ] || exit 67
[ "$SLURM_LOCALID" -lt "$COLLX_GPUS_PER_NODE" ] || exit 67
. /ix/experimental/CollectiveX/runtime/common.sh || exit 68
if [ "${COLLX_NODES:-1}" -gt 1 ] && [ "${COLLX_TRANSPORT:-}" != mnnvl ]; then
  if [ -z "${COLLX_SOCKET_IFNAME:-}" ]; then
    COLLX_SOCKET_IFNAME="$(collx_default_route_interface)" || exit 68
    [[ "$COLLX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]] || exit 68
    export COLLX_SOCKET_IFNAME
  fi
  collx_apply_network_profile "$COLLX_NODES" "$COLLX_TRANSPORT" || exit 68
fi
export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_RANK="$SLURM_LOCALID" LOCAL_WORLD_SIZE="$COLLX_GPUS_PER_NODE"
exec python3 bench/run_ep.py "$@"
BASH
}

# inferencex-dash samples sacct JobName and joins it against the GHA job's runner_name, so the
# job name must be $RUNNER_NAME. Hand-driven runs get a fixed label.
collx_slurm_job_name() {
  local name="${RUNNER_NAME:-}"
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || name="collectivex"
  printf '%s' "$name"
}

# JOB_ID is recorded in the job root so workflow cleanup can release a launcher interrupted by
# Actions. The job name is prepended so a caller-supplied --job-name wins (salloc takes the last).
collx_salloc_jobid() {
  local log_label=scheduler-allocation log job_id root="${COLLX_JOB_ROOT:-}"
  set -- --job-name="$(collx_slurm_job_name)" "$@"
  case "${COLLX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  if ! log="$(collx_private_log_path "$log_label")"; then
    collx_log "ERROR: scheduler log is unavailable"
    return 1
  fi
  collx_log "scheduler-request=submit"
  if ! (salloc "$@" --no-shell) > "$log" 2>&1; then
    collx_log "ERROR: scheduler allocation failed"
    collx_log_tail "$log"
    return 1
  fi
  job_id="$(sed -nE \
      's/.*Granted job allocation ([1-9][0-9]*).*/\1/p' "$log" | head -n1)"
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  JOB_ID="$job_id"
  if [ -n "$root" ]; then
    collx_job_root_is_safe "$root" || return 1
    (umask 077; printf '%s\n' "$JOB_ID" > "$root/jobid") || return 1
  fi
}

collx_cleanup_allocation() {
  local root="${1:-${COLLX_JOB_ROOT:-}}" path="" job_id="${JOB_ID:-}" active
  if [ -n "$root" ]; then
    collx_job_root_is_safe "$root" || return 1
    path="$root/jobid"
    if [ -z "$job_id" ] && [ -f "$path" ]; then
      IFS= read -r job_id < "$path" || return 1
    fi
  fi
  [ -n "$job_id" ] || return 0
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  scancel "$job_id" >/dev/null 2>&1 || true
  for _ in {1..30}; do
    active="$(squeue -h -j "$job_id" -o %A 2>/dev/null)" || active=unknown
    if [ -z "$active" ]; then
      [ -z "$path" ] || rm -f -- "$path"
      return
    fi
    sleep 1
  done
  collx_log "ERROR: scheduled allocation did not terminate during cleanup"
  return 1
}

# Collectives are barriers, so one throttled device paces every rank. `--gres` makes the step
# see the devices it judges; `--time` bounds nvidia-smi wedging in D-state on sick hardware,
# which Python's own timeout cannot reap.
collx_validate_gpu_health_on_job() {
  local job_id="$1" nodes="$2" gpus_per_node="$3" log_label=gpu-health log
  case "${COLLX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(collx_private_log_path "$log_label")"
  export COLLX_GPU_HEALTH_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --gres=gpu:"$gpus_per_node" --time=5 --chdir=/tmp --input=all \
    --export="$(collx_host_exports)" python3 /dev/stdin gpu-health \
    < "$COLLX_RUNTIME_DIR/probe.py" >"$log" 2>&1
}

# A clean nvidia-smi inventory does not prove a cancelled workload released every CUDA context;
# retaining each primary context catches poisoned allocations before a shard fails every case.
collx_validate_cuda_context_on_job() {
  local job_id="$1" nodes="$2" gpus_per_node="$3" log_label=cuda-context log
  case "${COLLX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${COLLX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(collx_private_log_path "$log_label")"
  export COLLX_CUDA_CONTEXT_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --gres=gpu:"$gpus_per_node" --chdir=/tmp --input=all \
    --export="$(collx_host_exports)" python3 /dev/stdin cuda-context "$gpus_per_node" \
    < "$COLLX_RUNTIME_DIR/probe.py" >"$log" 2>&1
}
