# shellcheck shell=bash
# Sequential shard execution and launcher cleanup traps. Sourced by common.sh.

# Per-case benchmark inputs travel as run_ep.py argv decoded from the shard control (config.py
# case-args), never as env; launchers supply only allocation/container policy.
# shellcheck disable=SC2153
collx_run_shard() {
  local build_log expected_cases ci=0 failed_cases=0
  local runtime_log argv_file shard wrap
  local -a container_args ep_args
  [ "${NODES:-0}" -ge 1 ] && [ "${NGPUS:-0}" = "$((NODES * GPN))" ] \
    || collx_die "invalid shard launcher placement"
  [ -n "${JOB_ID:-}" ] && [ -n "${SQUASH_FILE:-}" ] \
    && [ -n "${CONTAINER_MOUNTS:-}" ] || collx_die "shard launcher is incomplete"
  wrap="$(collx_source_backend_env)"$'\n'"$(collx_slurm_rank_wrapper)"

  collx_resolve_slurm_rendezvous "$JOB_ID"
  collx_apply_network_profile "$NODES" "${COLLX_TRANSPORT:-}"
  mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"
  container_args=(--container-mounts="$CONTAINER_MOUNTS" --no-container-mount-home
    --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint)
  if declare -p COLLX_DISTRIBUTED_CONTAINER_ARGS >/dev/null 2>&1; then
    container_args+=("${COLLX_DISTRIBUTED_CONTAINER_ARGS[@]}")
  fi
  local container_name="cxep_${JOB_ID}"

  shard="${COLLX_SHARD_FILE:-}"
  [ -f "$shard" ] || shard="$COLLX_DIR/$shard"
  [ -f "$shard" ] || collx_die "shard control is unavailable"
  expected_cases="$(python3 "$COLLX_RUNTIME_DIR/config.py" case-count "$shard")" \
    && [[ "$expected_cases" =~ ^[1-9][0-9]*$ ]] \
    || collx_die "could not enumerate shard cases"

  collx_log "shard backend preparation: bench=$COLLX_BENCH nodes=$NODES"
  build_log="$(collx_private_log_path backend-prepare)"
  if ! srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 --chdir=/tmp \
    --container-name="$container_name" --container-image="$SQUASH_FILE" \
    "${container_args[@]}" --export=ALL \
    bash /ix/experimental/CollectiveX/runtime/prepare_backend.sh \
    </dev/null >"$build_log" 2>&1; then
    collx_log "ERROR: backend preparation failed"
    collx_log_tail "$build_log"
    return 1
  fi

  argv_file="$(mktemp)" || return 1
  while [ "$ci" -lt "$expected_cases" ]; do
    python3 "$COLLX_RUNTIME_DIR/config.py" case-args "$shard" "$ci" \
      "$RUNNER" "$TS" \
      "$NGPUS" "$NODES" "$GPN" "$SCALE_UP_DOMAIN" > "$argv_file" \
      || { rm -f "$argv_file"; collx_die "shard case $ci does not decode against this allocation"; }
    mapfile -d '' -t ep_args < "$argv_file"
    [ "${#ep_args[@]}" -gt 0 ] \
      || { rm -f "$argv_file"; collx_die "case $ci produced no benchmark arguments"; }
    collx_log "EP${NGPUS}[$((ci + 1))/$expected_cases] $COLLX_BENCH"
    runtime_log="$(collx_private_log_path "runtime-c$(printf '%03d' "$ci")")"
    # A hang guard, not a work budget: 900 killed FP8 prefill cases with complete artifacts, and
    # 1800 killed b200/h200 multi-node EP16 prefill (virtualized pools with degraded GPU-NIC p2p,
    # ~34 GB/s per node; see docs/methodology.md). 5400 stays inside the 300-minute allocation.
    if ! timeout -k 30 "${COLLX_RUN_TIMEOUT:-5400}" \
      srun --jobid="$JOB_ID" --nodes="$NODES" \
      --ntasks="$NGPUS" --ntasks-per-node="$GPN" --chdir=/tmp \
      --container-name="$container_name" --container-image="$SQUASH_FILE" \
      "${container_args[@]}" \
      --export=ALL \
      bash -c "$wrap" _ "${ep_args[@]}" \
      </dev/null >"$runtime_log" 2>&1; then
      collx_log "ERROR: case $ci failed"
      collx_log_tail "$runtime_log"
      failed_cases=$((failed_cases + 1))
    fi
    ci=$((ci + 1))
  done
  rm -f "$argv_file"
  [ "$failed_cases" = 0 ] || {
    collx_log "ERROR: $failed_cases/$expected_cases case(s) failed"
    return 1
  }
}

# With pyxis container_scope=global the named --container-writable container (cxep_<jobid>)
# survives job teardown, and its unpacked rootfs (tens of GB per node) accumulates until the
# next writable extraction fails with ENOSPC. Best-effort and bounded: teardown must never hang.
collx_remove_distributed_container() {
  local job_id="$1" nodes="${2:-1}"
  [ -n "$job_id" ] || return 0
  [ "$nodes" -ge 1 ] 2>/dev/null || return 0
  timeout 120 srun --jobid="$job_id" --nodes="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp enroot remove -f "pyxis_cxep_${job_id}" \
    </dev/null >/dev/null 2>&1 || true
}

collx_launcher_cleanup() {
  local rc="$1" stage_root="${MOUNT_SRC:-}"
  trap - EXIT HUP INT TERM
  if [ -n "${JOB_ID:-}" ]; then
    collx_remove_distributed_container "$JOB_ID" "${NODES:-1}"
    if ! collx_cleanup_allocation; then
      [ "$rc" != 0 ] || rc=1
      exit "$rc"
    fi
  fi
  if [ -n "${REPO_ROOT:-}" ] && [ -n "$stage_root" ] \
      && [ "$stage_root" != "$REPO_ROOT" ]; then
    if [ "$rc" != 0 ] && [ -d "$stage_root/experimental/CollectiveX" ]; then
      collx_collect_results "$stage_root" "$REPO_ROOT" || true
    fi
    if ! collx_cleanup_stage "$stage_root" "$REPO_ROOT"; then
      [ "$rc" != 0 ] || rc=1
    fi
  fi
  exit "$rc"
}

collx_install_launcher_fail_safe() {
  trap 'collx_launcher_cleanup "$?"' EXIT
  trap 'collx_launcher_cleanup 129' HUP
  trap 'collx_launcher_cleanup 130' INT
  trap 'collx_launcher_cleanup 143' TERM
}
