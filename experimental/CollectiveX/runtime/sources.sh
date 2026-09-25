# shellcheck shell=bash
# Pinned backend versions, source staging, and backend-cache mounts. Sourced by common.sh.

COLLX_DEEPEP_V2_REPO="https://github.com/deepseek-ai/DeepEP"
# Upstream DeepEP main. Carries #630 (single-node V2 init), #642 (fence.proxy.async in
# LOW_LATENCY_COMBINE_RECV; fixes the Blackwell low-latency combine corruption, DeepEP issue
# #700), #715 (system-scope release before the GIN barrier), #688 (NCCL Device API compat) and
# #640/#627 (pip-wheel SO-name resolution). The backend cache is keyed on this value.
COLLX_DEEPEP_V2_COMMIT="01dc3aaac82068020353dce2c302e38153c0bfaa"

# Must match the image's CUDA line: the cu12 wheel's r12 host library on cu130 images survives
# on sm90/sm100 but poisons the CUDA context during symmetric-heap init over MNNVL on sm103
# (gb300); every CUDA call after buffer creation fails cudaErrorUnknown. Part of the venv cache key.
COLLX_DEEPEP_V2_NVSHMEM_SPEC="nvidia-nvshmem-cu13==3.4.5"

# 2.10.0+cu130's bundled CUDA userland poisons the CUDA context during nvshmem symmetric-heap
# init over MNNVL on sm103/driver 580.159.03 (gb300); 2.11.0 is what the cu130 image itself
# ships. Part of the venv cache key.
COLLX_DEEPEP_V2_TORCH_SPEC="torch==2.11.0"

# Build-recipe generation for the DeepEP venv cache key: bump when build flags change without
# any pin changing, so venvs that already carry .ready are not reused with a stale recipe.
COLLX_DEEPEP_V2_BUILD_GEN="dlarch1"

COLLX_UCCL_REPO="https://github.com/uccl-project/uccl"
COLLX_UCCL_COMMIT="fc1b582031221645ea9fce58aeb57187713145e3"

# nccl-extensions (github.com/NVIDIA/nccl-extensions) owns nccl.ep since nccl4py stopped
# bundling it at 0.4; its combine-recv fence (the DeepEP #642 analogue) releases the low-latency
# ladder clamp. nccl4py is pinned alongside so a rebuild resolves the same tree. Two
# whitespace-separated pip specs: the install site word-splits this deliberately, and the whole
# string keys the shared cache dir.
COLLX_NCCL_EP_SPEC="nccl-extensions[cu13]==0.1.0 nccl4py[cu13]==0.5.0"

# Only the fixed /cx-cache mount enters the container; the operator host path does not.
collx_prepare_backend_cache() {
  local cache
  unset COLLX_PREPARED_BACKEND_CACHE
  cache="$(python3 "$COLLX_RUNTIME_DIR/probe.py" prepare-cache "$1")" || return 1
  [[ "$cache" = /* ]] || return 1
  export COLLX_PREPARED_BACKEND_CACHE="$cache"
}

# On b300 the NFS export can realize a new stage dir as UID 0 while git runs as the UID-mapped
# Actions user, tripping git's "dubious ownership" guard. HOME is this job's ephemeral dir, so a
# global exemption is scoped to the job (and reaches the submodule child git).
collx_prepare_source() {
  local mount_src="$1" name="$2" repository="$3" commit="$4" label="$5"
  local root source temporary log
  shift 5
  root="$mount_src/experimental/CollectiveX/.collx_sources"
  source="$root/$name-$commit"
  [ ! -d "$source" ] || return 0
  mkdir -p -- "$root" && chmod 700 "$root" || return 1
  temporary="$(mktemp -d "$root/.$name.XXXXXX")" || return 1
  log="$(collx_private_log_path "backend-source-$name")" || return 1
  git config --global --add safe.directory '*' >> "$log" 2>&1 || true
  if GIT_TERMINAL_PROMPT=0 git init -q "$temporary" > "$log" 2>&1 \
      && git -C "$temporary" remote add origin "$repository" >> "$log" 2>&1 \
      && GIT_TERMINAL_PROMPT=0 git -C "$temporary" fetch -q --no-tags --depth 1 \
        origin "$commit" >> "$log" 2>&1 \
      && git -C "$temporary" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD \
        >> "$log" 2>&1 \
      && [ "$(git -C "$temporary" rev-parse HEAD)" = "$commit" ] \
      && collx_prepare_submodules "$temporary" "$@" >> "$log" 2>&1 \
      && mv -- "$temporary" "$source" >> "$log" 2>&1; then
    return 0
  fi
  rm -rf -- "$temporary"
  collx_log "ERROR: $label source preparation failed"
  collx_log_tail "$log"
  return 1
}

collx_prepare_submodules() {
  local source="$1"
  shift
  [ "$#" -gt 0 ] || return 0
  GIT_TERMINAL_PROMPT=0 git -C "$source" submodule update -q --init --depth 1 "$@"
}

collx_prepare_deepep_source() {
  collx_prepare_source "$1" deepep-v2 "$COLLX_DEEPEP_V2_REPO" \
    "$COLLX_DEEPEP_V2_COMMIT" DeepEP third-party/fmt
}

# Skips the thirdparty submodules (rccl/mscclpp, for other targets) but keeps the whole tree:
# the ROCm path (common_hip.hpp) includes top-level util/gpu_rt.h.
collx_prepare_uccl_source() {
  collx_prepare_source "$1" uccl "$COLLX_UCCL_REPO" "$COLLX_UCCL_COMMIT" UCCL
}

collx_materialize_source() {
  local destination="$1" source
  [ -n "${COLLX_BACKEND_SOURCE_ROOT:-}" ] || return 1
  source="$COLLX_BACKEND_SOURCE_ROOT/$2"
  [ -d "$source" ] || return 1
  rm -rf -- "$destination" && cp -R -- "$source" "$destination"
}

collx_materialize_deepep_source() {
  collx_materialize_source "$1" "deepep-v2-$COLLX_DEEPEP_V2_COMMIT"
}

collx_materialize_uccl_source() {
  collx_materialize_source "$1" "uccl-$COLLX_UCCL_COMMIT"
}
