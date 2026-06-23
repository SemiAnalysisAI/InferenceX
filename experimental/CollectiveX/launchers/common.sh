# shellcheck shell=bash
# CollectiveX — shared launcher helpers (sourced, not executed).
#
# Cluster-generic scaffolding only (Slurm/container/build/staging); no
# model-serving. Logging goes to stderr so functions can `echo` a single
# result on stdout.

cx_log() { printf '[collectivex] %s\n' "$*" >&2; }
cx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

# Single multi-arch, digest-pinned container for ALL NVIDIA SKUs.
# This is the OCI image index for tag `v0.5.12-cu130`, covering BOTH linux/amd64
# (B200) and linux/arm64 (GB200); enroot import on each host pulls the matching
# arch from the index. (cu130 = CUDA 13, system nccl.h in /usr/include, torch 2.9.x.)
# Pinned by DIGEST ONLY (no tag): enroot mis-parses a combined `tag@sha256` ref
# and 400s at auth, so we use `repo@sha256:` — also the stricter pin.
# NOTE: DeepEP is NOT bundled here -> run_in_container.sh builds it via
# rebuild-deepep at job setup. (The arch-specific deepseek-v4-{blackwell,
# grace-blackwell} images DO bundle DeepEP — see CONTAINERS.md — but are not
# multi-arch and are not used by default.)
CX_IMAGE_MULTIARCH="lmsysorg/sglang@sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356"

cx_default_image() {
  case "$1" in
    b200*|gb200*|b300*|gb300*|h100*|h200*) echo "$CX_IMAGE_MULTIARCH" ;;
    *) cx_die "no default image for runner prefix: $1" ;;
  esac
}

# cx_ensure_squash <squash_dir> <image>  ->  echoes the squash file path.
# Imports via enroot only if a valid squash is not already present (flock-guarded,
# mirroring runners/launch_b200-dgxc.sh).
cx_ensure_squash() {
  local squash_dir="$1" image="$2"
  mkdir -p "$squash_dir" 2>/dev/null || true
  local key sq locks
  key="$(printf '%s' "$image" | sed 's#[/:@#]#_#g')"
  sq="$squash_dir/${key}.sqsh"
  locks="$squash_dir/.locks"; mkdir -p "$locks" 2>/dev/null || true
  (
    flock -w 900 9 || cx_die "lock timeout for $sq"
    if unsquashfs -l "$sq" >/dev/null 2>&1; then
      cx_log "squash present: $sq"
    else
      cx_log "enroot import docker://$image -> $sq (one-time, multi-GB)"
      rm -f "$sq"
      enroot import -o "$sq" "docker://$image" >&2 || cx_die "enroot import failed for $image"
      unsquashfs -l "$sq" >/dev/null 2>&1 || cx_die "import produced no valid squash: $sq"
    fi
  ) 9>"$locks/${key}.lock"
  echo "$sq"
}

# cx_stage_repo <repo_root> <stage_dir>  ->  echoes the mount-source root.
# Some clusters (e.g. GB200/watchtower) do not cross-mount the runner workspace
# to compute nodes. If CX_STAGE_DIR is set, rsync the CollectiveX tree onto that
# compute-visible shared FS and mount from there. No-op (echo repo_root) when
# stage_dir is empty or equals repo_root.
cx_stage_repo() {
  local repo_root="$1" stage_dir="${2:-}"
  if [ -z "$stage_dir" ] || [ "$stage_dir" = "$repo_root" ]; then
    echo "$repo_root"; return 0
  fi
  mkdir -p "$stage_dir/experimental" || cx_die "cannot create stage dir $stage_dir"
  cx_log "staging experimental/CollectiveX -> $stage_dir (compute-visible)"
  rsync -a --delete \
    --exclude='.nccl-tests/' --exclude='__pycache__/' --exclude='results/plots/' \
    "$repo_root/experimental/CollectiveX" "$stage_dir/experimental/" >&2 \
    || cx_die "rsync to stage dir failed"
  echo "$stage_dir"
}

# cx_build_nccl_tests <parent_dir> <mpi 0|1>  ->  echoes the build/ dir.
# Runs IN-CONTAINER (login nodes have no nvcc). Cached: skips if already built.
# CX_NCCL_HOME defaults to /usr (system nccl.h in /usr/include on the sglang
# cu130 images); override CX_CUDA_HOME / CX_NCCL_HOME / CX_MPI_HOME if needed.
cx_build_nccl_tests() {
  local parent="$1" mpi="${2:-0}" dir bin
  dir="$parent/nccl-tests"
  bin="$dir/build/all_reduce_perf"
  if [ -x "$bin" ]; then
    cx_log "nccl-tests already built: $dir/build"
    echo "$dir/build"; return 0
  fi
  mkdir -p "$parent"
  if [ ! -d "$dir/.git" ]; then
    cx_log "cloning nccl-tests -> $dir"
    git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git "$dir" >&2 \
      || cx_die "git clone nccl-tests failed"
  fi
  cx_log "building nccl-tests (MPI=$mpi, NCCL_HOME=${CX_NCCL_HOME:-/usr})"
  make -C "$dir" -j MPI="$mpi" \
       CUDA_HOME="${CX_CUDA_HOME:-/usr/local/cuda}" \
       NCCL_HOME="${CX_NCCL_HOME:-/usr}" \
       ${CX_MPI_HOME:+MPI_HOME="$CX_MPI_HOME"} >&2 \
    || cx_die "nccl-tests build failed (try a different CX_NCCL_HOME; need nccl.h + libnccl)"
  [ -x "$bin" ] || cx_die "nccl-tests build produced no binary at $bin"
  echo "$dir/build"
}
