# shellcheck shell=bash
# CollectiveX — shared launcher helpers (sourced, not executed).
#
# Cluster-generic scaffolding only (Slurm/container/build/staging); no
# model-serving. Logging goes to stderr so functions can `echo` a single
# result on stdout.

cx_log() { printf '[collectivex] %s\n' "$*" >&2; }
cx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

# Allocate via salloc (--no-shell is appended) and echo the GRANTED Slurm job id, parsed from
# salloc's OWN output. Use INSTEAD of `salloc ...; JOB_ID=$(squeue --name=<name> -h -o %A | head -1)`:
# that lookup is not unique per allocation, so under GHA-matrix concurrency (several cells calling
# salloc with the same --job-name on one cluster) it returns a SIBLING cell's job id. Observed on
# gb300: salloc granted 11354 but the name lookup returned a still-pending 11356 -> srun "Expired or
# invalid job 11356" -> the cell failed even though its own allocation was fine. Parsing salloc's own
# "Granted job allocation N" is race-free. salloc progress still streams live to the job log via tee.
cx_salloc_jobid() {
  local _t; _t="$(mktemp)"
  salloc "$@" --no-shell 2>&1 | tee "$_t" >&2 || true
  sed -n 's/.*Granted job allocation \([0-9][0-9]*\).*/\1/p' "$_t" | head -n1
  rm -f "$_t"
}

# Single multi-arch container for ALL NVIDIA SKUs: tag `v0.5.11-cu130` is an OCI
# image index covering linux/amd64 (B200) + linux/arm64 (GB200); enroot import
# pulls the matching arch. (cu130 = CUDA 13, system nccl.h in /usr/include, torch 2.9.x.)
# IMPORT BY TAG, not by digest: enroot's anonymous Docker Hub token scope is built
# from the tag; a bare `repo@sha256:` ref makes enroot prompt for a password and
# HANG in non-interactive CI (and a combined `tag@sha256` ref 400s). The expected
# multi-arch index digest is recorded for provenance/verification:
CX_IMAGE_DIGEST="sha256:061fb71f838e82000a1768c159654d526c2f17ebe751c21e7fc48ca53c8ef975"
# (v0.5.12-cu130 was rejected: its 62 layers overflow enroot's overlay-based
# squash creation on these nodes — "failed to mount overlay ... Invalid argument".
# v0.5.11-cu130 imports cleanly and is pre-staged on GB200.)
# DeepEP is NOT bundled here -> run_in_container.sh builds it via rebuild-deepep.
# (The arch-specific deepseek-v4-{blackwell,grace-blackwell} images DO bundle
# DeepEP — see CONTAINERS.md — but are not multi-arch and are not the default.)
CX_IMAGE_MULTIARCH="lmsysorg/sglang:v0.5.11-cu130"

# AMD (ROCm/CDNA): the multi-arch NVIDIA image above is x86_64+aarch64 CUDA and
# cannot run on MI355X. AMD uses a separate ROCm image that bundles MoRI (the
# AMD EP library). Single-arch (linux/amd64 host, ROCm runtime); not digest-
# pinned yet — pin once validated on the runner. See CONTAINERS.md.
CX_IMAGE_AMD_MORI="rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2"

# NIXL stack: the sglang multiarch image has neither the NIXL agent nor the device-EP build deps,
# and its Abseil (20220623) is what blocked the NIXL EP meson build (docs/gated.md). The dynamo
# tensorrtllm-runtime image (CUDA-13, 2026) ships NIXL + a modern Abseil/UCX — the container-switch
# the gated NIXL item calls for. Selected automatically for CX_BENCH=nixl on NVIDIA SKUs (override
# with CX_IMAGE). Listed in .github/configs/nvidia-master.yaml.
CX_IMAGE_NIXL="${CX_IMAGE_NIXL:-nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-dev.1-cuda13}"

cx_default_image() {
  # CX_BENCH=nixl needs the NIXL/dynamo container — switch automatically on NVIDIA SKUs (CX_BENCH is
  # already in the inherited env at this point). AMD keeps the MoRI image (no NIXL build there).
  if [ "${CX_BENCH:-}" = "nixl" ]; then
    case "$1" in
      b200*|gb200*|b300*|gb300*|h100*|h200*) echo "$CX_IMAGE_NIXL"; return ;;
    esac
  fi
  case "$1" in
    mi355x*|mi350x*|mi325x*|mi300x*) echo "$CX_IMAGE_AMD_MORI" ;;
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
      # </dev/null: never block on enroot's interactive password prompt (a missing
      # anonymous token must fail fast, not hang the CI job).
      enroot import -o "$sq" "docker://$image" </dev/null >&2 \
        || cx_die "enroot import failed for $image (anonymous auth needs a TAG ref, not a bare digest; or pre-stage the squash)"
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
  # Concurrency isolation. Under GHA the per-config concurrency fan-out runs many
  # same-SKU dispatches at once, all staging into the SAME shared base dir; a
  # shared dir + `rsync --delete` lets one job unlink/replace a file a peer is
  # mid-read of -> "error reading input file: Stale file handle" on the next
  # `srun ... run_in_container.sh`. Give each EXECUTING job its own subdir keyed on
  # the runner name (a self-hosted runner runs one job at a time, so concurrent
  # jobs never share a dir); sequential reuse on one runner is safe (the jobs do
  # not overlap, and --delete refreshes the tree). Outside GHA (no RUNNER_NAME /
  # GITHUB_RUN_ID) keep the single shared dir — SSH use is single-tenant.
  local tag="${RUNNER_NAME:-${GITHUB_RUN_ID:-}}"
  if [ -n "$tag" ]; then
    stage_dir="$stage_dir/job_$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
  fi
  mkdir -p "$stage_dir/experimental" || cx_die "cannot create stage dir $stage_dir"
  cx_log "staging experimental/CollectiveX -> $stage_dir (compute-visible)"
  rsync -a --delete \
    --exclude='.nccl-tests/' --exclude='__pycache__/' --exclude='results/plots/' \
    "$repo_root/experimental/CollectiveX" "$stage_dir/experimental/" >&2 \
    || cx_die "rsync to stage dir failed"
  echo "$stage_dir"
}

# cx_collect_results <mount_src> <repo_root>
# When the run used a staged (compute-visible) mount, copy result JSONs back to
# the original checkout's results/ so the workflow's upload-artifact (which reads
# the checkout, not the stage dir) finds them. No-op when no staging was used.
cx_collect_results() {
  local mount_src="$1" repo_root="$2" dst
  [ "$mount_src" = "$repo_root" ] && return 0
  dst="$repo_root/experimental/CollectiveX/results"
  mkdir -p "$dst"
  cp "$mount_src/experimental/CollectiveX/results/"*.json "$dst/" 2>/dev/null || true
  cx_log "copied results from stage dir -> $dst (for artifact upload)"
}

# cx_build_nccl_tests <parent_dir> <mpi 0|1>  ->  echoes the build/ dir.
# Runs IN-CONTAINER (login nodes have no nvcc). Cached: skips if already built.
# CX_NCCL_HOME defaults to /usr (system nccl.h in /usr/include on the sglang
# cu130 images); override CX_CUDA_HOME / CX_NCCL_HOME / CX_MPI_HOME if needed.
cx_build_nccl_tests() {
  local parent="$1" mpi="${2:-0}" dir bin sfx=""
  # Cache MPI=0 and MPI=1 builds in SEPARATE dirs. A single-node (MPI=0) binary
  # reused under `srun --mpi=pmix` runs as N standalone world=1 procs (busbw=0);
  # keying the cache by flavor prevents that cross-contamination.
  [ "$mpi" = "1" ] && sfx="-mpi"
  dir="$parent/nccl-tests$sfx"
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
  # MPI=1 needs MPI_HOME. On Debian/Ubuntu OpenMPI the headers live under
  # /usr/lib/<arch>/openmpi/include (NOT /usr/include), so MPI_HOME=/usr fails;
  # point it at that openmpi dir (libmpi resolves via the default linker path).
  # Works for both x86_64 (B200) and aarch64 (GB200). Override with CX_MPI_HOME.
  local mpi_home="${CX_MPI_HOME:-}"
  if [ "$mpi" = "1" ] && [ -z "$mpi_home" ]; then
    mpi_home="$(ls -d /usr/lib/*/openmpi 2>/dev/null | head -n1)"
  fi
  cx_log "building nccl-tests (MPI=$mpi, NCCL_HOME=${CX_NCCL_HOME:-/usr}${mpi_home:+, MPI_HOME=$mpi_home})"
  make -C "$dir" -j MPI="$mpi" \
       CUDA_HOME="${CX_CUDA_HOME:-/usr/local/cuda}" \
       NCCL_HOME="${CX_NCCL_HOME:-/usr}" \
       ${mpi_home:+MPI_HOME="$mpi_home"} >&2 \
    || cx_die "nccl-tests build failed (try a different CX_NCCL_HOME/CX_MPI_HOME; need nccl.h + libnccl)"
  [ -x "$bin" ] || cx_die "nccl-tests build produced no binary at $bin"
  echo "$dir/build"
}

# cx_build_rccl_tests <parent_dir> <mpi 0|1>  ->  echoes the build/ dir.
# AMD/ROCm counterpart of cx_build_nccl_tests: ROCm/rccl-tests is a fork of
# nccl-tests producing the SAME binary names (<op>_perf) and output format, so
# run_nccl.py parses it unchanged. `make` defaults to ROCm at /opt/rocm
# (amdclang++ + librccl); validated building in-container on MI355X. Override
# CX_ROCM_HOME / CX_RCCL_HOME / CX_MPI_HOME if the toolchain lives elsewhere.
cx_build_rccl_tests() {
  local parent="$1" mpi="${2:-0}" dir bin
  dir="$parent/rccl-tests"
  bin="$dir/build/all_reduce_perf"
  if [ -x "$bin" ]; then
    cx_log "rccl-tests already built: $dir/build"
    echo "$dir/build"; return 0
  fi
  mkdir -p "$parent"
  if [ ! -d "$dir/.git" ]; then
    cx_log "cloning rccl-tests -> $dir"
    git clone --depth 1 https://github.com/ROCm/rccl-tests.git "$dir" >&2 \
      || cx_die "git clone rccl-tests failed"
  fi
  cx_log "building rccl-tests (MPI=$mpi, ROCm ${CX_ROCM_HOME:-/opt/rocm})"
  make -C "$dir" -j MPI="$mpi" \
       ${CX_ROCM_HOME:+HIP_HOME="$CX_ROCM_HOME"} \
       ${CX_RCCL_HOME:+RCCL_HOME="$CX_RCCL_HOME"} \
       ${CX_MPI_HOME:+MPI_HOME="$CX_MPI_HOME"} >&2 \
    || cx_die "rccl-tests build failed (need ROCm + librccl; try CX_ROCM_HOME)"
  [ -x "$bin" ] || cx_die "rccl-tests build produced no binary at $bin"
  echo "$dir/build"
}
