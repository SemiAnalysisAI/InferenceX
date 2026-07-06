#!/usr/bin/env bash
# CollectiveX — generic in-container benchmark dispatcher (single-node).
#
# Runs INSIDE the container under `srun` for single-node shards. The GB EP8 launcher invokes
# run_ep.py directly across nodes. The SKU adapter handles allocation/container/transport-env;
# this script selects one EP backend from CX_BENCH and writes result JSON under results/.
#
# Required env (exported by the adapter): CX_RUNNER CX_NGPUS CX_TS CX_TOPO
# Selector: CX_BENCH = deepep | deepep-v2 | mori | uccl | nccl-ep | deepep-hybrid
# EP knobs passed to tests/run_ep.py:
#   CX_PHASE = decode | prefill | both (default decode)   <- picks the token sweep
#   CX_TOKENS_LADDER (space/comma sep; blank = phase default)
#   CX_HIDDEN CX_TOPK CX_EXPERTS CX_ROUTING CX_SEED CX_ITERS
set -euo pipefail

cd /ix/experimental/CollectiveX
# shellcheck source=../runtime/common.sh
source runtime/common.sh
mkdir -p results
cx_write_runtime_stage backend-setup || cx_die "cannot record runtime stage"

: "${CX_RUNNER:?CX_RUNNER not set}"
: "${CX_NGPUS:?CX_NGPUS not set}"
: "${CX_TS:?CX_TS not set}"
: "${CX_TOPO:?CX_TOPO not set}"
CX_BENCH="${CX_BENCH:-deepep}"
CX_TRANSPORT="${CX_TRANSPORT:-}"

cx_apply_timing_profile

cx_log "in-container: runner=$CX_RUNNER ngpus=$CX_NGPUS bench=$CX_BENCH topo=$CX_TOPO"

# Blank ladders use the phase default in tests/run_ep.py.
cx_ep_ladder() {
  printf '%s' "${CX_TOKENS_LADDER:-}"
}

# Canonical workload staging. Every SKU/backend generates identical canonical array bytes and
# content IDs in-container; the NPZ container bytes themselves are not an identity boundary. When CX_CANONICAL=1
# (and CX_WORKLOAD_DIR not already provided) we generate routing traces for the run's ladder
# into a NON-results dir (.cx_workloads/ — so the *.manifest.json never pollute the results glob) and
# point run_ep at it. Raw attempts remain diagnostic until the publisher validates full coverage.
cx_stage_canonical() {
  cx_bool_enabled "${CX_CANONICAL:-0}" || return 0
  [ -n "${CX_WORKLOAD_DIR:-}" ] && return 0
  local dir="$PWD/.cx_workloads"
  local ladder; ladder="$(cx_ep_ladder)"
  # cover both phase ladders when none is given, so either phase finds its files.
  [ -z "$ladder" ] && ladder="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
  cx_log "staging canonical workloads (routing=${CX_ROUTING:-uniform} ep=$CX_NGPUS ladder='$ladder')"
  python3 tests/make_workloads.py --out-dir "$dir" --routing "${CX_ROUTING:-uniform}" \
    --ep "$CX_NGPUS" --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" \
    --experts "${CX_EXPERTS:-256}" --seed "${CX_SEED:-67}" --tokens-ladder "$ladder" \
    || { cx_log "ERROR: canonical workload staging failed"; return 1; }
  export CX_WORKLOAD_DIR="$dir"
  cx_log "canonical workloads staged at $dir"
}

# run_ep_suite <backend>
# One tests/run_ep.py invocation per phase (decode/prefill/both); dispatch and
# combine are timed separately inside it. One JSON per (backend, phase).
# Preserve a failed case with its full scheduled identity instead of letting it vanish.
emit_failed_case() {  # backend phase rc
  cx_emit_ep_failed_case \
    "results/failed_${CX_RUNNER}_${1}_${2}_${CX_TS}.json" "$1" "$2" "$3" || true
}

run_ep_suite() {
    local backend="$1" phase phases ladder failure_kind rc=0 rc_run
  ladder="$(cx_ep_ladder)"
  phases="${CX_PHASE:-decode}"
  [ "$phases" = "both" ] && phases="decode prefill"
  if ! cx_stage_canonical; then
    for phase in $phases; do
      emit_failed_case "$backend" "$phase" 2
    done
    return 1
  fi
  for phase in $phases; do
    cx_log "ep backend=$backend phase=$phase ngpus=$CX_NGPUS ladder='${ladder:-<phase-default>}'"
    local out="results/${CX_RUNNER}_${backend}_${phase}_${CX_TS}.json"
    local -a EPARGS=(--backend "$backend" --mode "${CX_MODE:-normal}" --phase "$phase"
      --precision-profile "${CX_PRECISION_PROFILE:-}"
      --tokens-ladder "$ladder"
      --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}"
      --routing "${CX_ROUTING:-uniform}" --seed "${CX_SEED:-67}" --iters "${CX_ITERS:-8}"
      --trials "${CX_TRIALS:-64}" --warmup "${CX_WARMUP:-32}"
      --gpus-per-node "${CX_GPUS_PER_NODE:-0}" --scale-up-domain "${CX_SCALE_UP_DOMAIN:-0}"
      --scope "${CX_SCOPE:-scale-up}" --scale-up-transport "${CX_SCALE_UP_TRANSPORT:-unknown}"
      --scale-out-transport "${CX_SCALE_OUT_TRANSPORT:-}"
      --case-id "${CX_CASE_ID:-}" --suite "${CX_SUITE:-}" --workload-name "${CX_WORKLOAD_NAME:-}"
      --required-publication "${CX_REQUIRED_PUBLICATION:-}"
      --qualification-index "${CX_QUALIFICATION_INDEX:-1}"
      --runner "$CX_RUNNER" --topology-class "$CX_TOPO" --transport "$CX_TRANSPORT"
      --out "$out")
    cx_bool_enabled "${CX_EPLB:-0}" && EPARGS+=(--eplb)
    [ -n "${CX_WORKLOAD_DIR:-}" ] && EPARGS+=(--workload-dir "$CX_WORKLOAD_DIR")
    cx_write_runtime_stage execution || cx_die "cannot record runtime stage"
    if timeout -k 30 "${CX_RUN_TIMEOUT:-900}" \
      torchrun --nproc_per_node="$CX_NGPUS" tests/run_ep.py "${EPARGS[@]}"; then
      rc_run=0
    else
      rc_run=$?
    fi
    if [ "$rc_run" = 0 ] && cx_result_doc_is "$out" invalid; then
      cx_log "WARN: $backend $phase completed with invalid semantic evidence"
      rc=1
      continue
    fi
    if [ "$rc_run" = 0 ] && ! cx_result_doc_is "$out" success; then
      rc_run=1
    fi
    if [ "$rc_run" != 0 ]; then
      failure_kind=failed
      [ "$rc_run" != 124 ] && [ "$rc_run" != 137 ] || failure_kind="timed out"
      if [ "$failure_kind" = "timed out" ]; then
        cx_log "WARN: $backend $phase run timed out rc=$rc_run (limit=${CX_RUN_TIMEOUT:-900}s)"
      else
        cx_log "WARN: $backend $phase run failed rc=$rc_run"
      fi
      if cx_has_result_doc "$out"; then
        cx_demote_result_doc "$out" "$rc_run" \
          || { cx_quarantine_result_doc "$out"; emit_failed_case "$backend" "$phase" "$rc_run"; }
        cx_log "preserved benchmark output as a failed attempt"
      else
        cx_quarantine_result_doc "$out"
        emit_failed_case "$backend" "$phase" "$rc_run"
      fi
      rc=1
    fi
  done
  return "$rc"
}

# Resolve and verify the actual CUDA target before compiling source kernels.
cx_cuda_arch() {
  local expected detected
  case "$CX_RUNNER" in
    h100*|h200*) expected="9.0" ;;
    b200*|gb200*) expected="10.0" ;;
    b300*|gb300*) expected="10.3" ;;
    *) cx_log "ERROR: no CUDA target registered for $CX_RUNNER"; return 1 ;;
  esac
  detected="$(python3 - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
print(f"{major}.{minor}")
PY
)" || return 1
  [ "$detected" = "$expected" ] || {
    cx_log "ERROR: $CX_RUNNER expected CUDA target $expected, detected $detected"
    return 1
  }
  printf '%s' "$detected"
}

cx_nvidia_package_root() {
  local package="$1" component="$2"
  python3 - "$package" "$component" <<'PY'
from importlib import metadata
from pathlib import Path, PurePosixPath
import sys

package, component = sys.argv[1:]
try:
    distribution = metadata.distribution(package)
    prefix = f"nvidia/{component}/"
    entries = [str(entry).replace("\\", "/") for entry in distribution.files or ()]
    if not any(entry.startswith(prefix) for entry in entries):
        raise ValueError
    root = Path(distribution.locate_file(PurePosixPath("nvidia") / component)).resolve()
    if not root.is_dir():
        raise ValueError
except (metadata.PackageNotFoundError, OSError, TypeError, ValueError):
    raise SystemExit(1)
print(root, end="")
PY
}

cx_prepare_cuda_cccl() {
  local cccl="" candidate cuda_home nvcc
  nvcc="$(command -v nvcc)" \
    || { cx_log "ERROR: CUDA nvcc is unavailable"; return 1; }
  nvcc="$(readlink -f -- "$nvcc")" \
    || { cx_log "ERROR: CUDA nvcc cannot be resolved"; return 1; }
  case "$nvcc" in
    */bin/nvcc) cuda_home="${nvcc%/bin/nvcc}" ;;
    *) cx_log "ERROR: CUDA nvcc has an unexpected path"; return 1 ;;
  esac
  [ -x "$cuda_home/bin/nvcc" ] && [ -d "$cuda_home/include" ] \
    && [ -d "$cuda_home/lib64" ] \
    || { cx_log "ERROR: CUDA toolkit root is incomplete"; return 1; }
  for candidate in "$cuda_home"/targets/*/include/cccl; do
    if [ -d "$candidate" ]; then
      cccl="$candidate"
      break
    fi
  done
  [ -n "$cccl" ] || { cx_log "ERROR: CUDA CCCL headers are unavailable"; return 1; }
  export CUDA_HOME="$cuda_home" CX_CUDA_CCCL="$cccl"
  export CPATH="$cccl:${CPATH:-}"
  export NVCC_PREPEND_FLAGS="-I$cccl ${NVCC_PREPEND_FLAGS:-}"
}

cx_prepare_deepep_toolchain() {
  local packaged overlay path root temporary
  packaged="$(cx_nvidia_package_root nvidia-nvshmem-cu12 nvshmem)" \
    || { cx_log "ERROR: nvidia.nvshmem is unavailable"; return 1; }
  root="$(cx_deepep_v2_root)" || return 1
  overlay="$root/nvshmem-overlay"
  if ! (
    umask 077
    exec 8>"$root/nvshmem-overlay.lock" || exit 1
    flock 8 || exit 1
    if [ ! -d "$overlay" ]; then
      temporary="$root/.nvshmem-overlay.$$"
      rm -rf "$temporary" || exit 1
      mkdir -p "$temporary/lib" || exit 1
      ln -s "$packaged/include" "$temporary/include" || exit 1
      for path in "$packaged"/lib/*; do
        ln -s "$path" "$temporary/lib/${path##*/}" || exit 1
      done
      [ ! -e "$packaged/lib/libnvshmem_host.so.3" ] \
        || ln -sf "$packaged/lib/libnvshmem_host.so.3" \
          "$temporary/lib/libnvshmem_host.so" || exit 1
      mv "$temporary" "$overlay" || exit 1
    fi
    [ ! -L "$overlay" ] \
      && [ "$(readlink -f "$overlay/include")" = "$(readlink -f "$packaged/include")" ] \
      && [ -e "$overlay/lib/libnvshmem_host.so" ] \
      && [ -e "$overlay/lib/libnvshmem_device.a" ]
  ); then
    cx_log "ERROR: DeepEP V2 NVSHMEM overlay is invalid"
    return 1
  fi
  NVSHMEM_DIR="$overlay"
  export NVSHMEM_DIR
  cx_prepare_cuda_cccl || return 1
  export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
}

cx_probe_deepep() {
  local expected_record_sha256 expected_version expected_wheel_sha256
  if [ "${COLLECTIVEX_IMAGE:-}" != "$CX_IMAGE_MULTIARCH" ] \
      || [ "${COLLECTIVEX_IMAGE_DIGEST:-}" != "$CX_IMAGE_MULTIARCH_DIGEST" ] \
      || [ "${COLLECTIVEX_IMAGE_DIGEST_VERIFIED:-0}" != 1 ]; then
    cx_log "ERROR: DeepEP V1 requires the exact pinned multi-architecture image"
    return 1
  fi
  cx_cuda_arch >/dev/null || return 1
  case "$CX_RUNNER" in
    gb200|gb300)
      expected_version="1.1.0+814e508"
      expected_wheel_sha256="784dabec0877b6cf72619b7e93eda7e2f365648487bd37fc3ff6960e53669313"
      expected_record_sha256="2671cff7baf8c2c214ff4bac721af875d513130670bec57601998bd1aae82882"
      DEEPEP_COMMIT="814e508537c6ffc775d59f6f1b9ba43f3a65968c"
      ;;
    *)
      expected_version="1.2.1"
      expected_wheel_sha256="7c02c29306ea0fe2dd474618e72e0f310f260187a9c0700a656d2f6964e8c307"
      expected_record_sha256="6548e9c504a12b2471af4b7f4d9546321210a57a456b5dc55bd4a8dad0f932ac"
      DEEPEP_COMMIT="9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee"
      ;;
  esac
  export DEEPEP_COMMIT
  python3 - "$expected_version" "$expected_wheel_sha256" "$expected_record_sha256" <<'PY' || {
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import sys

import deep_ep
from deep_ep import Buffer

distribution = metadata.distribution("deep_ep")
assert distribution.version == sys.argv[1]
assert Buffer.__name__ == "Buffer"
recorded_files = {
    Path(distribution.locate_file(entry)).resolve() for entry in distribution.files or ()
}
buffer_module = sys.modules.get(Buffer.__module__)
assert Path(deep_ep.__file__).resolve() in recorded_files
assert buffer_module is not None and Path(buffer_module.__file__).resolve() in recorded_files
direct_url = json.loads(distribution.read_text("direct_url.json"))
assert direct_url["archive_info"]["hashes"]["sha256"] == sys.argv[2]
record_entry = next(
    entry for entry in distribution.files or ()
    if str(entry).endswith(".dist-info/RECORD")
)
record = distribution.locate_file(record_entry).read_bytes()
assert hashlib.sha256(record).hexdigest() == sys.argv[3]
for path, encoded_digest, size in csv.reader(io.StringIO(record.decode())):
    if not encoded_digest:
        continue
    algorithm, expected = encoded_digest.split("=", 1)
    assert algorithm == "sha256"
    payload = distribution.locate_file(path).read_bytes()
    observed = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
    assert observed == expected
    assert not size or len(payload) == int(size)
PY
    cx_log "ERROR: container DeepEP build does not match its pinned image contract"
    return 1
  }
  cx_log "DeepEP image build ready ($DEEPEP_COMMIT)"
}

# DeepEP V2 is PR #605's ElasticBuffer implementation with upstream PR #630's pure scale-up
# initialization fix and PR #640's exact libnccl mapping check. Canonical launchers stage the
# pinned source and mount a private cluster-local build cache at /cx-cache.
cx_deepep_v2_root() {
  local arch cpu base identity key image_digest
  arch="$(cx_cuda_arch)" || return 1
  cpu="$(uname -m)"
  [[ "$cpu" =~ ^[A-Za-z0-9._-]+$ ]] || return 1
  base="${CX_BACKEND_CACHE_ROOT:-}"
  [[ "$base" = /* ]] || return 1
  image_digest="${COLLECTIVEX_IMAGE_DIGEST:-manual-unverified}"
  [[ "$image_digest" = manual-unverified || "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || return 1
  # Bump the recipe generation whenever the build procedure changes. Benchmark-only
  # source revisions must reuse the same immutable environment instead of leaking GBs.
  identity="deepep-v2-cache-v3|$cpu|sm${arch/./}|image=$image_digest|recipe=aot-persistent-nvshmem-active-cuda-maxjobs16-v3|$CX_DEEPEP_V2_COMMIT|$CX_DEEPEP_V2_TREE|$CX_DEEPEP_V2_FMT_COMMIT|$CX_DEEPEP_V2_NCCL_CHECK_COMMIT|pip=26.1.2|setuptools=82.0.1|wheel=0.47.0|ninja=1.13.0|numpy=2.2.6|torch=2.10.0+cu130|nccl=2.30.4|nvshmem=3.3.9|max-jobs=16"
  key="$(printf '%s' "$identity" | sha256sum | awk '{print $1}')"
  [[ "$key" =~ ^[0-9a-f]{64}$ ]] || return 1
  printf '%s/deepep-v2-%s' "$base" "$key"
}

cx_activate_deepep_v2() {
  local root venv stage_root
  root="$(cx_deepep_v2_root)" || return 1
  venv="$root/venv"
  [ -x "$venv/bin/python" ] \
    || { cx_log "ERROR: DeepEP V2 venv interpreter is unavailable"; return 1; }
  export VIRTUAL_ENV="$venv"
  export PATH="$venv/bin:${PATH#"$venv/bin:"}"
  EP_NCCL_ROOT_DIR="$(cx_nvidia_package_root nvidia-nccl-cu13 nccl)" \
    || { cx_log "ERROR: DeepEP V2 NCCL package root is unavailable"; return 1; }
  EP_NVSHMEM_ROOT_DIR="$(cx_nvidia_package_root nvidia-nvshmem-cu12 nvshmem)" \
    || { cx_log "ERROR: DeepEP V2 NVSHMEM package root is unavailable"; return 1; }
  export EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR
  export LD_LIBRARY_PATH="$EP_NCCL_ROOT_DIR/lib:$EP_NVSHMEM_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"
  case "${CX_BACKEND_SOURCE_ROOT:-}" in
    /*/.cx_sources) stage_root="${CX_BACKEND_SOURCE_ROOT%/.cx_sources}" ;;
    *) cx_log "ERROR: DeepEP V2 job-local source root is unavailable"; return 1 ;;
  esac
  [ -d "$stage_root" ] && [ ! -L "$stage_root" ] \
    || { cx_log "ERROR: DeepEP V2 job-local stage is invalid"; return 1; }
  # JIT CUBINs are evidence from this shard, not part of the persistent AOT environment.
  # Keeping them on the isolated staged tree prevents a prior driver/topology attempt
  # from seeding a later run; all ranks and cases in this shard still share one cold build.
  export EP_JIT_CACHE_DIR="$stage_root/.cx_backend/deepep-v2-jit"
  export EP_REUSE_NCCL_COMM=1
  export DEEPEP_V2_PR=605 DEEPEP_V2_FIX_PR=630 DEEPEP_V2_NCCL_CHECK_FIX_PR=640
  DEEPEP_V2_COMMIT="$CX_DEEPEP_V2_COMMIT"
  DEEPEP_V2_TREE="$CX_DEEPEP_V2_TREE"
  DEEPEP_V2_FMT_COMMIT="$CX_DEEPEP_V2_FMT_COMMIT"
  DEEPEP_V2_NCCL_CHECK_COMMIT="$CX_DEEPEP_V2_NCCL_CHECK_COMMIT"
  export DEEPEP_V2_COMMIT DEEPEP_V2_TREE DEEPEP_V2_FMT_COMMIT
  export DEEPEP_V2_NCCL_CHECK_COMMIT
  [ ! -L "$stage_root/.cx_backend" ] && [ ! -L "$EP_JIT_CACHE_DIR" ] \
    || { cx_log "ERROR: DeepEP V2 JIT cache path is unsafe"; return 1; }
  if ! mkdir -p "$EP_JIT_CACHE_DIR" \
      || ! chmod 700 "$stage_root/.cx_backend" "$EP_JIT_CACHE_DIR"; then
    cx_log "ERROR: DeepEP V2 JIT cache is unavailable"
    return 1
  fi
  unset EP_SUPPRESS_NCCL_CHECK
}

cx_enable_deepep_v2_jit_reproducibility() {
  local seed="collectivex-deepep-v2-fa8a9b1" cccl
  [ -n "${CUDA_HOME:-}" ] \
    || { cx_log "ERROR: active CUDA toolkit is unavailable"; return 1; }
  cccl="${CX_CUDA_CCCL:-}"
  case "$cccl" in
    "$CUDA_HOME"/targets/*/include/cccl) ;;
    *) cx_log "ERROR: CUDA CCCL headers differ from the active toolkit"; return 1 ;;
  esac
  [ -d "$cccl" ] || { cx_log "ERROR: CUDA CCCL headers are unavailable"; return 1; }
  CPATH="$cccl"
  NVCC_PREPEND_FLAGS="--frandom-seed=$seed -I$cccl"
  DEEPEP_V2_JIT_RANDOM_SEED="$seed"
  EP_JIT_DUMP_SASS=1
  unset EP_JIT_DEBUG EP_JIT_DUMP_ASM EP_JIT_DUMP_PTX EP_JIT_WITH_LINEINFO
  unset EP_JIT_PTXAS_VERBOSE EP_JIT_PRINT_COMPILER_COMMAND EP_JIT_NVCC_COMPILER
  unset EP_JIT_CPP_STANDARD EP_JIT_PTXAS_CHECK EP_GIN_GDAKI_DEBUG EP_NUM_TOPK_IDX_BITS
  export CPATH DEEPEP_V2_JIT_RANDOM_SEED EP_JIT_DUMP_SASS NVCC_PREPEND_FLAGS
}

cx_probe_deepep_v2() {
  python3 - <<'PY'
import ctypes
import importlib.metadata as metadata
import inspect
import os

import torch

assert torch.__version__ == "2.10.0+cu130", torch.__version__
assert metadata.version("nvidia-nccl-cu13") == "2.30.4"
assert metadata.version("nvidia-nvshmem-cu12") == "3.3.9"
assert metadata.version("numpy") == "2.2.6"

import deep_ep
assert deep_ep.__version__ == "2.0.0", deep_ep.__version__
assert metadata.version("deep_ep") in {"2.0.0+fa8a9b1", "2.0.0+local"}, metadata.version("deep_ep")
assert inspect.isclass(deep_ep.ElasticBuffer)
assert deep_ep.ElasticBuffer.__name__ == "ElasticBuffer"
assert os.environ.get("EP_SUPPRESS_NCCL_CHECK") is None
with open("/proc/self/maps", encoding="utf-8") as handle:
    loaded_nccl = {
        os.path.realpath(line.rstrip().split()[-1])
        for line in handle
        if "libnccl.so" in line and os.path.isfile(line.rstrip().split()[-1])
    }
assert len(loaded_nccl) == 1
runtime_version = ctypes.c_int()
assert ctypes.CDLL(loaded_nccl.pop()).ncclGetVersion(ctypes.byref(runtime_version)) == 0
assert runtime_version.value == 23004, runtime_version.value
PY
}

cx_deepep_v2_content_sha256() {
  python3 - <<'PY'
import hashlib
from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import stat

distribution = metadata.distribution("deep_ep")
entries = sorted(distribution.files or (), key=lambda entry: entry.as_posix())
if not entries:
    raise SystemExit(1)
venv_path = Path(os.environ["VIRTUAL_ENV"]).absolute()
if venv_path.is_symlink() or not venv_path.is_dir():
    raise SystemExit(1)
venv = venv_path.resolve(strict=True)
digest = hashlib.sha256()
extension = False
for entry in entries:
    relative = PurePosixPath(entry.as_posix())
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or not (
            relative.parts[0] == "deep_ep"
            or relative.parts[0].startswith("deep_ep-")
            and relative.parts[0].endswith(".dist-info")
        )
    ):
        raise SystemExit(1)
    path = Path(distribution.locate_file(entry)).absolute()
    resolved = path.resolve(strict=True)
    try:
        path.relative_to(venv_path)
        resolved.relative_to(venv)
    except ValueError:
        raise SystemExit(1)
    parent = path.parent
    while parent != venv_path:
        if parent.is_symlink():
            raise SystemExit(1)
        parent = parent.parent
    item = os.lstat(path)
    if not stat.S_ISREG(item.st_mode):
        raise SystemExit(1)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (item.st_dev, item.st_ino):
            raise SystemExit(1)
        file_digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            file_digest.update(chunk)
    finally:
        os.close(descriptor)
    name = relative.as_posix()
    extension |= name.startswith("deep_ep/") and name.endswith(".so")
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(str(item.st_size).encode())
    digest.update(b"\0")
    digest.update(file_digest.digest())
if not extension:
    raise SystemExit(1)
print(digest.hexdigest(), end="")
PY
}

cx_deepep_v2_marker_content_sha256() {
  local root="$1" marker="$2" revision="$3" tree="$4" fmt_revision="$5" cache_key="$6"
  python3 - "$root" "$marker" "$revision" "$tree" "$fmt_revision" "$cache_key" <<'PY'
import os
import re
import stat
import sys

root, marker, revision, tree, fmt_revision, cache_key = sys.argv[1:]
try:
    root_item = os.lstat(root)
    marker_item = os.lstat(marker)
    children = [os.lstat(os.path.join(root, name)) for name in ("source", "venv")]
    if (
        not stat.S_ISDIR(root_item.st_mode)
        or stat.S_IMODE(root_item.st_mode) & 0o777 != 0o700
        or not stat.S_ISREG(marker_item.st_mode)
        or marker_item.st_uid != root_item.st_uid
        or stat.S_IMODE(marker_item.st_mode) & 0o777 != 0o600
        or marker_item.st_size > 1024
        or any(
            not stat.S_ISDIR(child.st_mode)
            or child.st_uid != root_item.st_uid
            or stat.S_IMODE(child.st_mode) & 0o022
            for child in children
        )
    ):
        raise OSError
    descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (marker_item.st_dev, marker_item.st_ino):
            raise OSError
        payload = os.read(descriptor, 1025)
    finally:
        os.close(descriptor)
    lines = payload.decode("ascii").splitlines()
    if lines[:4] != [revision, tree, fmt_revision, cache_key] or len(lines) != 5:
        raise ValueError
    if not re.fullmatch(r"[0-9a-f]{64}", lines[4]):
        raise ValueError
except (OSError, UnicodeError, ValueError):
    raise SystemExit(1)
print(lines[4], end="")
PY
}

cx_deepep_v2_cache_is_valid() {
  local root="$1" marker="$2" revision="$3" tree="$4" fmt_revision="$5" cache_key="$6"
  local expected_content actual_content
  expected_content="$(
    cx_deepep_v2_marker_content_sha256 \
      "$root" "$marker" "$revision" "$tree" "$fmt_revision" "$cache_key"
  )" || return 1
  [ -d "$root/source" ] && [ ! -L "$root/source" ] \
    && [ "$(cx_git_in_tree "$root/source" rev-parse 'HEAD^{tree}' 2>/dev/null)" = "$tree" ] \
    && [ "$(cx_git_in_tree "$root/source/third-party/fmt" rev-parse HEAD 2>/dev/null)" = "$fmt_revision" ] \
    || return 1
  cx_activate_deepep_v2 || return 1
  actual_content="$(cx_deepep_v2_content_sha256)" || return 1
  [ "$actual_content" = "$expected_content" ]
}

cx_build_deepep_v2() {
  local root venv source marker marker_tmp lock_path arch cache_key cache_ready content_sha256
  local revision="fa8a9b16898204afd347c663b89e65ef87dc6ce6"
  local tree="29809e75c5874e6609dac4804e7b651d5226959f"
  local fmt_revision="a4c7e17133ee9cb6a2f45545f6e974dd3c393efa"
  cx_verify_backend_cache_mount \
    || { cx_log "ERROR: DeepEP V2 cache mount identity validation failed"; return 1; }
  arch="$(cx_cuda_arch)" || return 1
  root="$(cx_deepep_v2_root)" || return 1
  cache_key="${root##*/deepep-v2-}"
  [[ "$cache_key" =~ ^[0-9a-f]{64}$ ]] || return 1
  venv="$root/venv"; source="$root/source"; marker="$root/.collectivex-complete"
  lock_path="${root}.lock"
  command -v flock >/dev/null || { cx_log "ERROR: flock is required for DeepEP V2"; return 1; }
  mkdir -p "${root%/*}" || return 1
  cx_log "DeepEP V2: preparing PR #605 with upstream PR #630 and #640 fixes ($revision)"
  if ! (
    [ ! -L "$lock_path" ] \
      || { cx_log "ERROR: DeepEP V2 cache lock is unsafe"; exit 1; }
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" \
      || { cx_log "ERROR: DeepEP V2 cache-lock-create failed"; exit 1; }
    exec 9<>"$lock_path" \
      || { cx_log "ERROR: DeepEP V2 cache-lock-open failed"; exit 1; }
    flock 9 \
      || { cx_log "ERROR: DeepEP V2 cache-lock-acquire failed"; exit 1; }
    cache_ready=0
    if [ -e "$marker" ] || [ -L "$marker" ]; then
      if (
        cx_deepep_v2_cache_is_valid \
          "$root" "$marker" "$revision" "$tree" "$fmt_revision" "$cache_key"
      ); then
        cache_ready=1
      else
        cx_log "ERROR: published DeepEP V2 cache failed integrity validation; refusing reset"
        exit 1
      fi
    fi
    if [ "$cache_ready" != 1 ]; then
      if [ -e "$root" ] || [ -L "$root" ]; then
        rm -rf "$root" \
          || { cx_log "ERROR: incomplete DeepEP V2 cache-reset failed"; exit 1; }
      fi
      mkdir -m 700 "$root" \
        || { cx_log "ERROR: DeepEP V2 cache-create failed"; exit 1; }
      python3 -m venv "$venv" \
        || { cx_log "ERROR: DeepEP V2 venv creation failed"; exit 1; }
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        "pip==26.1.2" "setuptools==82.0.1" "wheel==0.47.0" "ninja==1.13.0" \
        "numpy==2.2.6" "nvidia-nvshmem-cu12==3.3.9" >&2 2>&1 \
        || { cx_log "ERROR: DeepEP V2 build-tool installation failed"; exit 1; }
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple "torch==2.10.0" >&2 2>&1 \
        || { cx_log "ERROR: torch 2.10.0+cu130 installation failed"; exit 1; }
      # Torch pins NCCL 2.28.9; the PR #605 ElasticBuffer implementation requires 2.30.4.
      "$venv/bin/python" -m pip install -q --disable-pip-version-check --no-input \
        --force-reinstall --no-deps "nvidia-nccl-cu13==2.30.4" >&2 2>&1 \
        || { cx_log "ERROR: NCCL 2.30.4 installation failed"; exit 1; }
      cx_activate_deepep_v2 \
        || { cx_log "ERROR: DeepEP V2 environment activation failed"; exit 1; }
      cx_prepare_deepep_toolchain \
        || { cx_log "ERROR: DeepEP V2 toolchain preparation failed"; exit 1; }
      EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
      export EP_NVSHMEM_ROOT_DIR
      cx_materialize_backend_source deepep-v2 "$source" \
        || { cx_log "ERROR: DeepEP V2 staged source is invalid"; exit 1; }
      (cd "$source" && SOURCE_DATE_EPOCH="$(cx_git_in_tree "$source" show -s --format=%ct HEAD)" \
        TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
        python3 -m pip install -q --no-build-isolation --no-deps --force-reinstall .) >&2 2>&1 \
        || { cx_log "ERROR: DeepEP V2 build failed"; exit 1; }
      cx_probe_deepep_v2 \
        || { cx_log "ERROR: DeepEP V2 ElasticBuffer/runtime probe failed"; exit 1; }
      content_sha256="$(cx_deepep_v2_content_sha256)" \
        || { cx_log "ERROR: DeepEP V2 installed-content hashing failed"; exit 1; }
      marker_tmp="$(mktemp "$root/.collectivex-complete.tmp.XXXXXX")" \
        || { cx_log "ERROR: DeepEP V2 cache-marker-create failed"; exit 1; }
      chmod 600 "$marker_tmp" \
        || { cx_log "ERROR: DeepEP V2 cache-marker-permission failed"; exit 1; }
      printf '%s\n%s\n%s\n%s\n%s\n' \
        "$revision" "$tree" "$fmt_revision" "$cache_key" "$content_sha256" > "$marker_tmp" \
        || { cx_log "ERROR: DeepEP V2 cache-marker-write failed"; exit 1; }
      mv -f -- "$marker_tmp" "$marker" \
        || { cx_log "ERROR: DeepEP V2 cache-marker-publish failed"; exit 1; }
    fi
    cx_deepep_v2_cache_is_valid \
      "$root" "$marker" "$revision" "$tree" "$fmt_revision" "$cache_key" \
      || { cx_log "ERROR: DeepEP V2 cache validation failed"; exit 1; }
  ); then
    cx_log "ERROR: shared DeepEP V2 environment is incomplete"
    return 1
  fi
  cx_activate_deepep_v2 || return 1
  cx_prepare_deepep_toolchain || return 1
  cx_enable_deepep_v2_jit_reproducibility || return 1
  EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
  export EP_NVSHMEM_ROOT_DIR
  cx_probe_deepep_v2 || { cx_log "ERROR: DeepEP V2 shared runtime probe failed"; return 1; }
  cx_log "DeepEP V2 ready ($DEEPEP_V2_COMMIT, ElasticBuffer, NCCL Device API; LSA/Gin selected by adapter)"
}

# Build the pinned DeepEP `hybrid-ep` implementation. MNNVL remains one scale-up
# domain; true x86 scale-out uses the upstream DOCA/RDMA build explicitly.
cx_configure_deepep_hybrid_build() {
  local interface device rdma_name
  local -a interfaces devices
  unset HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE
  if [ "${CX_NODES:-1}" -le 1 ] || [ "${CX_TRANSPORT:-}" = mnnvl ]; then
    export DEEPEP_HYBRID_BUILD_MODE=intradomain
    return 0
  fi
  [ "$(uname -m)" = x86_64 ] \
    || { cx_log "ERROR: hybrid-ep scale-out is registered only on x86_64"; return 1; }
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${NCCL_IB_HCA:-}" ] \
    || { cx_log "ERROR: hybrid-ep scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { cx_log "ERROR: configured hybrid-ep socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$NCCL_IB_HCA"
  for device in "${devices[@]}"; do
    rdma_name="$(cx_nccl_hca_device_name "$device")"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { cx_log "ERROR: configured hybrid-ep RDMA device is absent"; return 1; }
  done
  command -v make >/dev/null \
    || { cx_log "ERROR: make is required for hybrid-ep scale-out"; return 1; }
  [ -r /usr/include/infiniband/verbs.h ] && [ -r /usr/include/infiniband/mlx5dv.h ] \
    || { cx_log "ERROR: pinned hybrid-ep RDMA headers are unavailable"; return 1; }
  python3 - <<'PY' >/dev/null 2>&1 || {
import ctypes.util
import sys
sys.exit(0 if all(ctypes.util.find_library(name) for name in ("ibverbs", "mlx5")) else 1)
PY
    cx_log "ERROR: pinned hybrid-ep RDMA libraries are unavailable"
    return 1
  }
  export HYBRID_EP_MULTINODE=1 USE_NIXL=0 RDMA_CORE_HOME=/usr
  export DEEPEP_HYBRID_BUILD_MODE=multinode-doca
}

cx_deepep_hybrid_marker_content_sha256() {
  python3 - "$1" "$2" "$3" "$4" "${5:-}" <<'PY'
import os
import re
import stat
import sys

root, marker, revision, tree, build_mode = sys.argv[1:]
try:
    root_item = os.lstat(root)
    marker_item = os.lstat(marker)
    if (
        not stat.S_ISDIR(root_item.st_mode)
        or stat.S_IMODE(root_item.st_mode) & 0o777 != 0o700
        or not stat.S_ISREG(marker_item.st_mode)
        or marker_item.st_uid != root_item.st_uid
        or stat.S_IMODE(marker_item.st_mode) & 0o777 != 0o600
        or marker_item.st_size > 512
    ):
        raise OSError
    descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (marker_item.st_dev, marker_item.st_ino):
            raise OSError
        payload = os.read(descriptor, 513)
    finally:
        os.close(descriptor)
    lines = payload.decode("ascii").splitlines()
    expected = [revision, tree, build_mode] if build_mode else [revision, tree]
    if len(lines) != len(expected) + 1 or lines[:-1] != expected:
        raise ValueError
    if not re.fullmatch(r"[0-9a-f]{64}", lines[-1]):
        raise ValueError
except (OSError, UnicodeError, ValueError):
    raise SystemExit(1)
print(lines[-1], end="")
PY
}

cx_deepep_hybrid_cache_is_valid() {
  local root="$1" marker="$2" revision="$3" tree="$4" build_mode="${5:-}"
  local expected actual status extra
  expected="$(cx_deepep_hybrid_marker_content_sha256 \
    "$root" "$marker" "$revision" "$tree" "$build_mode")" || return 1
  [ "$(cx_git_in_tree "$root" rev-parse HEAD 2>/dev/null)" = "$revision" ] \
    && [ "$(cx_git_in_tree "$root" rev-parse 'HEAD^{tree}' 2>/dev/null)" = "$tree" ] \
    || return 1
  status="$(cx_git_in_tree "$root" status --porcelain --untracked-files=no \
    --ignore-submodules=none 2>/dev/null)" || return 1
  [ -z "$status" ] || return 1
  extra="$(cx_git_in_tree "$root" ls-files --others --exclude-standard -- \
    'deep_ep/*.py' 'deep_ep/*.so' 2>/dev/null)" || return 1
  [ -z "$extra" ] || return 1
  extra="$(cx_git_in_tree "$root" ls-files --others --ignored --exclude-standard -- \
    'deep_ep/*.py' 'deep_ep/*.so' 2>/dev/null)" || return 1
  [ -z "$extra" ] || return 1
  actual="$(cx_extension_pair_sha256 "$root" 'deep_ep_cpp*.so' 'hybrid_ep_cpp*.so')" \
    || return 1
  [ "$actual" = "$expected" ]
}

cx_build_deepep_hybrid() {
  local arch revision="$CX_DEEPEP_HYBRID_COMMIT" tree="$CX_DEEPEP_HYBRID_TREE"
  local build_root marker marker_tmp lock_path content_sha256 cache_ready build_mode
  export DEEPEP_COMMIT="$revision" DEEPEP_TREE="$tree"
  arch="$(cx_cuda_arch)" || return 1
  cx_configure_deepep_hybrid_build || return 1
  build_mode="$DEEPEP_HYBRID_BUILD_MODE"
  build_root="$PWD/.cx_backend/deepep-hybrid-${arch/./}-${build_mode}"
  marker="$build_root/.collectivex-complete"
  lock_path="${build_root}.lock"
  cx_log "DeepEP hybrid-ep: building $revision for CUDA target $arch"
  unset NVSHMEM_DIR
  cx_prepare_cuda_cccl || return 1
  command -v flock >/dev/null || { cx_log "ERROR: flock is required for hybrid-ep"; return 1; }
  mkdir -p "$PWD/.cx_backend" || return 1
  if ! (
    [ ! -L "$lock_path" ] || exit 1
    (umask 077; : >> "$lock_path") && chmod 600 "$lock_path" || exit 1
    exec 9<>"$lock_path" || exit 1
    flock 9 || exit 1
    cache_ready=0
    if [ -e "$marker" ] || [ -L "$marker" ]; then
      cx_deepep_hybrid_cache_is_valid \
        "$build_root" "$marker" "$revision" "$tree" "$build_mode" \
        || exit 1
      cache_ready=1
    fi
    if [ "$cache_ready" != 1 ]; then
      cx_materialize_backend_source deepep-hybrid "$build_root" \
        || { cx_log "ERROR: hybrid-ep staged source is invalid"; exit 1; }
      if [ "$build_mode" = multinode-doca ]; then
        [ "$(cx_git_in_tree "$build_root/third-party/nccl" rev-parse HEAD 2>/dev/null)" \
          = "$CX_DEEPEP_HYBRID_NCCL_COMMIT" ] \
          || { cx_log "ERROR: pinned hybrid-ep NCCL transport source is absent"; exit 1; }
      fi
      (cd "$build_root" && \
        SOURCE_DATE_EPOCH="$(cx_git_in_tree "$build_root" show -s --format=%ct HEAD)" \
        TORCH_CUDA_ARCH_LIST="$arch" MAX_JOBS=16 \
        python3 setup.py build_ext --inplace) >&2 2>&1 \
        || { cx_log "ERROR: hybrid-ep build failed"; exit 1; }
      content_sha256="$(cx_extension_pair_sha256 \
        "$build_root" 'deep_ep_cpp*.so' 'hybrid_ep_cpp*.so')" || exit 1
      marker_tmp="$(mktemp "$build_root/.collectivex-complete.tmp.XXXXXX")" || exit 1
      chmod 600 "$marker_tmp" || exit 1
      printf '%s\n%s\n%s\n%s\n' \
        "$revision" "$tree" "$build_mode" "$content_sha256" > "$marker_tmp" \
        || exit 1
      mv -f -- "$marker_tmp" "$marker" || exit 1
    fi
    cx_deepep_hybrid_cache_is_valid \
      "$build_root" "$marker" "$revision" "$tree" "$build_mode"
  ); then
    cx_log "ERROR: shared hybrid-ep build is incomplete"
    return 1
  fi
  export PYTHONPATH="$build_root:${PYTHONPATH:-}"
  python3 -c "import deep_ep; assert hasattr(deep_ep,'HybridEPBuffer'); print('built hybrid-ep deep_ep', getattr(deep_ep,'__version__','?'))" >&2 \
    || { cx_log "ERROR: hybrid-ep import / HybridEPBuffer missing after build"; return 1; }
  cx_log "DeepEP hybrid-ep ready ($DEEPEP_COMMIT, mode=$build_mode)"
}

# UCCL EP (uccl.ep.Buffer is a DeepEP-API clone). The prebuilt wheel is cu12; on a cu13
# image its kernels need a cu12 CUDA runtime on LD_LIBRARY_PATH (probe-confirmed). PEP-668
# images need PIP_BREAK_SYSTEM_PACKAGES. Best-effort; failure to import fails loudly.
cx_build_uccl() {
  if [ -f /tmp/.cx_built_uccl ]; then
    cx_log "UCCL EP already prepared this allocation — skip rebuild"
    python3 -c "import torch; from uccl_deepep import Buffer" 2>/dev/null || return 1
    return 0
  fi
  local version="0.1.1" tag="v0.1.1" node_id wrapper_root
  local wheel_sha256="390c1320918972206546e44d79b132988f2818ec07e23afcd0595f7183916cec"
  cx_log "UCCL EP: installing uccl==$version + cu12 runtime shim"
  export PIP_BREAK_SYSTEM_PACKAGES=1
  pip install -q --no-deps "sortedcontainers==2.4.0" "intervaltree==3.1.0" >&2 2>&1 \
    || { cx_log "ERROR: UCCL support dependency installation failed"; return 1; }
  printf 'uccl==%s --hash=sha256:%s\n' "$version" "$wheel_sha256" \
    | pip install -q --no-deps --only-binary=:all: --require-hashes -r /dev/stdin >&2 2>&1 \
    || { cx_log "ERROR: pip install uccl==$version failed"; return 1; }
  pip install -q --no-deps "nvidia-cuda-runtime-cu12==12.9.79" >&2 2>&1 \
    || { cx_log "ERROR: CUDA 12 runtime shim install failed"; return 1; }
  local cu12lib
  cu12lib="$(python3 -c "import nvidia.cuda_runtime as m, os; print(os.path.join(os.path.dirname(m.__file__),'lib'))" 2>/dev/null)"
  [ -n "$cu12lib" ] && export LD_LIBRARY_PATH="$cu12lib:${LD_LIBRARY_PATH:-}"
  local installed
  installed="$(python3 -c 'import importlib.metadata as m; print(m.version("uccl"))')" \
    || { cx_log "ERROR: cannot read installed UCCL version"; return 1; }
  [ "$installed" = "$version" ] \
    || { cx_log "ERROR: expected UCCL $version, installed $installed"; return 1; }
  UCCL_COMMIT="pkg-$installed"
  export UCCL_COMMIT
  # import torch FIRST: uccl.ep's C extension links libc10.so (torch), which is only on the loader
  # path once torch is imported (rpath). The adapter (ep_uccl.py) imports torch before uccl.ep too.
  python3 -c "import torch; from uccl.ep import Buffer; print('uccl.ep ready')" >&2 \
    || { cx_log "ERROR: uccl.ep import failed (cu12 runtime on LD_LIBRARY_PATH?)"; return 1; }
  # Vendor UCCL's DeepEP-API wrapper (ep/deep_ep_wrapper/deep_ep) under a NON-conflicting name
  # (uccl_deepep) so it doesn't shadow the container's real deep_ep. Its Buffer(group, num_nvl_bytes,
  # ...) takes a torch ProcessGroup (matching DeepEP + ep_uccl.py's calls) and runs the full
  # proxy/IPC-handle/runtime.sync bootstrap that the low-level uccl.ep.Buffer(rank,num_ranks) lacks.
  case "${SLURM_NODEID:-0}" in ""|*[!0-9]*) return 1 ;; esac
  node_id="${SLURM_NODEID:-0}"
  wrapper_root="$PWD/.cx_backend/uccl-wrapper-node-$node_id"
  case "$wrapper_root" in /ix/experimental/CollectiveX/.cx_backend/uccl-wrapper-node-*) ;; *) return 1 ;; esac
  rm -rf /tmp/uccl_src "$wrapper_root"
  # Pin the wrapper to the SAME tag as the installed wheel (pkg-0.1.1 -> v0.1.1): the wrapper's
  # dispatch calls into uccl.ep (get_rdma_buffer etc.), so a main-branch wrapper vs a 0.1.1 wheel
  # mismatches signatures. Match them.
  if git clone --depth 1 --branch "$tag" https://github.com/uccl-project/uccl /tmp/uccl_src >&2 2>&1 \
     && [ "$(git -C /tmp/uccl_src rev-parse HEAD)" = "73ee4f12ba71717d6de34ba06806e1baaabe3f42" ] \
     && [ -d /tmp/uccl_src/ep/deep_ep_wrapper/deep_ep ]; then
    mkdir -p "$wrapper_root/uccl_deepep"
    chmod 700 "$PWD/.cx_backend" "$wrapper_root" "$wrapper_root/uccl_deepep"
    cp /tmp/uccl_src/ep/deep_ep_wrapper/deep_ep/*.py "$wrapper_root/uccl_deepep/" 2>/dev/null
    export PYTHONPATH="$wrapper_root:${PYTHONPATH:-}"
    python3 -c "import torch; from uccl_deepep import Buffer; print('uccl_deepep wrapper ready')" >&2 \
      || { cx_log "ERROR: uccl_deepep wrapper import failed"; return 1; }
    export CX_UCCL_WRAPPER=1
    export UCCL_WRAPPER_COMMIT="73ee4f12ba71717d6de34ba06806e1baaabe3f42"
  else
    cx_log "ERROR: uccl deep_ep_wrapper not available"
    return 1
  fi
  : > /tmp/.cx_built_uccl
  cx_log "UCCL EP ready ($UCCL_COMMIT, wrapper=${CX_UCCL_WRAPPER:-0})"
}

# Rack build and rank steps may enter different container instances. Persist each node's
# loader/import path and build identity on the shared staged mount, then require it from every rank.
cx_persist_backend_env() {
  local root="$PWD/.cx_backend/env" node_id="${SLURM_NODEID:-0}" path temporary name
  local -a names=(PATH VIRTUAL_ENV LD_LIBRARY_PATH PYTHONPATH CUDA_HOME CPATH NVCC_PREPEND_FLAGS
    NVSHMEM_DIR DEEPEP_COMMIT DEEPEP_TREE
    EP_NCCL_ROOT_DIR EP_NVSHMEM_ROOT_DIR EP_JIT_CACHE_DIR EP_REUSE_NCCL_COMM
    EP_JIT_DUMP_SASS
    DEEPEP_V2_PR DEEPEP_V2_FIX_PR DEEPEP_V2_NCCL_CHECK_FIX_PR DEEPEP_V2_COMMIT
    DEEPEP_V2_TREE DEEPEP_V2_FMT_COMMIT DEEPEP_V2_NCCL_CHECK_COMMIT
    DEEPEP_V2_JIT_RANDOM_SEED
    HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE
    UCCL_COMMIT UCCL_WRAPPER_COMMIT CX_UCCL_WRAPPER)
  [[ "$node_id" =~ ^[0-9]+$ ]] || return 1
  mkdir -p "$root" || return 1
  chmod 700 "$root" || return 1
  temporary="$(mktemp "$root/.node-${node_id}.XXXXXX")" || return 1
  chmod 600 "$temporary" || { rm -f "$temporary"; return 1; }
  for name in "${names[@]}"; do
    if declare -p "$name" >/dev/null 2>&1; then
      printf 'export %s=%q\n' "$name" "${!name}" >> "$temporary" \
        || { rm -f "$temporary"; return 1; }
    fi
  done
  path="$root/node-${node_id}.sh"
  mv -f -- "$temporary" "$path" || { rm -f "$temporary"; return 1; }
}

# Validate private scale-out selectors on every allocated compute node before a
# backend can initialize or build transport code.
cx_probe_scaleout_network() {
  local interface device rdma_name
  local -a interfaces devices
  if [ "${CX_NODES:-1}" -le 1 ] || [ "${CX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  cx_restore_exact_hca_selector || return 1
  [ -n "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${NCCL_IB_HCA:-}" ] \
    || { cx_log "ERROR: scale-out network selectors are unavailable"; return 1; }
  IFS=, read -r -a interfaces <<< "$GLOO_SOCKET_IFNAME"
  for interface in "${interfaces[@]}"; do
    [ -d "/sys/class/net/$interface" ] \
      || { cx_log "ERROR: configured scale-out socket interface is absent"; return 1; }
  done
  IFS=, read -r -a devices <<< "$NCCL_IB_HCA"
  for device in "${devices[@]}"; do
    rdma_name="$(cx_nccl_hca_device_name "$device")"
    [ -d "/sys/class/infiniband/$rdma_name" ] \
      || { cx_log "ERROR: configured scale-out RDMA device is absent"; return 1; }
  done
}

# Prepare and probe one backend without running a benchmark. The same hook is used
# by normal in-container runs and by rack launchers' persistent build-only step.
cx_prepare_backend() {
  local backend="${1:-}"
  case "$backend" in
    deepep)
      cx_probe_deepep || return 1
      ;;
    deepep-v2)
      cx_build_deepep_v2 || return 1
      ;;
    deepep-hybrid)
      cx_build_deepep_hybrid || return 1
      ;;
    uccl)
      cx_build_uccl || return 1
      ;;
    mori)
      python3 -c "import mori" \
        || { cx_log "ERROR: MoRI backend import failed"; return 1; }
      ;;
    nccl-ep)
      ;;
    *)
      cx_log "ERROR: unknown backend preparation request"
      return 1
      ;;
  esac
}

prepare_backend_or_record() {
  local backend="$1" phases="${CX_PHASE:-decode}" phase
  cx_write_runtime_stage backend-setup || return 1
  if cx_prepare_backend "$backend"; then
    return 0
  fi
  cx_log "WARN: $backend preparation failed"
  [ "$phases" = "both" ] && phases="decode prefill"
  for phase in $phases; do
    CX_FAILURE_MODE=backend-setup emit_failed_case "$backend" "$phase" 6
  done
  return 1
}

# dispatch_bench runs the CURRENT CX_BENCH (+ CX_* config env) once. The sweep workflow runs many
# of these per allocation (SHARD mode below), reusing this single container + its built backend.
dispatch_bench() {
  case "$CX_BENCH" in
    nccl-ep)
      run_ep_suite "$CX_BENCH"
      ;;
    deepep|deepep-v2|deepep-hybrid|mori|uccl)
      prepare_backend_or_record "$CX_BENCH" && run_ep_suite "$CX_BENCH"
      ;;
    *)
      cx_die "unknown CX_BENCH=$CX_BENCH (want deepep|deepep-v2|mori|uccl|nccl-ep|deepep-hybrid)"
      ;;
  esac
}

run_precision_probe() {
  local fields probe_id backend sku ep mode profile out rc_run
  fields="$(cx_precision_probe_control_fields "$PWD")" || return 1
  IFS='|' read -r probe_id backend sku ep mode profile <<< "$fields"
  [ "$backend" = "$CX_BENCH" ] && [ "$sku" = "$CX_RUNNER" ] && [ "$ep" = "$CX_NGPUS" ] \
    || { cx_log "ERROR: precision probe control differs from runtime placement"; return 1; }
  out="results/${probe_id}.json"
  cx_write_runtime_stage execution || return 1
  if timeout -k 30 "${CX_RUN_TIMEOUT:-900}" torchrun --nproc_per_node="$CX_NGPUS" \
      tests/probe_precision.py --backend "$backend" --sku "$sku" --ep "$ep" \
      --mode "$mode" --precision-profile "$profile" --out "$out"; then
    rc_run=0
  else
    rc_run=$?
  fi
  [ "$rc_run" = 0 ] || return "$rc_run"
  python3 tests/probe_precision.py --validate-manifest "$out"
}

rc=0
cx_validate_shard_control "$PWD"
cx_load_network_control_mode "$PWD" || cx_die "cannot resolve network control mode"
cx_apply_network_profile "${CX_NODES:-1}" "${CX_TRANSPORT:-}"
# Build-only mode: rack launchers run the shared backend preparation hook once per
# node inside a persistent named container, then direct rank processes reuse it.
if [ -n "${CX_BUILD_ONLY:-}" ]; then
  if cx_probe_scaleout_network && cx_prepare_backend "${CX_BENCH:-}"; then
    cx_persist_backend_env || rc=1
  else
    rc=1
  fi
  cx_log "backend preparation: bench=${CX_BENCH:-unknown} rc=$rc"
  exit "$rc"
fi
if [ "${CX_PRECISION_PROBE:-0}" = 1 ]; then
  if cx_probe_scaleout_network && cx_prepare_backend "${CX_BENCH:-}"; then
    run_precision_probe || rc=$?
  else
    rc=1
  fi
elif [ -n "${CX_SHARD_FILE:-}" ]; then
  # SHARD/SWEEP mode (collectivex-sweep.yml): run EVERY case of this shard in THIS one allocation.
  # All cases share (sku, backend, nodes), so backend preparation is paid once and cached.
  ncases="$(python3 -c "import json;print(len(json.load(open('$CX_SHARD_FILE'))['cases']))")"
  cx_log "SHARD mode: $ncases case(s) in one allocation (shard=$CX_SHARD_FILE)"
  _cx_ts_base="$CX_TS"   # per-case CX_TS suffix below keeps each case's result file UNIQUE (else
                         # cases sharing backend+phase overwrite each other at the same timestamp).
  ci=0
  failed_cases=0
  while [ "$ci" -lt "$ncases" ]; do
    CX_TS="${_cx_ts_base}-c$(printf '%03d' "$ci")"
    export CX_TS
    # Map varying case fields plus the frozen v1 defaults into CX_* env.
    _exports="$(python3 - "$CX_SHARD_FILE" "$ci" <<'PY'
import json, sys, shlex
c = json.load(open(sys.argv[1]))["cases"][int(sys.argv[2])]
def g(k, d=""):
    v = c.get(k, d); return "" if v is None else str(v)
env = {
  "CX_BENCH": g("backend"),
  "CX_MODE": g("mode", "normal"),
  "CX_ROUTING": g("routing", "uniform"), "CX_PHASE": g("phase", "decode"),
  "CX_EP": g("ep", "1"),
  "CX_EPLB": "1" if c.get("eplb") else "",
  "CX_CASE_ID": g("case_id"), "CX_SUITE": g("suite"), "CX_WORKLOAD_NAME": g("workload"),
  "CX_REQUIRED_PUBLICATION": g("required_publication"),
  "CX_PRECISION_PROFILE": g("precision_profile"),
  "CX_HIDDEN": g("hidden"), "CX_TOPK": g("topk"), "CX_EXPERTS": g("experts"),
  "CX_TOKENS_LADDER": g("ladder"), "CX_CANONICAL": ("1" if c.get("canonical") else ""),
  "CX_NODES": g("nodes"), "CX_GPUS_PER_NODE": g("gpus_per_node"),
  "CX_SCALE_UP_DOMAIN": g("scale_up_domain"), "CX_SCOPE": g("scope"),
  "CX_SCALE_UP_TRANSPORT": g("scale_up_transport"),
  "CX_SCALE_OUT_TRANSPORT": g("scale_out_transport"),
  "CX_TRANSPORT": g("transport"), "CX_TOPO": g("topology_class"),
  "CX_SAMPLES_PER_POINT": g("samples_per_point"),
  "CX_WARMUP_SEMANTICS": g("warmup_semantics"),
}
lines = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
# Per-case timing "iters:trials:warmup" (fixed-512-v1 requires 8:64:32 everywhere);
# cases without one must fall back to the harness defaults, so UNSET rather than export-empty
# (an empty CX_ITERS would defeat the 8-iter default and break the run_ep argparse; NOTE no
# apostrophes in this heredoc — bash command-substitution scanning chokes on unbalanced quotes).
timing = g("timing")
if timing:
    parts = (timing.split(":") + ["", "", ""])[:3]
    for k, v in zip(("CX_ITERS", "CX_TRIALS", "CX_WARMUP"), parts):
        if v:
            lines.append(f"export {k}={shlex.quote(v)}")
else:
    lines.append("unset CX_ITERS CX_TRIALS CX_WARMUP 2>/dev/null || true")
print("\n".join(lines))
PY
)"
    eval "$_exports"
    cx_apply_network_profile "$CX_NODES" "$CX_TRANSPORT"
    # Each case has its OWN routing/dims -> its own canonical workload manifest. cx_stage_canonical
    # short-circuits when CX_WORKLOAD_DIR is already set, so without this unset the first case's
    # staged dir is reused for the rest and run_ep.py can't find the later cases' manifests
    # (FileNotFoundError .cx_workloads/<wid>.manifest.json). Unset so every case re-stages its own.
    unset CX_WORKLOAD_DIR 2>/dev/null || true
    cx_log "  [$((ci+1))/$ncases] $CX_BENCH $CX_MODE/$CX_PHASE routing=$CX_ROUTING eplb=${CX_EPLB:-0}"
    _cx_case_ts="$CX_TS"
    CX_TS="${_cx_case_ts}-a01"
    export CX_ATTEMPT_ID=1 CX_TS
    dispatch_bench || {
      failed_cases=$((failed_cases+1))
      cx_log "  [$((ci+1))/$ncases] $CX_BENCH case FAILED; failed-case record preserved"
    }
    export CX_TS="$_cx_case_ts"
    ci=$((ci + 1))
  done
  if [ "${failed_cases:-0}" -gt 0 ]; then
    cx_log "SHARD done: $failed_cases/$ncases case(s) failed"
    rc=1
  fi
  # The base timestamp matches every per-case file, so the final summary covers the whole shard.
  export CX_TS="$_cx_ts_base"
else
  _cx_single_ts="$CX_TS"
  CX_TS="${_cx_single_ts}-a01"
  export CX_ATTEMPT_ID=1 CX_TS
  dispatch_bench || rc=1
fi

# Summary table for the log; also fails the job if no valid results were produced.
if [ "${CX_PRECISION_PROBE:-0}" != 1 ]; then
  python3 summarize.py --results-dir results --runner "$CX_RUNNER" --ts "$CX_TS" || rc=1
fi
exit "$rc"
