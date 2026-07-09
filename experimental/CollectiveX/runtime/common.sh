# shellcheck shell=bash
# CollectiveX — shared launcher helpers (sourced, not executed).
#
# Cluster-generic scaffolding only (Slurm/container/build/staging); no
# model-serving. Logging goes to stderr so functions can `echo` a single
# result on stdout.

CX_SQUASH_FORMAT_VERSION="repro-v1"
CX_SQUASH_SOURCE_DATE_EPOCH=1
CX_DEEPEP_V2_COMMIT="fa8a9b16898204afd347c663b89e65ef87dc6ce6" # pragma: allowlist secret
CX_DEEPEP_V2_TREE="29809e75c5874e6609dac4804e7b651d5226959f" # pragma: allowlist secret
CX_DEEPEP_V2_FMT_COMMIT="a4c7e17133ee9cb6a2f45545f6e974dd3c393efa" # pragma: allowlist secret
# Consumed by run_in_container.sh after this helper is sourced.
# shellcheck disable=SC2034
CX_DEEPEP_V2_NCCL_CHECK_COMMIT="93d0564188f7a0a6288c6e316484861b0efa042e" # pragma: allowlist secret
CX_DEEPEP_HYBRID_COMMIT="e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8" # pragma: allowlist secret
CX_DEEPEP_HYBRID_TREE="d77aeab7f1bb52b615666fe178d26ced41fae08e" # pragma: allowlist secret
CX_DEEPEP_HYBRID_NCCL_COMMIT="1e0c869c39bb33f1034cb9920bd2a8a8406f04a3" # pragma: allowlist secret
unset COLLECTIVEX_OPERATOR_CONFIG_LOADED COLLECTIVEX_EPHEMERAL_CONFIG_PATH

cx_log() { printf '[collectivex] %s\n' "$*" >&2; }
cx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

# Keep a stable stage label while leaving diagnosis to the captured tool output.
cx_set_failure_stage() {
  local stage="$1"
  case "$stage" in
    setup|repository-stage|scheduler-allocation|container-import) ;;
    container-hash|container-launch|backend-setup|execution|artifact-collection) ;;
    *) cx_die "invalid launcher failure stage" ;;
  esac
  export CX_FAILSAFE_MODE="$stage"
}

cx_fail_stage() {
  local stage="$1" log_path="${2:-}" tail_lines="${CX_LOG_TAIL_LINES:-100}"
  cx_set_failure_stage "$stage"
  cx_log "ERROR: failure-stage=$stage"
  [[ "$tail_lines" =~ ^[1-9][0-9]{0,2}$ ]] || tail_lines=100
  if [ -n "$log_path" ] && [ -s "$log_path" ]; then
    cx_log "--- $stage log tail (last $tail_lines lines) ---"
    tail -n "$tail_lines" -- "$log_path" >&2 || true
    cx_log "--- end $stage log tail ---"
  fi
  return 1
}

cx_job_root_is_safe() {
  local root="$1"
  if [[ "$root" =~ ^/tmp/inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+$ ]]; then
    :
  elif [[ "$root" =~ ^/tmp/inferencex-collectivex-parent-([0-9]+)-([0-9]+)-([A-Za-z0-9._-]+)/inferencex-collectivex-([0-9]+)-([0-9]+)-([A-Za-z0-9._-]+)$ ]]; then
    [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[4]}" ] \
      && [ "${BASH_REMATCH[2]}" = "${BASH_REMATCH[5]}" ] \
      && [ "${BASH_REMATCH[3]}" = "${BASH_REMATCH[6]}" ] || return 1
  else
    return 1
  fi
  [ -d "$root" ] && [ ! -L "$root" ] \
    && [ "$(stat -c '%u:%a' "$root" 2>/dev/null)" = "$(id -u):700" ]
}

# Runner-local deployment settings are strict JSON kept outside the checkout.
# Only the selected runner's allowlisted values are exported; the document is
# never sourced or evaluated as shell.
cx_load_operator_config() {
  [ -n "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" ] \
    && [ "$COLLECTIVEX_OPERATOR_CONFIG_LOADED" = "$$" ] && return 0
  local config_path generated=0 parsed_path config_log key value validation_code
  unset CX_PARTITION CX_ACCOUNT CX_QOS CX_SQUASH_DIR CX_STAGE_DIR CX_ENROOT_CACHE_PATH
  unset ENROOT_CACHE_PATH
  unset CX_EXCLUDE_NODES CX_NODELIST CX_LOCK_DIR CX_MASTER_PORT
  unset CX_SOCKET_IFNAME CX_RDMA_DEVICES CX_IB_GID_INDEX CX_RDMA_SERVICE_LEVEL
  unset CX_RDMA_TRAFFIC_CLASS
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  config_path="${COLLECTIVEX_OPERATOR_CONFIG:-${XDG_CONFIG_HOME:-${HOME}/.config}/inferencex/collectivex.json}"
  if [ -n "${COLLECTIVEX_OPERATOR_CONFIG_CONTENT:-}" ]; then
    umask 077
    if cx_job_root_is_safe "${CX_JOB_ROOT:-}"; then
      config_path="$CX_JOB_ROOT/operator-config.json"
      (set -C; : > "$config_path") 2>/dev/null \
        || cx_die "cannot create ephemeral runner configuration"
    else
      config_path="$(mktemp /tmp/inferencex-collectivex-config.XXXXXX)" \
        || cx_die "cannot create ephemeral runner configuration"
    fi
    COLLECTIVEX_EPHEMERAL_CONFIG_PATH="$config_path"
    generated=1
    if ! printf '%s' "$COLLECTIVEX_OPERATOR_CONFIG_CONTENT" > "$config_path"; then
      unset COLLECTIVEX_OPERATOR_CONFIG_CONTENT
      rm -f -- "$config_path"
      unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
      cx_die "cannot materialize runner configuration"
    fi
  elif [ "${COLLECTIVEX_OPERATOR_CONFIG_REQUIRED:-0}" = 1 ]; then
    unset COLLECTIVEX_OPERATOR_CONFIG_CONTENT
    cx_die "runner configuration is unavailable"
  fi
  unset COLLECTIVEX_OPERATOR_CONFIG_CONTENT COLLECTIVEX_OPERATOR_CONFIG_REQUIRED
  if [ ! -e "$config_path" ]; then
    [ "${COLLECTIVEX_CANONICAL_GHA:-0}" != 1 ] \
      || cx_die "runner configuration is unavailable"
    COLLECTIVEX_OPERATOR_CONFIG_LOADED="$$"
    return 0
  fi
  umask 077
  parsed_path="$(mktemp /tmp/inferencex-collectivex-parsed.XXXXXX)" || {
    [ "$generated" = 0 ] || rm -f -- "$config_path"
    cx_die "cannot parse runner configuration"
  }
  config_log="$(cx_private_log_path operator-config)"
  if ! python3 - "$config_path" "${CX_RUNNER:-${CX_SHARD_SKU:-${CX_PUBLIC_RUNNER:-}}}" \
      > "$parsed_path" 2> "$config_log" <<'PY'
import json
import os
import posixpath
import re
import stat
import sys

RUNNERS = {
    "h100-dgxc", "h200-dgxc", "b200-dgxc", "b300",
    "gb200", "gb300", "mi300x", "mi325x", "mi355x",
}
FIELDS = {
    "partition": "CX_PARTITION",
    "account": "CX_ACCOUNT",
    "qos": "CX_QOS",
    "squash_dir": "CX_SQUASH_DIR",
    "stage_dir": "CX_STAGE_DIR",
    "enroot_cache_path": "CX_ENROOT_CACHE_PATH",
    "exclude_nodes": "CX_EXCLUDE_NODES",
    "nodelist": "CX_NODELIST",
    "lock_dir": "CX_LOCK_DIR",
    "socket_ifname": "CX_SOCKET_IFNAME",
    "rdma_devices": "CX_RDMA_DEVICES",
    "ib_gid_index": "CX_IB_GID_INDEX",
    "rdma_service_level": "CX_RDMA_SERVICE_LEVEL",
    "rdma_traffic_class": "CX_RDMA_TRAFFIC_CLASS",
}
NETWORK_FIELDS = {
    "socket_ifname", "rdma_devices", "ib_gid_index", "rdma_service_level",
    "rdma_traffic_class",
}
REQUIRED = {
    "h100-dgxc": {"partition", "account", "squash_dir"},
    "h200-dgxc": {"partition", "squash_dir"},
    "b200-dgxc": {"partition", "account", "squash_dir"},
    "b300": {"partition", "account", "squash_dir"},
    "gb200": {"partition", "account", "storage_roots"},
    "gb300": {"partition", "account", "squash_dir", "enroot_cache_path"},
    "mi300x": {"partition", "squash_dir"},
    "mi325x": {"partition", "squash_dir"},
    "mi355x": {"partition", "squash_dir"},
}
ALLOWED = {
    "h100-dgxc": REQUIRED["h100-dgxc"] | {"exclude_nodes", "stage_dir"} | NETWORK_FIELDS,
    "h200-dgxc": REQUIRED["h200-dgxc"] | {"account", "exclude_nodes", "stage_dir"} | NETWORK_FIELDS,
    "b200-dgxc": REQUIRED["b200-dgxc"] | {
        "exclude_nodes", "nodelist", "stage_dir", "qos",
    } | NETWORK_FIELDS,
    "b300": REQUIRED["b300"] | {"exclude_nodes", "stage_dir"} | NETWORK_FIELDS,
    "gb200": REQUIRED["gb200"] | NETWORK_FIELDS,
    "gb300": REQUIRED["gb300"] | {"stage_dir"} | NETWORK_FIELDS,
    "mi300x": REQUIRED["mi300x"] | {"exclude_nodes", "nodelist", "stage_dir", "lock_dir"} | NETWORK_FIELDS,
    "mi325x": REQUIRED["mi325x"] | {"exclude_nodes", "nodelist", "stage_dir", "lock_dir"} | NETWORK_FIELDS,
    "mi355x": REQUIRED["mi355x"] | {"exclude_nodes", "nodelist", "stage_dir", "lock_dir"} | NETWORK_FIELDS,
}
TOKEN = re.compile(r"^[A-Za-z0-9_.\[\],-]+$")
PATH = re.compile(r"^/[A-Za-z0-9._/+\-]+$")
IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
INTERFACES = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}(?:,[A-Za-z][A-Za-z0-9_.-]{0,31})*$")
RDMA_DEVICES = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}(?::[1-9][0-9]*)?(?:,[A-Za-z][A-Za-z0-9_.-]{0,31}(?::[1-9][0-9]*)?)*$")

def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError
        result[key] = value
    return result

def valid_path(value):
    return (
        isinstance(value, str) and len(value) <= 1024 and PATH.fullmatch(value)
        and posixpath.normpath(value) == value and not IPV4.search(value)
    )

def bounded_integer(value, maximum):
    if type(value) is int:
        result = value
    elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]{0,2}", value):
        result = int(value)
    else:
        raise ValueError
    if not 0 <= result <= maximum:
        raise ValueError
    return result

try:
    path, runner = sys.argv[1:]
    if runner not in RUNNERS:
        raise ValueError
    metadata = os.lstat(path)
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size > 65536
    ):
        raise ValueError
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError
        payload = b""
        while len(payload) <= 65536:
            chunk = os.read(descriptor, 65537 - len(payload))
            if not chunk:
                break
            payload += chunk
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    finally:
        os.close(descriptor)
    if (
        set(document) != {"schema_version", "runners"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError
    runners = document["runners"]
    if (
        not isinstance(runners, dict) or not runners or set(runners) - RUNNERS
        or runner not in runners
    ):
        raise ValueError
    selected = None
    for name, config in runners.items():
        if not isinstance(config, dict):
            raise ValueError
        if name == runner:
            missing = sorted(REQUIRED[name] - set(config))
            if missing:
                print(
                    "validation-missing-required-" + "-".join(missing),
                    file=sys.stderr,
                )
                raise SystemExit(1)
        if set(config) - ALLOWED[name]:
            raise ValueError
        for field, value in config.items():
            if field == "storage_roots":
                if (
                    not isinstance(value, list) or not 1 <= len(value) <= 16
                    or len(value) != len(set(value)) or not all(valid_path(item) for item in value)
                ):
                    raise ValueError
            elif field == "socket_ifname":
                if not isinstance(value, str) or not INTERFACES.fullmatch(value):
                    raise ValueError
            elif field == "rdma_devices":
                if not isinstance(value, str) or not RDMA_DEVICES.fullmatch(value):
                    raise ValueError
            elif field == "ib_gid_index":
                config[field] = bounded_integer(value, 255)
            elif field == "rdma_service_level":
                config[field] = bounded_integer(value, 15)
            elif field == "rdma_traffic_class":
                config[field] = bounded_integer(value, 255)
            elif field.endswith(("_dir", "_path")):
                if not valid_path(value):
                    raise ValueError
            elif (
                not isinstance(value, str) or not value or len(value) > 512
                or not TOKEN.fullmatch(value) or IPV4.search(value)
            ):
                raise ValueError
        if name == runner:
            selected = dict(config)
    if selected is None:
        raise ValueError
    roots = selected.pop("storage_roots", None)
    if roots is not None:
        for root in roots:
            squash = posixpath.join(root, "collectivex", "containers")
            stage = posixpath.join(root, "collectivex", "stage")
            probes = []
            try:
                for directory in (squash, stage):
                    os.makedirs(directory, mode=0o700, exist_ok=True)
                    probe = posixpath.join(directory, f".write-probe-{os.getpid()}")
                    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    os.close(fd)
                    probes.append(probe)
                selected.update(squash_dir=squash, stage_dir=stage)
                break
            except OSError:
                pass
            finally:
                for probe in probes:
                    try:
                        os.unlink(probe)
                    except OSError:
                        pass
        else:
            raise ValueError
    for field, value in selected.items():
        key = FIELDS[field]
        sys.stdout.buffer.write(
            key.encode() + b"\0" + str(value).encode() + b"\0"
        )
except (KeyError, OSError, TypeError, UnicodeError, ValueError):
    import traceback

    location = traceback.extract_tb(sys.exc_info()[2])[-1].lineno
    print(f"validation-line-{location}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    validation_code="$(head -n 1 "$config_log" 2>/dev/null || true)"
    rm -f -- "$parsed_path"
    [ "$generated" = 0 ] || rm -f -- "$config_path"
    unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
    unset COLLECTIVEX_OPERATOR_CONFIG COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL
    [[ "$validation_code" =~ ^validation-(line-[0-9]+|missing-required-[a-z0-9_-]+)$ ]] \
      || validation_code="validation-unknown"
    cx_die "runner-local configuration failed ($validation_code)"
  fi
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    printf -v "$key" '%s' "$value"
    export "${key?}"
  done < "$parsed_path"
  rm -f -- "$parsed_path"
  if [ "$generated" = 1 ] || [ "${COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL:-0}" = 1 ]; then
    rm -f -- "$config_path" || cx_die "cannot remove ephemeral runner configuration"
  fi
  unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
  unset COLLECTIVEX_OPERATOR_CONFIG COLLECTIVEX_OPERATOR_CONFIG_EPHEMERAL
  COLLECTIVEX_OPERATOR_CONFIG_LOADED="$$"
}

cx_private_log_path() {
  local label="$1" tag="${COLLECTIVEX_EXECUTION_ID:-manual_$$}" path
  path="$(python3 - "$tag" "$label" <<'PY' 2>/dev/null
import os
import re
import shutil
import stat
import sys
import time

tag, label = sys.argv[1:]
if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) for value in (tag, label)):
    raise SystemExit(1)
root = f"/tmp/inferencex-collectivex-{os.getuid()}"
job_root = os.environ.get("CX_JOB_ROOT", "")
job_parent = os.environ.get("CX_JOB_PARENT", "")
if (
    os.environ.get("COLLECTIVEX_CANONICAL_GHA") == "1"
    and job_parent
    and job_parent != "/tmp"
):
    if (
        not os.path.isabs(job_root)
        or os.path.dirname(job_root) != job_parent
        or not re.fullmatch(
            r"inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+",
            os.path.basename(job_root),
        )
    ):
        raise SystemExit(1)
    control = os.path.join(job_root, "control")
    control_metadata = os.stat(control, follow_symlinks=False)
    if (
        not stat.S_ISDIR(control_metadata.st_mode)
        or control_metadata.st_uid != os.getuid()
        or stat.S_IMODE(control_metadata.st_mode) != 0o700
    ):
        raise SystemExit(1)
    root = os.path.join(control, "private-logs")
old_umask = os.umask(0o077)
flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
try:
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        pass
    root_fd = os.open(root, flags)
    try:
        metadata = os.fstat(root_fd)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise OSError("unsafe root")
        cutoff = time.time() - 86400
        for entry in os.scandir(root):
            try:
                if (
                    entry.name != tag and entry.is_dir(follow_symlinks=False)
                    and entry.stat(follow_symlinks=False).st_mtime < cutoff
                ):
                    shutil.rmtree(entry.path)
            except OSError:
                pass
        try:
            os.mkdir(tag, 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        directory_fd = os.open(tag, flags, dir_fd=root_fd)
        try:
            metadata = os.fstat(directory_fd)
            if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                raise OSError("unsafe directory")
            log_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            log_fd = os.open(f"{label}.log", log_flags, 0o600, dir_fd=directory_fd)
            os.close(log_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(root_fd)
finally:
    os.umask(old_umask)
print(f"{root}/{tag}/{label}.log", end="")
PY
)" || cx_die "cannot create private runtime log"
  printf '%s' "$path"
}

# Manual successes delete diagnostics immediately. Canonical workflow logs survive
# until artifact upload succeeds; failed logs remain private for debugging, and a
# later run prunes abandoned directories older than 24 hours.
cx_cleanup_private_logs() {
  local rc="$1" tag="${COLLECTIVEX_EXECUTION_ID:-manual_$$}"
  [ "$rc" = 0 ] || return 0
  python3 - "$tag" <<'PY' >/dev/null 2>&1 || true
import os
import re
import shutil
import stat
import sys

tag = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
    raise SystemExit(1)
root = f"/tmp/inferencex-collectivex-{os.getuid()}"
job_root = os.environ.get("CX_JOB_ROOT", "")
job_parent = os.environ.get("CX_JOB_PARENT", "")
if (
    os.environ.get("COLLECTIVEX_CANONICAL_GHA") == "1"
    and job_parent
    and job_parent != "/tmp"
):
    if (
        not os.path.isabs(job_root)
        or os.path.dirname(job_root) != job_parent
        or not re.fullmatch(
            r"inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+",
            os.path.basename(job_root),
        )
    ):
        raise SystemExit(1)
    control = os.path.join(job_root, "control")
    control_metadata = os.stat(control, follow_symlinks=False)
    if (
        not stat.S_ISDIR(control_metadata.st_mode)
        or control_metadata.st_uid != os.getuid()
        or stat.S_IMODE(control_metadata.st_mode) != 0o700
    ):
        raise SystemExit(1)
    root = os.path.join(control, "private-logs")
flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
root_fd = os.open(root, flags)
try:
    metadata = os.fstat(root_fd)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise SystemExit(1)
finally:
    os.close(root_fd)
path = os.path.join(root, tag)
if os.path.isdir(path) and not os.path.islink(path):
    shutil.rmtree(path)
PY
}

# Explicit Slurm export boundary. Operator config, runner credentials, HOME,
# workspace paths, and unrelated service secrets never enter the container.
cx_container_exports() {
  printf '%s' 'COLLECTIVEX_SOURCE_SHA,COLLECTIVEX_ARTIFACT_NAME,COLLECTIVEX_EXECUTION_ID,COLLECTIVEX_IMAGE,COLLECTIVEX_SQUASH_SHA256,GITHUB_REF_NAME,GITHUB_REF,GITHUB_REPOSITORY,GITHUB_JOB,GITHUB_RUN_ID,GITHUB_RUN_ATTEMPT,GITHUB_SHA,CX_RUNNER,CX_BENCH,CX_NODES,CX_GPUS_PER_NODE,CX_SCALE_UP_DOMAIN,CX_SHARD_FILE,CX_SHARD_SKU,CX_NGPUS,CX_TS,CX_TOPO,CX_SCOPE,CX_TRANSPORT,CX_SCALE_UP_TRANSPORT,CX_SCALE_OUT_TRANSPORT,CX_MODE,CX_PHASE,CX_ROUTING,CX_CASE_ID,CX_SUITE,CX_WORKLOAD_NAME,CX_QUALIFICATION_INDEX,CX_VERSION,CX_HIDDEN,CX_TOPK,CX_EXPERTS,CX_TOKENS_LADDER,CX_CANONICAL,CX_ITERS,CX_TRIALS,CX_WARMUP,CX_SAMPLES_PER_POINT,CX_WARMUP_SEMANTICS,CX_SEED,CX_RUN_TIMEOUT,CX_NCCL_HOME,CX_ALLOW_MNNVL,CX_ATTEMPT_ID,CX_RUNTIME_MARKER,CX_MORI_KERNEL_TYPE,CX_WORKLOAD_DIR,CX_BACKEND_CACHE_ROOT,CX_BACKEND_SOURCE_ROOT,CX_SOCKET_IFNAME,CX_RDMA_DEVICES,CX_IB_GID_INDEX,CX_RDMA_SERVICE_LEVEL,CX_RDMA_TRAFFIC_CLASS,CX_RDMA_LINK_LAYER,MASTER_ADDR,MASTER_PORT,RANK,WORLD_SIZE,LOCAL_RANK,LOCAL_WORLD_SIZE,NCCL_NET,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NCCL_IB_HCA,NCCL_IB_GID_INDEX,NCCL_IB_SL,NVSHMEM_DISABLE_IB,NVSHMEM_REMOTE_TRANSPORT,NVSHMEM_ENABLE_NIC_PE_MAPPING,NVSHMEM_HCA_LIST,NVSHMEM_IB_GID_INDEX,NVSHMEM_IB_SL,NVSHMEM_IB_ENABLE_IBGDA,NVSHMEM_IBGDA_NIC_HANDLER,EP_NIC_NAME,EP_OVERRIDE_RDMA_SL,MORI_RDMA_DEVICES,MORI_RDMA_TC,MORI_IO_TC,MORI_RDMA_SL,MORI_IO_SL,HYBRID_EP_MULTINODE,USE_NIXL,RDMA_CORE_HOME,DEEPEP_HYBRID_BUILD_MODE,NCCL_CUMEM_ENABLE,NCCL_MNNVL_ENABLE,MC_FORCE_MNNVL,MORI_DISABLE_AUTO_XGMI,MORI_ENABLE_SDMA,MORI_APP_LOG_LEVEL,MORI_SHMEM_LOG_LEVEL,MORI_IO_LOG_LEVEL,MORI_COMMIT'
}

# Host-side utility steps need only the basic login paths. They never receive
# the complete Actions or runner environment.
cx_host_exports() {
  printf '%s' 'HOME,PATH,USER,XDG_CACHE_HOME,ENROOT_CACHE_PATH'
}

cx_prepare_runtime_marker() {
  local mount_src="$1" tag="${COLLECTIVEX_EXECUTION_ID:-${CX_TS:-}}" marker
  [[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || cx_die "cannot create runtime stage marker"
  marker=".shards/runtime-stage-${tag}.txt"
  mkdir -p "$mount_src/experimental/CollectiveX/.shards" >/dev/null 2>&1 \
    || cx_die "cannot create runtime stage marker"
  rm -f -- "$mount_src/experimental/CollectiveX/$marker" >/dev/null 2>&1 \
    || cx_die "cannot reset runtime stage marker"
  export CX_RUNTIME_MARKER="$marker"
}

cx_write_runtime_stage() {
  local stage="$1" marker="${CX_RUNTIME_MARKER:-}"
  [ -n "$marker" ] || return 0
  [[ "$marker" =~ ^\.shards/runtime-stage-[A-Za-z0-9][A-Za-z0-9._-]*\.txt$ ]] \
    || return 1
  case "$stage" in backend-setup|execution) ;; *) return 1 ;; esac
  printf '%s\n' "$stage" > "$marker"
}

cx_adopt_runtime_stage() {
  local mount_src="$1" marker="${CX_RUNTIME_MARKER:-}" stage=""
  [ -n "$marker" ] || return 0
  if [[ "$marker" =~ ^\.shards/runtime-stage-[A-Za-z0-9][A-Za-z0-9._-]*\.txt$ ]] \
      && [ -f "$mount_src/experimental/CollectiveX/$marker" ]; then
    IFS= read -r stage < "$mount_src/experimental/CollectiveX/$marker" || true
    rm -f -- "$mount_src/experimental/CollectiveX/$marker" >/dev/null 2>&1 || true
    case "$stage" in
      backend-setup|execution) cx_set_failure_stage "$stage" ;;
    esac
  fi
}

cx_require_vars() {
  local name
  local -a missing=()
  for name in "$@"; do
    [ -n "${!name:-}" ] || missing+=("$name")
  done
  [ "${#missing[@]}" -eq 0 ] || cx_die \
    "missing runner-local configuration: ${missing[*]} (set them in COLLECTIVEX_OPERATOR_CONFIG)"
}

cx_bool_enabled() {
  local normalized
  normalized="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$normalized" in
    1|true|yes) return 0 ;;
    *) return 1 ;;
  esac
}

cx_require_record_safe() {
  local value
  for value in "$@"; do
    case "$value" in
      *'|'*|*$'\n'*|*$'\r'*) cx_die "manual case field contains a record delimiter" ;;
    esac
  done
}

cx_nccl_hca_device_name() {
  local selector="${1#=}"
  printf '%s' "${selector%%:*}"
}

cx_export_gid_index_for_link_layer() {
  local link_layer="$1" scaleout="$2"
  unset NVSHMEM_IB_GID_INDEX NCCL_IB_GID_INDEX
  [ -n "${CX_IB_GID_INDEX:-}" ] || return 0
  case "$link_layer" in
    roce)
      export NVSHMEM_IB_GID_INDEX="$CX_IB_GID_INDEX"
      if [ "$scaleout" = 1 ]; then
        export NCCL_IB_GID_INDEX="$CX_IB_GID_INDEX"
      fi
      ;;
    infiniband) ;;
    *) cx_die "unsupported RDMA link layer" ;;
  esac
}

# Convert private, runner-local network selectors into the public library
# variables needed inside the container. Values are interface/HCA identifiers,
# never addresses; the rendezvous hostname is derived from the allocation.
cx_apply_network_profile() {
  local nodes="$1" transport="$2"
  local selector rdma_name rdma_names="" ep_nic=""
  local scaleout=0
  local -a selectors
  [[ "$nodes" =~ ^[1-9][0-9]*$ ]] || cx_die "invalid network placement"
  unset NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL
  unset NVSHMEM_DISABLE_IB NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset NVSHMEM_HCA_PE_MAPPING NVSHMEM_REMOTE_TRANSPORT
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  if [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; then
    scaleout=1
  elif [ "${CX_SHARD_SKU:-}" = b300 ] && [ "$nodes" = 1 ]; then
    # All 8 b300 GPUs share one NVLink domain, so single-node EP has no
    # cross-node peers and NVSHMEM must stay on intranode P2P/NVLink. DeepEP v1
    # low-latency kernels otherwise make NVSHMEM bring up the IBRC remote
    # transport, which loops on RoCE-HCA loopback QP setup (ibrc ibv_modify_qp
    # status 110) until the 900s case timeout. NVSHMEM_REMOTE_TRANSPORT=none is
    # the documented switch that disables the remote transport entirely (no
    # remote ranks exist to serve); DeepEP's allow_nvlink_for_low_latency_mode
    # then runs low-latency over NVLink. NVSHMEM_DISABLE_IB is not a recognized
    # NVSHMEM variable and had no effect, which is why the IBRC path still ran.
    export NVSHMEM_REMOTE_TRANSPORT=none
  fi
  [ "$scaleout" = 1 ] || return 0
  [ -n "${CX_RDMA_DEVICES:-}" ] \
    || cx_die "RDMA execution requires a private device selector"
  if [ "$scaleout" = 1 ] && [ -n "${CX_SOCKET_IFNAME:-}" ]; then
    [[ "$CX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(,[A-Za-z][A-Za-z0-9_.-]{0,31})*$ ]] \
      || cx_die "invalid private socket interface selector"
    export NCCL_SOCKET_IFNAME="$CX_SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$CX_SOCKET_IFNAME"
  fi
  if [ -n "${CX_RDMA_DEVICES:-}" ]; then
    [[ "$CX_RDMA_DEVICES" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*$ ]] \
      || cx_die "invalid private RDMA device selector"
    IFS=, read -r -a selectors <<< "$CX_RDMA_DEVICES"
    for selector in "${selectors[@]}"; do
      rdma_name="${selector%%:*}"
      rdma_names="${rdma_names}${rdma_names:+,}${rdma_name}"
      [ -n "$ep_nic" ] || ep_nic="$rdma_name"
    done
    export NVSHMEM_HCA_LIST="$CX_RDMA_DEVICES"
    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    if [ "$scaleout" = 1 ]; then
      if [ "${CX_SHARD_SKU:-}" = mi300x ] || [ "${CX_SHARD_SKU:-}" = mi325x ] \
          || [ "${CX_SHARD_SKU:-}" = mi355x ]; then
        unset NCCL_NET
      else
        export NCCL_NET=IB
      fi
      export NCCL_IB_HCA="=$CX_RDMA_DEVICES"
      export MORI_RDMA_DEVICES="$rdma_names" EP_NIC_NAME="$ep_nic"
    fi
  fi
  if [ -n "${CX_IB_GID_INDEX:-}" ]; then
    [[ "$CX_IB_GID_INDEX" =~ ^[0-9]+$ ]] && [ "$CX_IB_GID_INDEX" -le 255 ] \
      || cx_die "invalid private IB GID index"
  fi
  if [ -n "${CX_RDMA_SERVICE_LEVEL:-}" ]; then
    [[ "$CX_RDMA_SERVICE_LEVEL" =~ ^[0-9]+$ ]] && [ "$CX_RDMA_SERVICE_LEVEL" -le 15 ] \
      || cx_die "invalid private RDMA service level"
    export NVSHMEM_IB_SL="$CX_RDMA_SERVICE_LEVEL"
    if [ "$scaleout" = 1 ]; then
      export NCCL_IB_SL="$CX_RDMA_SERVICE_LEVEL"
      export EP_OVERRIDE_RDMA_SL="$CX_RDMA_SERVICE_LEVEL"
      export MORI_RDMA_SL="$CX_RDMA_SERVICE_LEVEL" MORI_IO_SL="$CX_RDMA_SERVICE_LEVEL"
    fi
  fi
  if [ -n "${CX_RDMA_TRAFFIC_CLASS:-}" ]; then
    [[ "$CX_RDMA_TRAFFIC_CLASS" =~ ^[0-9]+$ ]] && [ "$CX_RDMA_TRAFFIC_CLASS" -le 255 ] \
      || cx_die "invalid private RDMA traffic class"
    [ "$scaleout" = 1 ] \
      && export MORI_RDMA_TC="$CX_RDMA_TRAFFIC_CLASS" MORI_IO_TC="$CX_RDMA_TRAFFIC_CLASS"
  fi
  local nic_handler=gpu
  if [ "${CX_SHARD_SKU:-}" = b200-dgxc ] && [ "${CX_BENCH:-}" = deepep ]; then
    nic_handler=cpu
  fi
  export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER="$nic_handler"
  if [ -n "${CX_RDMA_LINK_LAYER:-}" ]; then
    case "$CX_RDMA_LINK_LAYER" in
      roce|infiniband) ;;
      *) cx_die "invalid validated RDMA link layer" ;;
    esac
    cx_export_gid_index_for_link_layer "$CX_RDMA_LINK_LAYER" "$scaleout"
  fi
}

# Slurm may remove NCCL's leading exact-match marker while propagating an
# inherited environment. Reconstruct it from the validated private selector at
# the container boundary instead of accepting a prefix-matched HCA list.
cx_restore_exact_hca_selector() {
  if [ "${CX_NODES:-1}" -le 1 ] || [ "${CX_TRANSPORT:-}" = mnnvl ]; then
    return 0
  fi
  [ -n "${CX_RDMA_DEVICES:-}" ] \
    || { cx_log "ERROR: scale-out RDMA selector is unavailable"; return 1; }
  [[ "$CX_RDMA_DEVICES" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*$ ]] \
    || { cx_log "ERROR: invalid scale-out RDMA selector"; return 1; }
  export NCCL_IB_HCA="=$CX_RDMA_DEVICES"
}

cx_default_route_interface() {
  python3 - <<'PY'
from pathlib import Path
for line in Path('/proc/net/route').read_text().splitlines()[1:]:
    fields = line.split()
    if len(fields) >= 4 and fields[1] == '00000000' and int(fields[3], 16) & 1:
        print(fields[0], end=''); break
PY
}

# Prove that the operator-pinned scale-out fabric exists on every allocated
# node before image import or backend initialization. Selector values and node
# diagnostics stay in the runner-private log.
cx_validate_network_profile_on_job() {
  local job_id="$1" nodes="$2" transport="$3" report_failure="${4:-1}"
  local log_label=network-profile log rc=0 scaleout=0 marker_count link_layer
  if [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; then
    scaleout=1
  fi
  [ "$scaleout" = 1 ] || return 0
  [[ "$job_id" =~ ^[1-9][0-9]*$ && "$nodes" =~ ^[1-9][0-9]*$ ]] \
    || return 1
  [ -n "${CX_RDMA_DEVICES:-}" ] || return 1
  case "${CX_NETWORK_VALIDATION_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${CX_NETWORK_VALIDATION_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(cx_private_log_path "$log_label")" || return 1
  CX_NETWORK_PROFILE_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp --input=all \
    --export="$(cx_host_exports),CX_SOCKET_IFNAME,CX_RDMA_DEVICES,CX_IB_GID_INDEX" \
    bash -s > "$log" 2>&1 <<'BASH' || rc=$?
set -euo pipefail
if [ -z "${CX_SOCKET_IFNAME:-}" ]; then
  CX_SOCKET_IFNAME="$(python3 - <<'PY'
from pathlib import Path

for line in Path('/proc/net/route').read_text().splitlines()[1:]:
    fields = line.split()
    if len(fields) >= 4 and fields[1] == '00000000' and int(fields[3], 16) & 1:
        print(fields[0], end='')
        break
PY
)"
  [ -n "$CX_SOCKET_IFNAME" ] \
    || { printf '[collectivex-private] socket-interface-1=default-route-missing\n'; exit 1; }
fi
[[ "$CX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(,[A-Za-z][A-Za-z0-9_.-]{0,31})*$ ]]
printf '[collectivex-private] socket-interface-selected=%s\n' "$CX_SOCKET_IFNAME"
[[ "$CX_RDMA_DEVICES" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?(,[A-Za-z][A-Za-z0-9_.-]{0,31}(:[1-9][0-9]*)?)*$ ]]
if [ -n "${CX_IB_GID_INDEX:-}" ]; then
  [[ "$CX_IB_GID_INDEX" =~ ^[0-9]+$ ]] && [ "$CX_IB_GID_INDEX" -le 255 ]
fi
if [ -n "${CX_SOCKET_IFNAME:-}" ]; then
  IFS=, read -r -a interfaces <<< "$CX_SOCKET_IFNAME"
  socket_ordinal=0
  for interface in "${interfaces[@]}"; do
    socket_ordinal=$((socket_ordinal + 1))
    [ -d "/sys/class/net/$interface" ] \
      || { printf '[collectivex-private] socket-interface-%s=missing\n' "$socket_ordinal"; exit 1; }
    state="$(cat "/sys/class/net/$interface/operstate")"
    [ "$state" = up ] || [ "$state" = unknown ] \
      || { printf '[collectivex-private] socket-interface-%s=down\n' "$socket_ordinal"; exit 1; }
  done
fi
check_port() {
  local port_path="$1" ordinal="$2" state gid link_layer
  [ -d "$port_path" ] \
    || { printf '[collectivex-private] rdma-port-%s=missing\n' "$ordinal"; return 1; }
  read -r state _ < "$port_path/state"
  [ "$state" = 4: ] \
    || { printf '[collectivex-private] rdma-port-%s=inactive\n' "$ordinal"; return 1; }
  if [ -n "${CX_IB_GID_INDEX:-}" ]; then
    [ -r "$port_path/gids/$CX_IB_GID_INDEX" ] \
      || { printf '[collectivex-private] rdma-port-%s=gid-missing\n' "$ordinal"; return 1; }
    gid="$(tr -d ':0[:space:]' < "$port_path/gids/$CX_IB_GID_INDEX")"
    [ -n "$gid" ] \
      || { printf '[collectivex-private] rdma-port-%s=gid-empty\n' "$ordinal"; return 1; }
  fi
  [ -r "$port_path/link_layer" ] \
    || { printf '[collectivex-private] rdma-port-%s=link-layer-missing\n' "$ordinal"; return 1; }
  link_layer="$(< "$port_path/link_layer")"
  case "$link_layer" in
    Ethernet) link_layer=roce ;;
    InfiniBand) link_layer=infiniband ;;
    *) printf '[collectivex-private] rdma-port-%s=link-layer-invalid\n' "$ordinal"; return 1 ;;
  esac
  [ -z "${profile:-}" ] || [ "$profile" = "$link_layer" ] \
    || { printf '[collectivex-private] rdma-port-%s=link-layer-mixed\n' "$ordinal"; return 1; }
  profile="$link_layer"
}
profile=""
IFS=, read -r -a devices <<< "$CX_RDMA_DEVICES"
ordinal=0
for selector in "${devices[@]}"; do
  ordinal=$((ordinal + 1))
  device="${selector%%:*}"
  configured_port=""
  [ "$selector" = "$device" ] || configured_port="${selector#*:}"
  ports="/sys/class/infiniband/$device/ports"
  [ -d "$ports" ] \
    || { printf '[collectivex-private] rdma-device-%s=missing\n' "$ordinal"; exit 1; }
  if [ -n "$configured_port" ]; then
    check_port "$ports/$configured_port" "$ordinal"
  else
    active=0
    for port_path in "$ports"/*; do
      if check_port "$port_path" "$ordinal"; then
        active=1
      fi
    done
    [ "$active" = 1 ]
  fi
done
[ -n "$profile" ]
printf '[collectivex-private] rdma-link-layer=%s\n' "$profile"
BASH
  if [ "$rc" != 0 ]; then
    marker="$(grep -aoE '(socket-interface|rdma-(device|port))-[0-9]+=(missing|down|inactive|default-route-missing|gid-missing|gid-empty|link-layer-missing|link-layer-invalid|link-layer-mixed)' "$log" \
      | tail -n 1 || true)"
    [ -z "$marker" ] || cx_log "ERROR: network-profile-$marker"
    [ "$report_failure" = 0 ] || cx_fail_stage setup "$log" || true
    return "$rc"
  fi
  socket_ifname="$(
    sed -nE 's/^\[collectivex-private\] socket-interface-selected=([A-Za-z][A-Za-z0-9_.-]{0,31})$/\1/p' "$log" \
      | sort -u
  )"
  marker_count="$(grep -Ec '^\[collectivex-private\] socket-interface-selected=' "$log")"
  socket_unique_count="$(printf '%s\n' "$socket_ifname" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [ "$socket_unique_count" -lt 1 ] || [ "$marker_count" != "$nodes" ]; then
    cx_log "ERROR: network-profile-socket-markers=$marker_count/$nodes unique=$socket_unique_count"
    return 1
  fi
  if [ "$socket_unique_count" = 1 ]; then
    export CX_SOCKET_IFNAME="$socket_ifname"
  else
    unset CX_SOCKET_IFNAME
  fi
  link_layer="$(
    sed -nE 's/^\[collectivex-private\] rdma-link-layer=(roce|infiniband)$/\1/p' "$log" \
      | sort -u
  )"
  marker_count="$(grep -Ec '^\[collectivex-private\] rdma-link-layer=(roce|infiniband)$' "$log")"
  case "$marker_count:$link_layer" in
    "$nodes":roce|"$nodes":infiniband) ;;
    *) [ "$report_failure" = 0 ] || cx_fail_stage setup "$log" || true; return 1 ;;
  esac
  export CX_RDMA_LINK_LAYER="$link_layer"
  cx_export_gid_index_for_link_layer "$link_layer" "$scaleout"
}

cx_allocation_nodes_csv() {
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

cx_resolve_slurm_rendezvous() {
  local job_id="$1" master_addr master_port socket_ifname="${CX_SOCKET_IFNAME:-}"
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || cx_die "invalid rendezvous allocation"
  # Query relative node zero directly so MASTER_ADDR always hosts global rank 0.
  # Prefer the address on the already validated cross-node socket interface;
  # a short hostname may resolve onto a management network that ranks cannot use.
  if [[ "$socket_ifname" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]]; then
    master_addr="$(srun --jobid="$job_id" --nodes=1 --ntasks=1 --relative=0 \
      --chdir=/tmp --export="$(cx_host_exports)" bash -s -- "$socket_ifname" \
      2>/dev/null <<'BASH' | head -n1
set -euo pipefail
ip -o -4 address show dev "$1" scope global \
  | awk 'NR == 1 {split($4, address, "/"); print address[1]}'
BASH
)"
    [[ "$master_addr" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] \
      || cx_die "could not resolve the allocated primary interface"
  else
    master_addr="$(srun --jobid="$job_id" --nodes=1 --ntasks=1 --relative=0 \
      --chdir=/tmp --export="$(cx_host_exports)" hostname -s 2>/dev/null | head -n1)"
    [[ "$master_addr" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
      || cx_die "could not resolve the allocated primary node"
  fi
  master_port="${CX_MASTER_PORT:-29551}"
  [[ "$master_port" =~ ^[1-9][0-9]*$ ]] && [ "$master_port" -le 65535 ] \
    || cx_die "invalid distributed rendezvous port"
  export MASTER_ADDR="$master_addr" MASTER_PORT="$master_port"
}

# Printed into `bash -c` for one Slurm task per GPU. Every rank derives its
# identity from Slurm rather than accepting caller-supplied rank values.
cx_slurm_rank_wrapper() {
  cat <<'BASH'
case "${SLURM_PROCID:-}:${SLURM_NTASKS:-}:${SLURM_LOCALID:-}:${SLURM_NODEID:-}" in
  *[!0-9:]*|:*|*::*|*:) exit 67 ;;
esac
[ "$SLURM_NTASKS" = "$CX_NGPUS" ] || exit 67
[ "$SLURM_LOCALID" -lt "$CX_GPUS_PER_NODE" ] || exit 67
. /ix/experimental/CollectiveX/runtime/common.sh || exit 68
if [ "${CX_NODES:-1}" -gt 1 ] && [ "${CX_TRANSPORT:-}" != mnnvl ]; then
  if [ -z "${CX_SOCKET_IFNAME:-}" ]; then
    CX_SOCKET_IFNAME="$(cx_default_route_interface)" || exit 68
    [[ "$CX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}$ ]] || exit 68
    export CX_SOCKET_IFNAME
  fi
  cx_apply_network_profile "$CX_NODES" "$CX_TRANSPORT" || exit 68
fi
cx_write_runtime_stage execution || exit 68
export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_RANK="$SLURM_LOCALID" LOCAL_WORLD_SIZE="$CX_GPUS_PER_NODE"
exec python3 bench/run_ep.py "$@"
BASH
}

# A set shard path is an execution contract, never a hint. Validate it before
# staging/allocation and again in-container so a missing or stale control file
# cannot silently fall back to a manual single-case run.
cx_validate_shard_control() {
  local cx_root="$1" shard="${CX_SHARD_FILE:-}" path expected_sku
  [ -n "$shard" ] || return 0
  expected_sku="${CX_SHARD_SKU:-}"
  [ -n "$expected_sku" ] || cx_die "CX_SHARD_SKU is required with CX_SHARD_FILE"
  [ -n "${CX_BENCH:-}" ] || cx_die "CX_BENCH is required with CX_SHARD_FILE"
  [[ "${CX_NODES:-}" =~ ^[1-9][0-9]*$ ]] \
    || cx_die "positive CX_NODES is required with CX_SHARD_FILE"
  path="$shard"
  [ -f "$path" ] || path="${cx_root%/}/$shard"
  [ -f "$path" ] || cx_die "shard control does not exist"
  [ -s "$path" ] || cx_die "shard control is empty"
  python3 "${cx_root%/}/sweep_matrix.py" \
    --validate-control "$path" --expect-sku "$expected_sku" \
    --expect-backend "$CX_BENCH" --expect-nodes "$CX_NODES" >/dev/null 2>&1 \
    || cx_die "invalid shard control"
}

# Load only the case mode needed to choose the allocation/network profile. A
# consolidated shard may contain normal and low-latency cases; selecting the
# strictest mode here makes allocation preflight prove every required device.
# The in-container dispatcher reapplies the profile for each individual case.
cx_load_network_control_mode() {
  local cx_root="$1" shard="${CX_SHARD_FILE:-}" path mode
  [ -n "$shard" ] || return 0
  path="$shard"
  [ -f "$path" ] || path="${cx_root%/}/$shard"
  [ -f "$path" ] || return 1
  mode="$(python3 - "$path" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    cases = json.load(stream)["cases"]
modes = {case["mode"] for case in cases}
if not modes or modes - {"normal", "low-latency"}:
    raise SystemExit("invalid shard mode")
print("low-latency" if "low-latency" in modes else "normal")
PY
)" || return 1
  case "$mode" in
    normal|low-latency) export CX_MODE="$mode" ;;
    *) return 1 ;;
  esac
}

cx_apply_timing_profile() {
  [ -n "${CX_TIMING:-}" ] || return 0
  local iters trials warmup extra
  IFS=: read -r iters trials warmup extra <<< "$CX_TIMING"
  [[ "$iters" =~ ^[1-9][0-9]*$ && "$trials" =~ ^[1-9][0-9]*$ \
    && "$warmup" =~ ^[1-9][0-9]*$ && -z "$extra" ]] \
    || cx_die "CX_TIMING must be positive iters:trials:warmup"
  export CX_ITERS="$iters" CX_TRIALS="$trials" CX_WARMUP="$warmup"
}

cx_scheduler_job_name() {
  local execution_id="${COLLECTIVEX_EXECUTION_ID:-manual-$$}" safe
  safe="$(printf '%s' "$execution_id" | tr -cs 'A-Za-z0-9_.-' '-')" || return 1
  safe="${safe#-}"; safe="${safe%-}"
  [ -n "$safe" ] || return 1
  if [ "${#safe}" -gt 120 ]; then
    safe="${safe:0:48}-${safe: -71}"
  fi
  printf 'cx-%s' "$safe"
}

# Return 0 after recovering one allocation ID, 2 after three successful empty
# observations, and 1 for every ambiguous or failed lookup. Callers inspect the
# state variables rather than the status because all missing-ID paths still fail.
cx_reconcile_salloc_jobid() {
  local job_name="$1" scheduler_user queue_output line delay attempt
  local -a ids=()
  scheduler_user="$(id -un 2>/dev/null)" || return 1
  [[ "$scheduler_user" =~ ^[A-Za-z0-9_.-]+$ \
    && "$job_name" =~ ^cx-[A-Za-z0-9_.-]{1,120}$ ]] || return 1
  for attempt in 1 2 3; do
    ids=()
    if ! queue_output="$(
      squeue -h --user="$scheduler_user" --name="$job_name" -o %A 2>/dev/null
    )"; then
      return 1
    fi
    while IFS= read -r line; do
      [[ "$line" =~ ^[[:space:]]*$ ]] && continue
      if [[ "$line" =~ ^[[:space:]]*([1-9][0-9]*)[[:space:]]*$ ]]; then
        ids+=("${BASH_REMATCH[1]}")
      else
        return 1
      fi
    done <<< "$queue_output"
    if [ "${#ids[@]}" -eq 1 ]; then
      JOB_ID="${ids[0]}"
      CX_ALLOCATION_UNCERTAIN=0
      return 0
    fi
    [ "${#ids[@]}" -eq 0 ] || return 1
    if [ "$attempt" -eq 3 ]; then
      CX_ALLOCATION_UNCERTAIN=0
      return 2
    fi
    delay=$((1 << (attempt - 1)))
    sleep "$delay" || return 1
  done
  return 1
}

cx_verify_salloc_jobid() {
  local job_id="$1" queue_output line count=0
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  queue_output="$(squeue -h -j "$job_id" -o %A 2>/dev/null)" || return 1
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*${job_id}[[:space:]]*$ ]] || return 1
    count=$((count + 1))
  done <<< "$queue_output"
  [ "$count" -eq 1 ]
}

# Allocate via salloc's stable grant message and assign JOB_ID in this shell.
# Raw scheduler output remains in the bounded private execution log.
cx_salloc_jobid() {
  local log_label=scheduler-allocation log job_id job_name argument salloc_rc=0
  case "${CX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${CX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  if ! log="$(cx_private_log_path "$log_label")"; then
    cx_log "ERROR: failure-stage=scheduler-allocation (private log unavailable)"
    return 1
  fi
  for argument in "$@"; do
    case "$argument" in
      --job-name|--job-name=*|-J|-J*)
        cx_log "ERROR: scheduler job names are managed by CollectiveX"
        return 1
        ;;
    esac
  done
  if ! job_name="$(cx_scheduler_job_name)"; then
    cx_log "ERROR: failure-stage=scheduler-allocation (invalid job name)"
    return 1
  fi
  CX_ALLOCATION_UNCERTAIN=1
  # salloc has no portable --parsable option. Parse the stable grant message
  # used by the production launchers, while also accepting a bare ID from
  # site wrappers. Contain shell-function wrappers that call exit so the
  # launcher can still reconcile and cancel an allocation.
  cx_log "scheduler-request=submit"
  (salloc "$@" --job-name="$job_name" --no-shell) > "$log" 2>&1 || salloc_rc=$?
  if ! job_id="$(sed -nE \
      -e 's/^([0-9]+)(;[^[:space:]]+)?$/\1/p; t found' \
      -e 's/.*Granted job allocation ([0-9]+).*/\1/p; t found' \
      -e 'b' -e ':found' -e 'q' "$log")"; then
    cx_log "ERROR: failure-stage=scheduler-allocation (cannot parse grant)"
    cx_reconcile_salloc_jobid "$job_name" || true
    [ -z "$JOB_ID" ] || cx_record_allocation_jobid "$JOB_ID" || true
    return 1
  fi
  if [ -n "$job_id" ]; then
    [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
    JOB_ID="$job_id"
    CX_ALLOCATION_UNCERTAIN=0
  fi
  if [ "$salloc_rc" != 0 ]; then
    if [ -n "$JOB_ID" ] && cx_verify_salloc_jobid "$JOB_ID"; then
      cx_log "scheduler-request=verified-grant"
      cx_record_allocation_jobid "$JOB_ID" || return 1
      return 0
    fi
    cx_log "ERROR: scheduler-request=rejected"
    if [ "$salloc_rc" -ge 128 ] && [ -z "$JOB_ID" ]; then
      cx_fail_stage scheduler-allocation "$log"
      return 1
    fi
    [ -n "$JOB_ID" ] || cx_reconcile_salloc_jobid "$job_name" || true
    [ -z "$JOB_ID" ] || cx_record_allocation_jobid "$JOB_ID" || true
    cx_fail_stage scheduler-allocation "$log"
    return 1
  fi
  if [ -z "$JOB_ID" ]; then
    cx_log "ERROR: scheduler-request=missing-grant"
    cx_reconcile_salloc_jobid "$job_name" || true
    cx_fail_stage scheduler-allocation "$log"
    return 1
  fi
  cx_record_allocation_jobid "$JOB_ID" || return 1
}

cx_record_allocation_jobid() {
  local job_id="$1" root="${CX_JOB_ROOT:-}" path temporary
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || return 1
  [ -n "$root" ] || return 0
  cx_job_root_is_safe "$root" || return 1
  path="$root/allocation-job-id"
  temporary="$(mktemp "$root/.allocation-job-id.XXXXXX")" || return 1
  chmod 600 "$temporary" || { rm -f -- "$temporary"; return 1; }
  printf '%s\n' "$job_id" > "$temporary" \
    || { rm -f -- "$temporary"; return 1; }
  mv -f -- "$temporary" "$path" || { rm -f -- "$temporary"; return 1; }
}

cx_clear_allocation_jobid() {
  local root="${CX_JOB_ROOT:-}" path
  [ -n "$root" ] || return 0
  cx_job_root_is_safe "$root" || return 1
  path="$root/allocation-job-id"
  [ ! -e "$path" ] || {
    [ -f "$path" ] && [ ! -L "$path" ] \
      && [ "$(stat -c '%u:%a' "$path" 2>/dev/null)" = "$(id -u):600" ] || return 1
    rm -f -- "$path"
  }
}

cx_cancel_job() {
  local job_id="$1" active delay
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
  scancel "$job_id" >/dev/null 2>&1 || true
  for delay in 1 2 4 8 16 32 64; do
    if ! active="$(squeue -h -j "$job_id" -o %A 2>/dev/null)"; then
      sleep "$delay"
      continue
    fi
    [ -n "$active" ] || return 0
    sleep "$delay"
  done
  cx_log "ERROR: scheduled allocation did not terminate during cleanup"
  return 1
}

cx_write_cleanup_guard() {
  local state="$1" root="${CX_JOB_ROOT:-}" safe unsafe
  cx_job_root_is_safe "$root" || return 0
  safe="$root/cleanup-safe"
  unsafe="$root/cleanup-unsafe"
  umask 077
  case "$state" in
    safe) : > "$safe" && rm -f -- "$unsafe" ;;
    unsafe) rm -f -- "$safe" && : > "$unsafe" ;;
    *) return 1 ;;
  esac
}

# A workflow cancellation may kill a foreground Slurm step before Bash can run
# the launcher trap. Reconcile the mode-0600 allocation record from an always()
# workflow step before isolated source cleanup is allowed.
cx_reconcile_recorded_allocation() {
  local root="$1" path job_id
  cx_job_root_is_safe "$root" || return 1
  export CX_JOB_ROOT="$root"
  path="$root/allocation-job-id"
  if [ ! -e "$path" ]; then
    [ -f "$root/cleanup-safe" ] && [ ! -e "$root/cleanup-unsafe" ]
    return
  fi
  [ -f "$path" ] && [ ! -L "$path" ] \
    && [ "$(stat -c '%u:%a' "$path" 2>/dev/null)" = "$(id -u):600" ] \
    || { cx_write_cleanup_guard unsafe || true; return 1; }
  IFS= read -r job_id < "$path" || {
    cx_write_cleanup_guard unsafe || true
    return 1
  }
  [[ "$job_id" =~ ^[1-9][0-9]*$ ]] || {
    cx_write_cleanup_guard unsafe || true
    return 1
  }
  if cx_cancel_job "$job_id" && cx_clear_allocation_jobid; then
    cx_write_cleanup_guard safe
    return
  fi
  cx_write_cleanup_guard unsafe || true
  return 1
}

# Single multi-arch container for ALL NVIDIA SKUs: tag `v0.5.11-cu130` is an OCI
# image index covering linux/amd64 (B200) + linux/arm64 (GB200); enroot import
# pulls the matching arch. (cu130 = CUDA 13, system nccl.h in /usr/include, torch 2.9.x.)
# Import uses the configured tag because Enroot cannot reliably import a
# digest-qualified Docker Hub reference non-interactively.
# (v0.5.12-cu130 was rejected: its 62 layers overflow enroot's overlay-based
# squash creation on these nodes — "failed to mount overlay ... Invalid argument".
# v0.5.11-cu130 imports cleanly.)
# Runtime setup verifies the image-bundled DeepEP build for the detected GPU target.
CX_IMAGE_MULTIARCH="lmsysorg/sglang:v0.5.11-cu130"

# AMD (ROCm/CDNA): single mi35x-tagged image bundles MoRI for all three CDNA
# SKUs (gfx942 mi300x/mi325x + gfx950 mi355x).
CX_IMAGE_AMD_MORI_MI325="rocm/sgl-dev:sglang-0.5.14-rocm720-mi35x-mori-0701"
CX_MORI_COMMIT_MI325="bf99bdf18fc69887a346913ca01c315c2aa9bd4c" # pragma: allowlist secret
cx_default_image() {
  case "$1" in
    mi300x*|mi325x*|mi355x*) echo "$CX_IMAGE_AMD_MORI_MI325" ;;
    b200*|gb200*|b300*|gb300*|h100*|h200*) echo "$CX_IMAGE_MULTIARCH" ;;
    *) cx_die "no default image for runner prefix: $1" ;;
  esac
}

cx_select_image() {
  local image="$1"
  [[ "$image" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]] \
    || cx_die "configured image reference is malformed"
  export COLLECTIVEX_IMAGE="$image"
}

# Create a per-UID cache under validated cluster-local storage. Only the fixed
# /cx-cache mount enters the container; the operator host path does not.
cx_prepare_backend_cache() {
  local cache
  unset CX_PREPARED_BACKEND_CACHE
  cache="$(python3 - "$1" <<'PY'
import os
import stat
import sys

parent = os.path.realpath(sys.argv[1])
marker_name = ".collectivex-cache-v1"
marker_value = b"collectivex-cache-v1\n"
flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
try:
    parent_fd = os.open(parent, flags)
    try:
        probe = f".collectivex-owner-probe-{os.getpid()}"
        os.mkdir(probe, 0o700, dir_fd=parent_fd)
        try:
            probe_fd = os.open(probe, flags, dir_fd=parent_fd)
            owner = os.fstat(probe_fd).st_uid
            os.close(probe_fd)
        finally:
            os.rmdir(probe, dir_fd=parent_fd)
        name = f".collectivex-backend-cache-v4-{os.getuid()}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        cache_fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            cache = os.fstat(cache_fd)
            if cache.st_uid != owner or stat.S_IMODE(cache.st_mode) != 0o700:
                raise OSError
            create = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                marker_fd = os.open(marker_name, create, 0o600, dir_fd=cache_fd)
                os.write(marker_fd, marker_value)
                os.close(marker_fd)
            except FileExistsError:
                pass
            marker_fd = os.open(
                marker_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=cache_fd,
            )
            try:
                marker = os.fstat(marker_fd)
                payload = os.read(marker_fd, len(marker_value) + 1)
                if (
                    not stat.S_ISREG(marker.st_mode)
                    or marker.st_uid != owner
                    or stat.S_IMODE(marker.st_mode) != 0o600
                    or payload != marker_value
                ):
                    raise OSError
            finally:
                os.close(marker_fd)
        finally:
            os.close(cache_fd)
    finally:
        os.close(parent_fd)
except OSError:
    raise SystemExit(1)
print(os.path.join(parent, name), end="")
PY
  )" || return 1
  [[ "$cache" = /* ]] || return 1
  export CX_PREPARED_BACKEND_CACHE="$cache"
}

cx_verify_backend_cache_mount() {
  python3 - "${CX_BACKEND_CACHE_ROOT:-}" <<'PY'
import os
import stat
import sys

root = sys.argv[1]
marker_value = b"collectivex-cache-v1\n"
try:
    if root != "/cx-cache" or os.path.realpath(root) != root:
        raise OSError
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, flags)
    try:
        cache = os.fstat(root_fd)
        if stat.S_IMODE(cache.st_mode) != 0o700:
            raise OSError
        marker_fd = os.open(
            ".collectivex-cache-v1",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        try:
            marker = os.fstat(marker_fd)
            payload = os.read(marker_fd, len(marker_value) + 1)
            if (
                not stat.S_ISREG(marker.st_mode)
                or marker.st_uid != cache.st_uid
                or stat.S_IMODE(marker.st_mode) != 0o600
                or payload != marker_value
            ):
                raise OSError
        finally:
            os.close(marker_fd)
    finally:
        os.close(root_fd)
except OSError:
    raise SystemExit(1)
PY
}

cx_git() {
  GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= "$@"
}

cx_git_in_tree() {
  local directory="$1" canonical
  shift
  [[ "$directory" = /* ]] && [ -d "$directory" ] && [ ! -L "$directory" ] \
    || return 1
  [[ "$directory" != *'*'* && "$directory" != *$'\n'* && "$directory" != *$'\r'* ]] \
    || return 1
  canonical="$(cd -P -- "$directory" && pwd -P)" || return 1
  cx_git -c "safe.directory=$canonical" -C "$canonical" "$@"
}

cx_fetch_revision() {
  local repository="$1" revision="$2" destination="$3" attempt
  for attempt in 1 2 3; do
    rm -rf -- "$destination"
    if cx_git init -q "$destination" \
        && cx_git_in_tree "$destination" remote add origin "$repository" \
        && cx_git_in_tree "$destination" fetch -q --no-tags --depth 1 origin "$revision" \
        && cx_git_in_tree "$destination" -c advice.detachedHead=false \
          checkout -q --detach FETCH_HEAD \
        && [ "$(cx_git_in_tree "$destination" rev-parse HEAD)" = "$revision" ]; then
      return 0
    fi
    [ "$attempt" = 3 ] || sleep $((attempt * 5))
  done
  return 1
}

cx_backend_source_pin() {
  case "$1" in
    deepep-v2)
      printf '%s|%s|%s' \
        "$CX_DEEPEP_V2_COMMIT" "$CX_DEEPEP_V2_TREE" "$CX_DEEPEP_V2_FMT_COMMIT"
      ;;
    deepep-hybrid)
      printf '%s|%s||%s' "$CX_DEEPEP_HYBRID_COMMIT" "$CX_DEEPEP_HYBRID_TREE" \
        "$CX_DEEPEP_HYBRID_NCCL_COMMIT"
      ;;
    *) return 1 ;;
  esac
}

cx_backend_source_path() {
  local root="$1" backend="$2" revision tree fmt nccl pin
  pin="$(cx_backend_source_pin "$backend")" || return 1
  IFS='|' read -r revision tree fmt nccl <<< "$pin"
  printf '%s/%s-%s' "$root" "$backend" "$revision"
}

cx_backend_source_is_valid() {
  local backend="$1" source="$2" revision tree fmt nccl pin status ignored
  pin="$(cx_backend_source_pin "$backend")" || return 1
  IFS='|' read -r revision tree fmt nccl <<< "$pin"
  [ -d "$source" ] && [ ! -L "$source" ] \
    && [ "$(cx_git_in_tree "$source" rev-parse HEAD 2>/dev/null)" = "$revision" ] \
    && [ "$(cx_git_in_tree "$source" rev-parse 'HEAD^{tree}' 2>/dev/null)" = "$tree" ] \
    || return 1
  status="$(GIT_OPTIONAL_LOCKS=0 cx_git_in_tree "$source" \
    status --porcelain --untracked-files=all --ignore-submodules=none \
    2>/dev/null)" || return 1
  if [ "$backend" = deepep-v2 ]; then
    [ "$status" = " M deep_ep/__init__.py" ] \
      && [ "$(grep -Fc "if 'libnccl' in line" "$source/deep_ep/__init__.py")" = 1 ] \
      || return 1
  else
    [ -z "$status" ] || return 1
  fi
  ignored="$(cx_git_in_tree "$source" ls-files --others --ignored --exclude-standard \
    2>/dev/null)" || return 1
  [ -z "$ignored" ] || return 1
  [ -z "$fmt" ] \
    || [ "$(cx_git_in_tree "$source/third-party/fmt" rev-parse HEAD 2>/dev/null)" = "$fmt" ] \
    || return 1
  [ -z "$nccl" ] \
    || [ "$(cx_git_in_tree "$source/third-party/nccl" rev-parse HEAD 2>/dev/null)" = "$nccl" ]
}

cx_apply_deepep_v2_nccl_check_fix() {
  local source="$1"
  python3 - "$source/deep_ep/__init__.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = "for so in [line.strip().split(' ')[-1] for line in f if 'nccl' in line]:"
new = "for so in [line.strip().split(' ')[-1] for line in f if 'libnccl' in line]:"
payload = path.read_text(encoding="utf-8")
if payload.count(old) != 1 or new in payload:
    raise SystemExit(1)
path.write_text(payload.replace(old, new), encoding="utf-8")
PY
}

# Acquire source before compute allocation, preferring the verified same-run GHA seed.
_cx_prepare_backend_source() {
  local mount_src="$1" backend="$2" root source temporary revision tree fmt nccl pin
  local root_mode stage_mode root_owner stage_owner
  root="$mount_src/experimental/CollectiveX/.cx_sources"
  CX_BACKEND_SOURCE_STEP="source mount creation"
  if [ ! -e "$root" ] && [ ! -L "$root" ]; then
    mkdir -m 700 -- "$root" || return 1
  fi
  CX_BACKEND_SOURCE_STEP="source mount ownership validation"
  [ -d "$mount_src" ] && [ ! -L "$mount_src" ] \
    && [ -d "$root" ] && [ ! -L "$root" ] || return 1
  stage_owner="$(stat -c '%u' "$mount_src" 2>/dev/null)" || return 1
  root_owner="$(stat -c '%u' "$root" 2>/dev/null)" || return 1
  [ "$root_owner" = "$stage_owner" ] || return 1
  stage_mode="$(stat -c '%a' "$mount_src" 2>/dev/null)" || return 1
  case "$stage_mode" in 700|[1-7]700) ;; *) return 1 ;; esac
  # Shared stage parents may retain harmless special bits despite mkdir -m.
  CX_BACKEND_SOURCE_STEP="source mount permission inspection"
  root_mode="$(stat -c '%a' "$root" 2>/dev/null)" || return 1
  case "$root_mode" in
    700|[1-7]700) ;;
    *)
      CX_BACKEND_SOURCE_STEP="source mount permission normalization"
      chmod 700 "$root" || return 1
      CX_BACKEND_SOURCE_STEP="source mount permission validation"
      root_mode="$(stat -c '%a' "$root" 2>/dev/null)" || return 1
      case "$root_mode" in 700|[1-7]700) ;; *) return 1 ;; esac
      ;;
  esac
  CX_BACKEND_SOURCE_STEP="git lookup"
  command -v git >/dev/null || return 1
  CX_BACKEND_SOURCE_STEP="source pin resolution"
  source="$(cx_backend_source_path "$root" "$backend")" || return 1
  if [ -e "$source" ] || [ -L "$source" ]; then
    CX_BACKEND_SOURCE_STEP="existing source validation"
    cx_backend_source_is_valid "$backend" "$source"
    return
  fi
  CX_BACKEND_SOURCE_STEP="source checkout creation"
  temporary="$(mktemp -d "$root/.${backend}.XXXXXX")" || return 1
  CX_BACKEND_SOURCE_STEP="source pin resolution"
  pin="$(cx_backend_source_pin "$backend")" || {
    rm -rf -- "$temporary"
    return 1
  }
  IFS='|' read -r revision tree fmt nccl <<< "$pin"
  CX_BACKEND_SOURCE_STEP="revision fetch"
  if ! cx_fetch_revision \
      https://github.com/deepseek-ai/DeepEP "$revision" "$temporary"; then
    rm -rf -- "$temporary"
    return 1
  fi
  CX_BACKEND_SOURCE_STEP="submodule fetch"
  if [ -n "$fmt" ] && ! cx_git_in_tree "$temporary" \
      -c "safe.directory=$temporary/third-party/fmt" \
      submodule update -q --init --depth 1 third-party/fmt; then
    rm -rf -- "$temporary"
    return 1
  fi
  if [ -n "$nccl" ] && ! cx_git_in_tree "$temporary" \
      -c "safe.directory=$temporary/third-party/nccl" \
      submodule update -q --init --depth 1 third-party/nccl; then
    rm -rf -- "$temporary"
    return 1
  fi
  if [ "$backend" = deepep-v2 ]; then
    CX_BACKEND_SOURCE_STEP="upstream NCCL check fix"
    cx_apply_deepep_v2_nccl_check_fix "$temporary" || {
      rm -rf -- "$temporary"
      return 1
    }
  fi
  CX_BACKEND_SOURCE_STEP="source publication validation"
  if ! cx_backend_source_is_valid "$backend" "$temporary" \
      || ! mv -- "$temporary" "$source"; then
    rm -rf -- "$temporary"
    return 1
  fi
}

cx_prepare_backend_source() {
  local log backend="$2" CX_BACKEND_SOURCE_STEP="initialization"
  log="$(cx_private_log_path "backend-source-$backend")" || return 1
  if _cx_prepare_backend_source "$@" > "$log" 2>&1; then
    return 0
  fi
  printf '%s failed\n' "$CX_BACKEND_SOURCE_STEP" >> "$log"
  cx_log "ERROR: backend-source-step=${CX_BACKEND_SOURCE_STEP// /-}"
  cx_fail_stage backend-setup "$log"
}

cx_materialize_backend_source() {
  local backend="$1" destination="$2" source parent temporary
  [ -n "${CX_BACKEND_SOURCE_ROOT:-}" ] || return 1
  source="$(cx_backend_source_path "$CX_BACKEND_SOURCE_ROOT" "$backend")" || return 1
  cx_backend_source_is_valid "$backend" "$source" || return 1
  parent="${destination%/*}"
  [ "$parent" != "$destination" ] && [ -d "$parent" ] && [ ! -L "$parent" ] \
    || return 1
  temporary="$(mktemp -d "$parent/.collectivex-source.XXXXXX")" || return 1
  if ! cp -R -- "$source/." "$temporary/" \
      || ! cx_backend_source_is_valid "$backend" "$temporary"; then
    rm -rf -- "$temporary"
    return 1
  fi
  if ! rm -rf -- "$destination" || ! mv -- "$temporary" "$destination"; then
    rm -rf -- "$temporary"
    return 1
  fi
  if ! cx_backend_source_is_valid "$backend" "$destination"; then
    rm -rf -- "$destination"
    return 1
  fi
  return 0
}

cx_prepare_implicit_stage_base() {
  python3 - "${1:-}" "${2:-}" <<'PY'
import os
from pathlib import Path
import pwd
import stat
import sys

def reject(reason):
    print(f"[collectivex] FATAL: implicit-stage-validation={reason}", file=sys.stderr)
    raise SystemExit(1)

try:
    configured_home = Path(sys.argv[1] or pwd.getpwuid(os.getuid()).pw_dir)
except (KeyError, OSError):
    reject("account-home")
if not configured_home.is_absolute():
    reject("path-shape")
home = Path(os.path.realpath(configured_home))
try:
    metadata = os.stat(home, follow_symlinks=False)
except OSError:
    reject("home-stat")
if not stat.S_ISDIR(metadata.st_mode):
    reject("home-type")
if metadata.st_uid != os.getuid():
    reject("home-owner")
if stat.S_IMODE(metadata.st_mode) & stat.S_IWGRP:
    reject("home-group-writable")
if stat.S_IMODE(metadata.st_mode) & stat.S_IWOTH:
    reject("home-world-writable")
home_owner = metadata.st_uid
try:
    isolation_key = sys.argv[2]
    suffix = ""
    if isolation_key:
        safe = "".join(character if character.isalnum() or character in "_.-" else "-"
                       for character in isolation_key).strip("-")
        if not safe:
            reject("isolation-key")
        if len(safe) > 48:
            safe = safe[:24] + "-" + safe[-23:]
        suffix = "-" + safe
    current = home / f".inferencex-collectivex-stage{suffix}"
    created = False
    try:
        os.mkdir(current, mode=0o700)
        created = True
    except FileExistsError:
        pass
    metadata = os.stat(current, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        reject("child-type")
    if metadata.st_uid not in {os.getuid(), home_owner} and not (
        isolation_key and created and metadata.st_uid == 0
    ):
        reject("child-owner")
    if Path(os.path.realpath(current)) != current:
        reject("child-symlink")
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        reject("child-writable")
    os.chmod(current, 0o700)
except OSError:
    reject("child-access")
print(current, end="")
PY
}

cx_prepare_runner_shared_stage_base() {
  local runner_temp="${RUNNER_TEMP:-}" runner_root
  case "$runner_temp" in
    /*/_work/_temp) runner_root="${runner_temp%/_work/_temp}" ;;
    *) cx_die "canonical AMD execution requires a standard shared runner temp" ;;
  esac
  [ -n "$runner_root" ] && [ "$runner_root" != "$runner_temp" ] \
    || cx_die "canonical AMD execution requires a shared runner root"
  cx_prepare_implicit_stage_base "$runner_root"
}

cx_lock_canonical_gha_env() {
  local runner="$1" expected_nodes expected_gpn expected_world trusted_lock_dir=""
  local trusted_stage_dir=""
  local trusted_qos=""
  local trusted_socket_ifname="" trusted_rdma_devices=""
  local trusted_ib_gid_index="" trusted_rdma_service_level=""
  local trusted_rdma_traffic_class=""
  [ "${COLLECTIVEX_CANONICAL_GHA:-0}" = 1 ] || return 0
  [ "${GITHUB_ACTIONS:-}" = true ] \
    || cx_die "canonical CollectiveX execution requires GitHub Actions"
  [ -n "${CX_SHARD_FILE:-}" ] && [ "${CX_SHARD_SKU:-}" = "$runner" ] \
    || cx_die "canonical CollectiveX execution requires a matched shard"
  [[ "${GITHUB_RUN_ID:-}" =~ ^[1-9][0-9]*$ \
    && "${GITHUB_RUN_ATTEMPT:-}" =~ ^[1-9][0-9]*$ \
    && "${COLLECTIVEX_SOURCE_SHA:-}" =~ ^[0-9a-f]{40,64}$ ]] \
    || cx_die "canonical CollectiveX workflow identity is incomplete"

  # cx_load_operator_config clears inherited values before setting this process marker.
  # Preserve only values parsed from that private strict document.
  if [ "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" = "$$" ]; then
    trusted_lock_dir="${CX_LOCK_DIR:-}"
    trusted_stage_dir="${CX_STAGE_DIR:-}"
    trusted_qos="${CX_QOS:-}"
    trusted_socket_ifname="${CX_SOCKET_IFNAME:-}"
    trusted_rdma_devices="${CX_RDMA_DEVICES:-}"
    trusted_ib_gid_index="${CX_IB_GID_INDEX:-}"
    trusted_rdma_service_level="${CX_RDMA_SERVICE_LEVEL:-}"
    trusted_rdma_traffic_class="${CX_RDMA_TRAFFIC_CLASS:-}"
  fi
  # The legacy B300 operator row contains a root-owned stage path. B300's
  # compute-visible account home is the canonical source for its private base.
  case "$runner" in b300|gb300) trusted_stage_dir="" ;; esac
  unset CX_NCCL_HOME CX_MASTER_PORT CX_MORI_KERNEL_TYPE CX_LOCK_DIR CX_STAGE_DIR CX_QOS
  unset CX_STAGE_PARENT_OWNER_OK
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  unset CX_SOCKET_IFNAME CX_RDMA_DEVICES CX_IB_GID_INDEX CX_RDMA_SERVICE_LEVEL
  unset CX_RDMA_TRAFFIC_CLASS
  unset NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL
  unset NVSHMEM_DISABLE_IB NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset NVSHMEM_HCA_PE_MAPPING NVSHMEM_REMOTE_TRANSPORT
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  unset HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE
  unset MORI_COMMIT MORI_DISABLE_AUTO_XGMI MORI_ENABLE_SDMA
  unset MORI_APP_LOG_LEVEL MORI_SHMEM_LOG_LEVEL MORI_IO_LOG_LEVEL
  unset NCCL_CUMEM_ENABLE NCCL_MNNVL_ENABLE MC_FORCE_MNNVL
  unset CX_BACKEND_CACHE_ROOT
  unset CX_PREPARED_BACKEND_CACHE CX_BACKEND_SOURCE_ROOT

  [ -n "${CX_SQUASH_DIR:-}" ] \
    || cx_die "canonical CollectiveX execution requires shared container storage"
  if [ -z "$trusted_stage_dir" ]; then
    case "$runner" in
      h100-dgxc)
        trusted_stage_dir="$(cx_prepare_implicit_stage_base "${CX_SQUASH_DIR%/*}")" \
          || cx_die "canonical CollectiveX execution cannot create an isolated shared stage directory"
        ;;
      b300)
        trusted_stage_dir="$(cx_prepare_implicit_stage_base "" \
          "${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-}}")" \
          || cx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      gb300)
        trusted_stage_dir="$(cx_prepare_implicit_stage_base "" \
          "${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-}}")" \
          || cx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      h200-dgxc|b200-dgxc)
        trusted_stage_dir="$(cx_prepare_implicit_stage_base)" \
          || cx_die "canonical CollectiveX execution cannot create an isolated stage directory"
        ;;
      mi300x|mi325x|mi355x)
        # AMD self-hosted runners and compute nodes share the runner filesystem,
        # while the image cache may be root-owned. Derive a runner-owned base
        # outside _work instead of weakening stage ownership validation.
        trusted_stage_dir="$(cx_prepare_runner_shared_stage_base)" \
          || cx_die "canonical AMD execution cannot create an isolated shared stage directory"
        ;;
      *) cx_die "canonical CollectiveX execution requires a configured shared stage directory" ;;
    esac
  elif [ "$runner" = mi300x ]; then
    # The MI300X runner home is a shared-filesystem symlink. Resolve the
    # operator-selected base once; cx_stage_path still validates the canonical
    # directory's ownership, permissions, overlap, and per-run child path.
    trusted_stage_dir="$(python3 -c \
      'import os,sys; p=os.path.realpath(sys.argv[1]); assert os.path.isdir(p); print(p,end="")' \
      "$trusted_stage_dir")" \
      || cx_die "canonical MI300X execution cannot resolve the shared stage directory"
  fi
  if [ "$runner" = b300 ]; then
    CX_STAGE_PARENT_OWNER_OK=1
  fi

  case "$runner" in
    h100-dgxc|h200-dgxc|b200-dgxc|b300)
      expected_nodes="${CX_NODES:-}"; expected_gpn=8
      [ "$expected_nodes" = 1 ] || [ "$expected_nodes" = 2 ] \
        || cx_die "canonical NVIDIA execution requires one or two nodes"
      CX_IMAGE="$CX_IMAGE_MULTIARCH"
      CX_NCCL_HOME=/usr
      ;;
    gb200|gb300)
      expected_nodes="${CX_NODES:-}"; expected_gpn=4
      [ "$expected_nodes" = 2 ] || [ "$expected_nodes" = 4 ] \
        || cx_die "canonical GB execution requires two or four trays"
      CX_IMAGE="$CX_IMAGE_MULTIARCH"
      CX_NCCL_HOME=/usr
      CX_MASTER_PORT=29551
      ;;
    mi300x|mi325x|mi355x)
      expected_nodes="${CX_NODES:-}"; expected_gpn=8
      [ "$expected_nodes" = 1 ] || [ "$expected_nodes" = 2 ] \
        || cx_die "canonical AMD execution requires one or two nodes"
      # All three CDNA SKUs run the same MoRI-bundled image (mi35x tag covers
      # gfx942 + gfx950); mi355x was migrated off the older 0227 image (sglang
      # 0.5.9), whose MoRI build hung during EpDispatchCombineOp construction.
      CX_IMAGE="$CX_IMAGE_AMD_MORI_MI325"
      if [ "$expected_nodes" = 2 ]; then
        CX_MORI_KERNEL_TYPE=internode-v1
      else
        CX_MORI_KERNEL_TYPE=asyncll
      fi
      MORI_COMMIT="$CX_MORI_COMMIT_MI325"
      MORI_DISABLE_AUTO_XGMI=0
      MORI_ENABLE_SDMA=1
      MORI_APP_LOG_LEVEL=info
      MORI_SHMEM_LOG_LEVEL=info
      MORI_IO_LOG_LEVEL=info
      ;;
    *) cx_die "canonical CollectiveX runner is not registered" ;;
  esac
  case "$runner:$trusted_lock_dir" in
    mi300x:?*|mi325x:?*|mi355x:?*) export CX_LOCK_DIR="$trusted_lock_dir" ;;
  esac
  CX_STAGE_DIR="$trusted_stage_dir"
  [ -z "$trusted_qos" ] || export CX_QOS="$trusted_qos"
  [ -z "$trusted_socket_ifname" ] \
    || export CX_SOCKET_IFNAME="$trusted_socket_ifname"
  [ -z "$trusted_rdma_devices" ] \
    || export CX_RDMA_DEVICES="$trusted_rdma_devices"
  [ -z "$trusted_ib_gid_index" ] \
    || export CX_IB_GID_INDEX="$trusted_ib_gid_index"
  [ -z "$trusted_rdma_service_level" ] \
    || export CX_RDMA_SERVICE_LEVEL="$trusted_rdma_service_level"
  [ -z "$trusted_rdma_traffic_class" ] \
    || export CX_RDMA_TRAFFIC_CLASS="$trusted_rdma_traffic_class"
  export CX_STAGE_DIR
  [ "${CX_NODES:-}" = "$expected_nodes" ] \
    && [ "${CX_GPUS_PER_NODE:-}" = "$expected_gpn" ] \
    || cx_die "canonical CollectiveX placement differs from the shard"
  expected_world=$((expected_nodes * expected_gpn))
  CX_NGPUS="$expected_world"
  CX_SEED=67
  case "$runner" in mi300x|mi325x|mi355x) CX_RUN_TIMEOUT=1800 ;; *) CX_RUN_TIMEOUT=900 ;; esac
  unset CX_PUBLIC_RUNNER CX_GB_PRODUCT CX_DRYRUN CX_TIMING CX_ALLOW_MNNVL
  unset CX_ENROOT_LOCAL_IMPORT COLLECTIVEX_IMAGE COLLECTIVEX_SQUASH_SHA256
  export CX_IMAGE CX_NGPUS CX_SEED CX_RUN_TIMEOUT
  case "$runner" in
    h100-dgxc|h200-dgxc|b200-dgxc|b300) export CX_NCCL_HOME ;;
    gb200|gb300) export CX_NCCL_HOME CX_MASTER_PORT ;;
    mi300x|mi325x|mi355x)
      export CX_MORI_KERNEL_TYPE MORI_COMMIT MORI_DISABLE_AUTO_XGMI MORI_ENABLE_SDMA
      export MORI_APP_LOG_LEVEL MORI_SHMEM_LOG_LEVEL MORI_IO_LOG_LEVEL
      ;;
  esac
}

cx_export_squash_identity() {
  local image="$1" digest log
  log="$(cx_private_log_path container-hash)"
  digest="$(sha256sum "$image" 2>> "$log" | awk '{print $1}')"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] \
    || { cx_fail_stage container-hash "$log"; return 1; }
  export COLLECTIVEX_SQUASH_SHA256="$digest"
}

cx_squash_path() {
  local squash_dir="$1" image="$2" key platform run_scope
  case "${CX_IMAGE_PLATFORM:-}" in
    linux/amd64) platform="" ;;
    linux/arm64) platform="_linux_arm64" ;;
    *) return 1 ;;
  esac
  run_scope="${GITHUB_RUN_ID:-${COLLECTIVEX_EXECUTION_ID:-manual}}-${GITHUB_RUN_ATTEMPT:-1}"
  run_scope="$(printf '%s' "$run_scope" | tr -cs 'A-Za-z0-9_.-' '-')" || return 1
  run_scope="${run_scope#-}"; run_scope="${run_scope%-}"
  [ -n "$run_scope" ] || return 1
  key="${CX_SQUASH_FORMAT_VERSION}${platform}_${run_scope}_$(
    printf '%s' "$image" | sed 's#[/:@#]#_#g'
  )"
  printf '%s' "$squash_dir/${key}.sqsh"
}

# cx_ensure_squash <squash_dir> <image>  ->  echoes the squash file path.
# Imports via Enroot only if a valid squash is not already present, under a lock.
cx_ensure_squash() {
  local squash_dir="$1" image="$2" key sq locks lock_fd log
  local enroot_local="" import_rc=0 machine
  log="$(cx_private_log_path container-import)"
  machine="$(uname -m)"
  case "${CX_IMAGE_PLATFORM:-}:$machine" in
    linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
    *) cx_fail_stage container-import "$log"; return 1 ;;
  esac
  mkdir -p "$squash_dir" 2>> "$log" \
    || { cx_fail_stage container-import "$log"; return 1; }
  sq="$(cx_squash_path "$squash_dir" "$image")" \
    || { cx_fail_stage container-import "$log"; return 1; }
  key="${sq##*/}"
  key="${key%.sqsh}"
  locks="$squash_dir/.locks"
  mkdir -p "$locks" 2>> "$log" \
    || { cx_fail_stage container-import "$log"; return 1; }
  { exec {lock_fd}>"$locks/${key}.lock"; } 2>> "$log" \
    || { cx_fail_stage container-import "$log"; return 1; }
  flock -w 900 "$lock_fd" 2>> "$log" \
    || { cx_fail_stage container-import "$log"; return 1; }
  if unsquashfs -l "$sq" >/dev/null 2>&1; then
    cx_log "container squash ready"
  else
    cx_log "importing configured container image"
    rm -f "$sq" 2>> "$log" \
      || { cx_fail_stage container-import "$log"; return 1; }
    # </dev/null: never block on an interactive password prompt.
    if [ "${CX_ENROOT_LOCAL_IMPORT:-0}" = 1 ]; then
      enroot_local="$(mktemp -d /tmp/inferencex-collectivex-enroot.XXXXXX)" \
        || { cx_fail_stage container-import "$log"; return 1; }
      (
        trap 'rm -rf -- "$enroot_local"' EXIT
        export ENROOT_TEMP_PATH="$enroot_local/tmp"
        export ENROOT_CACHE_PATH="$enroot_local/cache"
        export ENROOT_DATA_PATH="$enroot_local/data"
        export ENROOT_RUNTIME_PATH="$enroot_local/run"
        mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" \
          "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
        SOURCE_DATE_EPOCH="$CX_SQUASH_SOURCE_DATE_EPOCH" \
          enroot import -o "$sq" "docker://$image" </dev/null
      ) >> "$log" 2>&1 || import_rc=$?
      rm -rf -- "$enroot_local" >/dev/null 2>&1 || true
      [ "$import_rc" = 0 ] \
        || { cx_fail_stage container-import "$log"; return 1; }
    else
      SOURCE_DATE_EPOCH="$CX_SQUASH_SOURCE_DATE_EPOCH" \
        enroot import -o "$sq" "docker://$image" </dev/null >> "$log" 2>&1 \
        || { cx_fail_stage container-import "$log"; return 1; }
    fi
    unsquashfs -l "$sq" >> "$log" 2>&1 \
      || { cx_fail_stage container-import "$log"; return 1; }
  fi
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  echo "$sq"
}

# Import on an allocated compute node so multiarch tags resolve for the target
# architecture. The squash directory must be shared with the submit host.
cx_ensure_squash_on_job() {
  local job_id="$1" squash_dir="$2" image="$3" lock_dir="${4:-}" sq key lock
  local log_label=container-import log
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
  case "${CX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${CX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  sq="$(cx_squash_path "$squash_dir" "$image")" || return 1
  key="${sq##*/}"
  key="${key%.sqsh}"
  [ -n "$lock_dir" ] || lock_dir="$squash_dir/.locks"
  lock="$lock_dir/${key}.lock"
  log="$(cx_private_log_path "$log_label")"
  # Import (or verify) the squash on EVERY allocated node, not just one. On SKUs
  # whose squash_dir is node-local (e.g. mi355x/mi300x /var/lib/squash) a single-
  # node import leaves the remaining nodes without the squash, so the per-node
  # container-hash check and the benchmark itself then fail with "No such file"
  # on whichever node was missed. The per-node script below is flock-guarded and
  # idempotent: on shared-FS SKUs the first node imports and every other node
  # short-circuits on the unsquashfs check, so no redundant import occurs.
  if ! srun --jobid="$job_id" --nodes="${CX_NODES:-1}" --ntasks="${CX_NODES:-1}" \
      --ntasks-per-node=1 --chdir=/tmp \
      --export="$(cx_host_exports)" \
      bash -s -- "$sq" "$lock" "$image" "$CX_SQUASH_SOURCE_DATE_EPOCH" \
      "$CX_IMAGE_PLATFORM" \
      > "$log" 2>&1 <<'BASH'
set -euo pipefail
sq="$1"; lock="$2"; image="$3"; source_date_epoch="$4"; platform="$5"
machine="$(uname -m)"
case "$platform:$machine" in
  linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
  *) exit 13 ;;
esac
compute_home="$(mktemp -d /tmp/inferencex-collectivex-home.XXXXXX)"
trap 'rm -rf -- "$compute_home"' EXIT
export HOME="$compute_home" XDG_CACHE_HOME="$compute_home/.cache"
export ENROOT_TEMP_PATH="$compute_home/enroot-tmp"
export ENROOT_CACHE_PATH="$compute_home/enroot-cache"
export ENROOT_DATA_PATH="$compute_home/enroot-data"
export ENROOT_RUNTIME_PATH="$compute_home/enroot-run"
mkdir -p "$(dirname "$sq")" "$(dirname "$lock")" \
  "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"
exec 9>"$lock"
# Wait indefinitely: with a shared-FS squash_dir every node contends on the same
# lock, and a slow cold import must not spuriously time out the waiters. The lock
# is tied to fd 9, so a crashed importer releases it automatically. Node-local
# squash_dirs use independent per-node locks, so there is no contention there.
flock 9
if unsquashfs -l "$sq" >/dev/null 2>&1; then
  echo 'container squash ready'
else
  rm -f -- "$sq"
  SOURCE_DATE_EPOCH="$source_date_epoch" \
    enroot import -o "$sq" "docker://$image" </dev/null
  unsquashfs -l "$sq" >/dev/null 2>&1
fi
BASH
  then
    cx_fail_stage container-import "$log"
    return 1
  fi
  printf '%s' "$sq"
}

cx_preflight_allocation() {
  local job_id="$1" nodes="$2" mount_src="$3" squash="$4" shard="${5:-}"
  local log rc=0 runtime shard_path="" probe_root probe_token index
  runtime="$mount_src/experimental/CollectiveX/runtime/run_in_container.sh"
  [ -z "$shard" ] || shard_path="$mount_src/experimental/CollectiveX/$shard"
  log="$(cx_private_log_path allocation-preflight)"
  probe_root="$mount_src/.collectivex-preflight"
  probe_token="$probe_root/source"
  if [ -e "$probe_root" ] || [ -L "$probe_root" ] \
      || ! mkdir -m 700 "$probe_root"; then
    cx_fail_stage repository-stage "$log"
    return 1
  fi
  if ! printf '%s\n' "${COLLECTIVEX_EXECUTION_ID:-manual-$$}" > "$probe_token" \
      || ! chmod 600 "$probe_token"; then
    chmod 700 "$probe_root" >/dev/null 2>&1 || true
    rm -rf -- "$probe_root" >/dev/null 2>&1 || true
    cx_fail_stage repository-stage "$log"
    return 1
  fi
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp --input=all \
    --export="$(cx_host_exports)" bash -s -- "$runtime" "$shard_path" "$squash" \
    "$CX_IMAGE_PLATFORM" "$probe_root" \
    > "$log" 2>&1 <<'BASH' || rc=$?
set -euo pipefail
machine="$(uname -m)"
case "$4:$machine" in
  linux/amd64:x86_64|linux/amd64:amd64|linux/arm64:aarch64|linux/arm64:arm64) ;;
  *) exit 13 ;;
esac
test -r "$1" || exit 10
[ -z "$2" ] || test -r "$2" || exit 11
test -r "$3" || exit 12
unsquashfs -s "$3" >/dev/null 2>&1 || exit 12
case "${SLURM_NODEID:-}" in ""|*[!0-9]*) exit 10 ;; esac
[ -d "$5" ] && [ ! -L "$5" ] && [ -r "$5/source" ] || exit 10
(set -C; cat "$5/source" > "$5/node-$SLURM_NODEID") || exit 10
cmp -s -- "$5/source" "$5/node-$SLURM_NODEID" || exit 10
BASH
  if [ "$rc" = 0 ]; then
    for ((index = 0; index < nodes; index++)); do
      if ! cmp -s -- "$probe_token" "$probe_root/node-$index"; then
        rc=10
        break
      fi
    done
  fi
  if [ -d "$probe_root" ] && [ ! -L "$probe_root" ]; then
    chmod 700 "$probe_root" >/dev/null 2>&1 || rc=10
  fi
  rm -rf -- "$probe_root" >/dev/null 2>&1 || rc=10
  [ "$rc" = 0 ] && return 0
  case "$rc" in
    10|11) cx_fail_stage repository-stage "$log" ;;
    12) cx_fail_stage container-hash "$log" ;;
    *) cx_fail_stage container-launch "$log" ;;
  esac
  return 1
}

# A clean nvidia-smi inventory does not prove that a prior cancelled workload
# released every CUDA context. Retaining each primary context catches poisoned
# allocations before a full shard spends time failing every case.
cx_validate_cuda_context_on_job() {
  local job_id="$1" nodes="$2" gpus_per_node="$3" log_label=cuda-context log
  case "${CX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${CX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(cx_private_log_path "$log_label")"
  CX_CUDA_CONTEXT_LOG="$log"
  srun --jobid="$job_id" --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --gres=gpu:"$gpus_per_node" --chdir=/tmp --input=all \
    --export="$(cx_host_exports)" python3 - "$gpus_per_node" \
    >"$log" 2>&1 <<'PY'
import ctypes
import socket
import sys

expected = int(sys.argv[1])
cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuda.cuDeviceGetCount.restype = ctypes.c_int
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuDeviceGet.restype = ctypes.c_int
cuda.cuDevicePrimaryCtxRetain.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
]
cuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
cuda.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
cuda.cuDevicePrimaryCtxRelease.restype = ctypes.c_int


def check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA result {result}")


check(cuda.cuInit(0), "CUDA initialization")
count = ctypes.c_int()
check(cuda.cuDeviceGetCount(ctypes.byref(count)), "CUDA device count")
if count.value != expected:
    raise RuntimeError(
        f"CUDA device count mismatch on {socket.gethostname()}: "
        f"expected {expected}, found {count.value}"
    )

devices: list[int] = []
try:
    for ordinal in range(expected):
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        check(cuda.cuDeviceGet(ctypes.byref(device), ordinal), "CUDA device lookup")
        result = cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device.value)
        if result != 0:
            raise RuntimeError(
                f"CUDA primary context retain failed on {socket.gethostname()} "
                f"device {ordinal} with CUDA result {result}"
            )
        devices.append(device.value)
finally:
    for device in reversed(devices):
        check(cuda.cuDevicePrimaryCtxRelease(device), "CUDA primary context release")
PY
}

# Resolve the exact per-execution child before any copy starts, so the parent
# EXIT trap can remove an interrupted partial stage. The configured base must
# already exist on compute-visible storage and must not traverse symlinks.
cx_stage_path() {
  local repo_root="$1" stage_base="${2:-}" tag safe_tag stage_path
  tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual-$$}}"
  [[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || cx_die "invalid staging execution identity"
  safe_tag="$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
  if [ -z "$stage_base" ] || [ "$stage_base" = "$repo_root" ]; then
    [ -n "${CX_SQUASH_DIR:-}" ] \
      || cx_die "CollectiveX staging requires CX_STAGE_DIR or CX_SQUASH_DIR"
    stage_base="$CX_SQUASH_DIR"
    stage_path="${stage_base%/}/.collectivex-stage-$safe_tag"
  else
    stage_path="${stage_base%/}/job_$safe_tag"
  fi
  python3 - "$repo_root" "$stage_base" "$stage_path" \
    "${CX_JOB_ROOT:-}" "${GITHUB_WORKSPACE:-}" \
    "${CX_STAGE_PARENT_OWNER_OK:-0}" <<'PY'
import os
import stat
import sys

repo, base, child, job_root, workspace, allow_parent_owner = sys.argv[1:]
def reject(reason):
    print(f"[collectivex] FATAL: stage-base-validation={reason}", file=sys.stderr)
    raise SystemExit(1)

try:
    if (
        not os.path.isabs(repo)
        or os.path.realpath(repo) != repo
        or not os.path.isabs(base)
        or os.path.realpath(base) != base
        or not os.path.isabs(child)
        or os.path.dirname(child) != base.rstrip("/")
    ):
        reject("path-shape")
    if os.path.lexists(child):
        reject("child-exists")
    try:
        metadata = os.stat(base, follow_symlinks=False)
    except OSError:
        reject("base-stat")
    excluded = [repo]
    excluded.extend(path for path in (job_root, workspace) if path)
    for path in excluded:
        resolved = os.path.realpath(path)
        if os.path.commonpath((base, resolved)) == resolved:
            reject("overlap")
    if not stat.S_ISDIR(metadata.st_mode):
        reject("not-directory")
    if metadata.st_uid != os.getuid():
        if allow_parent_owner != "1":
            reject("owner")
        parent = os.path.dirname(base.rstrip("/"))
        parent_metadata = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or metadata.st_uid not in {parent_metadata.st_uid, 0}
            or stat.S_IMODE(parent_metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            reject("parent-owner")
    if stat.S_IMODE(metadata.st_mode) & stat.S_IWGRP:
        reject("group-writable")
    if stat.S_IMODE(metadata.st_mode) & stat.S_IWOTH:
        reject("world-writable")
    if not os.access(base, os.W_OK | os.X_OK):
        reject("access")
except ValueError:
    reject("path-shape")
print(child, end="")
PY
}

# Stage only the public benchmark tree into a pre-resolved, private execution
# child. A runner-owned marker makes recursive cleanup an explicit capability.
cx_stage_repo() {
  local repo_root="$1" stage_dir="$2" expected log tag marker
  cx_validate_shard_control "$repo_root/experimental/CollectiveX"
  expected="$(cx_stage_path "$repo_root" "${CX_STAGE_DIR:-}")" \
    || cx_die "configured stage base is unavailable or unsafe"
  [ "$stage_dir" = "$expected" ] \
    || cx_die "execution stage differs from the configured stage base"
  tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual-$$}}"
  if [ -e "$stage_dir" ] || [ -L "$stage_dir" ]; then
    cx_die "refusing to reuse a pre-existing execution stage"
  fi
  mkdir -m 700 "$stage_dir" 2>/dev/null \
    || cx_die "cannot create the configured stage directory"
  chmod 700 "$stage_dir" 2>/dev/null \
    || cx_die "cannot protect the configured stage directory"
  marker="$stage_dir/.collectivex-stage-v1"
  umask 077
  (set -C; printf 'collectivex-stage-v1\n%s\n' "$tag" > "$marker") 2>/dev/null \
    || cx_die "cannot claim the configured stage directory"
  chmod 600 "$marker" 2>/dev/null \
    || cx_die "cannot protect the configured stage directory"
  mkdir -m 700 "$stage_dir/experimental" 2>/dev/null \
    || cx_die "cannot create the configured stage directory"
  cx_log "staging CollectiveX on compute-visible storage"
  log="$(cx_private_log_path repository-stage)"
  if ! python3 - "$repo_root/experimental/CollectiveX" \
      "$stage_dir/experimental/CollectiveX" > "$log" 2>&1 <<'PY'
import os
from pathlib import Path
import shutil
import sys

def report_error(kind, value, trace):
    error_number = getattr(value, "errno", 0)
    print(f"collectivex-stage-copy-error={kind.__name__}:{error_number or 0}", file=sys.stderr)

sys.excepthook = report_error
source, target = map(Path, sys.argv[1:])
excluded = {
    Path("__pycache__"), Path("results"), Path(".cx_workloads"),
    Path(".cx_backend"), Path(".cx_sources"), Path("configs/platforms.yaml"),
    Path("private-infra.md"), Path("goal.md"), Path("notes.md"),
}
for root, directories, files in os.walk(source, followlinks=False):
    root_path = Path(root)
    relative_root = root_path.relative_to(source)
    directories[:] = [
        name for name in directories
        if relative_root / name not in excluded and not (root_path / name).is_symlink()
    ]
    destination = target / relative_root
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in files:
        relative = relative_root / name
        if relative in excluded:
            continue
        source_file = root_path / name
        if source_file.is_symlink() or not source_file.is_file():
            raise RuntimeError("unsupported source entry")
        with source_file.open("rb") as input_file, (destination / name).open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file)
PY
  then
    copy_error="$(grep -aoE 'collectivex-stage-copy-error=[A-Za-z]+:[0-9]+' "$log" \
      | tail -n 1 || true)"
    [ -z "$copy_error" ] || cx_log "ERROR: repository-stage-$copy_error"
    rm -rf -- "$stage_dir" >/dev/null 2>&1 \
      || cx_log "ERROR: cannot remove the incomplete execution stage"
    cx_fail_stage repository-stage "$log" || true
    return 1
  fi
}

# cx_collect_results <mount_src> <repo_root>
# When the run used a staged (compute-visible) mount, copy result JSONs back to
# the original checkout's results/ so the workflow's upload-artifact (which reads
# the checkout, not the stage dir) finds them. No-op when no staging was used.
cx_collect_results() {
  local mount_src="$1" repo_root="$2" dst log
  local -a files
  [ "$mount_src" = "$repo_root" ] && return 0
  log="$(cx_private_log_path "artifact-collection-$$-${RANDOM}")"
  dst="$repo_root/experimental/CollectiveX/results"
  mkdir -p "$dst" 2>> "$log" \
    || { cx_log "ERROR: cannot create checkout result directory"; return 1; }
  shopt -s nullglob
  files=("$mount_src/experimental/CollectiveX/results/"*.json)
  shopt -u nullglob
  [ "${#files[@]}" -gt 0 ] || { cx_log "ERROR: staged run produced no result JSON"; return 1; }
  cp -- "${files[@]}" "$dst/" >> "$log" 2>&1 \
    || { cx_log "ERROR: staged result collection failed"; return 1; }
  cx_log "collected staged results for artifact validation"
}

cx_cleanup_stage() {
  local mount_src="$1" repo_root="$2" base="${CX_STAGE_DIR:-}" tag safe_tag expected
  tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual-$$}}"
  safe_tag="$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
  [ "$mount_src" != "$repo_root" ] || return 0
  if [ -n "$base" ] && [ "$base" != "$repo_root" ]; then
    expected="${base%/}/job_$safe_tag"
  else
    [ -n "${CX_SQUASH_DIR:-}" ] \
      || { cx_log "ERROR: cannot identify the generated stage directory"; return 1; }
    expected="${CX_SQUASH_DIR%/}/.collectivex-stage-$safe_tag"
  fi
  if [ "$mount_src" != "$expected" ] || [ "$mount_src" = / ] \
      || { [ -n "$base" ] && [ "$mount_src" = "$base" ]; }; then
    cx_log "ERROR: refusing to remove an unrecognized stage directory"
    return 1
  fi
  if ! python3 - "$mount_src" "$tag" "${CX_STAGE_PARENT_OWNER_OK:-0}" <<'PY'
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
expected = f"collectivex-stage-v1\n{sys.argv[2]}\n"
allow_uid_mapped_root = sys.argv[3] == "1"
try:
    metadata = os.stat(root, follow_symlinks=False)
    marker = root / ".collectivex-stage-v1"
    owner_ok = metadata.st_uid == os.getuid()
    if not owner_ok and allow_uid_mapped_root and metadata.st_uid == 0:
        base = root.parent
        private_parent = base.parent
        base_metadata = os.stat(base, follow_symlinks=False)
        parent_metadata = os.stat(private_parent, follow_symlinks=False)
        owner_ok = (
            stat.S_ISDIR(base_metadata.st_mode)
            and base_metadata.st_uid == 0
            and stat.S_IMODE(base_metadata.st_mode) == 0o700
            and stat.S_ISDIR(parent_metadata.st_mode)
            and parent_metadata.st_uid != 0
            and not stat.S_IMODE(parent_metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
        )
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or not owner_ok
        or (stat.S_IMODE(metadata.st_mode) & 0o777) != 0o700
    ):
        raise OSError
    entries = list(root.iterdir())
    if marker.exists():
        marker_metadata = os.stat(marker, follow_symlinks=False)
        if (
            not stat.S_ISREG(marker_metadata.st_mode)
            or marker_metadata.st_uid != metadata.st_uid
            or stat.S_IMODE(marker_metadata.st_mode) != 0o600
        ):
            raise OSError
        marker_content = marker.read_text()
        if marker_content != expected and entries != [marker]:
            raise OSError
    elif entries:
        raise OSError
except (OSError, UnicodeError):
    raise SystemExit(1)
PY
  then
    cx_log "ERROR: refusing to remove an unowned stage directory"
    return 1
  fi
  rm -rf -- "$mount_src" >/dev/null 2>&1 || {
    cx_log "ERROR: cannot remove generated stage directory"
    return 1
  }
  cx_log "removed generated per-execution stage directory"
}

# Run one validated shard with one Slurm task per GPU. Launchers provide only
# allocation/container policy through globals and CX_DISTRIBUTED_CONTAINER_ARGS.
# shellcheck disable=SC2153
cx_run_distributed_shard() {
  local build_log build_rc cases_file expected_cases ci=0 failed_cases=0
  local ph mode routing hidden topk experts ladder suite workload
  local canonical case_id ep timing case_iters case_trials case_warmup case_stem
  local scope scale_up_transport scale_out_transport transport topology_class nodes gpn domain
  local workload_dir workload_ladder workload_log stage_rc attempt_tag out
  local runtime_log run_rc summary_log
  local -a container_args workload_args ep_args
  [ "${NODES:-0}" -gt 1 ] && [ "${NGPUS:-0}" = "$((NODES * GPN))" ] \
    || cx_die "invalid distributed launcher placement"
  [ -n "${JOB_ID:-}" ] && [ -n "${SQUASH_FILE:-}" ] \
    && [ -n "${CONTAINER_MOUNTS:-}" ] || cx_die "distributed launcher is incomplete"
  [ -n "${SOURCE_BACKEND_ENV:-}" ] && [ -n "${BACKEND_PROBE:-}" ] \
    && [ -n "${WRAP:-}" ] || cx_die "distributed rank wrapper is incomplete"

  cx_resolve_slurm_rendezvous "$JOB_ID"
  mkdir -p "$MOUNT_SRC/experimental/CollectiveX/results"
  container_args=(--container-mounts="$CONTAINER_MOUNTS" --no-container-mount-home
    --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint)
  if declare -p CX_DISTRIBUTED_CONTAINER_ARGS >/dev/null 2>&1; then
    container_args+=("${CX_DISTRIBUTED_CONTAINER_ARGS[@]}")
  fi
  local container_name="cxep_${JOB_ID}"

  cx_log "distributed backend preparation: bench=$CX_BENCH nodes=$NODES"
  cx_set_failure_stage backend-setup
  build_log="$(cx_private_log_path backend-prepare)"
  set +e
  srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 --chdir=/tmp \
    --container-name="$container_name" --container-image="$SQUASH_FILE" \
    "${container_args[@]}" --export="$(cx_container_exports),CX_BUILD_ONLY=1" \
    bash /ix/experimental/CollectiveX/runtime/run_in_container.sh \
    </dev/null >"$build_log" 2>&1
  build_rc=$?
  if [ "$build_rc" = 0 ]; then
    srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 --chdir=/tmp \
      --container-name="$container_name" --container-image="$SQUASH_FILE" \
      "${container_args[@]}" \
      --export="$(cx_container_exports)" bash -c "$BACKEND_PROBE" \
      </dev/null >>"$build_log" 2>&1
    build_rc=$?
  fi
  set -e
  if [ "$build_rc" != 0 ]; then
    cx_fail_stage backend-setup "$build_log" || true
    return "$build_rc"
  fi
  cx_set_failure_stage execution

  cases_file="$(mktemp)" || return 1
  local shard="${CX_SHARD_FILE:-}"
  [ -z "$shard" ] || [ -f "$shard" ] || shard="$CX_DIR/$shard"
  # Iterable benchmark version is a shard-level scalar; export it once so the
  # harness copies it verbatim into every emitted result for this shard.
  if [ -n "$shard" ] && [ -f "$shard" ]; then
    CX_VERSION="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['version'])" "$shard")"
    export CX_VERSION
  fi
  if [ -n "$shard" ]; then
    if [ ! -f "$shard" ] || ! python3 - "$shard" > "$cases_file" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    cases = json.load(handle)["cases"]
for case in cases:
    get = lambda key, default="": str(case.get(key) or default)
    fields = (
        get("phase", "decode"), get("mode", "normal"), get("routing", "uniform"),
        get("hidden", "7168"),
        get("topk", "8"), get("experts", "256"), get("ladder"),
        get("suite"), get("workload"),
        "1" if case.get("canonical") else "", get("case_id"), get("ep"),
        get("timing", "8:64:32"), get("nodes"), get("gpus_per_node"),
        get("scale_up_domain"), get("scope"), get("scale_up_transport"),
        get("scale_out_transport"), get("transport"), get("topology_class"),
    )
    print("|".join(fields))
PY
    then
      rm -f "$cases_file"
      cx_die "could not enumerate validated shard cases"
    fi
  else
    local phases="${CX_PHASE:-decode}" phase
    [ "$phases" = both ] && phases="decode prefill"
    cx_require_record_safe "$phases" "${CX_MODE:-normal}" "${CX_ROUTING:-uniform}" \
      "${CX_HIDDEN:-7168}" "${CX_TOPK:-8}" "${CX_EXPERTS:-256}" \
      "${CX_TOKENS_LADDER:-}" "${CX_SUITE:-}" "${CX_WORKLOAD_NAME:-}" \
      "${CX_CANONICAL:-}" "${CX_CASE_ID:-}" \
      "${CX_ITERS:-8}" "${CX_TRIALS:-64}" "${CX_WARMUP:-32}" \
      "${CX_SCOPE:-scale-up}" \
      "${CX_SCALE_UP_TRANSPORT:-unknown}" "${CX_SCALE_OUT_TRANSPORT:-}" \
      "${CX_TRANSPORT:-unknown}" "${CX_TOPO:-manual}"
    for phase in $phases; do
      (IFS='|'; printf '%s\n' "$phase|${CX_MODE:-normal}|${CX_ROUTING:-uniform}|${CX_HIDDEN:-7168}|${CX_TOPK:-8}|${CX_EXPERTS:-256}|${CX_TOKENS_LADDER:-}|${CX_SUITE:-}|${CX_WORKLOAD_NAME:-}|${CX_CANONICAL:-}|${CX_CASE_ID:-}|$NGPUS|${CX_ITERS:-8}:${CX_TRIALS:-64}:${CX_WARMUP:-32}|$NODES|$GPN|$SCALE_UP_DOMAIN|${CX_SCOPE:-scale-up}|${CX_SCALE_UP_TRANSPORT:-unknown}|${CX_SCALE_OUT_TRANSPORT:-}|${CX_TRANSPORT:-unknown}|${CX_TOPO:-manual}")
    done > "$cases_file"
  fi
  expected_cases="$(wc -l < "$cases_file" | tr -d ' ')"
  [ "$expected_cases" -gt 0 ] \
    || { rm -f "$cases_file"; cx_die "distributed case list is empty"; }

  while IFS='|' read -r ph mode routing hidden topk experts ladder suite workload \
      canonical case_id ep timing nodes gpn domain scope scale_up_transport \
      scale_out_transport transport topology_class; do
    [ -n "$ph" ] || continue
    ci=$((ci + 1))
    case_stem="${RUNNER}_${CX_BENCH}_${ph}_${TS}-c$(printf '%03d' "$ci")"
    IFS=: read -r case_iters case_trials case_warmup <<< "${timing:-8:64:32}"
    case_iters="${case_iters:-8}"
    case_trials="${case_trials:-64}"
    case_warmup="${case_warmup:-32}"
    ep="${ep:-$NGPUS}"
    export CX_MODE="$mode" CX_PHASE="$ph" CX_CASE_ID="$case_id" CX_SUITE="$suite"
    export CX_WORKLOAD_NAME="$workload"
    export CX_CANONICAL="$canonical" CX_EP="$ep"
    export CX_ROUTING="$routing" CX_TOKENS_LADDER="$ladder"
    export CX_HIDDEN="$hidden" CX_TOPK="$topk" CX_EXPERTS="$experts"
    export CX_NODES="$nodes" CX_GPUS_PER_NODE="$gpn" CX_SCALE_UP_DOMAIN="$domain"
    export CX_SCOPE="$scope" CX_SCALE_UP_TRANSPORT="$scale_up_transport"
    export CX_SCALE_OUT_TRANSPORT="$scale_out_transport"
    export CX_TRANSPORT="$transport" CX_TOPO="$topology_class"
    export CX_ITERS="$case_iters" CX_TRIALS="$case_trials" CX_WARMUP="$case_warmup"
    export CX_SAMPLES_PER_POINT="$((case_iters * case_trials))"
    export CX_WARMUP_SEMANTICS="full-roundtrip-before-each-component-trial-point-v1"
    cx_apply_network_profile "$NODES" "$transport"
    cx_log "EP${NGPUS}[$ci] id=${case_id:-manual} $mode/$ph $CX_BENCH"
    if [ "$ep" != "$NGPUS" ] || [ "$nodes" != "$NODES" ] || [ "$gpn" != "$GPN" ] \
        || [ "$domain" != "$SCALE_UP_DOMAIN" ]; then
      cx_log "ERROR: EP${NGPUS}[$ci] topology mismatch (ep=$ep nodes=$nodes gpn=$gpn domain=$domain); skipping case"
      failed_cases=$((failed_cases + 1))
      continue
    fi

    workload_dir=""
    if cx_bool_enabled "$canonical"; then
      workload_dir=".cx_workloads/c$(printf '%03d' "$ci")"
      workload_ladder="$ladder"
      [ -n "$workload_ladder" ] \
        || workload_ladder="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
      workload_args=(python3 bench/make_workloads.py --out-dir "$workload_dir"
        --routing "$routing" --ep "$ep" --hidden "$hidden" --topk "$topk"
        --experts "$experts" --seed "${CX_SEED:-67}" --tokens-ladder "$workload_ladder")
      workload_log="$(cx_private_log_path "workload-c$(printf '%03d' "$ci")")"
      set +e
      srun --jobid="$JOB_ID" --nodes=1 --ntasks=1 --chdir=/tmp \
        --container-name="$container_name" --container-image="$SQUASH_FILE" \
        "${container_args[@]}" \
        --export="$(cx_container_exports)" "${workload_args[@]}" \
        </dev/null >"$workload_log" 2>&1
      stage_rc=$?
      set -e
      if [ "$stage_rc" != 0 ]; then
        cx_log "ERROR: EP${NGPUS}[$ci] workload staging failed (rc=$stage_rc)"
        failed_cases=$((failed_cases + 1))
        continue
      fi
    fi

    ep_args=(--backend "$CX_BENCH" --mode "$mode" --phase "$ph" --routing "$routing"
      --gpus-per-node "$gpn" --scale-up-domain "$domain" --scope "$scope"
      --scale-up-transport "$scale_up_transport" --scale-out-transport "$scale_out_transport"
      --tokens-ladder "$ladder" --hidden "$hidden" --topk "$topk" --experts "$experts"
      --warmup "$case_warmup" --iters "$case_iters" --trials "$case_trials"
      --seed "${CX_SEED:-67}" --runner "$RUNNER" --topology-class "$topology_class"
      --transport "$transport" --case-id "$case_id" --suite "$suite"
      --workload-name "$workload"
      --qualification-index "${CX_QUALIFICATION_INDEX:-1}" --version "${CX_VERSION:-1}")
    [ -z "$workload_dir" ] || ep_args+=(--workload-dir "$workload_dir")
    export CX_ATTEMPT_ID=1
    attempt_tag=a01
    out="results/${case_stem}_${attempt_tag}.json"
    runtime_log="$(cx_private_log_path "runtime-c$(printf '%03d' "$ci")-$attempt_tag")"
    set +e
    timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JOB_ID" --nodes="$NODES" \
      --ntasks="$NGPUS" --ntasks-per-node="$GPN" --chdir=/tmp \
      --container-name="$container_name" --container-image="$SQUASH_FILE" \
      "${container_args[@]}" \
      --export="$(cx_container_exports)" \
      bash -c "$WRAP" _ "${ep_args[@]}" --out "$out" \
      </dev/null >"$runtime_log" 2>&1
    run_rc=$?
    set -e
    # Terminal-outcome emission and result-document gating were removed with
    # contracts.py; a case now counts as run purely on the distributed command's
    # return code. The rank-zero result the harness wrote (if any) is left in place
    # for the summary renderer, which validates nothing.
    if [ "$run_rc" != 0 ]; then
      cx_fail_stage execution "$runtime_log" || true
      failed_cases=$((failed_cases + 1))
    fi
  done < "$cases_file"
  rm -f "$cases_file"
  [ "$ci" -eq "$expected_cases" ] \
    || cx_die "enumerated $expected_cases cases but executed $ci"
  if [ "$failed_cases" -ne 0 ]; then
    summary_log="$(cx_private_log_path shard-summary)"
    printf 'SHARD done: %s/%s case(s) failed\n' "$failed_cases" "$expected_cases" \
      > "$summary_log"
    cx_fail_stage execution "$summary_log" || true
    return 1
  fi
  return 0
}

# Remove this allocation's persistent pyxis container before the allocation is
# released. The cluster runs pyxis with container_scope=global, so a named
# --container-writable container (the distributed path's cxep_<jobid>) survives
# job teardown and its unpacked rootfs — tens of GB per node — would otherwise
# accumulate on every allocated node's local image store until it fills and the
# next writable extraction fails with ENOSPC. Best-effort and bounded: teardown
# must never hang or fail on this. Single-node legs use an unnamed, ephemeral
# container that pyxis reclaims on its own, so only NODES>1 needs removal.
cx_remove_distributed_container() {
  local job_id="$1" nodes="${2:-1}"
  [ -n "$job_id" ] || return 0
  [ "$nodes" -gt 1 ] 2>/dev/null || return 0
  timeout 120 srun --jobid="$job_id" --nodes="$nodes" --ntasks-per-node=1 \
    --chdir=/tmp enroot remove -f "pyxis_cxep_${job_id}" \
    </dev/null >/dev/null 2>&1 || true
}

cx_launcher_cleanup() {
  local rc="$1" stage_root="${MOUNT_SRC:-}" source_root allocation_stopped=1
  source_root="${stage_root:-${REPO_ROOT:-}}"
  trap - EXIT HUP INT TERM
  if [ -n "${COLLECTIVEX_EPHEMERAL_CONFIG_PATH:-}" ]; then
    rm -f -- "$COLLECTIVEX_EPHEMERAL_CONFIG_PATH" >/dev/null 2>&1 || true
    unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
  fi
  if [ -n "${JOB_ID:-}" ]; then
    cx_remove_distributed_container "$JOB_ID" "${NODES:-1}"
    if ! cx_cancel_job "$JOB_ID"; then
      allocation_stopped=0
      [ "$rc" != 0 ] || rc=1
    elif ! cx_clear_allocation_jobid; then
      allocation_stopped=0
      [ "$rc" != 0 ] || rc=1
    fi
  elif [ "${CX_ALLOCATION_UNCERTAIN:-0}" = 1 ]; then
    allocation_stopped=0
    [ "$rc" != 0 ] || rc=1
  fi
  if [ "$allocation_stopped" = 1 ]; then
    cx_write_cleanup_guard safe || true
  else
    cx_write_cleanup_guard unsafe || true
  fi
  [ "$allocation_stopped" = 1 ] || source_root="${REPO_ROOT:-$source_root}"
  if [ "$rc" != 0 ] \
      && [ -n "${REPO_ROOT:-}" ] && [ -n "${CX_BENCH:-}" ]; then
    cx_log "ERROR: terminal-failure-stage=${CX_FAILSAFE_MODE:-setup}"
    [ -d "$source_root/experimental/CollectiveX" ] || source_root="$REPO_ROOT"
    [ "$source_root" = "$REPO_ROOT" ] \
      || cx_collect_results "$source_root" "$REPO_ROOT" || true
  fi
  if [ "$allocation_stopped" = 1 ] && [ -n "${REPO_ROOT:-}" ] \
      && [ -n "$stage_root" ] && [ "$stage_root" != "$REPO_ROOT" ]; then
    if ! cx_cleanup_stage "$stage_root" "$REPO_ROOT"; then
      [ "$rc" != 0 ] || rc=1
    fi
  fi
  [ "${COLLECTIVEX_CANONICAL_GHA:-0}" = 1 ] || cx_cleanup_private_logs "$rc"
  exit "$rc"
}

cx_install_launcher_fail_safe() {
  CX_ALLOCATION_UNCERTAIN=0
  trap 'cx_launcher_cleanup "$?"' EXIT
  trap 'cx_launcher_cleanup 129' HUP
  trap 'cx_launcher_cleanup 130' INT
  trap 'cx_launcher_cleanup 143' TERM
}
