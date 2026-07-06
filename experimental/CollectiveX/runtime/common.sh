# shellcheck shell=bash
# CollectiveX — shared launcher helpers (sourced, not executed).
#
# Cluster-generic scaffolding only (Slurm/container/build/staging); no
# model-serving. Logging goes to stderr so functions can `echo` a single
# result on stdout.

_CX_COMMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CX_SQUASH_FORMAT_VERSION="repro-v1"
CX_SQUASH_SOURCE_DATE_EPOCH=1
CX_DEEPEP_V2_COMMIT="fa8a9b16898204afd347c663b89e65ef87dc6ce6" # pragma: allowlist secret
CX_DEEPEP_V2_TREE="29809e75c5874e6609dac4804e7b651d5226959f" # pragma: allowlist secret
CX_DEEPEP_V2_FMT_COMMIT="a4c7e17133ee9cb6a2f45545f6e974dd3c393efa" # pragma: allowlist secret
# Consumed by run_in_container.sh after this helper is sourced.
# shellcheck disable=SC2034
CX_DEEPEP_V2_NCCL_CHECK_COMMIT="93d0564188f7a0a6288c6e316484861b0efa042e" # pragma: allowlist secret
CX_DEEPEP_V2_INIT_SHA256="f090bbacc38c7d5dba29f07fd38a918eb820b6adac6f76903fe56273060e4870"
CX_DEEPEP_HYBRID_COMMIT="e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8" # pragma: allowlist secret
CX_DEEPEP_HYBRID_TREE="d77aeab7f1bb52b615666fe178d26ced41fae08e" # pragma: allowlist secret
CX_DEEPEP_HYBRID_NCCL_COMMIT="1e0c869c39bb33f1034cb9920bd2a8a8406f04a3" # pragma: allowlist secret
unset COLLECTIVEX_OPERATOR_CONFIG_LOADED COLLECTIVEX_EPHEMERAL_CONFIG_PATH

cx_log() { printf '[collectivex] %s\n' "$*" >&2; }
cx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

# Public failure telemetry is a closed vocabulary. Raw scheduler, container,
# host, and filesystem diagnostics stay in the mode-0600 private logs.
cx_set_failure_stage() {
  local stage="$1"
  case "$stage" in
    setup|repository-stage|registry-verification|scheduler-allocation|container-import) ;;
    container-hash|container-launch|backend-setup|execution|artifact-collection) ;;
    *) cx_die "invalid launcher failure stage" ;;
  esac
  export CX_FAILSAFE_MODE="$stage"
}

cx_fail_stage() {
  local stage="$1" log_path="${2:-}" diagnostic="unknown" probe_stage=""
  cx_set_failure_stage "$stage"
  if [ -n "$log_path" ] && [ -f "$log_path" ]; then
    probe_stage="$(grep -aoE 'precision-probe-stage=(distributed-init|runtime-context|backend-construction|construction-consensus|native-operation|operation-consensus|evidence-aggregation)' "$log_path" \
      | tail -n 1 | cut -d= -f2 || true)"
    if grep -aEqi 'no space left|disk quota|quota exceeded' "$log_path"; then
      diagnostic="storage-capacity"
    elif grep -aEqi 'permission denied|operation not permitted|read-only file system|source mount (creation|ownership validation|permission inspection|permission normalization|permission validation) failed' "$log_path"; then
      diagnostic="storage-permission"
    elif grep -aEqi 'outside one realized LSA domain|lsa(Size| team| domain).*(mismatch|invalid|expected)|ranks.*not in (one|the same) nvlink.domain' "$log_path" \
        || { [ "${CX_BENCH:-}" = deepep-v2 ] \
          && grep -aEqi 'nccl[.]cu:(111|112)([^0-9]|$)' "$log_path"; }; then
      diagnostic="accelerator-topology"
    elif grep -aEqi 'cuda driver version is insufficient|call requires newer driver|cudaErrorCallRequiresNewerDriver|CUDA_ERROR_SYSTEM_DRIVER_MISMATCH|unsupported toolchain' "$log_path"; then
      diagnostic="accelerator-driver"
    elif grep -aEqi 'cudaErrorDevicesUnavailable|CUDA_ERROR_DEVICE_UNAVAILABLE|CUDA-capable device\(s\) is/are busy or unavailable|primary context retain failed' "$log_path"; then
      diagnostic="accelerator-unavailable"
    elif grep -aEqi 'ncclDevCommCreate|ncclCommWindowRegister|ncclGetLsa(Device)?Pointer|Communicator does not support symmetric memory|Symmetric memory is not supported' "$log_path" \
        || { [ "${CX_BENCH:-}" = deepep-v2 ] \
          && grep -aEqi 'nccl[.]cu:(106|127|128|129|135)([^0-9]|$)' "$log_path"; }; then
      diagnostic="nccl-device-api"
    elif grep -aEqi 'NVCC (PTX )?compilation failed|cuobjdump failed|invalid device (kernel )?image|no kernel image is available' "$log_path"; then
      diagnostic="jit-toolchain"
    elif [ -n "$probe_stage" ] \
        && grep -aEqi 'cuda out of memory|CUDA_ERROR_OUT_OF_MEMORY|out of memory.*cuda' "$log_path"; then
      diagnostic="${probe_stage}-accelerator-memory"
    elif grep -aEqi 'cuda out of memory|CUDA_ERROR_OUT_OF_MEMORY|out of memory.*cuda' "$log_path"; then
      diagnostic="accelerator-memory"
    elif grep -aEqi 'does not match its pinned image contract|requires the exact pinned|version mismatch' "$log_path"; then
      diagnostic="backend-version"
    elif grep -aEqi 'nvshmem is unavailable|build-tool installation failed' "$log_path"; then
      diagnostic="backend-dependency"
    elif grep -aEqi 'revision fetch failed|submodule fetch failed|package installation failed|staged source is invalid|source (pin resolution|seed validation|seed copy|checkout creation|publication validation|existing source validation) failed' "$log_path"; then
      diagnostic="backend-source"
    elif grep -aEqi 'backend preparation failed|backend import failed|build (failed|is incomplete)|cache (mount identity )?validation failed|import failed' "$log_path"; then
      diagnostic="backend-build"
    elif grep -aEqi 'failed to mount|squashfs|enroot|pyxis|mount.*invalid argument|invalid argument.*mount' "$log_path"; then
      diagnostic="container-runtime"
    elif grep -aEqi 'command not found|not found on this runner|git lookup failed' "$log_path"; then
      diagnostic="missing-runtime"
    elif grep -aEqi 'too many requests|rate.?limit' "$log_path"; then
      diagnostic="registry-rate-limit"
    elif [ -n "$probe_stage" ] \
        && grep -aEqi 'timed out|operation timeout|wait timeout after|watchdog.*timeout|timeout: sending signal' "$log_path"; then
      diagnostic="${probe_stage}-timeout"
    elif grep -aEqi 'ncclRemoteError|remote process exited|connection closed by peer' "$log_path"; then
      diagnostic="collective-remote"
    elif grep -aEqi 'ncclSystemError|unhandled system error' "$log_path"; then
      diagnostic="collective-system"
    elif grep -aEqi 'ncclInternalError|internal check failed' "$log_path"; then
      diagnostic="collective-internal"
    elif grep -aEqi 'ncclInvalidUsage|invalid usage' "$log_path"; then
      diagnostic="collective-invalid-usage"
    elif grep -aEqi 'timed out|operation timeout|wait timeout after|watchdog.*timeout|timeout: sending signal|connection reset|could not resolve|TLS|certificate' "$log_path"; then
      diagnostic="network-or-timeout"
    elif grep -aEqi 'salloc:|srun:.*(unable to create step|step creation|invalid partition|invalid account)|unable to create step|job allocation' "$log_path"; then
      diagnostic="scheduler"
    elif [ -n "$probe_stage" ] \
        && grep -aEqi 'Traceback \(most recent call last\)|[A-Za-z]+(Error|Exception):' "$log_path"; then
      diagnostic="${probe_stage}-failed"
    elif grep -aEqi 'ModuleNotFoundError|ImportError:' "$log_path"; then
      diagnostic="python-import"
    elif grep -aEqi 'AttributeError:|TypeError:.*(unexpected|argument|operand)|has no attribute' "$log_path"; then
      diagnostic="backend-api"
    elif grep -aEqi 'FP8 dispatch payload is missing block-128 scales' "$log_path"; then
      diagnostic="precision-dispatch-scales-missing"
    elif grep -aEqi 'native FP8 dispatch payload has an invalid dtype or shape' "$log_path"; then
      diagnostic="precision-dispatch-payload-shape"
    elif grep -aEqi 'native FP8 dispatch scales have an invalid dtype or shape' "$log_path"; then
      diagnostic="precision-dispatch-scale-shape"
    elif grep -aEqi 'expert-packed FP8 receive count exceeds capacity' "$log_path"; then
      diagnostic="precision-receive-capacity"
    elif grep -aEqi 'expert-packed (FP8 receive counts|BF16 stage workspace) has an invalid shape' "$log_path"; then
      diagnostic="precision-receive-shape"
    elif grep -aEqi 'active torch build does not expose (torch[.])?float8|active torch build does not expose fp8-' "$log_path"; then
      diagnostic="precision-runtime-dtype"
    elif grep -aEqi 'omits EpDispatchCombineQuantType[.]Fp8DirectCast' "$log_path"; then
      diagnostic="precision-combine-api"
    elif grep -aEqi 'dispatch FP8 format differs from the pinned GPU architecture' "$log_path"; then
      diagnostic="precision-architecture-format"
    elif grep -aEqi 'native FP8 dispatch requires hidden divisible by 128' "$log_path"; then
      diagnostic="precision-hidden-alignment"
    elif grep -aEqi 'PrecisionError:|unsupported precision|precision profile.*(invalid|unsupported|differs)' "$log_path"; then
      diagnostic="precision-contract"
    elif grep -aEqi 'AssertionError:' "$log_path"; then
      diagnostic="python-assertion"
    elif grep -aEqi 'RuntimeError:' "$log_path"; then
      diagnostic="python-runtime"
    elif grep -aEqi 'ValueError:.*(fields differ from collectivex[.]precision-probe|probe format, record type, schema, or contract differs)' "$log_path"; then
      diagnostic="probe-schema-value"
    elif grep -aEqi 'ValueError:.*probe target|ValueError:.*probe topology' "$log_path"; then
      diagnostic="probe-target-value"
    elif grep -aEqi 'ValueError:.*probe result|ValueError:.*statically promote the registry' "$log_path"; then
      diagnostic="probe-result-value"
    elif grep -aEqi 'ValueError:.*probe privacy' "$log_path"; then
      diagnostic="probe-privacy-value"
    elif grep -aEqi 'ValueError:.*probe API' "$log_path"; then
      diagnostic="probe-api-value"
    elif grep -aEqi 'ValueError:.*probe completion' "$log_path"; then
      diagnostic="probe-completion-value"
    elif grep -aEqi 'ValueError:.*probe (source SHA|image digest|backend provenance|backend component)' "$log_path"; then
      diagnostic="probe-identity-value"
    elif grep -aEqi 'ValueError:.*probe precision correctness did not pass' "$log_path"; then
      diagnostic="probe-correctness-failed"
    elif grep -aEqi 'ValueError:.*probe scale shapes are invalid' "$log_path"; then
      diagnostic="probe-scale-shape"
    elif grep -aEqi 'ValueError:.*probe precision .* is not finite' "$log_path"; then
      diagnostic="probe-nonfinite"
    elif grep -aEqi 'ValueError:.*probe precision .* shapes are empty|ValueError:.*probe precision .* shape is invalid' "$log_path"; then
      diagnostic="probe-tensor-shape"
    elif grep -aEqi 'ValueError:.*probe (precision correctness|scale shapes|precision .* (input|output|semantic|scales))' "$log_path"; then
      diagnostic="probe-correctness-value"
    elif grep -aEqi 'ValueError:.*probe transport' "$log_path"; then
      diagnostic="probe-transport-value"
    elif grep -aEqi 'ValueError:' "$log_path" \
        && grep -aEqi 'probe_precision[.]py' "$log_path"; then
      diagnostic="probe-manifest-value"
    elif grep -aEqi 'ValueError:' "$log_path" \
        && grep -aEqi 'ep_harness[.]py' "$log_path"; then
      diagnostic="harness-value"
    elif grep -aEqi 'ValueError:' "$log_path" \
        && grep -aEqi 'workload[.]py|make_workloads[.]py' "$log_path"; then
      diagnostic="workload-value"
    elif grep -aEqi 'ValueError:' "$log_path" \
        && grep -aEqi 'run_ep[.]py' "$log_path"; then
      diagnostic="runner-value"
    elif grep -aEqi 'ValueError:' "$log_path" \
        && grep -aEqi 'ep_deepep[.]py' "$log_path"; then
      diagnostic="deepep-adapter-value"
    elif grep -aEqi 'ValueError:' "$log_path" \
        && grep -aEqi '/(torch|numpy)/|site-packages/(torch|numpy)' "$log_path"; then
      diagnostic="dependency-value"
    elif grep -aEqi 'ValueError:' "$log_path"; then
      diagnostic="python-value"
    elif grep -aEqi 'KeyError:' "$log_path"; then
      diagnostic="python-key"
    elif grep -aEqi '(FileNotFoundError|PermissionError|IsADirectoryError|NotADirectoryError|OSError):' "$log_path"; then
      diagnostic="python-os"
    elif grep -aEqi '(NotImplemented|System)Error:' "$log_path"; then
      diagnostic="python-system"
    elif grep -aEqi 'DistBackendError:' "$log_path"; then
      diagnostic="collective-backend"
    elif grep -aEqi 'CalledProcessError:' "$log_path"; then
      diagnostic="python-subprocess"
    elif grep -aEqi 'Traceback \(most recent call last\)' "$log_path"; then
      diagnostic="python-exception"
    elif [ -n "$probe_stage" ]; then
      diagnostic="${probe_stage}-failed"
    elif grep -aEqi 'SHARD done: [0-9]+/[0-9]+ case\(s\) failed|WARN: .* run failed rc=|completed with invalid semantic evidence' "$log_path"; then
      diagnostic="benchmark-case-failure"
    elif [ -s "$log_path" ]; then
      diagnostic="unclassified"
    else
      diagnostic="empty-log"
    fi
  fi
  cx_log "ERROR: failure-class=$stage diagnostic=$diagnostic"
  return 1
}

# Runner-local deployment settings are strict JSON kept outside the checkout.
# Only the selected runner's allowlisted values are exported; the document is
# never sourced or evaluated as shell.
cx_load_operator_config() {
  [ -n "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" ] \
    && [ "$COLLECTIVEX_OPERATOR_CONFIG_LOADED" = "$$" ] && return 0
  local config_path generated=0 parsed_path config_log key value
  local audit_salt_override validation_code
  audit_salt_override="${COLLECTIVEX_OPERATOR_AUDIT_SALT:-}"
  unset COLLECTIVEX_OPERATOR_AUDIT_SALT
  unset CX_PARTITION CX_ACCOUNT CX_QOS CX_SQUASH_DIR CX_STAGE_DIR CX_ENROOT_CACHE_PATH
  unset ENROOT_CACHE_PATH
  unset CX_EXCLUDE_NODES CX_NODELIST CX_LOCK_DIR CX_MASTER_PORT
  unset CX_SOCKET_IFNAME CX_RDMA_DEVICES CX_IB_GID_INDEX CX_RDMA_SERVICE_LEVEL
  unset CX_RDMA_TRAFFIC_CLASS
  unset CX_AUDIT_SALT
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  config_path="${COLLECTIVEX_OPERATOR_CONFIG:-${XDG_CONFIG_HOME:-${HOME}/.config}/inferencex/collectivex.json}"
  if [ -n "${COLLECTIVEX_OPERATOR_CONFIG_CONTENT:-}" ]; then
    umask 077
    if [[ "${CX_JOB_ROOT:-}" =~ ^/tmp/inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+$ ]] \
        && [ -d "$CX_JOB_ROOT" ] && [ ! -L "$CX_JOB_ROOT" ] \
        && [ "$(stat -c '%u:%a' "$CX_JOB_ROOT" 2>/dev/null)" = "$(id -u):700" ]; then
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
      "${COLLECTIVEX_CANONICAL_GHA:-0}" "$audit_salt_override" \
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
AUDIT_SALT = re.compile(r"^[0-9a-f]{64}$")

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
    path, runner, audit_required, audit_override = sys.argv[1:]
    if runner not in RUNNERS or audit_required not in {"0", "1"}:
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
        set(document) not in (
            {"schema_version", "runners"},
            {"schema_version", "audit_salt", "runners"},
        )
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError
    audit_salt = document.get("audit_salt")
    if (
        (audit_salt is not None and (
            not isinstance(audit_salt, str) or not AUDIT_SALT.fullmatch(audit_salt)
        ))
        or (audit_override and not AUDIT_SALT.fullmatch(audit_override))
        or (audit_salt is not None and audit_override and audit_salt != audit_override)
    ):
        raise ValueError
    audit_salt = audit_salt or audit_override or None
    if audit_required == "1" and audit_salt is None:
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
    if audit_salt is not None:
        sys.stdout.buffer.write(b"CX_AUDIT_SALT\0" + audit_salt.encode() + b"\0")
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
  printf '%s' 'COLLECTIVEX_SOURCE_SHA,COLLECTIVEX_ARTIFACT_NAME,COLLECTIVEX_EXECUTION_ID,COLLECTIVEX_CONTROL_SHA256,COLLECTIVEX_IMAGE,COLLECTIVEX_IMAGE_DIGEST,COLLECTIVEX_IMAGE_DIGEST_VERIFIED,COLLECTIVEX_SQUASH_SHA256,GITHUB_REF_NAME,GITHUB_REF,GITHUB_REPOSITORY,GITHUB_JOB,GITHUB_RUN_ID,GITHUB_RUN_ATTEMPT,GITHUB_SHA,CX_RUNNER,CX_BENCH,CX_NODES,CX_GPUS_PER_NODE,CX_SCALE_UP_DOMAIN,CX_SHARD_FILE,CX_SHARD_SKU,CX_PRECISION_PROBE,CX_NGPUS,CX_TS,CX_TOPO,CX_SCOPE,CX_TRANSPORT,CX_SCALE_UP_TRANSPORT,CX_SCALE_OUT_TRANSPORT,CX_MODE,CX_PHASE,CX_ROUTING,CX_EPLB,CX_CASE_ID,CX_SUITE,CX_WORKLOAD_NAME,CX_REQUIRED_PUBLICATION,CX_PRECISION_PROFILE,CX_QUALIFICATION_INDEX,CX_HIDDEN,CX_TOPK,CX_EXPERTS,CX_TOKENS_LADDER,CX_CANONICAL,CX_ITERS,CX_TRIALS,CX_WARMUP,CX_SAMPLES_PER_POINT,CX_WARMUP_SEMANTICS,CX_SEED,CX_RUN_TIMEOUT,CX_NCCL_HOME,CX_ALLOW_MNNVL,CX_ATTEMPT_ID,CX_RUNTIME_MARKER,CX_MORI_KERNEL_TYPE,CX_WORKLOAD_DIR,CX_BACKEND_CACHE_ROOT,CX_BACKEND_CACHE_SENTINEL_SHA256,CX_BACKEND_SOURCE_ROOT,CX_AUDIT_SALT,CX_SOCKET_IFNAME,CX_RDMA_DEVICES,CX_IB_GID_INDEX,CX_RDMA_SERVICE_LEVEL,CX_RDMA_TRAFFIC_CLASS,CX_RDMA_LINK_LAYER,MASTER_ADDR,MASTER_PORT,RANK,WORLD_SIZE,LOCAL_RANK,LOCAL_WORLD_SIZE,NCCL_NET,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NCCL_IB_HCA,NCCL_IB_GID_INDEX,NCCL_IB_SL,NVSHMEM_DISABLE_IB,NVSHMEM_ENABLE_NIC_PE_MAPPING,NVSHMEM_HCA_LIST,NVSHMEM_IB_GID_INDEX,NVSHMEM_IB_SL,NVSHMEM_IB_ENABLE_IBGDA,NVSHMEM_IBGDA_NIC_HANDLER,EP_NIC_NAME,EP_OVERRIDE_RDMA_SL,UCCL_SOCKET_IFNAME,UCCL_IB_GID_INDEX,UCCL_IB_SL,MORI_RDMA_DEVICES,MORI_RDMA_TC,MORI_IO_TC,MORI_RDMA_SL,MORI_IO_SL,HYBRID_EP_MULTINODE,USE_NIXL,RDMA_CORE_HOME,DEEPEP_HYBRID_BUILD_MODE,NCCL_CUMEM_ENABLE,NCCL_MNNVL_ENABLE,MC_FORCE_MNNVL,MORI_DISABLE_AUTO_XGMI,MORI_ENABLE_SDMA,MORI_APP_LOG_LEVEL,MORI_SHMEM_LOG_LEVEL,MORI_IO_LOG_LEVEL'
  printf '%s' ',MORI_COMMIT'
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

cx_require_single_node() {
  [ "${CX_NODES:-1}" = "1" ] || cx_die "$1 supports one-node EP only"
}

cx_nccl_hca_device_name() {
  local selector="${1#=}"
  printf '%s' "${selector%%:*}"
}

cx_export_gid_index_for_link_layer() {
  local link_layer="$1" scaleout="$2"
  unset NVSHMEM_IB_GID_INDEX NCCL_IB_GID_INDEX UCCL_IB_GID_INDEX
  [ -n "${CX_IB_GID_INDEX:-}" ] || return 0
  case "$link_layer" in
    roce)
      export NVSHMEM_IB_GID_INDEX="$CX_IB_GID_INDEX"
      if [ "$scaleout" = 1 ]; then
        export NCCL_IB_GID_INDEX="$CX_IB_GID_INDEX"
        export UCCL_IB_GID_INDEX="$CX_IB_GID_INDEX"
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
  local scaleout=0 single_node_rdma=0
  local -a selectors
  [[ "$nodes" =~ ^[1-9][0-9]*$ ]] || cx_die "invalid network placement"
  unset NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL
  unset NVSHMEM_DISABLE_IB NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset NVSHMEM_HCA_PE_MAPPING NVSHMEM_REMOTE_TRANSPORT
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset UCCL_SOCKET_IFNAME UCCL_IB_GID_INDEX UCCL_IB_SL MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  if [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; then
    scaleout=1
  elif [ "${CX_SHARD_SKU:-}" = b300 ] && [ "$nodes" = 1 ] \
      && [ "${CX_BENCH:-}" = deepep ] && [ "${CX_MODE:-}" = low-latency ]; then
    # DeepEP V1 low-latency kernels use pure RDMA even for EP8 within one
    # NVLink domain. Keep NCCL on NVLink, but bind NVSHMEM to the private HCA.
    single_node_rdma=1
  elif [ "${CX_SHARD_SKU:-}" = b300 ] && [ "$nodes" = 1 ]; then
    export NVSHMEM_DISABLE_IB=1
  fi
  [ "$scaleout" = 1 ] || [ "$single_node_rdma" = 1 ] || return 0
  [ -n "${CX_RDMA_DEVICES:-}" ] \
    || cx_die "RDMA execution requires a private device selector"
  if [ "$scaleout" = 1 ] && [ -n "${CX_SOCKET_IFNAME:-}" ]; then
    [[ "$CX_SOCKET_IFNAME" =~ ^[A-Za-z][A-Za-z0-9_.-]{0,31}(,[A-Za-z][A-Za-z0-9_.-]{0,31})*$ ]] \
      || cx_die "invalid private socket interface selector"
    export NCCL_SOCKET_IFNAME="$CX_SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$CX_SOCKET_IFNAME"
    export UCCL_SOCKET_IFNAME="$CX_SOCKET_IFNAME"
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
      export NCCL_NET=IB NCCL_IB_HCA="=$CX_RDMA_DEVICES"
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
      export NCCL_IB_SL="$CX_RDMA_SERVICE_LEVEL" UCCL_IB_SL="$CX_RDMA_SERVICE_LEVEL"
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
  local single_node_rdma=0
  if [ "$nodes" -gt 1 ] && [ "$transport" != mnnvl ]; then
    scaleout=1
  elif [ "${CX_SHARD_SKU:-}" = b300 ] && [ "$nodes" = 1 ] \
      && [ "${CX_BENCH:-}" = deepep ] && [ "${CX_MODE:-}" = low-latency ]; then
    single_node_rdma=1
  fi
  [ "$scaleout" = 1 ] || [ "$single_node_rdma" = 1 ] || return 0
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
case "${CX_PRECISION_PROBE:-0}" in
  1) exec python3 tests/probe_precision.py "$@" ;;
  0|'') exec python3 tests/run_ep.py "$@" ;;
  *) exit 67 ;;
esac
BASH
}

# A set shard path is an execution contract, never a hint. Validate it before
# staging/allocation and again in-container so a missing or stale control file
# cannot silently fall back to a manual single-case run.
cx_validate_shard_control() {
  local cx_root="$1" shard="${CX_SHARD_FILE:-}" path expected_sku control_sha256
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
  if [ "${CX_PRECISION_PROBE:-0}" = 1 ]; then
    python3 "${cx_root%/}/tests/probe_precision.py" \
      --validate-control "$path" --expect-sku "$expected_sku" \
      --expect-backend "$CX_BENCH" --expect-nodes "$CX_NODES" >/dev/null 2>&1 \
      || cx_die "invalid precision probe control"
  else
    python3 "${cx_root%/}/sweep_matrix.py" \
      --validate-control "$path" --expect-sku "$expected_sku" \
      --expect-backend "$CX_BENCH" --expect-nodes "$CX_NODES" >/dev/null 2>&1 \
      || cx_die "invalid shard control"
  fi
  control_sha256="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$control_sha256" =~ ^[0-9a-f]{64}$ ]] \
    || cx_die "cannot hash shard control"
  export COLLECTIVEX_CONTROL_SHA256="$control_sha256"
}

cx_precision_probe_control_fields() {
  local cx_root="$1" shard="${CX_SHARD_FILE:-}" path
  [ "${CX_PRECISION_PROBE:-0}" = 1 ] || return 1
  path="$shard"
  [ -f "$path" ] || path="${cx_root%/}/$shard"
  python3 - "$path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
document = json.loads(path.read_text())
target = document["target"]
values = (
    document["id"], target["backend"], target["sku"], target["ep"],
    target["mode"], target["precision_profile"],
)
if any("|" in str(value) or "\n" in str(value) for value in values):
    raise SystemExit("unsafe precision probe control field")
print("|".join(map(str, values)))
PY
}

# Load only the case mode needed to choose the allocation/network profile. A
# consolidated shard may contain normal and low-latency cases; selecting the
# strictest mode here makes allocation preflight prove every required device.
# The in-container dispatcher reapplies the profile for each individual case.
cx_load_network_control_mode() {
  local cx_root="$1" shard="${CX_SHARD_FILE:-}" path fields mode
  [ -n "$shard" ] || return 0
  path="$shard"
  [ -f "$path" ] || path="${cx_root%/}/$shard"
  [ -f "$path" ] || return 1
  if [ "${CX_PRECISION_PROBE:-0}" = 1 ]; then
    fields="$(cx_precision_probe_control_fields "$cx_root")" || return 1
    IFS='|' read -r _ _ _ _ mode _ <<< "$fields"
  else
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
  fi
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

# Use an opaque, execution-bound name so a missing grant message can be
# reconciled without exposing runner or shard details in public logs.
cx_scheduler_job_name() {
  local execution_id="${COLLECTIVEX_EXECUTION_ID:-manual-$$}" digest
  digest="$(printf '%s' "$execution_id" | sha256sum | awk '{print $1}')" \
    || return 1
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || return 1
  printf 'cx-%s' "${digest:0:24}"
}

# Return 0 after recovering one allocation ID, 2 after three successful empty
# observations, and 1 for every ambiguous or failed lookup. Callers inspect the
# state variables rather than the status because all missing-ID paths still fail.
cx_reconcile_salloc_jobid() {
  local job_name="$1" scheduler_user queue_output line delay attempt
  local -a ids=()
  scheduler_user="$(id -un 2>/dev/null)" || return 1
  [[ "$scheduler_user" =~ ^[A-Za-z0-9_.-]+$ \
    && "$job_name" =~ ^cx-[0-9a-f]{24}$ ]] || return 1
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

# Allocate via salloc's stable grant message and assign JOB_ID in this shell.
# Raw scheduler output remains in the bounded private execution log.
cx_salloc_jobid() {
  local log_label=scheduler-allocation log job_id job_name argument salloc_rc=0
  case "${CX_SALLOC_ATTEMPT:-1}" in
    1) ;;
    2|3) log_label+="-a${CX_SALLOC_ATTEMPT}" ;;
    *) return 1 ;;
  esac
  log="$(cx_private_log_path "$log_label")"
  for argument in "$@"; do
    case "$argument" in
      --job-name|--job-name=*|-J|-J*)
        cx_log "ERROR: scheduler job names are managed by CollectiveX"
        return 1
        ;;
    esac
  done
  job_name="$(cx_scheduler_job_name)" || return 1
  CX_ALLOCATION_UNCERTAIN=1
  # salloc has no portable --parsable option. Parse the stable grant message
  # used by the production launchers, while also accepting a bare ID from
  # site wrappers.
  salloc "$@" --job-name="$job_name" --no-shell > "$log" 2>&1 || salloc_rc=$?
  job_id="$(sed -nE \
    -e 's/^([0-9]+)(;[^[:space:]]+)?$/\1/p' \
    -e 's/.*Granted job allocation ([0-9]+).*/\1/p' \
    "$log" | head -n1)"
  if [ -n "$job_id" ]; then
    [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
    JOB_ID="$job_id"
    CX_ALLOCATION_UNCERTAIN=0
  fi
  if [ "$salloc_rc" != 0 ]; then
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
  [[ "$root" =~ ^/tmp/inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+$ ]] \
    && [ -d "$root" ] && [ ! -L "$root" ] \
    && [ "$(stat -c '%u:%a' "$root" 2>/dev/null)" = "$(id -u):700" ] || return 1
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
  [[ "$root" =~ ^/tmp/inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+$ ]] \
    && [ -d "$root" ] && [ ! -L "$root" ] \
    && [ "$(stat -c '%u:%a' "$root" 2>/dev/null)" = "$(id -u):700" ] || return 1
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
  [[ "$root" =~ ^/tmp/inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+$ ]] \
    && [ -d "$root" ] && [ ! -L "$root" ] \
    && [ "$(stat -c '%u:%a' "$root" 2>/dev/null)" = "$(id -u):700" ] || return 0
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
  [[ "$root" =~ ^/tmp/inferencex-collectivex-[0-9]+-[0-9]+-[A-Za-z0-9._-]+$ ]] \
    && [ -d "$root" ] && [ ! -L "$root" ] \
    && [ "$(stat -c '%u:%a' "$root" 2>/dev/null)" = "$(id -u):700" ] || return 1
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
# Import remains tag-based because Enroot cannot reliably import a digest-qualified
# Docker Hub reference non-interactively. The registry digest is resolved and checked
# immediately before import, then recorded as verified provenance.
CX_IMAGE_MULTIARCH_DIGEST="sha256:061fb71f838e82000a1768c159654d526c2f17ebe751c21e7fc48ca53c8ef975"
# (v0.5.12-cu130 was rejected: its 62 layers overflow enroot's overlay-based
# squash creation on these nodes — "failed to mount overlay ... Invalid argument".
# v0.5.11-cu130 imports cleanly.)
# Runtime setup verifies the image-bundled DeepEP build for the detected GPU target.
CX_IMAGE_MULTIARCH="lmsysorg/sglang:v0.5.11-cu130"

# AMD (ROCm/CDNA): separate single-arch images bundle MoRI.
CX_IMAGE_AMD_MORI="rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2"
CX_IMAGE_AMD_MORI_DIGEST="sha256:24c3b30d64475937abbb6498e3b29528649adcb836dde7a468979f767809b0e8"
CX_MORI_COMMIT_MI355="99bc0a3a6e7a70aacc6372cd9a4275ccfb4de567" # pragma: allowlist secret
CX_IMAGE_AMD_MORI_MI325="rocm/sgl-dev:sglang-0.5.14-rocm720-mi35x-mori-0701"
CX_IMAGE_AMD_MORI_MI325_DIGEST="sha256:ea42375343c2ef8f73b3bdb9e1b7b435556e3ca92aba5e3f74ada29ba217fabc"
CX_MORI_COMMIT_MI325="bf99bdf18fc69887a346913ca01c315c2aa9bd4c" # pragma: allowlist secret
cx_default_image() {
  case "$1" in
    mi300x*|mi325x*) echo "$CX_IMAGE_AMD_MORI_MI325" ;;
    mi355x*) echo "$CX_IMAGE_AMD_MORI" ;;
    b200*|gb200*|b300*|gb300*|h100*|h200*) echo "$CX_IMAGE_MULTIARCH" ;;
    *) cx_die "no default image for runner prefix: $1" ;;
  esac
}

cx_resolve_registry_digest() {
  local image="$1" repository reference token digest registry
  if [[ "$image" == *@* ]]; then
    cx_die "digest-qualified image overrides are unsupported; configure a tag and pinned digest"
  fi
  registry="${image%%/*}"
  if [[ "$image" == */* && ( "$registry" == *.* || "$registry" == *:* || "$registry" = localhost ) ]]; then
    case "$registry" in
      docker.io|registry-1.docker.io) image="${image#*/}" ;;
      *) cx_die "only Docker Hub images are supported by the registry verifier" ;;
    esac
  fi
  repository="${image%:*}"
  reference="${image##*:}"
  [ "$repository" != "$image" ] || { repository="$image"; reference=latest; }
  [ -n "$repository" ] && [ -n "$reference" ] \
    || cx_die "configured image reference is malformed"
  [[ "$repository" == */* ]] || repository="library/$repository"
  token="$(curl -fsSLG --connect-timeout 10 --max-time 30 --retry 2 \
    --retry-delay 1 --retry-all-errors 'https://auth.docker.io/token' \
    --data-urlencode 'service=registry.docker.io' \
    --data-urlencode "scope=repository:${repository}:pull" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')" \
    || cx_die "cannot authenticate to the image registry"
  digest="$(curl -fsSI --connect-timeout 10 --max-time 30 --retry 2 \
    --retry-delay 1 --retry-all-errors \
    -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json' \
    "https://registry-1.docker.io/v2/${repository}/manifests/${reference}" \
    | tr -d '\r' | awk 'tolower($1)=="docker-content-digest:" {print $2; exit}')" \
    || cx_die "cannot resolve the configured image digest"
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || cx_die "registry returned an invalid image digest"
  printf '%s' "$digest"
}

cx_verify_registry_image() {
  local image="$1" expected actual
  expected="${CX_IMAGE_DIGEST:-$(cx_default_image_digest "$image")}"
  [[ "$expected" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || cx_die "a pinned digest is required for the configured image"
  actual="$(cx_resolve_registry_digest "$image")"
  [ "$actual" = "$expected" ] \
    || cx_die "configured image tag no longer matches its pinned digest"
  export COLLECTIVEX_IMAGE="$image" COLLECTIVEX_IMAGE_DIGEST="$actual"
  export COLLECTIVEX_IMAGE_DIGEST_VERIFIED=1
}

cx_default_image_digest() {
  case "$1" in
    "$CX_IMAGE_MULTIARCH") printf '%s' "$CX_IMAGE_MULTIARCH_DIGEST" ;;
    "$CX_IMAGE_AMD_MORI") printf '%s' "$CX_IMAGE_AMD_MORI_DIGEST" ;;
    "$CX_IMAGE_AMD_MORI_MI325") printf '%s' "$CX_IMAGE_AMD_MORI_MI325_DIGEST" ;;
  esac
}

# Canonical workflow runs must not inherit benchmark controls from a persistent
# self-hosted runner service. Manual/SSH diagnostics retain their explicit
# overrides by leaving COLLECTIVEX_CANONICAL_GHA unset.
cx_gha_workspace_stage_root() {
  local workspace="${GITHUB_WORKSPACE:-}"
  python3 - "$workspace" <<'PY'
import os
import stat
import sys

workspace = sys.argv[1]
try:
    if (
        not os.path.isabs(workspace)
        or os.path.realpath(workspace) != workspace
        or not os.path.isdir(workspace)
    ):
        raise OSError
    metadata = os.stat(workspace, follow_symlinks=False)
    # GitHub runner workspaces are runner-owned but commonly writable by the
    # trusted runner-service group. Keep the child mode 0700 and reject world write.
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & stat.S_IWOTH:
        raise OSError
except OSError:
    raise SystemExit(1)
print(workspace, end="")
PY
}

# Create a per-UID cache under validated cluster-local storage. Only the fixed
# /cx-cache mount enters the container; the operator host path does not.
cx_prepare_backend_cache() {
  local stage_parent="$1" cache info sentinel_sha256
  unset CX_PREPARED_BACKEND_CACHE CX_BACKEND_CACHE_SENTINEL_SHA256
  info="$(python3 - "$stage_parent" <<'PY'
import hashlib
import os
import secrets
import stat
import sys

configured_parent = sys.argv[1]
try:
    if (
        not os.path.isabs(configured_parent)
        or "\n" in configured_parent
        or "\r" in configured_parent
    ):
        raise OSError
    parent = os.path.realpath(configured_parent)
    if not os.path.isdir(parent):
        raise OSError
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, flags)
    try:
        probe_name = f".collectivex-owner-probe-{os.getpid()}-{secrets.token_hex(8)}"
        os.mkdir(probe_name, 0o700, dir_fd=parent_fd)
        try:
            probe_fd = os.open(probe_name, flags, dir_fd=parent_fd)
            try:
                probe = os.fstat(probe_fd)
                if stat.S_IMODE(probe.st_mode) & 0o777 != 0o700:
                    raise OSError
                realized_owner = probe.st_uid
            finally:
                os.close(probe_fd)
        finally:
            os.rmdir(probe_name, dir_fd=parent_fd)
        for generation in (3, 4):
            name = f".collectivex-backend-cache-v{generation}-{os.getuid()}"
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            try:
                cache_fd = os.open(name, flags, dir_fd=parent_fd)
                try:
                    metadata = os.fstat(cache_fd)
                    if (
                        metadata.st_uid != realized_owner
                        or stat.S_IMODE(metadata.st_mode) & 0o777 != 0o700
                    ):
                        raise OSError
                    sentinel_name = ".collectivex-mount-sentinel-v1"
                    temporary_name = (
                        f"{sentinel_name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
                    )
                    create_flags = (
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    payload = secrets.token_bytes(32)
                    temporary_fd = os.open(
                        temporary_name, create_flags, 0o600, dir_fd=cache_fd
                    )
                    try:
                        try:
                            view = memoryview(payload)
                            try:
                                while view:
                                    written = os.write(temporary_fd, view)
                                    if written <= 0:
                                        raise OSError
                                    view = view[written:]
                                os.fsync(temporary_fd)
                            finally:
                                view.release()
                        finally:
                            os.close(temporary_fd)
                        try:
                            os.link(
                                temporary_name,
                                sentinel_name,
                                src_dir_fd=cache_fd,
                                dst_dir_fd=cache_fd,
                                follow_symlinks=False,
                            )
                        except FileExistsError:
                            pass
                    finally:
                        try:
                            os.unlink(temporary_name, dir_fd=cache_fd)
                        except FileNotFoundError:
                            pass
                    sentinel_fd = os.open(
                        sentinel_name,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=cache_fd,
                    )
                    try:
                        sentinel = os.fstat(sentinel_fd)
                        payload = os.read(sentinel_fd, 33)
                        if (
                            not stat.S_ISREG(sentinel.st_mode)
                            or sentinel.st_uid != realized_owner
                            or stat.S_IMODE(sentinel.st_mode) & 0o777 != 0o600
                            or sentinel.st_size != 32
                            or len(payload) != 32
                        ):
                            raise OSError
                        sentinel_sha256 = hashlib.sha256(payload).hexdigest()
                    finally:
                        os.close(sentinel_fd)
                finally:
                    os.close(cache_fd)
            except OSError:
                if generation == 3:
                    continue
                raise
            break
    finally:
        os.close(parent_fd)
except OSError:
    raise SystemExit(1)
print(sentinel_sha256, os.path.join(parent, name), end="")
PY
)" || return 1
  sentinel_sha256="${info%% *}"
  cache="${info#* }"
  [ "$cache" != "$info" ] && [[ "$sentinel_sha256" =~ ^[0-9a-f]{64}$ ]] \
    && [[ "$cache" = /* ]] || return 1
  export CX_PREPARED_BACKEND_CACHE="$cache"
  export CX_BACKEND_CACHE_SENTINEL_SHA256="$sentinel_sha256"
}

cx_verify_backend_cache_mount() {
  python3 - "${CX_BACKEND_CACHE_ROOT:-}" \
    "${CX_BACKEND_CACHE_SENTINEL_SHA256:-}" <<'PY'
import hashlib
import os
import re
import stat
import sys

root, expected = sys.argv[1:]
try:
    if (
        not os.path.isabs(root)
        or os.path.realpath(root) != root
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
    ):
        raise OSError
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, flags)
    try:
        root_item = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_item.st_mode)
            or stat.S_IMODE(root_item.st_mode) & 0o777 != 0o700
        ):
            raise OSError
        sentinel_fd = os.open(
            ".collectivex-mount-sentinel-v1",
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        try:
            sentinel = os.fstat(sentinel_fd)
            payload = os.read(sentinel_fd, 33)
            if (
                not stat.S_ISREG(sentinel.st_mode)
                or sentinel.st_uid != root_item.st_uid
                or stat.S_IMODE(sentinel.st_mode) & 0o777 != 0o600
                or sentinel.st_size != 32
                or len(payload) != 32
                or hashlib.sha256(payload).hexdigest() != expected
            ):
                raise OSError
        finally:
            os.close(sentinel_fd)
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
      && [ "$(sha256sum "$source/deep_ep/__init__.py" | awk '{print $1}')" = \
        "$CX_DEEPEP_V2_INIT_SHA256" ] \
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
  python3 - "$source/deep_ep/__init__.py" "$CX_DEEPEP_V2_INIT_SHA256" <<'PY'
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = sys.argv[2]
old = "for so in [line.strip().split(' ')[-1] for line in f if 'nccl' in line]:"
new = "for so in [line.strip().split(' ')[-1] for line in f if 'libnccl' in line]:"
payload = path.read_text(encoding="utf-8")
if payload.count(old) != 1 or new in payload:
    raise SystemExit(1)
path.write_text(payload.replace(old, new), encoding="utf-8")
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise SystemExit(1)
PY
}

cx_extension_pair_sha256() {
  python3 - "$1" "$2" "$3" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
digest = hashlib.sha256()
try:
    if root.is_symlink() or not root.is_dir():
        raise OSError
    for pattern in sys.argv[2:]:
        matches = list(root.glob(pattern))
        if len(matches) != 1 or matches[0].is_symlink():
            raise OSError
        path = matches[0]
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError
            file_digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(path.name.encode("utf-8") + b"\0")
            digest.update(str(metadata.st_size).encode("ascii") + b"\0")
            digest.update(file_digest.digest())
        finally:
            os.close(descriptor)
except (OSError, UnicodeError):
    raise SystemExit(1)
print(digest.hexdigest(), end="")
PY
}

# Acquire source before compute allocation, preferring the verified same-run GHA seed.
_cx_prepare_backend_source() {
  local mount_src="$1" backend="$2" root source temporary revision tree fmt nccl pin
  local root_mode stage_mode root_owner stage_owner
  local seed_root="${CX_BACKEND_SOURCE_SEED_ROOT:-}" seed seed_mode
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
  if [ -n "$seed_root" ]; then
    CX_BACKEND_SOURCE_STEP="source seed validation"
    [[ "$seed_root" = /* ]] && [ -d "$seed_root" ] && [ ! -L "$seed_root" ] \
      || return 1
    seed_mode="$(stat -c '%a' "$seed_root" 2>/dev/null)" || return 1
    case "$seed_mode" in 700|[1-7]700) ;; *) return 1 ;; esac
    seed="$(cx_backend_source_path "$seed_root" "$backend")" || return 1
    cx_backend_source_is_valid "$backend" "$seed" || return 1
    CX_BACKEND_SOURCE_STEP="source seed copy"
    temporary="$(mktemp -d "$root/.${backend}.XXXXXX")" || return 1
    if ! cp -R -- "$seed/." "$temporary/" \
        || ! cx_backend_source_is_valid "$backend" "$temporary" \
        || ! mv -- "$temporary" "$source"; then
      rm -rf -- "$temporary"
      return 1
    fi
    return
  fi
  if [ "${COLLECTIVEX_CANONICAL_GHA:-0}" = 1 ]; then
    CX_BACKEND_SOURCE_STEP="source seed validation"
    return 1
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
import hashlib
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
        suffix = "-" + hashlib.sha256(isolation_key.encode("utf-8")).hexdigest()[:16]
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
  local trusted_audit_salt=""
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
    trusted_audit_salt="${CX_AUDIT_SALT:-}"
  fi
  # The legacy B300 operator row contains a root-owned stage path. B300's
  # compute-visible account home is the canonical source for its private base.
  case "$runner" in b300|gb300) trusted_stage_dir="" ;; esac
  unset CX_NCCL_HOME CX_MASTER_PORT CX_MORI_KERNEL_TYPE CX_LOCK_DIR CX_STAGE_DIR CX_QOS
  unset CX_STAGE_PARENT_OWNER_OK
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  unset CX_SOCKET_IFNAME CX_RDMA_DEVICES CX_IB_GID_INDEX CX_RDMA_SERVICE_LEVEL
  unset CX_RDMA_TRAFFIC_CLASS
  unset CX_AUDIT_SALT
  unset NCCL_NET NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  unset NCCL_IB_GID_INDEX NCCL_IB_SL
  unset NVSHMEM_DISABLE_IB NVSHMEM_ENABLE_NIC_PE_MAPPING
  unset NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_IB_SL
  unset NVSHMEM_IB_ENABLE_IBGDA NVSHMEM_IBGDA_NIC_HANDLER
  unset NVSHMEM_HCA_PE_MAPPING NVSHMEM_REMOTE_TRANSPORT
  unset EP_NIC_NAME EP_OVERRIDE_RDMA_SL
  unset UCCL_SOCKET_IFNAME UCCL_IB_GID_INDEX UCCL_IB_SL MORI_RDMA_DEVICES
  unset MORI_RDMA_TC MORI_IO_TC MORI_RDMA_SL MORI_IO_SL
  unset HYBRID_EP_MULTINODE USE_NIXL RDMA_CORE_HOME DEEPEP_HYBRID_BUILD_MODE
  unset MORI_COMMIT MORI_DISABLE_AUTO_XGMI MORI_ENABLE_SDMA
  unset MORI_APP_LOG_LEVEL MORI_SHMEM_LOG_LEVEL MORI_IO_LOG_LEVEL
  unset NCCL_CUMEM_ENABLE NCCL_MNNVL_ENABLE MC_FORCE_MNNVL
  unset CX_BACKEND_CACHE_ROOT CX_BACKEND_CACHE_SENTINEL_SHA256
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
  [[ "$trusted_audit_salt" =~ ^[0-9a-f]{64}$ ]] \
    || cx_die "canonical CollectiveX execution requires a private audit salt"
  if [ "$runner" = b300 ]; then
    CX_STAGE_PARENT_OWNER_OK=1
  fi

  case "$runner" in
    h100-dgxc|h200-dgxc|b200-dgxc|b300)
      expected_nodes="${CX_NODES:-}"; expected_gpn=8
      [ "$expected_nodes" = 1 ] || [ "$expected_nodes" = 2 ] \
        || cx_die "canonical NVIDIA execution requires one or two nodes"
      CX_IMAGE="$CX_IMAGE_MULTIARCH"
      CX_IMAGE_DIGEST="$CX_IMAGE_MULTIARCH_DIGEST"
      CX_NCCL_HOME=/usr
      ;;
    gb200|gb300)
      expected_nodes="${CX_NODES:-}"; expected_gpn=4
      [ "$expected_nodes" = 2 ] || [ "$expected_nodes" = 4 ] \
        || cx_die "canonical GB execution requires two or four trays"
      CX_IMAGE="$CX_IMAGE_MULTIARCH"
      CX_IMAGE_DIGEST="$CX_IMAGE_MULTIARCH_DIGEST"
      CX_NCCL_HOME=/usr
      CX_MASTER_PORT=29551
      ;;
    mi300x|mi325x)
      expected_nodes="${CX_NODES:-}"; expected_gpn=8
      [ "$expected_nodes" = 1 ] || [ "$expected_nodes" = 2 ] \
        || cx_die "canonical AMD execution requires one or two nodes"
      CX_IMAGE="$CX_IMAGE_AMD_MORI_MI325"
      CX_IMAGE_DIGEST="$CX_IMAGE_AMD_MORI_MI325_DIGEST"
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
    mi355x)
      expected_nodes="${CX_NODES:-}"; expected_gpn=8
      [ "$expected_nodes" = 1 ] || [ "$expected_nodes" = 2 ] \
        || cx_die "canonical AMD execution requires one or two nodes"
      CX_IMAGE="$CX_IMAGE_AMD_MORI"
      CX_IMAGE_DIGEST="$CX_IMAGE_AMD_MORI_DIGEST"
      if [ "$expected_nodes" = 2 ]; then
        CX_MORI_KERNEL_TYPE=internode-v1
      else
        CX_MORI_KERNEL_TYPE=intranode
      fi
      MORI_COMMIT="$CX_MORI_COMMIT_MI355"
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
  CX_AUDIT_SALT="$trusted_audit_salt"
  export CX_STAGE_DIR CX_AUDIT_SALT
  [ "${CX_NODES:-}" = "$expected_nodes" ] \
    && [ "${CX_GPUS_PER_NODE:-}" = "$expected_gpn" ] \
    || cx_die "canonical CollectiveX placement differs from the shard"
  expected_world=$((expected_nodes * expected_gpn))
  CX_NGPUS="$expected_world"
  CX_SEED=67
  case "$runner" in mi300x|mi325x|mi355x) CX_RUN_TIMEOUT=1800 ;; *) CX_RUN_TIMEOUT=900 ;; esac
  unset CX_PUBLIC_RUNNER CX_GB_PRODUCT CX_DRYRUN CX_TIMING CX_ALLOW_MNNVL
  unset CX_ENROOT_LOCAL_IMPORT COLLECTIVEX_IMAGE COLLECTIVEX_IMAGE_DIGEST
  unset COLLECTIVEX_IMAGE_DIGEST_VERIFIED COLLECTIVEX_SQUASH_SHA256
  export CX_IMAGE CX_IMAGE_DIGEST CX_NGPUS CX_SEED CX_RUN_TIMEOUT
  case "$runner" in
    h100-dgxc|h200-dgxc|b200-dgxc|b300) export CX_NCCL_HOME ;;
    gb200|gb300) export CX_NCCL_HOME CX_MASTER_PORT ;;
    mi300x|mi325x)
      export CX_MORI_KERNEL_TYPE MORI_COMMIT MORI_DISABLE_AUTO_XGMI MORI_ENABLE_SDMA
      export MORI_APP_LOG_LEVEL MORI_SHMEM_LOG_LEVEL MORI_IO_LOG_LEVEL
      ;;
    mi355x) export CX_MORI_KERNEL_TYPE MORI_COMMIT ;;
  esac
}

cx_reverify_registry_image() {
  local image="$1" actual
  [[ "${COLLECTIVEX_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    && [ "${COLLECTIVEX_IMAGE_DIGEST_VERIFIED:-0}" = 1 ] || return 1
  actual="$(cx_resolve_registry_digest "$image")" || return 1
  [ "$actual" = "$COLLECTIVEX_IMAGE_DIGEST" ] || {
    cx_log "ERROR: configured image tag changed during container import"
    return 1
  }
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
  local squash_dir="$1" image="$2" key platform
  [[ "${COLLECTIVEX_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || return 1
  case "${CX_IMAGE_PLATFORM:-}" in
    linux/amd64) platform="" ;;
    linux/arm64) platform="_linux_arm64" ;;
    *) return 1 ;;
  esac
  key="${CX_SQUASH_FORMAT_VERSION}${platform}_${COLLECTIVEX_IMAGE_DIGEST#sha256:}_$(
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
  if ! cx_reverify_registry_image "$image" >> "$log" 2>&1; then
    flock -u "$lock_fd" >/dev/null 2>&1 || true
    exec {lock_fd}>&-
    cx_fail_stage container-import "$log"
    return 1
  fi
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  echo "$sq"
}

# Import on an allocated compute node so multiarch tags resolve for the target
# architecture. The squash directory must be shared with the submit host.
cx_ensure_squash_on_job() {
  local job_id="$1" squash_dir="$2" image="$3" lock_dir="${4:-}" sq key lock log
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
  sq="$(cx_squash_path "$squash_dir" "$image")" || return 1
  key="${sq##*/}"
  key="${key%.sqsh}"
  [ -n "$lock_dir" ] || lock_dir="$squash_dir/.locks"
  lock="$lock_dir/${key}.lock"
  log="$(cx_private_log_path container-import)"
  if ! srun --jobid="$job_id" --nodes=1 --ntasks=1 --chdir=/tmp \
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
flock -w 900 9
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
  if ! cx_reverify_registry_image "$image" >> "$log" 2>&1; then
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

# Return success only when a benchmark output is a complete JSON result object.
# Callers use this before synthesizing a terminal outcome so an emitted invalid result
# is not shadowed by a second record for the same attempt.
cx_has_result_doc() {
  local path="$1"
  python3 "$_CX_COMMON_ROOT/contracts.py" probe "$path" >/dev/null 2>&1
}

cx_result_doc_is() {
  local path="$1" expected="$2"
  python3 "$_CX_COMMON_ROOT/contracts.py" probe "$path" --status "$expected" \
    >/dev/null 2>&1
}

# A rank-zero result can be written before another rank or backend teardown fails. Preserve its
# measurements, but make the distributed command's nonzero terminal status authoritative.
cx_demote_result_doc() {
  local path="$1" rc="$2"
  python3 "$_CX_COMMON_ROOT/contracts.py" demote "$path" --return-code "$rc"
}

cx_quarantine_result_doc() {
  python3 "$_CX_COMMON_ROOT/contracts.py" quarantine-invalid "$1"
}

# cx_emit_ep_failed_case <out> <backend> <phase> <return-code>
# Preserve failures from rack launchers that invoke run_ep.py directly and therefore cannot use
# run_in_container.sh's emitter. Case identity is read from the exported CX_* variables.
cx_emit_ep_failed_case() {
  local out="$1" backend="$2" phase="$3" rc="$4"
  local -a args=(emit-terminal --out "$out" --backend "$backend" --phase "$phase"
    --return-code "$rc")
  [ -z "${CX_FAILURE_MODE:-}" ] || args+=(--failure-mode "$CX_FAILURE_MODE")
  if ! python3 "$_CX_COMMON_ROOT/contracts.py" "${args[@]}"
  then
    cx_log "ERROR: could not preserve terminal outcome"
    return 1
  fi
}

cx_case_attempt_exists() {
  local out_dir="$1" case_id="$2"
  python3 - "$_CX_COMMON_ROOT" "$out_dir" "$case_id" <<'PY'
import pathlib, sys

sys.path.insert(0, sys.argv[1])
import contracts

sample_paths = set()
referenced_samples = set()
found = False

def quarantine(path, document):
    sample = document.get("sample_artifact") if isinstance(document, dict) else None
    if (
        isinstance(sample, dict)
        and isinstance(sample.get("path"), str)
        and pathlib.Path(sample["path"]).name == sample["path"]
    ):
        sample_path = path.with_name(sample["path"])
        if sample_path.is_file():
            sample_path.replace(sample_path.with_name(sample_path.name + ".quarantine"))
    if path.is_file():
        path.replace(path.with_name(path.name + ".quarantine"))

for path in pathlib.Path(sys.argv[2]).glob("*.json"):
    document = None
    try:
        document = contracts.strict_load(path)
        if not isinstance(document, dict):
            continue
        if document.get("format") == contracts.RAW_FORMAT:
            document = contracts.load_raw_attempt(path)
            referenced_samples.add(path.with_name(document["sample_artifact"]["path"]))
        elif document.get("format") == contracts.TERMINAL_FORMAT:
            document = contracts.validate_terminal_document(document)
        elif document.get("format") == contracts.SAMPLES_FORMAT:
            contracts.validate_samples_document(document)
            sample_paths.add(path)
            continue
        else:
            continue
    except (contracts.ContractError, OSError, ValueError):
        quarantine(path, document)
        continue
    if document["identity"]["case_id"] == sys.argv[3]:
        found = True
for orphan in sample_paths - referenced_samples:
    quarantine(orphan, {})
raise SystemExit(0 if found else 1)
PY
}

# Emit one setup-failure record per requested case. Rack launchers call this when
# backend preparation fails before rank processes can start.
cx_emit_setup_failures() {
  local root="$1" out_dir="$2" backend="$3" rc="$4" shard="${CX_SHARD_FILE:-}" path
  local phase case_id suite workload required routing eplb ep hidden topk experts nodes
  local gpn domain ladder canonical timing mode scope scale_up_transport scale_out_transport
  local warmup_semantics precision_profile
  local transport topology_class
  local cases_file expected emitted=0 covered=0
  mkdir -p "$out_dir" || return 1
  export CX_FAILURE_MODE="${CX_FAILSAFE_MODE:-setup}" CX_ATTEMPT_ID=1
  if [ -z "$shard" ]; then
    local phases="${CX_PHASE:-decode}"
    [ "$phases" = both ] && phases="decode prefill"
    for phase in $phases; do
      if [ -n "${CX_CASE_ID:-}" ] && cx_case_attempt_exists "$out_dir" "$CX_CASE_ID"; then
        continue
      fi
      cx_emit_ep_failed_case "$out_dir/failed_${backend}_${phase}_${CX_TS:-setup}-a01.json" \
        "$backend" "$phase" "$rc" || return 1
    done
    unset CX_FAILURE_MODE
    return 0
  fi
  path="$shard"
  [ -f "$path" ] || path="${root%/}/$shard"
  [ -f "$path" ] || {
    unset CX_FAILURE_MODE
    cx_log "ERROR: cannot emit setup failures without shard control"
    return 1
  }
  export COLLECTIVEX_CONTROL_SHA256
  COLLECTIVEX_CONTROL_SHA256="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$COLLECTIVEX_CONTROL_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    unset CX_FAILURE_MODE COLLECTIVEX_CONTROL_SHA256
    cx_log "ERROR: cannot hash shard for setup-failure records"
    return 1
  }
  cases_file="$(mktemp)" || return 1
  if ! python3 - "$path" > "$cases_file" <<'PY'
import json, sys

with open(sys.argv[1]) as handle:
    cases = json.load(handle)["cases"]
for case in cases:
    fields = (
        case["phase"], case["mode"], case["case_id"], case["suite"], case["workload"],
        case["required_publication"], case["routing"], "1" if case["eplb"] else "",
        case["ep"], case["hidden"], case["topk"], case["experts"], case["nodes"],
        case["gpus_per_node"], case["scale_up_domain"], case["scope"],
        case["scale_up_transport"], case.get("scale_out_transport") or "",
        case["transport"], case["topology_class"], case["ladder"],
        case["warmup_semantics"],
        "1" if case["canonical"] else "", case["timing"],
        case.get("precision_profile") or "",
    )
    print("|".join(map(str, fields)))
PY
  then
    rm -f "$cases_file"
    unset CX_FAILURE_MODE
    return 1
  fi
  expected="$(wc -l < "$cases_file" | tr -d ' ')"
  [ "$expected" -gt 0 ] || { rm -f "$cases_file"; unset CX_FAILURE_MODE; return 1; }
  while IFS='|' read -r phase mode case_id suite workload required routing eplb ep hidden topk \
      experts nodes gpn domain scope scale_up_transport scale_out_transport transport \
      topology_class ladder warmup_semantics canonical timing precision_profile; do
    export CX_CASE_ID="$case_id" CX_SUITE="$suite" CX_WORKLOAD_NAME="$workload"
    export CX_REQUIRED_PUBLICATION="$required" CX_ROUTING="$routing" CX_EPLB="$eplb"
    export CX_EP="$ep" CX_NGPUS="$ep" CX_HIDDEN="$hidden" CX_TOPK="$topk" CX_EXPERTS="$experts"
    export CX_MODE="$mode" CX_NODES="$nodes" CX_GPUS_PER_NODE="$gpn"
    export CX_SCALE_UP_DOMAIN="$domain" CX_SCOPE="$scope"
    export CX_SCALE_UP_TRANSPORT="$scale_up_transport"
    export CX_SCALE_OUT_TRANSPORT="$scale_out_transport"
    export CX_TRANSPORT="$transport" CX_TOPO="$topology_class"
    export CX_TOKENS_LADDER="$ladder" CX_CANONICAL="$canonical"
    export CX_PRECISION_PROFILE="$precision_profile"
    export CX_WARMUP_SEMANTICS="$warmup_semantics"
    IFS=: read -r CX_ITERS CX_TRIALS CX_WARMUP <<< "$timing"
    export CX_ITERS CX_TRIALS CX_WARMUP CX_SAMPLES_PER_POINT="$((CX_ITERS * CX_TRIALS))"
    if cx_case_attempt_exists "$out_dir" "$case_id"; then
      covered=$((covered + 1))
      continue
    fi
    cx_emit_ep_failed_case "$out_dir/failed_${case_id}-a01.json" "$backend" "$phase" "$rc" || return 1
    emitted=$((emitted + 1))
  done < "$cases_file"
  rm -f "$cases_file"
  unset CX_FAILURE_MODE
  [ "$((emitted + covered))" -eq "$expected" ] || {
    cx_log "ERROR: covered $((emitted + covered))/$expected terminal cases"
    return 1
  }
}

# Run one validated shard with one Slurm task per GPU. Launchers provide only
# allocation/container policy through globals and CX_DISTRIBUTED_CONTAINER_ARGS.
# shellcheck disable=SC2153
cx_run_distributed_shard() {
  local build_log build_rc cases_file expected_cases ci=0 failed_cases=0
  local ph mode routing eplb hidden topk experts ladder suite workload required_pub
  local canonical case_id ep timing case_iters case_trials case_warmup case_stem
  local scope scale_up_transport scale_out_transport transport topology_class nodes gpn domain
  local precision_profile
  local workload_dir workload_ladder workload_log stage_rc attempt_tag out failure_out
  local runtime_log run_rc expected_out case_ok summary_log
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
    [ "${CX_PRECISION_PROBE:-0}" != 1 ] || return "$build_rc"
    cx_emit_setup_failures "$CX_DIR" "$MOUNT_SRC/experimental/CollectiveX/results" \
      "$CX_BENCH" "$build_rc"
    return "$build_rc"
  fi
  cx_set_failure_stage execution

  if [ "${CX_PRECISION_PROBE:-0}" = 1 ]; then
    local fields probe_id backend sku ep mode profile
    fields="$(cx_precision_probe_control_fields "$CX_DIR")" || return 1
    IFS='|' read -r probe_id backend sku ep mode profile <<< "$fields"
    [ "$backend" = "$CX_BENCH" ] && [ "$sku" = "$RUNNER" ] && [ "$ep" = "$NGPUS" ] \
      || cx_die "precision probe control differs from runtime placement"
    out="results/${probe_id}.json"
    expected_out="$MOUNT_SRC/experimental/CollectiveX/$out"
    runtime_log="$(cx_private_log_path precision-probe)"
    set +e
    timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JOB_ID" --nodes="$NODES" \
      --ntasks="$NGPUS" --ntasks-per-node="$GPN" --chdir=/tmp \
      --container-name="$container_name" --container-image="$SQUASH_FILE" \
      "${container_args[@]}" \
      --export="$(cx_container_exports)" \
      bash -c "$WRAP" _ --backend "$backend" --sku "$sku" --ep "$ep" \
      --mode "$mode" --precision-profile "$profile" --out "$out" \
      </dev/null >"$runtime_log" 2>&1
    run_rc=$?
    set -e
    if [ "$run_rc" = 124 ] || [ "$run_rc" = 137 ]; then
      printf '[collectivex] precision probe timed out rc=%s limit=%ss\n' \
        "$run_rc" "${CX_RUN_TIMEOUT:-900}" >> "$runtime_log"
    fi
    if [ "$run_rc" != 0 ] || ! python3 "$CX_DIR/tests/probe_precision.py" \
        --validate-manifest "$expected_out" >/dev/null 2>&1; then
      [ "$run_rc" != 0 ] || run_rc=1
      cx_fail_stage execution "$runtime_log" || true
      return "$run_rc"
    fi
    return 0
  fi

  cases_file="$(mktemp)" || return 1
  local shard="${CX_SHARD_FILE:-}"
  [ -z "$shard" ] || [ -f "$shard" ] || shard="$CX_DIR/$shard"
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
        "1" if case.get("eplb") else "", get("hidden", "7168"),
        get("topk", "8"), get("experts", "256"), get("ladder"),
        get("suite"), get("workload"), get("required_publication"),
        "1" if case.get("canonical") else "", get("case_id"), get("ep"),
        get("timing", "8:64:32"), get("nodes"), get("gpus_per_node"),
        get("scale_up_domain"), get("scope"), get("scale_up_transport"),
        get("scale_out_transport"), get("transport"), get("topology_class"),
        get("precision_profile"),
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
      "${CX_EPLB:-}" "${CX_HIDDEN:-7168}" "${CX_TOPK:-8}" "${CX_EXPERTS:-256}" \
      "${CX_TOKENS_LADDER:-}" "${CX_SUITE:-}" "${CX_WORKLOAD_NAME:-}" \
      "${CX_REQUIRED_PUBLICATION:-}" "${CX_CANONICAL:-}" "${CX_CASE_ID:-}" \
      "${CX_PRECISION_PROFILE:-}" \
      "${CX_ITERS:-8}" "${CX_TRIALS:-64}" "${CX_WARMUP:-32}" \
      "${CX_SCOPE:-scale-up}" \
      "${CX_SCALE_UP_TRANSPORT:-unknown}" "${CX_SCALE_OUT_TRANSPORT:-}" \
      "${CX_TRANSPORT:-unknown}" "${CX_TOPO:-manual}"
    for phase in $phases; do
      (IFS='|'; printf '%s\n' "$phase|${CX_MODE:-normal}|${CX_ROUTING:-uniform}|${CX_EPLB:-}|${CX_HIDDEN:-7168}|${CX_TOPK:-8}|${CX_EXPERTS:-256}|${CX_TOKENS_LADDER:-}|${CX_SUITE:-}|${CX_WORKLOAD_NAME:-}|${CX_REQUIRED_PUBLICATION:-}|${CX_CANONICAL:-}|${CX_CASE_ID:-}|$NGPUS|${CX_ITERS:-8}:${CX_TRIALS:-64}:${CX_WARMUP:-32}|$NODES|$GPN|$SCALE_UP_DOMAIN|${CX_SCOPE:-scale-up}|${CX_SCALE_UP_TRANSPORT:-unknown}|${CX_SCALE_OUT_TRANSPORT:-}|${CX_TRANSPORT:-unknown}|${CX_TOPO:-manual}|${CX_PRECISION_PROFILE:-}")
    done > "$cases_file"
  fi
  expected_cases="$(wc -l < "$cases_file" | tr -d ' ')"
  [ "$expected_cases" -gt 0 ] \
    || { rm -f "$cases_file"; cx_die "distributed case list is empty"; }

  while IFS='|' read -r ph mode routing eplb hidden topk experts ladder suite workload \
      required_pub canonical case_id ep timing nodes gpn domain scope scale_up_transport \
      scale_out_transport transport topology_class precision_profile; do
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
    export CX_REQUIRED_PUBLICATION="$required_pub" CX_CANONICAL="$canonical" CX_EP="$ep"
    export CX_PRECISION_PROFILE="$precision_profile"
    export CX_ROUTING="$routing" CX_EPLB="$eplb" CX_TOKENS_LADDER="$ladder"
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
      export CX_ATTEMPT_ID=1
      failure_out="$MOUNT_SRC/experimental/CollectiveX/results/failed_${case_stem}-a01.json"
      cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" 5
      failed_cases=$((failed_cases + 1))
      continue
    fi

    workload_dir=""
    if cx_bool_enabled "$canonical"; then
      workload_dir=".cx_workloads/c$(printf '%03d' "$ci")"
      workload_ladder="$ladder"
      [ -n "$workload_ladder" ] \
        || workload_ladder="1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
      workload_args=(python3 tests/make_workloads.py --out-dir "$workload_dir"
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
        export CX_ATTEMPT_ID=1
        failure_out="$MOUNT_SRC/experimental/CollectiveX/results/failed_${case_stem}-a01.json"
        cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" "$stage_rc"
        failed_cases=$((failed_cases + 1))
        continue
      fi
    fi

    ep_args=(--backend "$CX_BENCH" --mode "$mode" --phase "$ph" --routing "$routing"
      --precision-profile "$precision_profile"
      --gpus-per-node "$gpn" --scale-up-domain "$domain" --scope "$scope"
      --scale-up-transport "$scale_up_transport" --scale-out-transport "$scale_out_transport"
      --tokens-ladder "$ladder" --hidden "$hidden" --topk "$topk" --experts "$experts"
      --warmup "$case_warmup" --iters "$case_iters" --trials "$case_trials"
      --seed "${CX_SEED:-67}" --runner "$RUNNER" --topology-class "$topology_class"
      --transport "$transport" --case-id "$case_id" --suite "$suite"
      --workload-name "$workload" --required-publication "$required_pub"
      --qualification-index "${CX_QUALIFICATION_INDEX:-1}")
    cx_bool_enabled "$eplb" && ep_args+=(--eplb)
    [ -z "$workload_dir" ] || ep_args+=(--workload-dir "$workload_dir")
    export CX_ATTEMPT_ID=1
    attempt_tag=a01
    out="results/${case_stem}_${attempt_tag}.json"
    failure_out="$MOUNT_SRC/experimental/CollectiveX/results/failed_${case_stem}-${attempt_tag}.json"
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
    expected_out="$MOUNT_SRC/experimental/CollectiveX/$out"
    case_ok=0
    if [ "$run_rc" = 0 ] && cx_result_doc_is "$expected_out" success; then
      case_ok=1
    elif [ "$run_rc" = 0 ] && cx_result_doc_is "$expected_out" invalid; then
      cx_log "ERROR: EP${NGPUS}[$ci] completed with invalid semantic evidence"
    else
      [ "$run_rc" != 0 ] || run_rc=1
      if cx_has_result_doc "$expected_out"; then
        cx_demote_result_doc "$expected_out" "$run_rc" \
          || { cx_quarantine_result_doc "$expected_out"; cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" "$run_rc"; }
      else
        cx_quarantine_result_doc "$expected_out"
        cx_emit_ep_failed_case "$failure_out" "$CX_BENCH" "$ph" "$run_rc"
      fi
    fi
    if [ "$case_ok" = 0 ]; then
      [ "$run_rc" = 0 ] || cx_fail_stage execution "$runtime_log" || true
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

cx_launcher_cleanup() {
  local rc="$1" stage_root="${MOUNT_SRC:-}" source_root out_dir allocation_stopped=1
  source_root="${stage_root:-${REPO_ROOT:-}}"
  trap - EXIT HUP INT TERM
  if [ -n "${COLLECTIVEX_EPHEMERAL_CONFIG_PATH:-}" ]; then
    rm -f -- "$COLLECTIVEX_EPHEMERAL_CONFIG_PATH" >/dev/null 2>&1 || true
    unset COLLECTIVEX_EPHEMERAL_CONFIG_PATH
  fi
  if [ -n "${JOB_ID:-}" ]; then
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
  if [ "$rc" != 0 ] && [ "${CX_PRECISION_PROBE:-0}" = 1 ]; then
    cx_log "ERROR: precision-probe-failure-class=${CX_FAILSAFE_MODE:-setup}"
  fi
  if [ "$rc" != 0 ] && [ "${CX_PRECISION_PROBE:-0}" != 1 ] \
      && [ -n "${REPO_ROOT:-}" ] && [ -n "${CX_BENCH:-}" ]; then
    cx_log "ERROR: terminal-failure-class=${CX_FAILSAFE_MODE:-setup}"
    [ -d "$source_root/experimental/CollectiveX" ] || source_root="$REPO_ROOT"
    out_dir="$source_root/experimental/CollectiveX/results"
    cx_emit_setup_failures \
      "$source_root/experimental/CollectiveX" "$out_dir" "$CX_BENCH" "$rc" || true
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
