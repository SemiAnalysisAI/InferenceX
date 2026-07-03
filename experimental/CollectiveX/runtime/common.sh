# shellcheck shell=bash
# CollectiveX — shared launcher helpers (sourced, not executed).
#
# Cluster-generic scaffolding only (Slurm/container/build/staging); no
# model-serving. Logging goes to stderr so functions can `echo` a single
# result on stdout.

cx_log() { printf '[collectivex] %s\n' "$*" >&2; }
cx_die() { printf '[collectivex] FATAL: %s\n' "$*" >&2; exit 1; }

# Runner-local deployment settings are deliberately kept outside the checkout.
# The file is trusted shell input owned by the runner operator.
cx_load_operator_config() {
  [ -n "${COLLECTIVEX_OPERATOR_CONFIG_LOADED:-}" ] && return 0
  local config="${COLLECTIVEX_OPERATOR_CONFIG:-${XDG_CONFIG_HOME:-${HOME}/.config}/inferencex/collectivex.env}"
  if [ -r "$config" ]; then
    # shellcheck disable=SC1090
    source "$config"
  fi
  export COLLECTIVEX_OPERATOR_CONFIG_LOADED=1
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

cx_require_single_node() {
  [ "${CX_NODES:-1}" = "1" ] || cx_die "$1 supports one-node EP only"
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

cx_load_operator_config

# Allocate via salloc (--no-shell is appended) and echo the GRANTED Slurm job id, parsed from
# salloc's OWN output. Use INSTEAD of `salloc ...; JOB_ID=$(squeue --name=<name> -h -o %A | head -1)`:
# that lookup is not unique per allocation, so concurrent cells can resolve a sibling allocation.
# Parsing salloc's own "Granted job allocation N" is race-free; raw scheduler output stays private.
cx_salloc_jobid() {
  local _t; _t="$(mktemp)"
  salloc "$@" --no-shell >"$_t" 2>&1 || true
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
CX_IMAGE_MULTIARCH_DIGEST="sha256:061fb71f838e82000a1768c159654d526c2f17ebe751c21e7fc48ca53c8ef975"
# (v0.5.12-cu130 was rejected: its 62 layers overflow enroot's overlay-based
# squash creation on these nodes — "failed to mount overlay ... Invalid argument".
# v0.5.11-cu130 imports cleanly.)
# DeepEP is NOT bundled here -> run_in_container.sh builds it via rebuild-deepep.
# The arch-specific deepseek-v4-{blackwell,grace-blackwell} images do bundle
# DeepEP, but are not multi-arch and are not the default.
CX_IMAGE_MULTIARCH="lmsysorg/sglang:v0.5.11-cu130"

# AMD (ROCm/CDNA): the multi-arch NVIDIA image above is x86_64+aarch64 CUDA and
# cannot run on MI355X. AMD uses a separate ROCm image that bundles MoRI (the
# AMD EP library). Single-arch (linux/amd64 host, ROCm runtime); not digest-
# pinned yet, so it is not promotion-eligible.
CX_IMAGE_AMD_MORI="rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2"
cx_default_image() {
  case "$1" in
    mi355x*|mi325x*) echo "$CX_IMAGE_AMD_MORI" ;;
    b200*|gb200*|b300*|gb300*|h100*|h200*) echo "$CX_IMAGE_MULTIARCH" ;;
    *) cx_die "no default image for runner prefix: $1" ;;
  esac
}

cx_default_image_digest() {
  [ "$1" = "$CX_IMAGE_MULTIARCH" ] && printf '%s' "$CX_IMAGE_MULTIARCH_DIGEST"
}

# cx_ensure_squash <squash_dir> <image>  ->  echoes the squash file path.
# Imports via enroot only if a valid squash is not already present (flock-guarded,
# mirroring runners/launch_b200-dgxc.sh).
cx_ensure_squash() {
  local squash_dir="$1" image="$2"
  mkdir -p "$squash_dir" 2>/dev/null || true
  local key sq locks lock_fd
  key="$(printf '%s' "$image" | sed 's#[/:@#]#_#g')"
  sq="$squash_dir/${key}.sqsh"
  locks="$squash_dir/.locks"; mkdir -p "$locks" 2>/dev/null || true
  { exec {lock_fd}>"$locks/${key}.lock"; } 2>/dev/null \
    || cx_die "cannot open the configured squash lock"
  flock -w 900 "$lock_fd" || cx_die "configured squash lock timed out"
  if unsquashfs -l "$sq" >/dev/null 2>&1; then
    cx_log "container squash ready"
  else
    cx_log "importing configured container image"
    rm -f "$sq" 2>/dev/null || cx_die "cannot replace the configured squash"
    # </dev/null: never block on an interactive password prompt.
    enroot import -o "$sq" "docker://$image" </dev/null >/dev/null 2>&1 \
      || cx_die "configured container image import failed"
    unsquashfs -l "$sq" >/dev/null 2>&1 \
      || cx_die "configured container image produced an invalid squash"
  fi
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  echo "$sq"
}

# cx_stage_repo <repo_root> <stage_dir>  ->  echoes the mount-source root.
# Some deployments do not cross-mount the runner workspace to compute nodes. If
# CX_STAGE_DIR is set, rsync the CollectiveX tree onto a compute-visible shared
# filesystem and mount from there. No-op (echo repo_root) when
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
  # a workflow-provided execution id. Outside GHA, keep the single shared dir.
  local tag="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-}}"
  if [ -n "$tag" ]; then
    stage_dir="$stage_dir/job_$(printf '%s' "$tag" | tr -c 'A-Za-z0-9._-' '_')"
  fi
  mkdir -p "$stage_dir/experimental" 2>/dev/null \
    || cx_die "cannot create the configured stage directory"
  cx_log "staging CollectiveX on compute-visible storage"
  rsync -a --delete --delete-excluded \
    --exclude='__pycache__/' --exclude='results/' --exclude='.cx_workloads/' \
    --exclude='configs/platforms.yaml' --exclude='private-infra.md' \
    --exclude='goal.md' --exclude='notes.md' \
    "$repo_root/experimental/CollectiveX" "$stage_dir/experimental/" >/dev/null 2>&1 \
    || cx_die "staging CollectiveX failed"
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
  cx_log "collected staged results for artifact validation"
}

# Return success only when a benchmark output is a complete JSON result object.
# Callers use this before synthesizing a failed-case so an emitted invalid result
# is not shadowed by a second record for the same attempt.
cx_has_result_doc() {
  local path="$1"
  [ -f "$path" ] || return 1
  python3 - "$path" <<'PY' >/dev/null 2>&1
import json
import sys

try:
    with open(sys.argv[1]) as fh:
        doc = json.load(fh)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

is_result = (
    isinstance(doc, dict)
    and doc.get("schema_version") is not None
    and doc.get("family") is not None
    and any(key in doc for key in ("publication_status", "status", "record_type"))
)
raise SystemExit(0 if is_result else 1)
PY
}

# A rank-zero result can be written before another rank or backend teardown fails. Preserve its
# measurements, but make the distributed command's nonzero terminal status authoritative.
cx_demote_result_doc() {
  local path="$1" rc="$2"
  python3 - "$path" "$rc" <<'PY'
import json
import os
import sys

path, rc_text = sys.argv[1:3]
with open(path) as fh:
    doc = json.load(fh)
if not isinstance(doc, dict):
    raise SystemExit(1)
validity = doc.get("validity")
if not isinstance(validity, dict):
    validity = {}
doc["validity"] = {**validity, "execution_status": "failed"}
doc["publication_status"] = "failed"
doc["status"] = "invalid"
doc["post_emit_failure"] = {"return_code": int(rc_text)}
tmp = f"{path}.tmp"
with open(tmp, "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
PY
}

# cx_emit_ep_failed_case <out> <backend> <phase> <return-code>
# Preserve failures from rack launchers that invoke run_ep.py directly and therefore cannot use
# run_in_container.sh's emitter. Case identity is read from the exported CX_* variables.
cx_emit_ep_failed_case() {
  local out="$1" backend="$2" phase="$3" rc="$4"
  python3 - "$out" "$backend" "$phase" "$rc" <<'PY' || \
    cx_log "WARN: could not preserve failed-case record"
import datetime as dt
import json
import os
import sys

out, backend, phase, rc_text = sys.argv[1:5]
rc = int(rc_text)


def env(name, default=""):
    return os.environ.get(name, default)


def integer(name, default):
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def enabled(name):
    return env(name).lower() in {"1", "true", "yes"}


failure_mode = {
    5: "unsupported", 124: "timeout", 137: "timeout", 134: "deadlock",
}.get(rc, "unknown")
case = {
    "case_id": env("CX_CASE_ID") or None,
    "suite": env("CX_SUITE") or None,
    "workload": env("CX_WORKLOAD_NAME") or None,
    "required_publication": env("CX_REQUIRED_PUBLICATION") or None,
    "backend": backend,
    "phase": phase,
    "ep": integer("CX_EP", integer("CX_NGPUS", 1)),
    "gpus_per_node": integer("CX_GPUS_PER_NODE", integer("CX_NGPUS", 1)),
    "scale_up_domain": integer("CX_SCALE_UP_DOMAIN", integer("CX_NGPUS", 1)),
    "dispatch_dtype": env("CX_DISPATCH_DTYPE", "bf16"),
    "mode": env("CX_MODE", "normal"),
    "contract": env("CX_MEASUREMENT_CONTRACT", "layout-and-dispatch-v1"),
    "routing": env("CX_ROUTING", "uniform"),
    "eplb": enabled("CX_EPLB"),
    "combine_quant_mode": env("CX_COMBINE_QUANT_MODE", "none"),
    "resource_mode": env("CX_RESOURCE_MODE", "tuned"),
    "activation_profile": env("CX_ACTIVATION_PROFILE", "normal"),
    "placement": env("CX_PLACEMENT", "packed"),
    "routing_step": env("CX_ROUTING_STEP", "0"),
    "uneven_tokens": env("CX_UNEVEN_TOKENS", "none"),
    "tokens_ladder": env("CX_TOKENS_LADDER"),
    "canonical": enabled("CX_CANONICAL"),
    "sampling_contract": "fixed-512-v1",
    "samples_per_point": integer("CX_SAMPLES_PER_POINT", 512),
    "iters": integer("CX_ITERS", 8),
    "trials": integer("CX_TRIALS", 64),
    "warmup": integer("CX_WARMUP", 32),
    "warmup_semantics": env(
        "CX_WARMUP_SEMANTICS", "full-roundtrip-per-trial-point-v1"
    ),
}
record = {
    "schema_version": 5,
    "family": "moe",
    "record_type": "failed-case",
    "generated_by": "runtime/common.sh",
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "attempt_id": env("CX_ATTEMPT_ID", "1"),
    "case_id": case["case_id"],
    "suite": case["suite"],
    "workload_name": case["workload"],
    "required_publication": case["required_publication"],
    "runner": env("CX_RUNNER"),
    "backend": backend,
    "mode": case["mode"],
    "phase": phase,
    "ep_size": case["ep"],
    "measurement_contract": case["contract"],
    "resource_mode": case["resource_mode"],
    "topology_class": env("CX_TOPO"),
    "status": "failed",
    "publication_status": "failed",
    "rows": [],
    "failure": {
        "failure_mode": failure_mode,
        "return_code": rc,
        "case": case,
        "evidence": "",
    },
}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(record, fh, indent=2)
print(f"preserved failed-case record ({failure_mode})")
PY
}
