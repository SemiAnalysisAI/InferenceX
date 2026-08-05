#!/usr/bin/env bash
# Self-hosted launcher for the SPUR cluster (Crusoe-operated, 126x MI355X in
# partition amd-spur). The Actions runner lives on the login node
# (crs-m2m-cpu-spur-009), which has NO GPUs and NO docker -- it only
# orchestrates. The recipe runs on a worker, reached via srun.
#
# Why this cannot reuse launch_mi355x-amds.sh:
#
#   SPUR is not Slurm. It is Crusoe's Rust reimplementation of the slurm CLI
#   (`spur` 0.6.0); srun/sinfo/squeue are compat wrappers talking to spurctld.
#   Three differences break the amds path outright:
#
#     1. `salloc` DOES NOT EXIST. Only srun/sbatch/squeue/sinfo/scancel/scontrol
#        are provided. The amds salloc -> enroot import -> pyxis srun flow has
#        no entry point here.
#     2. There is NO `--export` flag of any kind, so nothing from this shell's
#        environment reaches the job. Every value is instead interpolated
#        literally into the inner script generated below, which srun then runs
#        by path off NFS. Nothing depends on env inheritance.
#     3. SPUR does have native container flags (--container-image/-mounts/-env),
#        but exposes no knob for /dev/kfd, /dev/dri, --group-add, --shm-size or
#        ipc=host -- all mandatory for ROCm. So we srun a plain bash script and
#        use docker inside it, matching slurm_container_runtime=docker.
#
# Weights: /shared_nfs (150 T) is mounted on both login and worker nodes and
# already holds a complete HF hub cache, including the 1.5 TB moonshotai/Kimi-K3
# checkpoint and both DSpark drafters. It is READ-ONLY to us, so we mount it
# read-only, run HF offline against it, and hand the recipe a MODEL_PATH that
# points straight at the snapshot -- kimik3_fp4_mi355x.sh then takes its
# "already present" branch and never calls `hf download` against read-only
# storage. Node-local /mnt/m2m_nobackup (28 T NVMe) supplies the writable
# scratch that HF, aiperf and vLLM still need.
#
# Env parity: on the amds slurm path `srun --export=ALL` hands the container the
# entire benchmark-tmpl.yml env block. Here there is no such mechanism, so every
# variable in that block is written to a docker --env-file below. Keep the two
# in sync when benchmark-tmpl.yml gains a variable.
set -uo pipefail
umask 077
set -x

# The spur CLI finds spurctld via SPUR_CONTROLLER_ADDR, which is exported from
# /etc/profile.d/spur.sh -- and profile.d is sourced by LOGIN shells only. A
# systemd-managed Actions runner is not a login shell, so without this every
# srun dies with "failed to connect to spurctld: ... Connection refused". Source
# it explicitly rather than assuming an interactive environment.
if [[ -z "${SPUR_CONTROLLER_ADDR:-}" && -r /etc/profile.d/spur.sh ]]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/spur.sh
fi
if [[ -z "${SPUR_CONTROLLER_ADDR:-}" ]]; then
    echo "[spuraim] FATAL: SPUR_CONTROLLER_ADDR is unset and /etc/profile.d/spur.sh" \
         "is unreadable; srun cannot reach spurctld." >&2
    exit 78
fi
export SPUR_CONTROLLER_ADDR

for required_command in srun squeue scancel python3 timeout; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "[spuraim] FATAL: required command is unavailable: $required_command" >&2
        exit 69
    fi
done

SPUR_ACCOUNT="${SPUR_ACCOUNT:-amd-aifw-aim}"
SPUR_QOS="${SPUR_QOS:-amd-aifw-aim-qos}"
SPUR_PARTITION="${SPUR_PARTITION:-amd-spur}"
SPUR_CPUS_PER_TASK="${SPUR_CPUS_PER_TASK:-128}"
SPUR_TIME_LIMIT="${SPUR_TIME_LIMIT:-480}"

# Node denylist. crsuse2-m2m-071 is in the idle set but its docker daemon is
# dead ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock"),
# and idle nodes get picked first, so without this the scheduler steers us
# straight at it. `${VAR-default}` (no colon) is deliberate: setting
# SPUR_EXCLUDE_NODES= clears the denylist, unset gets the default.
SPUR_EXCLUDE_NODES="${SPUR_EXCLUDE_NODES-crsuse2-m2m-071}"

# Non-exclusive by default, based on evidence rather than convenience.
#
# Exclusive was briefly made the default to remove co-tenancy as a confound
# after run 30870535817 lost a worker. That confound has since been disproven
# twice: the non-DSpark control (30872688837) came up on a shared node, and the
# k=2 aiter arm (30874551315) failed on SPUR with the SAME 8/10 request error
# rate and ~16s duration as the upstream mia1 arm (30873376524) on a dedicated
# fleet. Co-tenancy is not what breaks these runs.
#
# Meanwhile exclusive is now a real cost: with the partition at 84 alloc / 60
# mix / 68 resv and 67+ jobs queued, demanding a whole node means waiting, and
# TP8 already needs all 8 GPUs so exclusivity buys almost nothing extra. Set
# SPUR_EXCLUSIVE=1 when a run genuinely needs an uncontended node for numbers.
SPUR_EXCLUSIVE="${SPUR_EXCLUSIVE:-0}"

# Per-runner port offset (last char of runner name), same scheme as amds.
PORT_OFFSET="${RUNNER_NAME: -1}"
[[ "$PORT_OFFSET" =~ ^[0-9]$ ]] || PORT_OFFSET=0
export PORT=$(( 8888 + PORT_OFFSET ))

# GPUs to request. benchmark-tmpl.yml exports GPU_COUNT=TP*PP_SIZE*PCP_SIZE;
# fall back to TP for manual invocation.
GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-none}" == "mtp" ]] && printf '_mtp' || printf '')

# Recipe-path resolution mirrors launch_mi355x-amds.sh.
SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_mi355x"
SCRIPT_FW="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
SCRIPT_FALLBACK="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
if [[ -f "$SCRIPT_FW" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_FW"
else
    BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
fi
echo "[spuraim] recipe: $BENCHMARK_SCRIPT"

# ---------------------------------------------------------------------------
# HF cache resolution (done here on the login node -- /shared_nfs is mounted
# here too, so we can resolve the snapshot before the job is even scheduled).
# ---------------------------------------------------------------------------
SHARED_HF_ROOT="${SPUR_SHARED_HF_ROOT:-/shared_nfs/huggingface}"
SHARED_HF_HUB="$SHARED_HF_ROOT/hub"
NODE_SCRATCH="${SPUR_NODE_SCRATCH:-/mnt/m2m_nobackup/$(id -un)}"

# Resolve an HF repo id to its snapshot dir in the shared read-only hub.
# Prints the path, or nothing if that repo is not staged there.
# org/name -> models--org--name
resolve_shared_snapshot() {
    local repo="$1"
    local dir="$SHARED_HF_HUB/models--${repo//\//--}"
    [[ -d "$dir/snapshots" ]] || return 0
    # Prefer the ref the hub points at; fall back to the sole snapshot.
    if [[ -f "$dir/refs/main" ]]; then
        local rev; rev="$(<"$dir/refs/main")"
        if [[ -d "$dir/snapshots/$rev" ]]; then printf '%s' "$dir/snapshots/$rev"; return 0; fi
    fi
    find "$dir/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1 | tr -d '\n'
}

# HF_HUB_CACHE always points at WRITABLE node-local scratch, and HF is NOT run
# offline. The shared hub is read-only, so using it as HF_HUB_CACHE plus
# HF_HUB_OFFLINE=1 looks tempting but breaks the run: the agentic recipes also
# pull a trace DATASET (semianalysisai/cc-traces-*), which is not staged there,
# and offline mode blocks it --
#   "Local entry not found. Cannot reach .../datasets/semianalysisai/... :
#    offline mode is enabled."
# So instead of routing everything through the shared hub, we take only the two
# things that are actually expensive out of it, by absolute path, and let
# everything small resolve normally over the network.
CONTAINER_HF_HUB="$NODE_SCRATCH/hf_hub"

RESOLVED_MODEL_PATH="$(resolve_shared_snapshot "$MODEL")"
if [[ -n "$RESOLVED_MODEL_PATH" ]]; then
    # MODEL_PATH non-empty makes the recipe skip its `hf download`, so the
    # 1.5 TB checkpoint is read straight off shared NFS and never copied.
    echo "[spuraim] model staged on shared NFS: $RESOLVED_MODEL_PATH"
else
    echo "[spuraim] WARNING: $MODEL is NOT on $SHARED_HF_HUB; it will be" \
         "downloaded to node-local $CONTAINER_HF_HUB (re-downloaded per fresh" \
         "node). Pre-stage large checkpoints on $SHARED_HF_HUB." >&2
fi

# Same treatment for the speculative-decoding drafter. The recipe defaults
# SPEC_DRAFT_MODEL to a repo id; handing it the shared snapshot path avoids a
# per-node download. The recipe branches on the drafter name with globs
# (*RadixArk*/*Inferact*), and those still match inside the snapshot path, so
# substituting a path does not change its behaviour. A caller-supplied
# SPEC_DRAFT_MODEL always wins.
RESOLVED_DRAFT_PATH=""
if [[ "${SPEC_DECODING:-none}" == "mtp" && -z "${SPEC_DRAFT_MODEL:-}" ]]; then
    RESOLVED_DRAFT_PATH="$(resolve_shared_snapshot "${SPUR_DEFAULT_DRAFTER:-Inferact/Kimi-K3-DSpark}")"
    [[ -n "$RESOLVED_DRAFT_PATH" ]] && \
        echo "[spuraim] drafter staged on shared NFS: $RESOLVED_DRAFT_PATH"
fi

# Unique per invocation so lifecycle cleanup can target only this allocation.
JOB_NAME="ix-${RUNNER_NAME}-$$-${EXP_NAME:-job}"
JOB_NAME="${JOB_NAME:0:60}"
CONTAINER="spuraim_${RUNNER_NAME}_$$"

# Reclaim an allocation left by an earlier crash of this exact runner. This is
# deliberately scoped by both user and runner-name prefix; it cannot touch a
# sibling runner or another user's job.
SPUR_JOB_PREFIX="ix-${RUNNER_NAME}-"
mapfile -t STALE_JOB_IDS < <(
    squeue -h -u "$(id -un)" -o '%.18i %.200j' 2>/dev/null | \
        awk -v prefix="$SPUR_JOB_PREFIX" 'index($2, prefix) == 1 { print $1 }'
)
if [[ ${#STALE_JOB_IDS[@]} -gt 0 ]]; then
    echo "[spuraim] reclaiming stale allocation(s): ${STALE_JOB_IDS[*]}"
    scancel "${STALE_JOB_IDS[@]}" || true
fi

STAGE_DIR="$GITHUB_WORKSPACE/.spuraim"
install -d -m 700 "$STAGE_DIR" || {
    echo "[spuraim] FATAL: cannot create private staging directory: $STAGE_DIR" >&2
    exit 73
}
# A killed runner can leave these behind. Each Actions runner owns a separate
# workspace and executes one job at a time, so anything left here is stale.
find "$STAGE_DIR" -maxdepth 1 -type f \
    \( -name 'env.*.list' -o -name 'inner.*.sh' -o -name 'heartbeat.*.status' \) \
    -delete || {
        echo "[spuraim] FATAL: cannot remove stale staging files from $STAGE_DIR" >&2
        exit 73
    }
ENV_FILE="$STAGE_DIR/env.$$.list"
INNER="$STAGE_DIR/inner.$$.sh"
HEARTBEAT_FILE="$STAGE_DIR/heartbeat.$$.status"

SPUR_HEARTBEAT_SECONDS="${SPUR_HEARTBEAT_SECONDS:-60}"
if ! [[ "$SPUR_HEARTBEAT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[spuraim] FATAL: SPUR_HEARTBEAT_SECONDS must be a positive integer." >&2
    exit 64
fi

MONITOR_PID=""
SRUN_ACTIVE=0

stop_monitor() {
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
}

cleanup_outer() {
    local rc=$?
    trap - EXIT INT TERM
    stop_monitor
    if [[ "$SRUN_ACTIVE" == "1" ]]; then
        echo "[spuraim] cancelling allocation for interrupted job $JOB_NAME" >&2
    fi
    # The name is unique to this invocation. This also catches the rare case
    # where the srun client exited but SPUR kept the allocation alive.
    scancel -n "$JOB_NAME" >/dev/null 2>&1 || true
    rm -f "$ENV_FILE" "$INNER" "$HEARTBEAT_FILE"
    exit "$rc"
}

handle_signal() {
    local signal="$1"
    echo "[spuraim] received $signal; stopping monitor and allocation" >&2
    case "$signal" in
        INT) exit 130 ;;
        TERM) exit 143 ;;
    esac
}

trap cleanup_outer EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

# ---------------------------------------------------------------------------
# docker --env-file: strict KEY=VALUE, one per line, value taken LITERALLY to
# end of line (no quote or backslash processing). That is exactly what we want
# -- it sidesteps the quoting hell of pushing ~50 values through
# login-shell -> srun -> bash -> docker.
# ---------------------------------------------------------------------------
emit_env() {
    local key="$1"
    local value="$2"
    if ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "[spuraim] FATAL: invalid env key: $key" >&2
        exit 65
    fi
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        echo "[spuraim] FATAL: env value for $key contains a newline." >&2
        exit 65
    fi
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
}

compact_json_env() {
    local key="$1"
    local value="$2"
    if [[ -z "$value" || "$value" == "none" || "$value" == "null" ]]; then
        printf '%s' "$value"
        return 0
    fi
    JSON_ENV_VALUE="$value" python3 -c 'import json, os
print(json.dumps(json.loads(os.environ["JSON_ENV_VALUE"]), separators=(",", ":")))' || {
        echo "[spuraim] FATAL: $key must contain valid JSON." >&2
        return 65
    }
}

# Do not xtrace secrets or metadata while constructing the env file.
set +x
: > "$ENV_FILE"
chmod 600 "$ENV_FILE"
KV_OFFLOAD_BACKEND_METADATA_COMPACT="$(compact_json_env \
    KV_OFFLOAD_BACKEND_METADATA "${KV_OFFLOAD_BACKEND_METADATA:-}")" || exit $?
ROUTER_METADATA_COMPACT="$(compact_json_env \
    ROUTER_METADATA "${ROUTER_METADATA:-}")" || exit $?
emit_env HF_HUB_CACHE          "$CONTAINER_HF_HUB"
emit_env HF_HOME               "$NODE_SCRATCH/hf_home"
emit_env HF_TOKEN              "${HF_TOKEN:-}"
[[ -n "$RESOLVED_MODEL_PATH" ]] && emit_env MODEL_PATH "$RESOLVED_MODEL_PATH"
if [[ -n "${SPEC_DRAFT_MODEL:-}" ]]; then
    emit_env SPEC_DRAFT_MODEL "$SPEC_DRAFT_MODEL"
elif [[ -n "$RESOLVED_DRAFT_PATH" ]]; then
    emit_env SPEC_DRAFT_MODEL "$RESOLVED_DRAFT_PATH"
fi
emit_env PORT                  "$PORT"
emit_env RANDOM_RANGE_RATIO    "${RANDOM_RANGE_RATIO:-0.8}"
emit_env MODEL                 "$MODEL"
emit_env MODEL_PREFIX          "${MODEL_PREFIX:-}"
emit_env EXP_NAME              "${EXP_NAME:-}"
emit_env PRECISION             "${PRECISION:-}"
emit_env FRAMEWORK             "${FRAMEWORK:-}"
emit_env IMAGE                 "${IMAGE:-}"
emit_env TP                    "$TP"
emit_env PP_SIZE               "${PP_SIZE:-1}"
emit_env DCP_SIZE              "${DCP_SIZE:-1}"
emit_env PCP_SIZE              "${PCP_SIZE:-1}"
emit_env EP_SIZE               "${EP_SIZE:-1}"
emit_env DP_ATTENTION          "${DP_ATTENTION:-false}"
emit_env CONC                  "$CONC"
emit_env ISL                   "${ISL:-0}"
emit_env OSL                   "${OSL:-0}"
emit_env MAX_MODEL_LEN         "${MAX_MODEL_LEN:-0}"
emit_env SPEC_DECODING         "${SPEC_DECODING:-none}"
emit_env DISAGG                "${DISAGG:-false}"
emit_env SCENARIO_TYPE         "${SCENARIO_TYPE:-}"
emit_env SCENARIO_SUBDIR       "${SCENARIO_SUBDIR:-}"
emit_env IS_AGENTIC            "${IS_AGENTIC:-0}"
emit_env KV_OFFLOADING         "${KV_OFFLOADING:-}"
emit_env KV_OFFLOAD_BACKEND    "${KV_OFFLOAD_BACKEND:-}"
emit_env KV_OFFLOAD_BACKEND_METADATA "$KV_OFFLOAD_BACKEND_METADATA_COMPACT"
emit_env ROUTER_METADATA       "$ROUTER_METADATA_COMPACT"
emit_env KV_P2P_TRANSFER       "${KV_P2P_TRANSFER:-}"
emit_env TOTAL_CPU_DRAM_GB     "${TOTAL_CPU_DRAM_GB:-0}"
emit_env DURATION              "${DURATION:-3600}"
emit_env RUN_EVAL              "${RUN_EVAL:-false}"
emit_env EVAL_ONLY             "${EVAL_ONLY:-false}"
emit_env EVAL_LIMIT            "${EVAL_LIMIT:-}"
emit_env SWEBENCH_GEN_MODE     "${SWEBENCH_GEN_MODE:-}"
emit_env SWEBENCH_USE_MODAL    "${SWEBENCH_USE_MODAL:-true}"
emit_env MODAL_TOKEN_ID        "${MODAL_TOKEN_ID:-}"
emit_env MODAL_TOKEN_SECRET    "${MODAL_TOKEN_SECRET:-}"
# AIPERF_EXPERIMENTAL_FAST is in benchmark-tmpl.yml's env block but is missing
# from launch_m1517docker.sh; forwarded here so this launcher is at full parity.
emit_env AIPERF_EXPERIMENTAL_FAST "${AIPERF_EXPERIMENTAL_FAST:-}"
emit_env AIPERF_FAILED_REQUEST_THRESHOLD "${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
emit_env AIPERF_DATASET_MMAP_CACHE_DIR /aiperf_mmap_cache
emit_env RESULT_DIR            "${RESULT_DIR:-/workspace/results}"
emit_env RESULT_FILENAME       "${RESULT_FILENAME:-}"
emit_env RUNNER_TYPE           "${RUNNER_TYPE:-}"
emit_env VLLM_CACHE_ROOT       /vllm_cache
emit_env VLLM_ALLREDUCE_USE_SYMM_MEM 0
emit_env PYTHONDONTWRITEBYTECODE "${PYTHONDONTWRITEBYTECODE:-1}"
emit_env PYTHONPYCACHEPREFIX   "${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
emit_env PYTHONHASHSEED        0
set -x

# ---------------------------------------------------------------------------
# Inner script: runs ON THE WORKER. Values are baked in as shell-quoted
# assignments (printf %q) ahead of a fully-quoted body, so nothing here relies
# on env inheritance -- which SPUR cannot provide.
# ---------------------------------------------------------------------------
{
    printf '#!/usr/bin/env bash\n'
    printf '# Generated by runners/launch_spuraim.sh -- runs on the SPUR worker.\n'
    printf 'IMAGE=%q\n'             "$IMAGE"
    printf 'CONTAINER=%q\n'         "$CONTAINER"
    printf 'ENV_FILE=%q\n'          "$ENV_FILE"
    printf 'HEARTBEAT_FILE=%q\n'    "$HEARTBEAT_FILE"
    printf 'HEARTBEAT_SECONDS=%q\n' "$SPUR_HEARTBEAT_SECONDS"
    printf 'WORKSPACE=%q\n'         "$GITHUB_WORKSPACE"
    printf 'BENCHMARK_SCRIPT=%q\n'  "$BENCHMARK_SCRIPT"
    printf 'SHARED_HF_ROOT=%q\n'    "$SHARED_HF_ROOT"
    printf 'NODE_SCRATCH=%q\n'      "$NODE_SCRATCH"
    printf 'CONTAINER_HF_HUB=%q\n'  "$CONTAINER_HF_HUB"
    printf 'HOST_UID=%q\n'          "$(id -u)"
    printf 'HOST_GID=%q\n'          "$(id -g)"
    cat <<'INNER_EOF'
set -uo pipefail
umask 077
set -x

write_worker_heartbeat() {
    local phase="$1"
    local temp="${HEARTBEAT_FILE}.tmp.${BASHPID}"
    printf '%s node=%s phase=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(hostname)" "$phase" > "$temp"
    mv -f "$temp" "$HEARTBEAT_FILE"
}

HEARTBEAT_PID=""
worker_heartbeat() {
    set +x
    local container_state
    while true; do
        if container_state="$(docker inspect --format '{{.State.Status}}' \
                "$CONTAINER" 2>/dev/null)"; then
            write_worker_heartbeat "container-${container_state}"
        else
            write_worker_heartbeat "worker-active-container-not-created"
        fi
        sleep "$HEARTBEAT_SECONDS"
    done
}

cleanup_worker() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "$HEARTBEAT_PID" ]]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    # The benchmark container runs as root against the bind-mounted checkout.
    # Repair ownership here (including cancellation), not only on normal return,
    # or the next actions/checkout clean can fail with EACCES.
    timeout 120s docker run --rm -v "$WORKSPACE":/workspace --entrypoint chown \
        "$IMAGE" -R "${HOST_UID}:${HOST_GID}" /workspace 2>/dev/null || \
        echo "[spuraim/worker] WARN: workspace chown failed; next checkout may need manual cleanup"
    rm -f "$HEARTBEAT_FILE" "${HEARTBEAT_FILE}.tmp."*
    exit "$rc"
}

handle_worker_signal() {
    case "$1" in
        INT) exit 130 ;;
        TERM) exit 143 ;;
    esac
}

trap cleanup_worker EXIT
trap 'handle_worker_signal INT' INT
trap 'handle_worker_signal TERM' TERM

write_worker_heartbeat worker-starting
echo "[spuraim/worker] node=$(hostname)"

# Docker health is NOT uniform across this cluster -- some nodes have a dead
# daemon even while sitting `idle`. Fail fast and legibly rather than dying
# later inside the recipe.
if ! timeout 30s docker info >/dev/null 2>&1; then
    write_worker_heartbeat docker-unavailable
    echo "[spuraim/worker] FATAL: docker daemon unreachable on $(hostname)." >&2
    echo "[spuraim/worker] Add this node to SPUR_EXCLUDE_NODES and retry." >&2
    exit 125
fi
write_worker_heartbeat docker-ready
worker_heartbeat &
HEARTBEAT_PID=$!

mkdir -p "$NODE_SCRATCH/hf_home" "$NODE_SCRATCH/aiperf-cache" "$NODE_SCRATCH/vllm-cache" \
         "$CONTAINER_HF_HUB"

# The scheduler already masked our GPU slice via ROCR_VISIBLE_DEVICES; forward
# it rather than computing a device list. We are non-exclusive, so this mask is
# what keeps us off a co-tenant's GPUs. Do NOT also set HIP_VISIBLE_DEVICES --
# it would be re-indexed against the already-masked set.
ROCR_ARG=()
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    ROCR_ARG=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES")
    echo "[spuraim/worker] ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
else
    echo "[spuraim/worker] WARN: scheduler set no ROCR_VISIBLE_DEVICES mask." >&2
fi

# Resolve video/render as NUMERIC gids from the host. `--group-add render` by
# name requires the group to exist inside the image and hard-fails the run if
# it does not ("unable to find group render: no matching entries in group
# file") -- true of plain base images, and not something a launcher should
# depend on. Numeric gids are always accepted.
GROUP_ARG=()
for g in video render; do
    gid="$(getent group "$g" | cut -d: -f3)"
    if [[ -n "$gid" ]]; then
        GROUP_ARG+=(--group-add "$gid")
    else
        echo "[spuraim/worker] WARN: host has no '$g' group; GPU access may fail." >&2
    fi
done

# Two HF mounts, and they are not interchangeable:
#   - the shared hub, :ro, is where MODEL_PATH and the drafter live. Read-only
#     so a stray write fails at the mount boundary instead of half-succeeding.
#   - node-local scratch, writable, is HF_HUB_CACHE: where the trace dataset
#     and anything else not pre-staged gets downloaded.
HF_MOUNT=(-v "$SHARED_HF_ROOT:$SHARED_HF_ROOT:ro"
          -v "$CONTAINER_HF_HUB:$CONTAINER_HF_HUB")

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# No shared image cache on this cluster (no enroot squashfs equivalent), so a
# fresh node pays a full pull.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull "$IMAGE" || { echo "[spuraim/worker] docker pull failed for $IMAGE" >&2; exit 1; }
fi

# --shm-size=0 with --ipc=host gives the container the host's /dev/shm (1.4 T),
# which is what the LMCache arms size their L1 pool from.
docker run --rm --name "$CONTAINER" \
    --network host \
    --device /dev/kfd --device /dev/dri \
    --ipc=host --shm-size=0 \
    "${GROUP_ARG[@]}" \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    "${ROCR_ARG[@]}" \
    --env-file "$ENV_FILE" \
    -v "$WORKSPACE":/workspace \
    "${HF_MOUNT[@]}" \
    -v "$NODE_SCRATCH/hf_home":"$NODE_SCRATCH/hf_home" \
    -v "$NODE_SCRATCH/aiperf-cache":/aiperf_mmap_cache \
    -v "$NODE_SCRATCH/vllm-cache":/vllm_cache \
    -w /workspace \
    --entrypoint bash \
    "$IMAGE" \
    "$BENCHMARK_SCRIPT"
RC=$?

echo "[spuraim/worker] recipe exit=$RC"
exit $RC
INNER_EOF
} > "$INNER"
chmod +x "$INNER"

EXCLUDE_ARG=()
if [[ -n "$SPUR_EXCLUDE_NODES" ]]; then
    EXCLUDE_ARG=(-x "$SPUR_EXCLUDE_NODES")
    echo "[spuraim] excluding nodes: $SPUR_EXCLUDE_NODES"
fi
EXCLUSIVE_ARG=()
[[ "$SPUR_EXCLUSIVE" == "1" ]] && EXCLUSIVE_ARG=(--exclusive)

# SPUR buffers worker output, which previously made a healthy two-hour run look
# hung in the Actions UI. Poll scheduler state on the login node and combine it
# with an atomic heartbeat written by the worker over shared NFS. Do not cancel
# and resubmit PENDING jobs: AIFW-AIM queue age must be preserved.
monitor_spur_job() {
    set +x
    local started_at=$SECONDS
    local scheduler_status
    local worker_status
    while true; do
        scheduler_status="$(squeue -h -n "$JOB_NAME" \
            -o '%.18i|%.12T|%.12M|%.80R' 2>/dev/null | head -n1)"
        [[ -n "$scheduler_status" ]] || scheduler_status="not-yet-visible"
        if [[ -r "$HEARTBEAT_FILE" ]]; then
            worker_status="$(<"$HEARTBEAT_FILE")"
        else
            worker_status="not-started"
        fi
        echo "[spuraim/heartbeat] wall=$((SECONDS - started_at))s" \
             "scheduler=[$scheduler_status] worker=[$worker_status]"
        sleep "$SPUR_HEARTBEAT_SECONDS"
    done
}

echo "[spuraim] submitting $JOB_NAME; queued jobs retain their queue age"
monitor_spur_job &
MONITOR_PID=$!
SRUN_ACTIVE=1
srun -A "$SPUR_ACCOUNT" --qos="$SPUR_QOS" -p "$SPUR_PARTITION" \
    -N1 --gres="gpu:$GPU_COUNT" -c "$SPUR_CPUS_PER_TASK" \
    -t "$SPUR_TIME_LIMIT" -J "$JOB_NAME" \
    "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \
    bash "$INNER"
RC=$?
SRUN_ACTIVE=0
stop_monitor
echo "[spuraim] job exit=$RC"
exit $RC
