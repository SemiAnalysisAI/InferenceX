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
set -x

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

# NOT --exclusive. Only ~8 of 233 nodes are fully idle at any time (86 alloc,
# 66 mix, 64 resv), so demanding whole nodes means queueing behind the cluster
# instead of running. Dropping it lets us land on any `mix` node with enough
# free GPUs; the scheduler's ROCR_VISIBLE_DEVICES mask (forwarded below) keeps
# us to our own slice. The tradeoff is co-tenant noise -- this lane is a
# functional/bring-up harness, not a source of comparable perf numbers.
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

# Resolve MODEL -> the read-only snapshot dir, if that repo is staged.
# org/name -> models--org--name
MODEL_REPO_DIR="$SHARED_HF_HUB/models--${MODEL//\//--}"
RESOLVED_MODEL_PATH=""
if [[ -d "$MODEL_REPO_DIR/snapshots" ]]; then
    # Prefer the ref the hub points at; fall back to the sole snapshot.
    _ref_file="$MODEL_REPO_DIR/refs/main"
    if [[ -f "$_ref_file" ]]; then
        _rev="$(<"$_ref_file")"
        [[ -d "$MODEL_REPO_DIR/snapshots/$_rev" ]] && RESOLVED_MODEL_PATH="$MODEL_REPO_DIR/snapshots/$_rev"
    fi
    if [[ -z "$RESOLVED_MODEL_PATH" ]]; then
        _only="$(find "$MODEL_REPO_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1)"
        [[ -n "$_only" ]] && RESOLVED_MODEL_PATH="$_only"
    fi
fi

if [[ -n "$RESOLVED_MODEL_PATH" ]]; then
    # Shared read-only hub has the model. Run HF offline against it; the recipe
    # skips its `hf download` because MODEL_PATH is a non-empty dir.
    HF_MODE="shared-ro"
    CONTAINER_HF_HUB="$SHARED_HF_HUB"
    HF_OFFLINE_VAL=1
    echo "[spuraim] model staged on shared NFS: $RESOLVED_MODEL_PATH"
else
    # Not staged. Fall back to a writable node-local hub and let HF download.
    # Node-local means a fresh node re-downloads -- fine for small models, and
    # the explicit reason large ones should be pre-staged on /shared_nfs.
    HF_MODE="node-local-rw"
    CONTAINER_HF_HUB="$NODE_SCRATCH/hf_hub"
    HF_OFFLINE_VAL=0
    echo "[spuraim] WARNING: $MODEL is NOT on $SHARED_HF_HUB; falling back to a" \
         "writable node-local cache at $CONTAINER_HF_HUB (will download)."
fi

JOB_NAME="ix-${RUNNER_NAME}-${EXP_NAME:-job}"
JOB_NAME="${JOB_NAME:0:60}"
CONTAINER="spuraim_${RUNNER_NAME}_$$"

STAGE_DIR="$GITHUB_WORKSPACE/.spuraim"
mkdir -p "$STAGE_DIR"
ENV_FILE="$STAGE_DIR/env.$$.list"
INNER="$STAGE_DIR/inner.$$.sh"

# ---------------------------------------------------------------------------
# docker --env-file: strict KEY=VALUE, one per line, value taken LITERALLY to
# end of line (no quote or backslash processing). That is exactly what we want
# -- it sidesteps the quoting hell of pushing ~50 values through
# login-shell -> srun -> bash -> docker.
# ---------------------------------------------------------------------------
emit_env() { printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"; }

: > "$ENV_FILE"
emit_env HF_HUB_CACHE          "$CONTAINER_HF_HUB"
emit_env HF_HOME               "$NODE_SCRATCH/hf_home"
emit_env HF_HUB_OFFLINE        "$HF_OFFLINE_VAL"
emit_env HF_TOKEN              "${HF_TOKEN:-}"
[[ -n "$RESOLVED_MODEL_PATH" ]] && emit_env MODEL_PATH "$RESOLVED_MODEL_PATH"
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
emit_env KV_OFFLOAD_BACKEND_METADATA "${KV_OFFLOAD_BACKEND_METADATA:-}"
emit_env ROUTER_METADATA       "${ROUTER_METADATA:-}"
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
    printf 'WORKSPACE=%q\n'         "$GITHUB_WORKSPACE"
    printf 'BENCHMARK_SCRIPT=%q\n'  "$BENCHMARK_SCRIPT"
    printf 'SHARED_HF_ROOT=%q\n'    "$SHARED_HF_ROOT"
    printf 'NODE_SCRATCH=%q\n'      "$NODE_SCRATCH"
    printf 'CONTAINER_HF_HUB=%q\n'  "$CONTAINER_HF_HUB"
    printf 'HF_MODE=%q\n'           "$HF_MODE"
    printf 'HOST_UID=%q\n'          "$(id -u)"
    printf 'HOST_GID=%q\n'          "$(id -g)"
    cat <<'INNER_EOF'
set -uo pipefail
set -x

echo "[spuraim/worker] node=$(hostname) hf_mode=$HF_MODE"

# Docker health is NOT uniform across this cluster -- some nodes have a dead
# daemon even while sitting `idle`. Fail fast and legibly rather than dying
# later inside the recipe.
if ! docker info >/dev/null 2>&1; then
    echo "[spuraim/worker] FATAL: docker daemon unreachable on $(hostname)." >&2
    echo "[spuraim/worker] Add this node to SPUR_EXCLUDE_NODES and retry." >&2
    exit 125
fi

mkdir -p "$NODE_SCRATCH/hf_home" "$NODE_SCRATCH/aiperf-cache" "$NODE_SCRATCH/vllm-cache"
[[ "$HF_MODE" == "node-local-rw" ]] && mkdir -p "$CONTAINER_HF_HUB"

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

# Shared hub is read-only; mount it :ro so a stray write fails loudly at the
# mount boundary instead of half-succeeding.
HF_MOUNT=(-v "$SHARED_HF_ROOT:$SHARED_HF_ROOT:ro")
if [[ "$HF_MODE" == "node-local-rw" ]]; then
    HF_MOUNT=(-v "$CONTAINER_HF_HUB:$CONTAINER_HF_HUB")
fi

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
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
    --group-add video --group-add render \
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

# The recipe runs as root in the container and writes results/ into the
# bind-mounted checkout. docker gives us no --container-remap-root, so the next
# job's actions/checkout `clean: true` (non-root) would hit EACCES. Chown back.
docker run --rm -v "$WORKSPACE":/workspace --entrypoint chown \
    "$IMAGE" -R "${HOST_UID}:${HOST_GID}" /workspace 2>/dev/null || \
    echo "[spuraim/worker] WARN: workspace chown failed; next checkout may need manual cleanup"

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

srun -A "$SPUR_ACCOUNT" --qos="$SPUR_QOS" -p "$SPUR_PARTITION" \
    -N1 --gres="gpu:$GPU_COUNT" -c "$SPUR_CPUS_PER_TASK" \
    -t "$SPUR_TIME_LIMIT" -J "$JOB_NAME" \
    "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \
    bash "$INNER"
RC=$?

rm -f "$ENV_FILE" "$INNER"
echo "[spuraim] job exit=$RC"
exit $RC
