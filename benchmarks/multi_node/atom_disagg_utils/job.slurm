#!/bin/bash
#SBATCH --job-name=1p2d_atom-disagg-serving
#SBATCH -N 3
#SBATCH -n 3
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
# --output and --error are set by submit.sh via BENCHMARK_LOGS_DIR
# NOTE: -N / -n above are sentinel values for direct sbatch invocation; the
# actual node count is overridden by submit.sh based on PREFILL_NODES +
# DECODE_NODES (canonical topology for dsv4-fp4-mi355x-atom-disagg is 1P2D,
# i.e. 3 nodes).

# Multi-node ATOM disaggregated benchmark SLURM job script.
#
# Structurally identical to benchmarks/multi_node/amd_utils/job.slurm but
# adapted for the atom-disagg track:
#   - SGLANG_WS_PATH -> ATOM_WS_PATH (points at this directory inside Docker)
#   - models.yaml lives in atom_disagg_utils/ and uses a simpler schema
#     (atom has fewer server knobs than SGLang)
#   - server.sh launches atom.entrypoints.openai_server in prefill/decode
#     roles instead of sglang.launch_server
# Per-node srun + Docker invocation is unchanged from amd_utils/job.slurm.

echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# =============================================================================
# Model validation from models.yaml
# =============================================================================
MODELS_YAML="$(pwd)/models.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "Error: models.yaml not found at $MODELS_YAML"
    exit 1
fi

if ! grep -q "^${MODEL_NAME}:" "$MODELS_YAML"; then
    echo "Error: Model '$MODEL_NAME' not found in models.yaml"
    echo "Available models:"
    grep -E '^[A-Za-z]' "$MODELS_YAML" | sed 's/:.*$//' | sed 's/^/  - /'
    exit 1
fi
echo "Model found: $MODEL_NAME"

RUN_FILE="server.sh"
echo "Runfile set: $RUN_FILE"

if [[ -z "${DOCKER_IMAGE_NAME:-}" ]]; then
    echo "Error: DOCKER_IMAGE_NAME is not set."
    exit 1
fi

# Repo root: $(pwd) is atom_disagg_utils/, go up 3 levels.
export DI_REPO_DIR=$(cd "$(pwd)/../../.." && pwd)

xP="${xP:-1}"
yD="${yD:-1}"

PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-false}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-false}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-false}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-false}"

BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-0.8}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-1}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MODEL_NAME="${MODEL_NAME:-None}"

if [[ -z "$MODEL_DIR" ]]; then
    NODENAME=$(hostname -s)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        MODEL_DIR="/nfsdata"
    elif [[ $NODENAME == mia1* ]]; then
        MODEL_DIR="/it-share/data"
    else
        MODEL_DIR="/nfsdata"
    fi
    echo "[INFO] Auto-detected MODEL_DIR=$MODEL_DIR from hostname $NODENAME"
fi
export MODEL_DIR

echo "Looking for model: $MODEL_NAME"
echo "Checking model availability across all allocated nodes..."

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)

echo "Total allocated nodes: $TOTAL_NODES"
echo "Nodes: $(echo "$ALL_NODES" | tr '\n' ' ')"

check_model_path() {
    local path=$1
    local check_name=$2
    echo "Checking $check_name: $path"
    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES /bin/bash -c "
        if [ -d '$path' ]; then
            echo \"\$(hostname): OK Found $path\"
            exit 0
        else
            echo \"\$(hostname): X Missing $path\"
            exit 1
        fi
    "
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "OK $check_name available on ALL nodes"
        return 0
    else
        echo "X $check_name NOT available on all nodes"
        return 1
    fi
}

if check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    echo "Selected MODEL_PATH: $MODEL_PATH"
else
    echo "FATAL: Model '$MODEL_NAME' not found at $MODEL_DIR/$MODEL_NAME on all nodes" >&2
    exit 1
fi
echo "Final MODEL_PATH: $MODEL_PATH"
echo ""

NUM_NODES="${NUM_NODES}"

# Extract first NUM_NODES from allocation and refresh SLURM env.
FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n $NUM_NODES)
NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd, | sed 's/,/,/g')

export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$HOME}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-1p2d_atom-disagg-serving}"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "Selected nodes for execution:"
echo "$SELECTED_NODES"
echo ""

USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
NODE0_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'ip route get 1.1.1.1')
NODE0_ADDR=$(echo "$NODE0_ADDR" | awk '/src/ {print $7}')

IPS=()
for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'ip route get 1.1.1.1')
    IP=$(echo "$IP" | awk '/src/ {print $7}')
    IPS+=("$IP")
done
echo "Selected node IPs: ${IPS[*]}" | sed 's/ /,/g'

DOCKER_MOUNT_PATH="/workspace"
ATOM_WS_PATH="${DOCKER_MOUNT_PATH}/benchmarks/multi_node/atom_disagg_utils"

NNODES=$NUM_NODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "NODE0_ADDR is ${NODE0_ADDR}"
echo "NNODES is ${NNODES}"
echo "REPO Directory is ${DI_REPO_DIR}"
echo "USER_NAME is ${USER_NAME}"

# nicctl is used by env.sh for QoS detection inside the container, mount it
# from host where available (consistent with amd_utils/job.slurm).
NICCTL_HOST_PATH=""
if command -v nicctl >/dev/null 2>&1; then
    NICCTL_HOST_PATH=$(which nicctl)
fi

# Bug 7 workaround: rocm/atom-dev's libibverbs install ships every standard
# vendor provider (bnxt/mlx5/...) EXCEPT the Pensando ionic provider used by
# the mia1 cluster. The host has libionic + the ionic.driver registration
# file. Bind-mount them in so libibverbs can enumerate rdma0..rdma7 and
# mori::io::RdmaManager doesn't trip its `availDevices.size() > 0` assert.
# Build IONIC_MOUNT_OPT as a flat -v string so it can be interpolated into
# the docker run heredoc below. Harmless on hosts without ionic (e.g.
# mlx5 clusters): IONIC_MOUNT_OPT stays empty.
IONIC_MOUNT_OPT=""
_ionic_paths=(
    /etc/libibverbs.d/ionic.driver
    /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
    /usr/lib/x86_64-linux-gnu/libionic.so
    /usr/lib/x86_64-linux-gnu/libionic.so.1
)
# Glob in the versioned .so (e.g. libionic.so.1.0.54.0-149.g3304be71) since
# the suffix is host-specific. The bind-mount needs to use the literal name
# so the symlink chain resolves inside the container.
shopt -s nullglob
for f in /usr/lib/x86_64-linux-gnu/libionic.so.1.*; do
    _ionic_paths+=("$f")
done
shopt -u nullglob
for f in "${_ionic_paths[@]}"; do
    if [[ -e "$f" ]]; then
        IONIC_MOUNT_OPT+=" -v $f:$f:ro"
    fi
done
echo "[ionic] IONIC_MOUNT_OPT=${IONIC_MOUNT_OPT}"

# Patches resolution (PR #812 overlay, V4 model_type config) is deferred
# until after SELECTED_NODELIST_SRUN is computed so we can use it to stage
# the patches to every allocated node's local /tmp first. See the "patches"
# block below this one.

export TQDM_MININTERVAL=20
export DI_REPO_DIR=$DI_REPO_DIR
export ATOM_WS_PATH=$ATOM_WS_PATH
export NNODES=$NNODES
export NODE0_ADDR=$NODE0_ADDR
export MODEL_PATH=$MODEL_PATH
export MODEL_DIR=$MODEL_DIR
export xP=$xP
export yD=$yD
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export PREFILL_TP_SIZE=$PREFILL_TP_SIZE
export PREFILL_ENABLE_EP=$PREFILL_ENABLE_EP
export PREFILL_ENABLE_DP=$PREFILL_ENABLE_DP
export DECODE_TP_SIZE=$DECODE_TP_SIZE
export DECODE_ENABLE_EP=$DECODE_ENABLE_EP
export DECODE_ENABLE_DP=$DECODE_ENABLE_DP
export GPUS_PER_NODE=$GPUS_PER_NODE
export BENCH_INPUT_LEN=$BENCH_INPUT_LEN
export BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN
export BENCH_RANDOM_RANGE_RATIO=$BENCH_RANDOM_RANGE_RATIO
export BENCH_NUM_PROMPTS_MULTIPLIER=$BENCH_NUM_PROMPTS_MULTIPLIER
export BENCH_MAX_CONCURRENCY=$BENCH_MAX_CONCURRENCY
export DRY_RUN="${DRY_RUN:-0}"
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$(pwd)/benchmark_logs}"

export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_CONC="${EVAL_CONC:-}"
export FRAMEWORK="${FRAMEWORK:-atom-disagg}"
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"
export SPEC_DECODING="${SPEC_DECODING:-}"
export IS_MULTINODE="${IS_MULTINODE:-true}"

SANITIZED_USER=$(echo "$USER_NAME" | tr -c 'a-zA-Z0-9_.-' '_')
export DOCKER_CONT_NAME="container_sbatch_atom_${SANITIZED_USER}_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$ATOM_WS_PATH/${RUN_FILE}"

SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

# --- Stage the in-repo patches to a node-local /tmp directory ----------------
# Both Bug 8 (PR #812 overlay) and Bug 9 (V4 model_type config.json patch)
# need files that are bind-mounted into the docker container later. The
# canonical source of truth lives in-repo under
# `benchmarks/multi_node/atom_disagg_utils/_patches/`; bind-mounting straight
# from NFS works in principle but tickles a couple of subtle issues
# (root-squashed read of an NFS-only path, atime/mtime drift during long
# runs). So we copy the patches once per node to a node-local /tmp dir;
# this keeps everything in one cleanable place and is the only thing that
# previously needed a manual `rsync` step.
#
# Source is configurable via PATCHES_REPO_DIR for testing; default is the
# canonical in-repo location. PATCHES_STAGE_DIR is the host-side destination
# (must be on a real local fs, not NFS).
PATCHES_REPO_DIR="${PATCHES_REPO_DIR:-${DI_REPO_DIR}/benchmarks/multi_node/atom_disagg_utils/_patches}"
PATCHES_STAGE_DIR="${PATCHES_STAGE_DIR:-/tmp/atom_disagg_patches_${SLURM_JOB_ID:-local}}"
if [[ -d "$PATCHES_REPO_DIR" ]]; then
    echo "[patches] staging from ${PATCHES_REPO_DIR} -> ${PATCHES_STAGE_DIR} on every allocated node"
    srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c "
        mkdir -p '${PATCHES_STAGE_DIR}'
        cp -a '${PATCHES_REPO_DIR}/pr812-overlay'  '${PATCHES_STAGE_DIR}/' 2>/dev/null || true
        cp -a '${PATCHES_REPO_DIR}/v4_model_patch' '${PATCHES_STAGE_DIR}/' 2>/dev/null || true
        echo \"[patches] \$(hostname): \$(find ${PATCHES_STAGE_DIR} -type f 2>/dev/null | wc -l) files staged\"
    "
else
    echo "[patches] WARNING: in-repo patch dir not found at ${PATCHES_REPO_DIR} -- falling back to /tmp/pr812-overlay + /tmp/v4_model_patch if they exist"
fi

# Bug 8 workaround: ROCm/ATOM PR #812 ("Support Deepseek v4 pd disaggregation")
# is open at the time of writing; rocm/atom-dev:nightly_202605171131-mooncake_build
# (and earlier nightlies) does not contain it. Bind-mount the PR #812
# overlay (10 patched .py files) over the image's /app/ATOM/atom/... tree
# so atom learns to register the V4 compressor-state KV tensors via the new
# get_kv_transfer_tensors() hook. The overlay is loaded read-only.
# Source resolution order:
#   1. ${PATCHES_STAGE_DIR}/pr812-overlay/applied/atom   (staged from repo above)
#   2. /tmp/pr812-overlay/atom                            (legacy manual rsync)
#   3. ${PR812_OVERLAY_HOST_ROOT}                         (explicit override)
# Harmless on hosts where none of the above exist (PR812_MOUNT_OPT empty).
if [[ -z "${PR812_OVERLAY_HOST_ROOT:-}" ]]; then
    if [[ -d "${PATCHES_STAGE_DIR}/pr812-overlay/applied/atom" ]]; then
        PR812_OVERLAY_HOST_ROOT="${PATCHES_STAGE_DIR}/pr812-overlay/applied/atom"
    elif [[ -d "/tmp/pr812-overlay/atom" ]]; then
        PR812_OVERLAY_HOST_ROOT="/tmp/pr812-overlay/atom"
    fi
fi
PR812_MOUNT_OPT=""
if [[ -n "${PR812_OVERLAY_HOST_ROOT:-}" && -d "$PR812_OVERLAY_HOST_ROOT" ]]; then
    while IFS= read -r -d '' f; do
        rel="${f#${PR812_OVERLAY_HOST_ROOT}/}"
        PR812_MOUNT_OPT+=" -v $f:/app/ATOM/atom/${rel}:ro"
    done < <(find "$PR812_OVERLAY_HOST_ROOT" -type f -name '*.py' -print0)
fi
echo "[pr812] source=${PR812_OVERLAY_HOST_ROOT:-<none>}  mount_count=$(grep -o ' -v ' <<<"$PR812_MOUNT_OPT" | wc -l)"

# Bug 9 workaround: deepseek-ai/DeepSeek-V4-Pro's config.json declares
# `"model_type": "deepseek_v3"` even though its `architectures` entry is
# `DeepseekV4ForCausalLM` and the weights/structure are V4. atom's
# `ModelRunner.is_deepseek_v4()` checks `model_type == "deepseek_v4"`,
# so with the unpatched config V4 mode is never enabled -- the selector
# falls back to `AiterMLABackend` and `AiterMLAMetadataBuilder` (no
# `compress_plans`), while the model code (`atom/models/deepseek_v4.py`)
# always reads `attn_md.compress_plans` and crashes.
# Bind-mount a patched config.json (same content; only model_type flipped
# to "deepseek_v4") over the HF cache snapshot path so V4 mode lights up.
# Source resolution order matches the overlay above.
if [[ -z "${V4_CFG_HOST:-}" ]]; then
    if [[ -f "${PATCHES_STAGE_DIR}/v4_model_patch/config.json" ]]; then
        V4_CFG_HOST="${PATCHES_STAGE_DIR}/v4_model_patch/config.json"
    elif [[ -f "/tmp/v4_model_patch/config.json" ]]; then
        V4_CFG_HOST="/tmp/v4_model_patch/config.json"
    fi
fi
V4_CFG_CONTAINER_TARGET=""
if [[ -n "${V4_CFG_HOST:-}" && -f "$V4_CFG_HOST" ]]; then
    # The container sees the model at /models/${MODEL_NAME} via the host
    # `${MODEL_DIR}:/models` mount + the host symlink
    # ${MODEL_DIR}/${MODEL_NAME} -> models--deepseek-ai--${MODEL_NAME}/snapshots/<sha>/.
    # docker bind-mount resolves through symlinks at the host side, so we
    # need to target the symlink-resolved path that lives inside ${MODEL_DIR}.
    _resolved=$(readlink -f "${MODEL_DIR}/${MODEL_NAME}/config.json" 2>/dev/null || true)
    if [[ -n "$_resolved" && "$_resolved" == "${MODEL_DIR}"/* ]]; then
        _rel="${_resolved#${MODEL_DIR}/}"
        V4_CFG_CONTAINER_TARGET="/models/${_rel}"
    fi
fi
V4_CFG_MOUNT_OPT=""
if [[ -n "$V4_CFG_CONTAINER_TARGET" ]]; then
    V4_CFG_MOUNT_OPT="-v ${V4_CFG_HOST}:${V4_CFG_CONTAINER_TARGET}:ro"
fi
echo "[v4-cfg] source=${V4_CFG_HOST:-<none>}  target=${V4_CFG_CONTAINER_TARGET:-<none>}"

cleanup() {
  echo "[${SLURM_JOB_ID}] termination received on $(hostname); cleaning stale logs folder..."
  sudo rm -rf ${SLURM_SUBMIT_DIR}/logs 2>/dev/null || true
  echo "[${SLURM_JOB_ID}] cleanup done."
}
trap cleanup INT TERM HUP

# Force NFS cache refresh on all nodes before running Docker.
echo "Refreshing NFS caches on all nodes..."
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
    sync
    ls -la '"$DI_REPO_DIR"'/benchmarks/multi_node/atom_disagg_utils > /dev/null 2>&1
    stat '"$DI_REPO_DIR"'/benchmarks/multi_node/atom_disagg_utils/server.sh > /dev/null 2>&1
    cat '"$DI_REPO_DIR"'/benchmarks/multi_node/atom_disagg_utils/server.sh > /dev/null 2>&1
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    echo "NFS cache refreshed on $(hostname)"
'

NICCTL_MOUNT_OPT=""
if [[ -n "$NICCTL_HOST_PATH" ]]; then
    NICCTL_MOUNT_OPT="-v $NICCTL_HOST_PATH:/usr/sbin/nicctl"
fi

srun \
  --nodelist="$SELECTED_NODELIST_SRUN" \
  --kill-on-bad-exit=1 \
  --signal=TERM@30 \
  --unbuffered \
  bash -lc "
set -euo pipefail

echo \"Rank \$SLURM_PROCID on \$(hostname)\"

# Auto-detect docker invocation. SLURM-launched bash has no TTY, so 'sudo
# docker' would require NOPASSWD sudo on every compute node — which is
# rarely the case. Additionally, SLURM does not load the user's
# supplementary groups via PAM, so even users listed in /etc/group:docker
# do not get GID 3000 in their process credentials and a plain 'docker'
# call fails with 'permission denied' on /var/run/docker.sock. Workaround:
# detect docker-group membership via getent (which reads /etc/group
# directly) and acquire GID 3000 via 'sg docker -c <cmd>'. Mirrors the
# dsv4 sglang-disagg Bug 3 workaround documented in
# scripts/dsv4/dsv4_disagg_enablement/SUMMARY.md.
DOCKER_USER=\$(whoami)
if id -nG 2>/dev/null | grep -qw docker; then
    DOCKER_WRAP=''           # docker GID already in supplementary groups
elif getent group docker 2>/dev/null | awk -F: '{print \$4}' | tr ',' '\n' | grep -qx \"\$DOCKER_USER\"; then
    DOCKER_WRAP='sg-docker'  # need 'sg docker -c' to acquire GID 3000
elif sudo -n true 2>/dev/null; then
    DOCKER_WRAP='sudo-n'     # fall back to passwordless sudo
else
    echo \"ERROR: user \$DOCKER_USER cannot reach docker daemon on \$(hostname)\" >&2
    echo \"       (not in docker group via /etc/group, and sudo requires a password).\" >&2
    exit 1
fi
echo \"[docker] wrap=\$DOCKER_WRAP user=\$DOCKER_USER host=\$(hostname)\"

# Write the docker invocation to a per-rank file in the per-job artifact
# directory so we can dispatch it through 'sg docker -c \"bash <script>\"'
# (which only accepts a single command string) without messy nested
# quoting. Putting it under /tmp/slurm_job-\${SLURM_JOB_ID}/ instead of
# directly in /tmp keeps it grouped with the other run artifacts
# (server_*.log, prefill_*.log, decode_*.log, atom_isl_*_osl_*/*.json)
# and lets it be cleaned up with the rest of the job's tmpfiles instead
# of accumulating one orphan-script-per-job in the top of /tmp.
DOCKER_RUN_DIR=/tmp/slurm_job-\${SLURM_JOB_ID}
mkdir -p \"\$DOCKER_RUN_DIR\"
DOCKER_RUN_SCRIPT=\$DOCKER_RUN_DIR/docker_run_\${SLURM_PROCID}.sh
cat > \"\$DOCKER_RUN_SCRIPT\" <<DOCKER_SH
#!/bin/bash
set -euo pipefail

# Pre-clean (idempotent) — same docker daemon as the upcoming docker run.
docker ps -aq --filter \"name=^container_sbatch_atom_\" | xargs -r docker rm -f >/dev/null 2>&1 || true
docker ps -aq | xargs -r docker stop >/dev/null 2>&1 || true

exec docker run --rm \\
    --init \
    --stop-timeout 10 \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --device=/dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --device=/dev/infiniband/uverbs4 \
    --device=/dev/infiniband/uverbs5 \
    --device=/dev/infiniband/uverbs6 \
    --device=/dev/infiniband/uverbs7 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v ${MODEL_DIR}:/models \
    -v \$HOME/.ssh:/root/.ssh \
    ${NICCTL_MOUNT_OPT} \
    ${IONIC_MOUNT_OPT} \
    ${PR812_MOUNT_OPT} \
    ${V4_CFG_MOUNT_OPT} \
    --shm-size 128G \
    -v /tmp:/run_logs \
    -v ${BENCHMARK_LOGS_DIR}:/benchmark_logs \
    -v ${DI_REPO_DIR}:${DOCKER_MOUNT_PATH} \
    -e SLURM_JOB_ID=\$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=\$SLURM_JOB_NODELIST \
    -e NNODES=\$NNODES \
    -e NODE_RANK=\$SLURM_PROCID \
    -e NODE0_ADDR=\$NODE0_ADDR \
    -e MODEL_DIR=/models \
    -e ATOM_WS_PATH=${ATOM_WS_PATH} \
    -e GPUS_PER_NODE=\$GPUS_PER_NODE \
    -e xP=\$xP \
    -e yD=\$yD \
    -e MODEL_NAME=\$MODEL_NAME \
    -e IPADDRS=\$IPADDRS \
    -e PREFILL_TP_SIZE=\$PREFILL_TP_SIZE \
    -e PREFILL_ENABLE_EP=\$PREFILL_ENABLE_EP \
    -e PREFILL_ENABLE_DP=\$PREFILL_ENABLE_DP \
    -e DECODE_TP_SIZE=\$DECODE_TP_SIZE \
    -e DECODE_ENABLE_EP=\$DECODE_ENABLE_EP \
    -e DECODE_ENABLE_DP=\$DECODE_ENABLE_DP \
    -e BENCH_INPUT_LEN=\$BENCH_INPUT_LEN \
    -e BENCH_OUTPUT_LEN=\$BENCH_OUTPUT_LEN \
    -e BENCH_RANDOM_RANGE_RATIO=\$BENCH_RANDOM_RANGE_RATIO \
    -e BENCH_NUM_PROMPTS_MULTIPLIER=\$BENCH_NUM_PROMPTS_MULTIPLIER \
    -e BENCH_MAX_CONCURRENCY=\$BENCH_MAX_CONCURRENCY \
    -e TQDM_MININTERVAL=\$TQDM_MININTERVAL \
    -e DRY_RUN=\$DRY_RUN \
    -e BENCHMARK_LOGS_DIR=/benchmark_logs \
    -e RUN_EVAL=\$RUN_EVAL \
    -e EVAL_ONLY=\$EVAL_ONLY \
    -e EVAL_CONC=\$EVAL_CONC \
    -e FRAMEWORK=\$FRAMEWORK \
    -e PRECISION=\$PRECISION \
    -e MODEL_PREFIX=\$MODEL_PREFIX \
    -e RUNNER_TYPE=\$RUNNER_TYPE \
    -e RESULT_FILENAME=\"\$RESULT_FILENAME\" \\
    -e SPEC_DECODING=\"\$SPEC_DECODING\" \\
    -e IS_MULTINODE=\"\$IS_MULTINODE\" \\
    --name \"$DOCKER_CONT_NAME\" \\
    \"$DOCKER_IMAGE_NAME\" bash -lc 'set -o pipefail; mkdir -p /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'; '\"$RUN_FILE_FULL\"' 2>&1 | tee /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'/server_'\"\$(hostname)\"'.log'
DOCKER_SH
chmod +x \"\$DOCKER_RUN_SCRIPT\"

# Dispatch the docker run script via the detected wrapper. 'exec' replaces
# the current shell so the docker run's exit code flows back to srun.
case \"\$DOCKER_WRAP\" in
    '')          exec bash \"\$DOCKER_RUN_SCRIPT\" ;;
    'sg-docker') exec sg docker -c \"bash \$DOCKER_RUN_SCRIPT\" ;;
    'sudo-n')    exec sudo -n bash \"\$DOCKER_RUN_SCRIPT\" ;;
    *) echo \"ERROR: unknown DOCKER_WRAP=\$DOCKER_WRAP\" >&2; exit 1 ;;
esac
"

srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
if id -nG 2>/dev/null | grep -qw docker; then
    DOCKER_CMD=docker
elif sudo -n true 2>/dev/null; then
    DOCKER_CMD="sudo -n docker"
else
    DOCKER_CMD=docker  # last resort; cleanup is best-effort anyway
fi
$DOCKER_CMD rm -f '"$DOCKER_CONT_NAME"' 2>/dev/null || true'
