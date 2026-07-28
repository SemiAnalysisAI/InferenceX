#!/usr/bin/env bash
set -eo pipefail

export HF_HUB_CACHE_MOUNT="/raid/hf-hub-cache/"
export PORT=8888

PARTITION="compute"
SQUASH_FILE="/home/gharunner/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

# Route spec-decoding=mtp configs to the _mtp benchmark script (parity with
# the h200 launchers, which have carried SPEC_SUFFIX since #392).
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

set -x

# Temporary diagnostic branch only: import the K3 image to node-local /raid
# and verify its ROCm userspace on one GPU without loading model weights.
if [[ "${PROFILE:-0}" == "1" ]]; then
    DIAG_ALLOC_LOG=$(mktemp)
    DIAG_JOB_ID=""

    cleanup_diag() {
        local rc=$?
        trap - EXIT INT TERM
        if [[ -n "$DIAG_JOB_ID" ]]; then
            scancel "$DIAG_JOB_ID" 2>/dev/null || true
            for _ in $(seq 1 30); do
                if ! squeue -j "$DIAG_JOB_ID" --noheader 2>/dev/null | grep -q "$DIAG_JOB_ID"; then
                    break
                fi
                sleep 1
            done
        fi
        rm -f "$DIAG_ALLOC_LOG"
        exit "$rc"
    }
    trap cleanup_diag EXIT INT TERM

    set +e
    timeout 120s salloc \
        --partition="$PARTITION" \
        --exclude=chi-mi300x-049,chi-mi300x-121 \
        --nodes=1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --time=30 \
        --no-shell \
        --job-name="$RUNNER_NAME" 2>&1 | tee "$DIAG_ALLOC_LOG"
    alloc_rc=${PIPESTATUS[0]}
    set -e

    DIAG_JOB_ID=$(sed -nE 's/.*(Pending|Granted) job allocation ([0-9]+).*/\2/p' "$DIAG_ALLOC_LOG" | tail -n1)
    if [[ "$alloc_rc" -ne 0 || -z "$DIAG_JOB_ID" ]]; then
        echo "DIAG allocation failed: rc=$alloc_rc job_id=${DIAG_JOB_ID:-missing}"
        exit 1
    fi

    scontrol show job -o "$DIAG_JOB_ID"
    srun \
        --jobid="$DIAG_JOB_ID" \
        --nodes=1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        bash -lc '
            set -euo pipefail
            host=$(hostname -f 2>/dev/null || hostname)
            image=/raid/squash/vllm_vllm-openai-rocm_kimi-k3.sqsh
            lock="${image}.lock"
            mkdir -p "$(dirname "$image")"
            exec 9>"$lock"
            flock -w 1200 9

            cleanup_partial() {
                if ! unsquashfs -s "$image" >/dev/null 2>&1; then
                    rm -f "$image"
                fi
            }
            trap cleanup_partial EXIT

            if unsquashfs -l "$image" >/dev/null 2>&1; then
                echo "K3_RAID_IMAGE_IMPORT host=$host status=already-valid"
            else
                rm -f "$image"
                enroot import -o "$image" docker://vllm/vllm-openai-rocm:kimi-k3
                unsquashfs -l "$image" >/dev/null
                echo "K3_RAID_IMAGE_IMPORT host=$host status=imported bytes=$(stat -c %s "$image")"
            fi
            trap - EXIT
        '

    srun \
        --jobid="$DIAG_JOB_ID" \
        --nodes=1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --gpus-per-task=1 \
        --kill-on-bad-exit=1 \
        --container-image=/raid/squash/vllm_vllm-openai-rocm_kimi-k3.sqsh \
        --container-mounts=/dev/kfd:/dev/kfd,/dev/dri:/dev/dri \
        --container-mount-home \
        --container-writable \
        --container-remap-root \
        --no-container-entrypoint \
        bash -lc '
            set -euo pipefail
            python - <<"PY"
import importlib
import importlib.util
import json
import pathlib

import torch

assert torch.cuda.is_available()
props = torch.cuda.get_device_properties(0)
arch = getattr(props, "gcnArchName", "unknown")

vllm = importlib.import_module("vllm")
aiter = importlib.import_module("aiter")
vllm_root = pathlib.Path(vllm.__file__).resolve().parent
kimi_paths = sorted(
    str(path.relative_to(vllm_root))
    for path in vllm_root.rglob("*")
    if "kimi" in path.name.lower() and "k3" in path.name.lower()
)
kimi_spec = importlib.util.find_spec("vllm.model_executor.models.kimi_k3")

evidence = {
    "torch_version": torch.__version__,
    "hip_version": torch.version.hip,
    "device_count": torch.cuda.device_count(),
    "device_name": props.name,
    "gcn_arch": arch,
    "vllm_version": getattr(vllm, "__version__", "unknown"),
    "aiter_path": str(pathlib.Path(aiter.__file__).resolve()),
    "kimi_module": kimi_spec is not None,
    "kimi_paths": kimi_paths[:20],
}
print("K3_RAID_CONTAINER_PREFLIGHT " + json.dumps(evidence, sort_keys=True))
assert evidence["hip_version"]
assert evidence["device_count"] >= 1
assert arch == "unknown" or "gfx942" in arch
assert evidence["kimi_module"] or evidence["kimi_paths"]
PY
        '

    echo "K3_RAID_CONTAINER_PREFLIGHT complete job_id=$DIAG_JOB_ID"
    exit 42
fi

# Exclude known-bad nodes; let Slurm pick from anything else:
#   chi-mi300x-049: persistent /nvme_home disk-full
#   chi-mi300x-121: provisioning incomplete; missing /raid and Enroot storage
JOB_ID=$(set +o pipefail; salloc --partition=$PARTITION --exclude=chi-mi300x-049,chi-mi300x-121 --gres=gpu:$GPU_COUNT --cpus-per-task=256 --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT

# Use flock to serialize concurrent imports to the same squash file
srun --jobid=$JOB_ID --job-name="$RUNNER_NAME" bash -c "
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
    if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
    fi
"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh

scancel $JOB_ID
