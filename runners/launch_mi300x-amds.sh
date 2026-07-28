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

# Temporary diagnostic branch only: prove that the MI300X controller can
# create a two-node allocation, launch one Slurm task per node, and inspect K3
# staging without importing a container or loading model weights.
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
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --gres=gpu:1 \
        --cpus-per-task=1 \
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
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        bash -lc '
            set -euo pipefail
            host=$(hostname -f 2>/dev/null || hostname)
            gpu_count=$(find /dev/dri -maxdepth 1 -name "renderD*" 2>/dev/null | wc -l)
            if [[ -e /dev/kfd ]]; then
                kfd=present
            else
                kfd=missing
            fi
            printf "MI300X_TWO_NODE_SMOKE host=%s procid=%s gpu_count=%s kfd=%s\n" \
                "$host" "${SLURM_PROCID:-unknown}" "$gpu_count" "$kfd"
            [[ "$gpu_count" -ge 8 ]]
            [[ "$kfd" == "present" ]]
        '

    srun \
        --jobid="$DIAG_JOB_ID" \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        bash -lc '
            set -euo pipefail
            host=$(hostname -f 2>/dev/null || hostname)
            image=/home/gharunner/gharunners/squash/vllm_vllm-openai-rocm_kimi-k3.sqsh
            cache=/raid/hf-hub-cache/models--moonshotai--Kimi-K3
            if [[ ! -d "$cache" && -d /raid/hf-hub-cache/hub/models--moonshotai--Kimi-K3 ]]; then
                cache=/raid/hf-hub-cache/hub/models--moonshotai--Kimi-K3
            fi

            image_status=missing
            if [[ -r "$image" ]] && unsquashfs -s "$image" >/dev/null 2>&1; then
                image_status=valid
            elif [[ -e "$image" ]]; then
                image_status=invalid
            fi

            snapshot_status=missing
            snapshot_count=0
            safetensor_count=0
            broken_links=0
            if [[ -d "$cache/snapshots" ]]; then
                snapshot_count=$(find "$cache/snapshots" -mindepth 1 -maxdepth 1 -type d | wc -l)
                latest_snapshot=$(find "$cache/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)
                if [[ -n "$latest_snapshot" ]]; then
                    safetensor_count=$(find "$latest_snapshot" -maxdepth 1 -name "*.safetensors" | wc -l)
                    broken_links=$(find -L "$latest_snapshot" -maxdepth 1 -type l | wc -l)
                    if [[ -f "$latest_snapshot/config.json" && "$safetensor_count" -gt 0 && "$broken_links" -eq 0 ]]; then
                        snapshot_status=present
                    else
                        snapshot_status=incomplete
                    fi
                fi
            fi

            raid_free_bytes=$(df -PB1 /raid | awk "NR==2 {print \$4}")
            printf "K3_STAGING_PREFLIGHT host=%s image=%s cache=%s snapshots=%s safetensors=%s broken_links=%s raid_free_bytes=%s\n" \
                "$host" "$image_status" "$snapshot_status" "$snapshot_count" \
                "$safetensor_count" "$broken_links" "$raid_free_bytes"
        '

    srun \
        --jobid="$DIAG_JOB_ID" \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        bash -lc '
            set -euo pipefail
            image=/home/gharunner/gharunners/squash/vllm_vllm-openai-rocm_kimi-k3.sqsh
            lock="${image}.lock"
            exec 9>"$lock"
            flock -w 1200 9
            if unsquashfs -l "$image" >/dev/null 2>&1; then
                echo "K3_IMAGE_IMPORT host=$(hostname) status=already-valid"
            else
                rm -f "$image"
                enroot import -o "$image" docker://vllm/vllm-openai-rocm:kimi-k3
                unsquashfs -l "$image" >/dev/null
                echo "K3_IMAGE_IMPORT host=$(hostname) status=imported"
            fi
        '

    srun \
        --jobid="$DIAG_JOB_ID" \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --gpus-per-task=1 \
        --kill-on-bad-exit=1 \
        --container-image=/home/gharunner/gharunners/squash/vllm_vllm-openai-rocm_kimi-k3.sqsh \
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
print("K3_CONTAINER_PREFLIGHT " + json.dumps(evidence, sort_keys=True))
assert evidence["hip_version"]
assert evidence["device_count"] >= 1
assert arch == "unknown" or "gfx942" in arch
assert evidence["kimi_module"] or evidence["kimi_paths"]
PY
        '

    echo "MI300X_TWO_NODE_SMOKE complete job_id=$DIAG_JOB_ID"
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
