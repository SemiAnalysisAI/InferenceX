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
        --nodelist=chi-mi300x-043 \
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
            image=/raid/hf-hub-cache/inferencex/squash/vllm_vllm-openai-rocm_kimi-k3.sqsh
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
        --container-image=/raid/hf-hub-cache/inferencex/squash/vllm_vllm-openai-rocm_kimi-k3.sqsh \
        --container-mounts=/dev/kfd:/dev/kfd,/dev/dri:/dev/dri \
        --container-mount-home \
        --container-writable \
        --container-remap-root \
        --no-container-entrypoint \
        bash -lc '
            set -euo pipefail
            python - <<"PY"
import importlib.metadata
import os

import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_2stage_cfgs, get_padded_M
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import GateMode

assert torch.cuda.is_available()
assert get_gfx() == "gfx942", get_gfx()

try:
    aiter_version = importlib.metadata.version("amd-aiter")
except importlib.metadata.PackageNotFoundError:
    aiter_version = getattr(aiter, "__version__", "unknown")

print(
    "K3_MXFP4_METADATA_ENV"
    f" gfx={get_gfx()} cu={get_cu_num()} aiter={aiter_version}"
    f" torch={torch.__version__} hip={torch.version.hip}"
)

def describe(stage):
    func = getattr(stage, "func", stage)
    keywords = getattr(stage, "keywords", {})
    module_name = getattr(func, "__module__", "")
    function_name = getattr(func, "__name__", repr(func))
    return f"{module_name}.{function_name} {keywords}"

supported = {}
for mode, activation_dtype in (
    ("a16w4", dtypes.bf16),
    ("a8w4", dtypes.fp8),
):
    os.environ["AITER_SITUV2_A8W4"] = "1" if mode == "a8w4" else "0"
    try:
        metadata = get_2stage_cfgs(
            get_padded_M(1),
            3584,
            384,
            896,
            16,
            dtypes.bf16,
            activation_dtype,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Situv2,
            False,
            0,
            0,
            True,
            GateMode.SEPARATED,
        )
        supported[mode] = True
        print(
            f"K3_MXFP4_METADATA mode={mode} status=supported"
            f" block_m={metadata.block_m}"
            f" stage1={describe(metadata.stage1)}"
            f" stage2={describe(metadata.stage2)}"
        )
    except Exception as exc:
        supported[mode] = False
        print(
            f"K3_MXFP4_METADATA mode={mode} status=unsupported"
            f" error_type={type(exc).__name__} error={exc}"
        )

if any(supported.values()):
    print("K3_MXFP4_METADATA_RESULT status=kernel-launch-required")
else:
    print("K3_MXFP4_METADATA_RESULT status=no-gfx942-dispatch")
PY

            echo "K3_MXFP4_RUNTIME_ENV_BEGIN"
            env | sort | grep -E "^(AITER|ATOM|VLLM|ROCM|HIP).*MOE|^AITER_SITUV2_A8W4=" || true
            echo "K3_MXFP4_RUNTIME_ENV_END"

            echo "K3_MXFP4_SOURCE_CONTRACT_BEGIN"
            grep -R -n \
                --include="*.py" \
                -E "AITER_SITUV2_A8W4|ATOM_MOE_GU_ITLV|GateMode\\.(INTERLEAVE|SEPARATED)|gate_mode=|shuffle_weight_a16w4|ActivationType\\.Situv2" \
                /usr/local/lib/python3.12/dist-packages/vllm \
                /usr/local/lib/python3.12/dist-packages/aiter \
                2>/dev/null | head -n 400 || true
            echo "K3_MXFP4_SOURCE_CONTRACT_END"
        '

    echo "K3_MXFP4_METADATA_PREFLIGHT complete job_id=$DIAG_JOB_ID"
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
