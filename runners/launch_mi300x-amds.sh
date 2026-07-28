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
    AITER_OVERLAY_REF="192f8db9c6d6f841f3bafc4e382a7cd2a361e88c"
    AITER_OVERLAY_DIR="$GITHUB_WORKSPACE/.aiter-k3-gfx942-overlay"

    download_aiter_overlay() {
        local relative_path="$1"
        local expected_sha256="$2"
        local output="$AITER_OVERLAY_DIR/$(basename "$relative_path")"
        local url="https://raw.githubusercontent.com/edwingao28/aiter/${AITER_OVERLAY_REF}/${relative_path}"

        python3 - "$url" "$output" "$expected_sha256" <<'PY'
import hashlib
import os
import pathlib
import sys
import urllib.request

url, output, expected = sys.argv[1:]
destination = pathlib.Path(output)
destination.parent.mkdir(parents=True, exist_ok=True)
payload = urllib.request.urlopen(url, timeout=120).read()
actual = hashlib.sha256(payload).hexdigest()
if actual != expected:
    raise SystemExit(
        f"AITER overlay checksum mismatch for {url}: "
        f"expected={expected} actual={actual}"
    )
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_bytes(payload)
os.replace(temporary, destination)
print(
    f"K3_AITER_OVERLAY_DOWNLOAD file={destination.name} "
    f"sha256={actual} bytes={len(payload)}"
)
PY
    }

    download_aiter_overlay \
        "aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py" \
        "ed809fce18da00bf98de6385b7b934bd97fbc52cc9dc8b6306280e235b4d3fae"
    download_aiter_overlay \
        "aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py" \
        "12ecefe55e188232166ddc34fe182464c26d1bff278196f7f72078ac5cab9cdd"
    download_aiter_overlay \
        "aiter/ops/flydsl/kernels/lds_dma_policy.py" \
        "cfeca166acba58f789c61cb78a77a5cb8fad12a71cfbd3cbe428ea8c83a5fdc9"
    download_aiter_overlay \
        "aiter/ops/flydsl/kernels/mfma_policy.py" \
        "dac7aa6b2cf0e25adb2119072ca80fd708855c637efc3b98cfd8f998762d8f04"

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
        --container-mounts="$GITHUB_WORKSPACE:/workspace,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
        --container-mount-home \
        --container-writable \
        --container-remap-root \
        --no-container-entrypoint \
        bash -lc '
            set -euo pipefail
            export FLYDSL_RUNTIME_CACHE_DIR=/tmp/inferencex-k3-flydsl-192f8db9c6d6f841f3bafc4e382a7cd2a361e88c-random
            mkdir -p "$FLYDSL_RUNTIME_CACHE_DIR"
            echo "K3_FLYDSL_CACHE dir=$FLYDSL_RUNTIME_CACHE_DIR"
            aiter_root=$(python -c \
                "import pathlib, aiter; print(pathlib.Path(aiter.__file__).resolve().parent)")
            install -m 0644 \
                /workspace/.aiter-k3-gfx942-overlay/mixed_moe_gemm_2stage.py \
                "$aiter_root/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"
            install -m 0644 \
                /workspace/.aiter-k3-gfx942-overlay/mfma_preshuffle_pipeline.py \
                "$aiter_root/ops/flydsl/kernels/mfma_preshuffle_pipeline.py"
            install -m 0644 \
                /workspace/.aiter-k3-gfx942-overlay/lds_dma_policy.py \
                "$aiter_root/ops/flydsl/kernels/lds_dma_policy.py"
            install -m 0644 \
                /workspace/.aiter-k3-gfx942-overlay/mfma_policy.py \
                "$aiter_root/ops/flydsl/kernels/mfma_policy.py"
            python -m compileall -q \
                "$aiter_root/ops/flydsl/kernels/mixed_moe_gemm_2stage.py" \
                "$aiter_root/ops/flydsl/kernels/mfma_preshuffle_pipeline.py" \
                "$aiter_root/ops/flydsl/kernels/lds_dma_policy.py" \
                "$aiter_root/ops/flydsl/kernels/mfma_policy.py"
            echo "K3_AITER_OVERLAY_APPLIED root=$aiter_root ref=192f8db9c6d6f841f3bafc4e382a7cd2a361e88c"

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

            export AITER_SITUV2_A8W4=0
            export K3_PROBE_PATTERN=random
            timeout --signal=TERM --kill-after=30s 1200s \
                python /workspace/scripts/diagnostics/k3_mxfp4_gfx942_probe.py
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
