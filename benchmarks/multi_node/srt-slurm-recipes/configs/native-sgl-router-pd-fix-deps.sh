#!/usr/bin/env bash
set -euo pipefail

# Temporary exact-source pin for the upstream SGLang Model Gateway one-token
# P/D response fix. The shared wheel cache and flock make the source build
# happen once even though srt-slurm runs the recipe setup on every role.
SGLANG_REPOSITORY="https://github.com/cquil11/sglang.git"
SGLANG_REF="fix/pd-one-token-prefill-response"
SGLANG_COMMIT="9072e0dc5aaf1962b4e2c6f1a51094356b5d3324"
WHEEL_CACHE_ROOT="/router_wheels/sglang-router-${SGLANG_COMMIT}"
LOCK_FILE="${WHEEL_CACHE_ROOT}.lock"

mkdir -p "${WHEEL_CACHE_ROOT}"
exec 9>"${LOCK_FILE}"
flock -w 3600 9

shopt -s nullglob
router_wheels=("${WHEEL_CACHE_ROOT}"/sglang_router-*.whl)
if (( ${#router_wheels[@]} == 0 )); then
    build_dir=$(mktemp -d /tmp/sglang-router-build.XXXXXX)
    trap 'rm -rf "${build_dir}"' EXIT

    git clone --filter=blob:none --no-checkout --single-branch \
        --branch "${SGLANG_REF}" "${SGLANG_REPOSITORY}" "${build_dir}"
    git -C "${build_dir}" sparse-checkout init --cone
    git -C "${build_dir}" sparse-checkout set sgl-model-gateway
    git -C "${build_dir}" checkout --detach "${SGLANG_COMMIT}"
    if [[ "$(git -C "${build_dir}" rev-parse HEAD)" != "${SGLANG_COMMIT}" ]]; then
        echo "ERROR: SGLang checkout does not match ${SGLANG_COMMIT}" >&2
        exit 1
    fi

    python3 -m pip install --break-system-packages --no-cache-dir --upgrade \
        "maturin>=1.0,<2.0"
    (
        cd "${build_dir}/sgl-model-gateway/bindings/python"
        CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}" \
            maturin build --release --out "${WHEEL_CACHE_ROOT}"
    )
    router_wheels=("${WHEEL_CACHE_ROOT}"/sglang_router-*.whl)
fi

if (( ${#router_wheels[@]} != 1 )); then
    echo "ERROR: expected exactly one cached SGLang Router wheel, found ${#router_wheels[@]}" >&2
    exit 1
fi

# Install the released package first to resolve its Python dependencies, then
# replace only the native package with the wheel built from the exact fix SHA.
python3 -m pip install --break-system-packages --no-cache-dir --upgrade \
    "sglang-router==0.3.2"
python3 -m pip install --break-system-packages --no-cache-dir \
    --force-reinstall --no-deps "${router_wheels[0]}"
python3 -c 'import sglang_router; print(sglang_router.__file__)'
echo "Installed SGLang Router from ${SGLANG_COMMIT}"
