#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars TOKENSPEED_WHEEL_URL TOKENSPEED_KERNEL_WHEEL_URL TOKENSPEED_ENGINE_WHEEL_URL TOKENSPEED_SCHEDULER_VERSION TOKENSPEED_FLASHINFER_VERSION TOKENSPEED_CUDA_VERSION ROUTER_VERSION

pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --upgrade \
    --index-url https://pypi.org/simple \
    "$TOKENSPEED_WHEEL_URL" "$TOKENSPEED_KERNEL_WHEEL_URL" "$TOKENSPEED_ENGINE_WHEEL_URL" \
    "tokenspeed-scheduler==$TOKENSPEED_SCHEDULER_VERSION" \
    "tokenspeed-smg==$ROUTER_VERSION"
python3 -m flashinfer download-kernels \
    --cuda-version "$TOKENSPEED_CUDA_VERSION" \
    --flashinfer-version "$TOKENSPEED_FLASHINFER_VERSION"
python3 -m pip check
