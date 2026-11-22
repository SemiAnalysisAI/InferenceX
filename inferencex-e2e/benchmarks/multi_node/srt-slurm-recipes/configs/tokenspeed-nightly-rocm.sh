#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars TOKENSPEED_WHEEL_URL TOKENSPEED_KERNEL_WHEEL_URL TOKENSPEED_KERNEL_AMD_WHEEL_URL TOKENSPEED_ENGINE_WHEEL_URL TOKENSPEED_SCHEDULER_VERSION ROUTER_VERSION

pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
# The runner image provides ROCm torch. Upgrading only what the pinned wheels
# need keeps it: PyPI's torch is a CUDA build.
"${pip_install[@]}" --upgrade --upgrade-strategy only-if-needed \
    --index-url https://pypi.org/simple \
    "$TOKENSPEED_WHEEL_URL" "$TOKENSPEED_KERNEL_WHEEL_URL" \
    "$TOKENSPEED_KERNEL_AMD_WHEEL_URL" "$TOKENSPEED_ENGINE_WHEEL_URL" \
    "tokenspeed-scheduler==$TOKENSPEED_SCHEDULER_VERSION" \
    "tokenspeed-smg==$ROUTER_VERSION"
python3 -c 'import torch, sys; sys.exit(0 if torch.version.hip else "torch is not a ROCm build")'
python3 -m pip check
