#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars TOKENSPEED_WHEEL_URL TOKENSPEED_KERNEL_WHEEL_URL TOKENSPEED_SCHEDULER_VERSION ROUTER_VERSION

python3 -m pip install --upgrade \
    --index-url https://pypi.org/simple \
    --extra-index-url https://lightseek.org/whl/nightly \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "$TOKENSPEED_WHEEL_URL" "$TOKENSPEED_KERNEL_WHEEL_URL" \
    "tokenspeed-scheduler==$TOKENSPEED_SCHEDULER_VERSION" \
    "tokenspeed-smg==$ROUTER_VERSION"
python3 -m pip check
