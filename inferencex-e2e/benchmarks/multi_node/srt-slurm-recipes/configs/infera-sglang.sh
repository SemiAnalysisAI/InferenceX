#!/usr/bin/env bash
# Install Infera, at the master config's router version, into the stock SGLang image for the
# infera-sglang points. The release wheel carries the infera-router binary and the
# infera.engine.sglang worker wrapper, so the master's router version is the Infera version.
# Only nats-py is missing from the lmsysorg ROCm image; --no-deps keeps every package the
# image ships (torch, sglang, transformers, ...) at its pinned version.
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars ROUTER_VERSION
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-cache-dir --no-deps \
    "nats-py==2.16.0" \
    "amd-infera==${ROUTER_VERSION}" \
    --find-links "https://github.com/AMD-AGI/Infera/releases/expanded_assets/v${ROUTER_VERSION}"
python3 -c "import shutil, sys, infera.engine.sglang; sys.exit(shutil.which('infera-router') is None)"
