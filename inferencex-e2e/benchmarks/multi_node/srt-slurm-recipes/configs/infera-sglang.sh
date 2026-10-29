#!/usr/bin/env bash
# Install Infera, at the master config's router version, into the stock SGLang image for the
# infera-sglang points, then apply the SGLang fixes the pinned image still lacks
# (infera-sglang-patches/apply.sh, waiver docs/waiver/4629.md). The Linux x86_64 release
# wheel bundles the infera-router binary next to the infera.engine.sglang worker wrapper, so
# the master's router version is the Infera version. Only nats-py is missing from the
# lmsysorg ROCm image; --no-deps keeps every package the image ships (torch, sglang,
# transformers, ...) at its pinned version.
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
# The router the launcher will exec: the wheel's bundled binary, not one on PATH.
python3 -c "import infera.engine.sglang; from infera.server.launch_rust import _find_binary; _find_binary()"
# INFERA_SGLANG_PATCHES=0 only for the installer unit test, which has no SGLang tree.
if [[ "${INFERA_SGLANG_PATCHES:-1}" != 0 ]]; then
    bash "$(dirname "${BASH_SOURCE[0]}")/infera-sglang-patches/apply.sh"
fi
