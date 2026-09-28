#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
# Install missing disagg dependencies at container start. Each installer is
# idempotent and gated on $ENGINE.

_SETUP_START=$(date +%s)
_SETUP_INSTALLED=()

# Pinned by the recipe (TILERT_VERSION); the rest are fixed properties of the
# TileRT 0.1.x runtime rather than caller configuration.
TILERT_PACKAGE=tilert
TILERT_HTTP_DEPS="fastapi uvicorn httpx"
TILERT_TRANSPORT_DEPS="mooncake-transfer-engine-rocm>=0.3.13"
TILERT_TRANSFORMERS_SPEC="transformers>=4.56"

_tilert_resolve_python() {
    if [[ -n "${PY:-}" ]] && command -v "$PY" >/dev/null 2>&1; then :; else
        PY=""
        local c
        for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
    fi
    [[ -n "$PY" ]] || { echo "[SETUP] ERROR: neither python3 nor python found"; exit 1; }
    export PY
    echo "[SETUP] interpreter PY=$PY ($(command -v "$PY"))"
}

_tilert_installed_version() {
    "$PY" - "$1" <<'PYEOF' 2>/dev/null
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    pass
PYEOF
}

_tilert_pip() {
    "$PY" -m pip install --quiet --no-cache-dir "$@"
}

_tilert_install_missing() {
    local probe="$1"; shift
    [[ $# -gt 0 ]] || return 0
    if "$PY" -c "import $probe" 2>/dev/null; then
        echo "[SETUP] $probe already present, skipping ($*)"
        return 0
    fi
    echo "[SETUP] installing $* (probe module '$probe' missing)"
    _tilert_pip "$@" || { echo "[SETUP] ERROR: failed to install: $*"; exit 1; }
    _SETUP_INSTALLED+=("$*")
}

install_tilert_container_tools() {
    if command -v ip >/dev/null 2>&1 && command -v curl >/dev/null 2>&1 \
        && command -v ibv_devices >/dev/null 2>&1; then
        echo "[SETUP] Container RDMA/net tools already present"
        return 0
    fi
    echo "[SETUP] Installing iproute2 + curl + ibverbs userspace in container..."
    apt-get update -q -y && apt-get install -q -y --no-install-recommends \
        iproute2 curl ibverbs-utils libibverbs1 librdmacm1 ibverbs-providers \
        && rm -rf /var/lib/apt/lists/*
    if ! command -v ip >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
        echo "[SETUP] ERROR: failed to install iproute2/curl"; exit 1
    fi
    _SETUP_INSTALLED+=("iproute2+curl+ibverbs")
}

_tilert_install_wheel() {
    local mode="$1"  # full | no-deps
    local have; have="$(_tilert_installed_version tilert)"
    if [[ "$have" == "$TILERT_VERSION" ]]; then
        echo "[SETUP] tilert $have already installed, skipping"
        return 0
    fi
    [[ -n "$have" ]] && echo "[SETUP] tilert $have installed, switching to pinned $TILERT_VERSION"
    if [[ "$mode" == "no-deps" ]]; then
        echo "[SETUP] installing $TILERT_PIP_SPEC --no-deps (connector plugin + router on top of the image's vLLM)"
        _tilert_pip --no-deps "$TILERT_PIP_SPEC" || { echo "[SETUP] ERROR: failed to install $TILERT_PIP_SPEC (--no-deps)"; exit 1; }
    else
        echo "[SETUP] installing $TILERT_PIP_SPEC (TileRT ROCm build, official PyPI wheel)"
        _tilert_pip "$TILERT_PIP_SPEC" || { echo "[SETUP] ERROR: failed to install $TILERT_PIP_SPEC"; exit 1; }
    fi
    have="$(_tilert_installed_version tilert)"
    [[ "$have" == "$TILERT_VERSION" ]] || {
        echo "[SETUP] ERROR: tilert is ${have:-not installed} after install, expected $TILERT_VERSION"; exit 1; }
    _SETUP_INSTALLED+=("$TILERT_PACKAGE==$TILERT_VERSION($mode)")
}

install_tilert_decode() {
    install_tilert_container_tools
    _tilert_install_wheel full
    _tilert_install_missing uvicorn $TILERT_HTTP_DEPS
    _tilert_install_missing mooncake.engine "$TILERT_TRANSPORT_DEPS"
    _tilert_install_missing transformers "$TILERT_TRANSFORMERS_SPEC"
    "$PY" -c "import tilert.pd_vllm.decode_server" 2>/dev/null || {
        echo "[SETUP] ERROR: import tilert.pd_vllm.decode_server failed:"
        "$PY" -c "import tilert.pd_vllm.decode_server" 2>&1 | tail -3
        exit 1; }
    echo "[SETUP] tilert.pd_vllm.decode_server imports OK"
}

install_tilert_prefill() {
    local vllm_v; vllm_v="$(_tilert_installed_version vllm)"
    if [[ -z "$vllm_v" ]]; then
        echo "[SETUP] ERROR: no vLLM in the prefill image (PREFILL_IMAGE must be a vllm/vllm-openai-rocm image)."
        exit 1
    fi
    echo "[SETUP] prefill-side vLLM $vllm_v"
    install_tilert_container_tools
    _tilert_install_wheel no-deps
    _tilert_install_missing mooncake.engine "$TILERT_TRANSPORT_DEPS"
    "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>/dev/null || {
        echo "[SETUP] WARN: import tilert.pd_vllm.prefill_connector failed (vLLM will report again when loading the connector plugin):"
        "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>&1 | tail -3; }
}

if [[ "$ENGINE" == "tilert" ]]; then
    check_env_vars TILERT_VERSION
    TILERT_PIP_SPEC="$TILERT_PACKAGE==$TILERT_VERSION"
    _tilert_resolve_python
    case "${TILERT_ROLE:-}" in
        decode)  install_tilert_decode ;;
        prefill) install_tilert_prefill ;;
        *) echo "[SETUP] ERROR: ENGINE=tilert needs TILERT_ROLE=decode|prefill (got '${TILERT_ROLE:-}')"; exit 1 ;;
    esac
fi

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present ($(( _SETUP_END - _SETUP_START ))s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
