#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
check_env_vars ROCM_PATH UCX_HOME RIXL_HOME
# Install missing disagg dependencies at container start; sourced by server_vllm.sh
# and server_sglang.sh so PATH / LD_LIBRARY_PATH exports persist. Each installer is
# idempotent and gated on $ENGINE (vllm-disagg / sglang-disagg).

_SETUP_START=$(date +%s)
_SETUP_INSTALLED=()

# ibv_devinfo (ibverbs-utils) and ip (iproute2) for in-container NIC/RDMA checks.
install_recipe_deps() {
    if command -v ibv_devinfo >/dev/null 2>&1 && command -v ip >/dev/null 2>&1; then
        echo "[SETUP] Container RDMA/net tools already present"
        return 0
    fi

    echo "[SETUP] Installing ibv_devinfo + iproute2 in container..."
    apt-get update -q -y && apt-get install -q -y \
        ibverbs-utils iproute2 \
        && rm -rf /var/lib/apt/lists/*

    if ! command -v ibv_devinfo >/dev/null 2>&1 || ! command -v ip >/dev/null 2>&1; then
        echo "[SETUP] ERROR: Failed to install ibv_devinfo/iproute2"; exit 1
    fi
    _SETUP_INSTALLED+=("ibverbs-utils+iproute2")
}

# ROCm vLLM lacks the quark dependency needed for MXFP4 models:
# https://github.com/vllm-project/vllm/issues/35633
install_amd_quark() {
    if python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] amd-quark already present"
        return 0
    fi

    echo "[SETUP] Installing amd-quark for MXFP4 quantization support..."
    pip install --quiet amd-quark

    if ! python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] WARN: amd-quark install failed (non-fatal for non-MXFP4 models)"
        return 0
    fi
    _SETUP_INSTALLED+=("amd-quark")
}

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
        && command -v ibv_devices >/dev/null 2>&1 && command -v patch >/dev/null 2>&1; then
        echo "[SETUP] Container RDMA/net tools already present"
        return 0
    fi
    echo "[SETUP] Installing iproute2 + curl + patch + ibverbs userspace in container..."
    apt-get update -q -y && apt-get install -q -y --no-install-recommends \
        iproute2 curl patch ibverbs-utils libibverbs1 librdmacm1 ibverbs-providers \
        && rm -rf /var/lib/apt/lists/*
    if ! command -v ip >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1 || ! command -v patch >/dev/null 2>&1; then
        echo "[SETUP] ERROR: failed to install iproute2/curl/patch"; exit 1
    fi
    _SETUP_INSTALLED+=("iproute2+curl+patch+ibverbs")
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

# KV prefix reuse across AgentX turns. tilert 0.1.6.post1 resets the decode
# sequence and copies the whole prompt's KV into all eight rank caches on every
# request, one layer after another on the default streams: 2.4 s at the p50
# AgentX context, although most turns only extend the previous prompt. The
# patch keeps the last prompt's KV resident, copies only the rows from the
# first differing page onward after checking a sample of the kept rows against
# the transfer, and fans the copies out on one stream per device pair. The
# prefill connector sends the prompt ids when the decode hello asks for them.
# Engine-patch waiver: docs/waiver/3376.md. Applied when the recipe sets
# TILERT_PD_PREFIX_REUSE=1; with 0 the wheel runs as shipped.
_TILERT_PD_REUSE_PATCH="$(dirname "${BASH_SOURCE[0]}")/patches/tilert-0.1.6.post1-pd-prefix-reuse.patch"

_tilert_apply_pd_reuse_patch() {
    [[ "$TILERT_PD_PREFIX_REUSE" == "1" ]] || { echo "[SETUP] KV prefix reuse off (TILERT_PD_PREFIX_REUSE=$TILERT_PD_PREFIX_REUSE); tilert unpatched"; return 0; }
    local dir
    dir="$("$PY" -c 'import os, tilert.pd_vllm as m; print(os.path.dirname(m.__file__))')" || { echo "[SETUP] ERROR: cannot locate tilert.pd_vllm"; exit 1; }
    if grep -q "_REUSE_ENV = 'TILERT_PD_PREFIX_REUSE'" "$dir/profiles/glm5_rocm_engine.py"; then
        echo "[SETUP] tilert KV prefix reuse patch already applied in $dir"
        return 0
    fi
    [[ -f "$_TILERT_PD_REUSE_PATCH" ]] || { echo "[SETUP] ERROR: missing $_TILERT_PD_REUSE_PATCH"; exit 1; }
    echo "[SETUP] applying $(basename "$_TILERT_PD_REUSE_PATCH") to $dir"
    patch -p1 -s -N -d "$dir" < "$_TILERT_PD_REUSE_PATCH" || { echo "[SETUP] ERROR: patch failed to apply"; exit 1; }
    "$PY" -m py_compile "$dir"/*.py "$dir"/profiles/*.py || { echo "[SETUP] ERROR: patched tilert.pd_vllm does not compile"; exit 1; }
    _SETUP_INSTALLED+=("tilert-pd-prefix-reuse.patch")
}

install_tilert_decode() {
    install_tilert_container_tools
    _tilert_install_wheel full
    _tilert_apply_pd_reuse_patch
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
    _tilert_apply_pd_reuse_patch
    _tilert_install_missing mooncake.engine "$TILERT_TRANSPORT_DEPS"
    "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>/dev/null || {
        echo "[SETUP] WARN: import tilert.pd_vllm.prefill_connector failed (vLLM will report again when loading the connector plugin):"
        "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>&1 | tail -3; }
}

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    install_recipe_deps
    install_amd_quark

    export ROCM_PATH
    export UCX_HOME
    export RIXL_HOME
    export PATH="${UCX_HOME}/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH}"
    export LD_LIBRARY_PATH="${UCX_HOME}/lib:${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
elif [[ "$ENGINE" == "tilert" ]]; then
    check_env_vars TILERT_VERSION TILERT_PD_PREFIX_REUSE
    if [[ "$TILERT_PD_PREFIX_REUSE" == "1" && "$TILERT_VERSION" != "0.1.6.post1" ]]; then
        echo "[SETUP] ERROR: $_TILERT_PD_REUSE_PATCH targets tilert 0.1.6.post1, got $TILERT_VERSION; rebase the patch or set TILERT_PD_PREFIX_REUSE=0"
        exit 1
    fi
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
