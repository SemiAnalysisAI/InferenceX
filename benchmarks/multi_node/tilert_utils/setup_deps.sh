#!/bin/bash

TILERT_VERSION="${TILERT_VERSION:-0.1.5.post2}"
TILERT_PIP_INDEX_URL="${TILERT_PIP_INDEX_URL:-}"

TILERT_HTTP_DEPS="${TILERT_HTTP_DEPS:-fastapi uvicorn httpx}"
TILERT_NIXL_VERSION="${TILERT_NIXL_VERSION:-1.3.1}"
TILERT_TRANSPORT_DEPS="${TILERT_TRANSPORT_DEPS:-nixl==$TILERT_NIXL_VERSION}"

_SETUP_INSTALLED=()

activate_tilert_env() {
    local env_dir=/opt/conda/envs/tilert
    [[ -d "$env_dir" ]] || return 0
    if [[ "$(command -v python)" == "$env_dir/bin/python" ]]; then
        echo "[SETUP] conda env 'tilert' already active"
        return 0
    fi
    echo "[SETUP] activating conda env 'tilert' (enroot does not run the image ENTRYPOINT)"
    # shellcheck disable=SC1091
    if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
        . /opt/conda/etc/profile.d/conda.sh && conda activate tilert
    fi
    [[ "$(command -v python)" == "$env_dir/bin/python" ]] || export PATH="$env_dir/bin:$PATH"
    echo "[SETUP] python -> $(command -v python)"
}

_installed_version() {
    "$PY" - "$1" <<'PY' 2>/dev/null
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    pass
PY
}

_pip_args() {
    local a=(--quiet --no-cache-dir)
    [[ -n "$TILERT_PIP_INDEX_URL" ]] && a+=(--index-url "$TILERT_PIP_INDEX_URL")
    printf '%s\n' "${a[@]}"
}

install_tilert_decode() {
    mapfile -t _pa < <(_pip_args)
    local have; have="$(_installed_version tilert)"
    if [[ "$have" == "$TILERT_VERSION" ]]; then
        echo "[SETUP] tilert $have already installed, skipping"
    else
        [[ -n "$have" ]] && echo "[SETUP] tilert $have installed, switching to pinned $TILERT_VERSION"
        echo "[SETUP] installing tilert==$TILERT_VERSION (official PyPI release wheel)"
        "$PY" -m pip install "${_pa[@]}" "tilert==$TILERT_VERSION" || {
            echo "[SETUP] ERROR: failed to install tilert==$TILERT_VERSION"; exit 1; }
        have="$(_installed_version tilert)"
        [[ "$have" == "$TILERT_VERSION" ]] || {
            echo "[SETUP] ERROR: still not $TILERT_VERSION after install (actual: ${have:-not installed})"; exit 1; }
        _SETUP_INSTALLED+=("tilert==$TILERT_VERSION")
    fi
    _install_missing "uvicorn" "$TILERT_HTTP_DEPS"
    _install_missing "nixl"    "$TILERT_TRANSPORT_DEPS"
    local tv; tv="$(_installed_version transformers)"
    if [[ -n "$tv" && "${tv%%.*}" -lt 5 ]]; then
        echo "[SETUP] transformers $tv < 5 -- cannot load the official checkpoint's TokenizersBackend, upgrading"
        "$PY" -m pip install "${_pa[@]}" -U "transformers>=5.4.0" || {
            echo "[SETUP] ERROR: failed to upgrade transformers"; exit 1; }
        _SETUP_INSTALLED+=("transformers>=5.4.0(upgrade from $tv)")
    fi
    "$PY" -c "import tilert.pd_vllm.decode_server" 2>/dev/null || {
        echo "[SETUP] ERROR: import tilert.pd_vllm.decode_server failed; actual error:"
        "$PY" -c "import tilert.pd_vllm.decode_server" 2>&1 | tail -3
        exit 1; }
    echo "[SETUP] tilert.pd_vllm.decode_server imports OK"
}

_install_missing() {
    local probe="$1" pkgs="$2"
    [[ -n "$pkgs" ]] || return 0
    if "$PY" -c "import $probe" 2>/dev/null; then
        echo "[SETUP] $probe already present, skipping ($pkgs)"
        return 0
    fi
    echo "[SETUP] installing $pkgs (probe module $probe missing)"
    mapfile -t _pa < <(_pip_args)
    # shellcheck disable=SC2086
    "$PY" -m pip install "${_pa[@]}" $pkgs || {
        echo "[SETUP] ERROR: failed to install: $pkgs"; exit 1; }
    _SETUP_INSTALLED+=("$pkgs")
}

install_tilert_prefill() {
    local vllm_v; vllm_v="$(_installed_version vllm)"
    if [[ -z "$vllm_v" ]]; then
        echo "[SETUP] ERROR: no vLLM in the prefill image."
        echo "[SETUP]        prefill needs an image with V1 disaggregation + GLM-5/5.1 (DSA) +"
        echo "[SETUP]        --kv-cache-dtype fp8_ds_mla support; the official tilert image has no vLLM,"
        echo "[SETUP]        and its dependency set conflicts with vLLM, so P and D must use different images."
        exit 1
    fi
    echo "[SETUP] prefill-side vLLM $vllm_v"
    local have; have="$(_installed_version tilert)"
    if [[ "$have" == "$TILERT_VERSION" ]]; then
        echo "[SETUP] tilert $have already installed, skipping"
    else
        echo "[SETUP] installing tilert==$TILERT_VERSION --no-deps (connector plugin only; leaves transformers untouched)"
        mapfile -t _pa < <(_pip_args)
        "$PY" -m pip install "${_pa[@]}" --no-deps "tilert==$TILERT_VERSION" || {
            echo "[SETUP] ERROR: failed to install tilert==$TILERT_VERSION (--no-deps)"; exit 1; }
        have="$(_installed_version tilert)"
        [[ "$have" == "$TILERT_VERSION" ]] || {
            echo "[SETUP] ERROR: still not $TILERT_VERSION after install (actual: ${have:-not installed})"; exit 1; }
        _SETUP_INSTALLED+=("tilert==$TILERT_VERSION(--no-deps)")
    fi
    _install_missing "nixl" "$TILERT_TRANSPORT_DEPS"
    "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>/dev/null || {
        echo "[SETUP] WARN: import tilert.pd_vllm.prefill_connector failed (vLLM will report again when loading the connector plugin):"
        "$PY" -c "import tilert.pd_vllm.prefill_connector" 2>&1 | tail -3; }
}

resolve_python() {
    if [[ -n "${PY:-}" ]] && command -v "$PY" >/dev/null 2>&1; then :
    else
        PY=""
        for c in python python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
    fi
    [[ -n "$PY" ]] || { echo "[SETUP] ERROR: neither python nor python3 found"; exit 1; }
    export PY
    echo "[SETUP] interpreter PY=$PY ($(command -v "$PY"))"
}

activate_tilert_env
resolve_python
case "${TILERT_ROLE:-}" in
    decode)        install_tilert_decode ;;
    prefill)       install_tilert_prefill ;;
    *)             echo "[SETUP] ERROR: unknown ROLE='${TILERT_ROLE:-}'"; exit 1 ;;
esac
if (( ${#_SETUP_INSTALLED[@]} )); then
    echo "[SETUP] installed this run: ${_SETUP_INSTALLED[*]}"
else
    echo "[SETUP] nothing to install (dependencies already satisfied)"
fi
