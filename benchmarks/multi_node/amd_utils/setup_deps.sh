#!/bin/bash
# =============================================================================
# setup_deps.sh — Install missing vLLM disagg dependencies at container start.
#
# Base image: vllm/vllm-openai-rocm:v0.18.0
# Sourced by server.sh so PATH / LD_LIBRARY_PATH exports persist.
# Idempotent: each component is skipped if already present.
#
# Build steps run in subshells to avoid CWD pollution between installers.
# =============================================================================

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
UCX_HOME="${UCX_HOME:-/usr/local/ucx}"
RIXL_HOME="${RIXL_HOME:-/usr/local/rixl}"

_SETUP_START=$(date +%s)
_SETUP_INSTALLED=()

git_clone_retry() {
    local url="$1" dest="$2" max_tries=3 try=1
    while (( try <= max_tries )); do
        if git clone --quiet "$url" "$dest" 2>/dev/null; then return 0; fi
        echo "[SETUP] git clone attempt $try/$max_tries failed for $url, retrying in 10s..."
        rm -rf "$dest"
        sleep 10
        (( try++ ))
    done
    echo "[SETUP] git clone failed after $max_tries attempts: $url"
    return 1
}


# ---------------------------------------------------------------------------
# 5. Container RDMA/net tools
#    - ibv_devinfo comes from ibverbs-utils
#    - iproute2 provides the `ip` command
#    Used for in-container NIC/RDMA validation and routing checks.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 6b. amd-quark (MXFP4 quantization support for Kimi-K2.5-MXFP4 and similar)
#     Required due to ROCm vLLM missing the quark dependency:
#     https://github.com/vllm-project/vllm/issues/35633
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 8. Broadcom bnxt RDMA userspace libraries (libbnxt_re verbs provider)
#     Required on nodes with Broadcom Thor2 NICs (bcm5760x) when the base
#     image does not ship the bnxt_re verbs provider.
# ---------------------------------------------------------------------------
install_bnxt_rdma() {
    local existing
    existing=$(find /usr/local/lib /usr/lib64 /usr/lib -name "libbnxt_re-rdmav*.so" 2>/dev/null)
    if [[ -n "$existing" ]]; then
        echo "[SETUP] Existing bnxt RDMA libraries found (will override):"
        for lib in $existing; do
            echo "  $lib ($(ls -l "$lib" 2>/dev/null | awk '{print $5, $6, $7, $8}'))"
        done
    fi

    echo "[SETUP] Installing bnxt RDMA build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        wget unzip autoconf automake libtool pkg-config \
        libibverbs-dev librdmacm-dev ibverbs-utils \
        && rm -rf /var/lib/apt/lists/*

    echo "[SETUP] Downloading and building Broadcom bnxt RDMA userspace libraries..."
    (
        set -e
        cd /tmp
        wget -q https://docs.broadcom.com/docs-and-downloads/ethernet-network-adapters/NXE/Thor2/GCA1/bcm5760x_230.2.52.0a.zip
        unzip -q bcm5760x_230.2.52.0a.zip
        cd bcm5760x_230.2.52.0a/drivers_linux/bnxt_rocelib/
        results=$(find . -name "libbnxt*.tar.gz")
        tar -xf $results
        untar_dir=$(find . -maxdepth 1 -type d -name "libbnxt*" ! -name "*.tar.gz" | head -n 1)
        cd "$untar_dir"
        sh autogen.sh
        ./configure
        make -j"$(nproc)"
        find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \; 2>/dev/null || true
        make install all
        echo /usr/local/lib >> /etc/ld.so.conf
        ldconfig
        cp -f bnxt_re.driver /etc/libibverbs.d/
    )
    rm -rf /tmp/bcm5760x_230.2.52.0a /tmp/bcm5760x_230.2.52.0a.zip

    if ! ibv_devices 2>/dev/null; then
        echo "[SETUP] WARN: ibv_devices failed after bnxt install (may be OK if no Broadcom NIC on this node)"
    fi
    _SETUP_INSTALLED+=("bnxt-rdma")
}

# =============================================================================
# Run installers
# =============================================================================

install_bnxt_rdma
install_recipe_deps
install_amd_quark

# =============================================================================
# Export paths (persists for server.sh since this file is sourced)
# =============================================================================

export ROCM_PATH="${ROCM_PATH}"
export UCX_HOME="${UCX_HOME}"
export RIXL_HOME="${RIXL_HOME}"
export PATH="${UCX_HOME}/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH}"
export LD_LIBRARY_PATH="${UCX_HOME}/lib:${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present (${_SETUP_END}s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
