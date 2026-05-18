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
# 1. UCX (ROCm fork — required for GPU-direct RDMA via Nixl)
# ---------------------------------------------------------------------------
install_ucx() {
    if [[ -x "${UCX_HOME}/bin/ucx_info" ]]; then
        echo "[SETUP] UCX already present at ${UCX_HOME}"
        return 0
    fi

    echo "[SETUP] Installing UCX build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        autoconf automake libtool pkg-config \
        librdmacm-dev rdmacm-utils libibverbs-dev ibverbs-utils ibverbs-providers \
        infiniband-diags perftest ethtool rdma-core strace \
        && rm -rf /var/lib/apt/lists/*

    echo "[SETUP] Building UCX from source (ROCm/ucx @ da3fac2a)..."
    (
        set -e
        mkdir -p /usr/local/src && cd /usr/local/src
        git_clone_retry https://github.com/ROCm/ucx.git ucx && cd ucx
        git checkout da3fac2a
        ./autogen.sh && mkdir -p build && cd build
        ../configure \
            --prefix="${UCX_HOME}" \
            --enable-shared --disable-static \
            --disable-doxygen-doc --enable-optimizations \
            --enable-devel-headers --enable-mt \
            --with-rocm="${ROCM_PATH}" --with-verbs --with-dm
        make -j"$(nproc)" && make install
    )
    rm -rf /usr/local/src/ucx

    if [[ ! -x "${UCX_HOME}/bin/ucx_info" ]]; then
        echo "[SETUP] ERROR: UCX build failed"; exit 1
    fi
    _SETUP_INSTALLED+=("UCX")
}

# ---------------------------------------------------------------------------
# 2. RIXL (ROCm fork of NIXL — KV cache transfer for disaggregated vLLM)
# ---------------------------------------------------------------------------
install_rixl() {
    if python3 -c "import rixl" 2>/dev/null; then
        echo "[SETUP] RIXL Python bindings already present"
        return 0
    fi

    echo "[SETUP] Installing RIXL build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
        libcpprest-dev libaio-dev \
        && rm -rf /var/lib/apt/lists/*
    pip3 install --quiet meson "pybind11[global]"

    echo "[SETUP] Building RIXL from source (ROCm/RIXL @ f33a5599)..."
    (
        set -e
        git_clone_retry https://github.com/ROCm/RIXL.git /opt/rixl && cd /opt/rixl
        git checkout f33a5599
        meson setup build --prefix="${RIXL_HOME}" \
            -Ducx_path="${UCX_HOME}" \
            -Drocm_path="${ROCM_PATH}"
        cd build && ninja && ninja install
        cd /opt/rixl
        pip install --quiet \
            --config-settings=setup-args="-Drocm_path=${ROCM_PATH}" \
            --config-settings=setup-args="-Ducx_path=${UCX_HOME}" .
    )
    rm -rf /opt/rixl

    if ! python3 -c "import rixl" 2>/dev/null; then
        echo "[SETUP] ERROR: RIXL build failed"; exit 1
    fi
    _SETUP_INSTALLED+=("RIXL")
}

# ---------------------------------------------------------------------------
# 3. etcd (distributed KV store for vLLM disagg service discovery)
# ---------------------------------------------------------------------------
install_etcd() {
    if [[ -x /usr/local/bin/etcd/etcd ]]; then
        echo "[SETUP] etcd already present"
        return 0
    fi

    local version="v3.6.0-rc.5"
    echo "[SETUP] Downloading etcd ${version}..."
    wget -q "https://github.com/etcd-io/etcd/releases/download/${version}/etcd-${version}-linux-amd64.tar.gz" \
        -O /tmp/etcd.tar.gz
    mkdir -p /usr/local/bin/etcd
    tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1
    rm /tmp/etcd.tar.gz
    _SETUP_INSTALLED+=("etcd")
}

# ---------------------------------------------------------------------------
# 4. libionic1 (Pensando ionic RDMA verbs provider for RoCEv2 KV transfer)
#    Harmless on non-Pensando nodes (shared lib is simply unused).
# ---------------------------------------------------------------------------
install_libionic() {
    if dpkg -l libionic1 2>/dev/null | grep -q '^ii'; then
        echo "[SETUP] libionic1 already installed"
        return 0
    fi

    echo "[SETUP] Downloading and installing libionic1..."
    wget -q "https://repo.radeon.com/amdainic/pensando/ubuntu/1.117.5/pool/main/r/rdma-core/libionic1_54.0-149.g3304be71_amd64.deb" \
        -O /tmp/libionic1.deb
    dpkg -i /tmp/libionic1.deb || true
    rm -f /tmp/libionic1.deb
    _SETUP_INSTALLED+=("libionic1")
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
# 6. MoRI (Modular RDMA Interface — EP dispatch/combine kernels for MoE)
#    Required for --all2all-backend mori (Expert Parallelism via RDMA).
#    GPU kernels are JIT-compiled on first use; no hipcc needed at install.
#
#    v0.18.0 ships MoRI 0.1.dev185+g2d02c6a98, but it STILL has the PCI
#    topology bug (TopoSystemPci::Load assertion failure on Broadcom
#    PEX890xx switches).  Always rebuild from our target commit b645fc8
#    which includes the dsp2dev subordinate-range fix.
# ---------------------------------------------------------------------------
install_mori() {
    local MORI_TARGET_COMMIT="b645fc8"
    local MORI_MARKER="/usr/local/lib/python3.*/dist-packages/.mori_commit_${MORI_TARGET_COMMIT}"

    if ls $MORI_MARKER &>/dev/null; then
        echo "[SETUP] MoRI @ $MORI_TARGET_COMMIT already installed (marker found)"
        return 0
    fi

    echo "[SETUP] Installing MoRI build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        libopenmpi-dev openmpi-bin libpci-dev \
        && rm -rf /var/lib/apt/lists/*

    echo "[SETUP] Building MoRI from source (ROCm/mori @ $MORI_TARGET_COMMIT)..."
    echo "[SETUP]   (overriding image-provided version to fix PCI topology bug)"
    (
        set -e
        git_clone_retry https://github.com/ROCm/mori.git /opt/mori && cd /opt/mori
        git checkout "$MORI_TARGET_COMMIT"
        pip install --quiet --force-reinstall .
    )
    rm -rf /opt/mori

    if ! python3 -c "import mori" 2>/dev/null; then
        echo "[SETUP] ERROR: MoRI build failed"; exit 1
    fi
    touch $(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/.mori_commit_${MORI_TARGET_COMMIT}
    _SETUP_INSTALLED+=("MoRI@$MORI_TARGET_COMMIT")
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
# 7. Patch vLLM MoRI-EP + FP8 incompatibility (present in v0.17.1 & v0.18.0)
#    vLLM asserts MoRI requires AITER fused_moe, but AITER's FP8 kernel
#    uses defer_input_quant=True which MoRI's prepare/finalize rejects.
#    Patch: remove both the AITER requirement assertion and the
#    defer_input_quant NotImplementedError so non-AITER kernels work.
# ---------------------------------------------------------------------------
patch_mori_fp8_compat() {
    python3 -c '
import re, os, sys
patched = []

# Patch layer.py: remove AITER requirement assertion(s) for MoRI
try:
    import vllm.model_executor.layers.fused_moe.layer as lm
    f = lm.__file__
    src = open(f).read()
    if "[PATCHED] AITER requirement removed for MoRI-EP + FP8" in src:
        print("[SETUP] layer.py MoRI-FP8 patch already applied")
    elif "Mori needs to be used with aiter" in src:
        # v0.19+: two consecutive assertions inside `if self.moe_config.use_mori_kernels:`
        new = re.sub(
            r"assert self\.rocm_aiter_fmoe_enabled,\s*\([^)]*Mori needs[^)]*\)\s*"
            r"assert not self\.aiter_fmoe_shared_expert_enabled,\s*\([^)]*\)",
            "pass  # [PATCHED] AITER requirement removed for MoRI-EP + FP8",
            src, flags=re.DOTALL)
        if new == src:
            # v0.17.1/v0.18.0: only the first assertion existed
            new = re.sub(
                r"assert self\.rocm_aiter_fmoe_enabled,\s*\([^)]*Mori needs[^)]*\)",
                "pass  # [PATCHED] AITER requirement removed for MoRI-EP + FP8",
                src, flags=re.DOTALL)
        if new != src:
            open(f, "w").write(new)
            patched.append("layer.py")
        else:
            print("[SETUP] ERROR: layer.py pattern found but regex had no effect", file=sys.stderr)
            sys.exit(1)
    else:
        print("[SETUP] ERROR: layer.py AITER assertion pattern not found — vLLM API may have changed", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"[SETUP] ERROR patch layer.py: {e}", file=sys.stderr)
    sys.exit(1)

# prepare_finalize/mori.py (v0.19+) already handles defer_input_quant correctly
# (skips FP8 quant when True). No patch needed for that file.
# Added in 0.18.1: https://github.com/vllm-project/vllm/commit/6a9cceb219fcbd6b1eb540ddfdc77ec160f0e209

if patched:
    print(f"[SETUP] Patched: {chr(44).join(patched)}")
else:
    print("[SETUP] No MoRI-FP8 patches needed")
' || exit 1
    _SETUP_INSTALLED+=("MoRI-FP8-patch")
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
patch_mori_fp8_compat

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
