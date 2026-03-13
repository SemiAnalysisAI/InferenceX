#!/bin/bash
# =============================================================================
# setup_deps.sh — Install missing vLLM disagg dependencies at container start.
#
# Base image: vllm/vllm-openai-rocm:v0.17.1
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
        git clone --quiet https://github.com/ROCm/ucx.git && cd ucx
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
        git clone --quiet https://github.com/ROCm/RIXL.git /opt/rixl && cd /opt/rixl
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
# 5. vllm-router (Rust-based proxy for PD disaggregation)
#    Only needed on NODE_RANK=0 (proxy node).
# ---------------------------------------------------------------------------
install_vllm_router() {
    if pip show vllm-router &>/dev/null; then
        echo "[SETUP] vllm-router already installed"
        return 0
    fi

    echo "[SETUP] Installing Rust toolchain..."
    if ! command -v cargo &>/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        export PATH="/root/.cargo/bin:${PATH}"
    fi

    echo "[SETUP] Installing vllm-router via pip..."
    pip install --quiet vllm-router

    if ! pip show vllm-router &>/dev/null; then
        echo "[SETUP] ERROR: vllm-router install failed"; exit 1
    fi
    _SETUP_INSTALLED+=("vllm-router")
}

# ---------------------------------------------------------------------------
# 6. MoRI (Modular RDMA Interface — EP dispatch/combine kernels for MoE)
#    Required for --all2all-backend mori (Expert Parallelism via RDMA).
#    GPU kernels are JIT-compiled on first use; no hipcc needed at install.
# ---------------------------------------------------------------------------
install_mori() {
    if python3 -c "import mori" 2>/dev/null; then
        echo "[SETUP] MoRI Python bindings already present"
        return 0
    fi

    echo "[SETUP] Installing MoRI build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        libopenmpi-dev openmpi-bin libpci-dev \
        && rm -rf /var/lib/apt/lists/*

    echo "[SETUP] Building MoRI from source (ROCm/mori @ b645fc8)..."
    (
        set -e
        git clone --quiet https://github.com/ROCm/mori.git /opt/mori && cd /opt/mori
        git checkout b645fc8
        pip install --quiet .
    )
    rm -rf /opt/mori

    if ! python3 -c "import mori" 2>/dev/null; then
        echo "[SETUP] ERROR: MoRI build failed"; exit 1
    fi
    _SETUP_INSTALLED+=("MoRI")
}

# ---------------------------------------------------------------------------
# 7. Patch vLLM v0.17.1 MoRI-EP + FP8 incompatibility
#    v0.17.1 asserts MoRI requires AITER fused_moe, but AITER's FP8 kernel
#    uses defer_input_quant=True which MoRI's prepare/finalize rejects.
#    Patch: remove both the AITER requirement assertion and the
#    defer_input_quant NotImplementedError so non-AITER kernels work.
# ---------------------------------------------------------------------------
patch_mori_fp8_compat() {
    python3 -c '
import re, os, sys
patched = []

# 1. Patch layer.py: remove multi-line AITER assertion for MoRI
try:
    import vllm.model_executor.layers.fused_moe.layer as lm
    f = lm.__file__
    src = open(f).read()
    if "Mori needs to be used with aiter" in src:
        new = re.sub(
            r"assert self\.rocm_aiter_fmoe_enabled,\s*\([^)]*Mori needs[^)]*\)",
            "pass  # [PATCHED] AITER requirement removed for MoRI-EP + FP8",
            src, flags=re.DOTALL)
        if new != src:
            open(f, "w").write(new)
            patched.append("layer.py")
except Exception as e:
    print(f"[SETUP] WARN patch layer.py: {e}", file=sys.stderr)

# 2. Patch mori_prepare_finalize.py: remove defer_input_quant restriction
try:
    import vllm.model_executor.layers.fused_moe.mori_prepare_finalize as mm
    f = mm.__file__
    src = open(f).read()
    if "defer_input_quant" in src:
        new = re.sub(
            r"raise NotImplementedError\([^)]*defer_input_quant[^)]*\)",
            "pass  # [PATCHED] defer_input_quant check removed for MoRI-EP + FP8",
            src)
        if new != src:
            open(f, "w").write(new)
            patched.append("mori_prepare_finalize.py")
except Exception as e:
    print(f"[SETUP] WARN patch mori_pf: {e}", file=sys.stderr)

if patched:
    print(f"[SETUP] Patched: {chr(44).join(patched)}")
else:
    print("[SETUP] No MoRI-FP8 patches needed")
'
    _SETUP_INSTALLED+=("MoRI-FP8-patch")
}

# =============================================================================
# Run installers
# =============================================================================

install_ucx
install_rixl
install_etcd
install_libionic
install_mori
patch_mori_fp8_compat

if [[ "${NODE_RANK:-0}" -eq 0 ]]; then
    install_vllm_router
fi

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
