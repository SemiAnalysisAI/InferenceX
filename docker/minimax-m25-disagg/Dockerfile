# MiniMax M2.5 PD Disaggregation Docker Image
#
# Extends the public vLLM ROCm image with:
#   1. WideEP + MoRI support for MiniMax M2.5 (minimax_m2.py patch)
#   2. Pre-installed runtime deps (UCX, RIXL, etcd, MoRI)
#   3. Disagg orchestration scripts baked in
#
# Build:
#   docker build -t minimax-m25-disagg:latest -f docker/minimax-m25-disagg/Dockerfile .
#
# The image still sources setup_deps.sh at startup for idempotent patching
# (scheduler KV reaper, MoRI-IO read mode, etc.) but the heavy build steps
# (UCX, RIXL) are cached in the image layer.

ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.18.0
FROM ${BASE_IMAGE}

ARG ROCM_PATH=/opt/rocm
ARG UCX_HOME=/usr/local/ucx
ARG RIXL_HOME=/usr/local/rixl

# ----------------------------------------------------------------
# 1. Patch vLLM: MiniMax M2.5 WideEP + MoRI + EPLB support
# ----------------------------------------------------------------
COPY docker/minimax-m25-disagg/patches/minimax_m2.py /tmp/patches/minimax_m2.py
RUN VLLM_MODELS=$(python3 -c "import vllm.model_executor.models; import os; print(os.path.dirname(vllm.model_executor.models.__file__))") && \
    cp /tmp/patches/minimax_m2.py "${VLLM_MODELS}/minimax_m2.py" && \
    echo "[DOCKER] Patched minimax_m2.py -> ${VLLM_MODELS}/minimax_m2.py" && \
    rm -rf /tmp/patches

# ----------------------------------------------------------------
# 2. Pre-install UCX build deps (speeds up setup_deps.sh at runtime)
# ----------------------------------------------------------------
RUN apt-get update -q -y && apt-get install -q -y --no-install-recommends \
        autoconf automake libtool pkg-config \
        librdmacm-dev rdmacm-utils libibverbs-dev ibverbs-utils ibverbs-providers \
        infiniband-diags perftest ethtool rdma-core strace \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------
# 3. Pre-build UCX (ROCm fork) — the longest step in setup_deps.sh
# ----------------------------------------------------------------
RUN git clone --quiet https://github.com/ROCm/ucx.git /usr/local/src/ucx && \
    cd /usr/local/src/ucx && \
    git checkout da3fac2a && \
    ./autogen.sh && mkdir -p build && cd build && \
    ../configure \
        --prefix="${UCX_HOME}" \
        --enable-shared --disable-static \
        --disable-doxygen-doc --enable-optimizations \
        --enable-devel-headers --enable-mt \
        --with-rocm="${ROCM_PATH}" --with-verbs --with-dm && \
    make -j"$(nproc)" && make install && \
    rm -rf /usr/local/src/ucx

# ----------------------------------------------------------------
# 4. Pre-install etcd
# ----------------------------------------------------------------
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then ETCD_ARCH=amd64; else ETCD_ARCH=arm64; fi && \
    ETCD_VER=v3.5.21 && \
    curl -fsSL "https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-${ETCD_ARCH}.tar.gz" | \
    tar xz -C /usr/local/bin --strip-components=1 \
        "etcd-${ETCD_VER}-linux-${ETCD_ARCH}/etcd" \
        "etcd-${ETCD_VER}-linux-${ETCD_ARCH}/etcdctl" && \
    etcd --version

# ----------------------------------------------------------------
# 5. Pre-install RIXL (Nixl KV transfer)
# ----------------------------------------------------------------
RUN pip install --no-cache-dir nixl && \
    python3 -c "import nixl; print('RIXL installed:', nixl.__file__)" || \
    echo "[DOCKER] WARN: nixl pip install failed, will fallback to setup_deps.sh"

# ----------------------------------------------------------------
# 6. Copy disagg orchestration scripts into the image
# ----------------------------------------------------------------
COPY benchmarks/multi_node/vllm_disagg_utils/ /opt/vllm_disagg/
COPY benchmarks/multi_node/minimaxm2.5_fp8_mi355x_vllm-disagg.sh /opt/vllm_disagg/

# ----------------------------------------------------------------
# 7. Environment
# ----------------------------------------------------------------
ENV UCX_HOME=${UCX_HOME} \
    RIXL_HOME=${RIXL_HOME} \
    ROCM_PATH=${ROCM_PATH} \
    PATH="${UCX_HOME}/bin:/usr/local/bin:${PATH}" \
    LD_LIBRARY_PATH="${UCX_HOME}/lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPYCACHEPREFIX=/tmp/pycache

WORKDIR /workspace
