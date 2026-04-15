# Thin layer that adds Broadcom bnxt_rocelib RDMA support to any SGLang ROCm image.
# Usage:
#   docker build --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.9-rocm700-mi30x \
#     -t semianalysiswork/sgl-bnxt-cdna3:v0.5.9-bnxt-lite \
#     -f add-bnxt.Dockerfile .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install RDMA build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libibumad-dev rdma-core ibverbs-utils infiniband-diags \
    gcc make libtool autoconf librdmacm-dev rdmacm-utils \
    perftest ethtool libibverbs-dev strace \
    && rm -rf /var/lib/apt/lists/*

# Install Broadcom bnxt_rocelib
ARG BCM_DRIVER=bcm5760x_231.2.63.0a.zip
COPY ${BCM_DRIVER} /tmp/${BCM_DRIVER}
RUN cd /tmp && \
    case "${BCM_DRIVER}" in \
      *.zip) apt-get update && apt-get install -y unzip && unzip -o ./${BCM_DRIVER} ;; \
      *.tar.gz) tar zxf ./${BCM_DRIVER} ;; \
      *) echo "ERROR: unsupported archive: ${BCM_DRIVER}" && exit 1 ;; \
    esac && \
    DIR_NAME="${BCM_DRIVER%.*}" && \
    # Handle double extension (.tar.gz)
    case "${BCM_DRIVER}" in *.tar.gz) DIR_NAME="${BCM_DRIVER%.tar.gz}" ;; esac && \
    cd /tmp/${DIR_NAME}/drivers_linux/bnxt_rocelib && \
    BCM_LIB=$(ls -1 *.tar.gz) && \
    tar zxf ${BCM_LIB} && \
    cd "${BCM_LIB%.tar.gz}" && \
    sh ./autogen.sh && \
    sh ./configure && \
    make -j8 && \
    # Backup inbox drivers and install
    find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \; 2>/dev/null || true && \
    make install && \
    echo /usr/local/lib >> /etc/ld.so.conf && \
    ldconfig && \
    # Cleanup
    rm -rf /tmp/${BCM_DRIVER} /tmp/${DIR_NAME} && \
    echo "bnxt_rocelib installed successfully"

# Upgrade transformers for glm_moe_dsa and qwen3_5_moe model type support
RUN pip install --no-cache-dir -U \
    "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2" \
    && python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
