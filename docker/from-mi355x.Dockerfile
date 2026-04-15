# Build MI325X bnxt image from the MI355X Qwen3.5 disagg image.
# Copies the working SGLang code and adds Broadcom bnxt_rocelib.
#
# The MI355X image has commit 20fc36a2e that supports non-MLA PD disagg.
# This commit was force-pushed out of upstream, so we extract from the image.

ARG BASE_IMAGE=rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2
FROM ${BASE_IMAGE}

# Install RDMA build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libibumad-dev rdma-core ibverbs-utils infiniband-diags \
    gcc make libtool autoconf librdmacm-dev rdmacm-utils \
    perftest ethtool libibverbs-dev strace unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Broadcom bnxt_rocelib
ARG BCM_DRIVER=bcm5760x_231.2.63.0a.zip
COPY ${BCM_DRIVER} /tmp/${BCM_DRIVER}
RUN cd /tmp && \
    case "${BCM_DRIVER}" in \
      *.zip) unzip -o ./${BCM_DRIVER} ;; \
      *.tar.gz) tar zxf ./${BCM_DRIVER} ;; \
    esac && \
    DIR_NAME="${BCM_DRIVER%.*}" && \
    case "${BCM_DRIVER}" in *.tar.gz) DIR_NAME="${BCM_DRIVER%.tar.gz}" ;; esac && \
    cd /tmp/${DIR_NAME}/drivers_linux/bnxt_rocelib && \
    BCM_LIB=$(ls -1 *.tar.gz) && \
    tar zxf ${BCM_LIB} && \
    cd "${BCM_LIB%.tar.gz}" && \
    sh ./autogen.sh && \
    sh ./configure && \
    make -j8 && \
    find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \; 2>/dev/null || true && \
    make install && \
    echo /usr/local/lib >> /etc/ld.so.conf && \
    ldconfig && \
    rm -rf /tmp/${BCM_DRIVER} /tmp/${DIR_NAME} && \
    echo "bnxt_rocelib installed"

# Install tiktoken for GLM-5 tokenizer support
RUN pip install --no-cache-dir tiktoken sentencepiece
