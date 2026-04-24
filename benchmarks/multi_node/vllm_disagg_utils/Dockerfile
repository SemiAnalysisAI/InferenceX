# Image combining:
#  - nightly vllm
#  - router PRs https://github.com/vllm-project/router/pull/138 + https://github.com/vllm-project/router/pull/114
#    for MoRI and streaming support 
# Also installs
#  - RDMA libs for Thor2
#  - libssl1.1 required for router. This is in the original router image but since we copy the router binary into the vllm image we need to reinstall it. 
# 
# Build with `docker build -t vllm-router-rocm:0.1.0 --no-cache .` to use in run_P.sh

FROM ghcr.io/simondanielsson/vllm-router:dev-streaming-cn-cjy AS router

# From 4/24/2026, 0.19+
FROM vllm/vllm-openai-rocm:nightly-100c7b65e7579c8caf4ee0b04a6410b2796b905c

# RDMA drivers for broadcom devices
RUN apt-get update && apt-get install -y autoconf libibverbs-dev ibverbs-utils libtool unzip wget
RUN wget https://docs.broadcom.com/docs-and-downloads/ethernet-network-adapters/NXE/Thor2/GCA1/bcm5760x_230.2.52.0a.zip && \
    unzip bcm5760x_230.2.52.0a.zip && \
    cd bcm5760x_230.2.52.0a/drivers_linux/bnxt_rocelib/ && \
    results=$(find -name "libbnxt*.tar.gz") && tar -xf $results && \
    untar_dir=$(find . -maxdepth 1 -type d -name "libbnxt*" ! -name "*.tar.gz" | head -n 1) && cd $untar_dir && sh autogen.sh && ./configure && make && \
    find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \; && \
    make install all && sudo sh -c "echo /usr/local/lib >> /etc/ld.so.conf" && \
    sudo ldconfig && \
    cp -f bnxt_re.driver /etc/libibverbs.d/ && \
    ibv_devices
RUN rm -rf bcm5760x_230.2.52.0a && \
    rm -rf bcm5760x_230.2.52.0a.zip

# libssl.so.1.1 is absent in Ubuntu 22.04+ (the vLLM base); router binary needs it
RUN apt-get install -y libssl1.1 2>/dev/null || \
    ( wget -q http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb -O /tmp/libssl1.1.deb && \
      dpkg -i /tmp/libssl1.1.deb && \
      rm /tmp/libssl1.1.deb )

COPY --from=router /usr/local/bin/vllm-router /app/vllm-router
