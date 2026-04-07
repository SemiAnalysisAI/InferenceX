FROM vllm/vllm-openai-rocm:v0.18.1

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

COPY vllm-router /app/
