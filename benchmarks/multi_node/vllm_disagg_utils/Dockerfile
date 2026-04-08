# Build with `docker build -t vllm-router-rocm:0.1.0 --no-cache .` to use in run_P.sh

# Router builder stage: Ubuntu 22.04 matches the vllm-openai-rocm base image,
# ensuring shared libraries (OpenSSL 3.x, glibc 2.35) are compatible.
FROM ubuntu:22.04 AS rust-builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libzmq3-dev \
    protobuf-compiler \
    python3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:$PATH"

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --branch feat/moriio-support https://github.com/mpashkovskii/router.git .

RUN cargo build --release

FROM vllm/vllm-openai-rocm:v0.18.0

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

COPY --from=rust-builder /app/target/release/vllm-router /app/vllm-router
