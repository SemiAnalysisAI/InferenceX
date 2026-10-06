ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY --chmod=0755 amd-metrics-exporter /home/amd/bin/server
ENV AMD_GPU_GET_CACHE_TTL=0s
