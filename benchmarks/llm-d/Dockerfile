# Combined image for the InferenceX llm-d-vllm framework.
#
# Base = ghcr.io/llm-d/llm-d-cuda which already ships vLLM + DeepEP +
# NVSHMEM + GDRCopy. We add the EPP, the routing-sidecar, and Envoy on top
# so every node in a SLURM allocation can play any role (prefill, decode,
# or coordinator) from a single image.
#
# Configs (epp-config.yaml, envoy.yaml, per-topology recipes) are NOT
# baked in. They are mounted at runtime by job.slurm so config-only
# iteration does not require an image rebuild. See
# benchmarks/multi_node/llm-d/job.slurm for the expected mount layout.

FROM ghcr.io/llm-d/llm-d-cuda:v0.7.0

COPY --from=ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:main \
       /app/epp /usr/local/bin/epp

COPY --from=ghcr.io/llm-d/llm-d-router-disagg-sidecar-dev:main \
       /app/pd-sidecar /usr/local/bin/pd-sidecar

COPY --from=envoyproxy/envoy:distroless-v1.33.2 \
     /usr/local/bin/envoy /usr/local/bin/
