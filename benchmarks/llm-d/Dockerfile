# Combined image for the InferenceX llmd-vllm framework.
#
# Base = vllm/vllm-openai (vLLM with the OpenAI-compatible API server).
# We add the EPP, the routing-sidecar, and Envoy on top so every node in
# a SLURM allocation can play any role (prefill, decode, or coordinator)
# from a single image.
#
# Configs (epp-config.yaml, envoy.yaml, per-topology recipes) are NOT
# baked in. They are mounted at runtime by job.slurm so config-only
# iteration does not require an image rebuild. See
# benchmarks/multi_node/llm-d/job.slurm for the expected mount layout.

# All FROM-referenced images are declared as global-scope ARGs. BuildKit
# does NOT expand a build-arg inside `COPY --from=${VAR}` directly ("variable
# expansion is not supported for --from"); the supported pattern is to bind
# each ARG to a named FROM stage in the global scope and COPY --from that
# stage alias. Defaults MUST match benchmarks/llm-d/binaries.env, which is the
# single source of truth (it also drives the mounted-binary path via
# extract-binaries.sh). Override at build time from that file, e.g.:
#   source benchmarks/llm-d/binaries.env
#   docker build \
#     --build-arg EPP_FROM_IMAGE="$EPP_FROM_IMAGE" \
#     --build-arg ROUTING_SIDECAR_IMAGE="$ROUTING_SIDECAR_IMAGE" \
#     --build-arg ENVOY_FROM_IMAGE="$ENVOY_FROM_IMAGE" ...
ARG VLLM_BASE=vllm/vllm-openai:v0.23.0
ARG EPP_FROM_IMAGE=ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.9.0
ARG ROUTING_SIDECAR_IMAGE=ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.9.0
ARG ENVOY_FROM_IMAGE=envoyproxy/envoy:distroless-v1.33.2

# Stage aliases for the three add-on binary sources (see note above).
FROM ${EPP_FROM_IMAGE} AS epp_src
FROM ${ROUTING_SIDECAR_IMAGE} AS sidecar_src
FROM ${ENVOY_FROM_IMAGE} AS envoy_src

FROM ${VLLM_BASE}
# Force-upgrade nvidia-cutlass-dsl only when explicitly requested. The
# v0.22 base needed it because its bundled wheel lacked cute.arch.fmin
# for the DSV4 sparse-attention code; stock v0.23.0 (what upstream
# wide-ep-lws runs) ships a working cutlass, so we leave it untouched
# there. Build the v0.22 base with --build-arg UPGRADE_CUTLASS_DSL=1.
ARG UPGRADE_CUTLASS_DSL=0
RUN if [ "$UPGRADE_CUTLASS_DSL" = "1" ]; then \
        pip install --no-cache-dir --upgrade nvidia-cutlass-dsl; \
    fi

COPY --from=epp_src \
       /app/epp /usr/local/bin/epp

COPY --from=sidecar_src \
       /app/pd-sidecar /usr/local/bin/pd-sidecar

COPY --from=envoy_src \
     /usr/local/bin/envoy /usr/local/bin/
