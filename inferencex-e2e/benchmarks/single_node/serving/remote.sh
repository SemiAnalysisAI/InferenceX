#!/usr/bin/env bash
# One remote-endpoint serving point inside its CPU-only Slurm allocation. infx/launch/drivers/serving.py
# sbatches this file with --chdir=<workspace>; every input is env (see the driver's job_env).
# No engine, sidecars or ports: one client step in the client image probes the endpoint and replays
# the scenario against it. SERVING_API_KEY reaches the step through the environment; never trace it.
set -euo pipefail

: "${SERVING_ENGINE_IMAGE:?}" "${SERVING_HF_CACHE_HOST:?}"
WS="$PWD"
mkdir -p "$WS/results"
HF_TARGET="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"

srun --container-image="$SERVING_ENGINE_IMAGE" \
    --container-mounts="$WS:/workspace,$SERVING_HF_CACHE_HOST:${HF_TARGET%/}" \
    --no-container-mount-home --container-remap-root --container-writable \
    --container-workdir=/workspace --no-container-entrypoint \
    bash -c 'set -eo pipefail
: "${RESULT_DIR:?}" "${MODEL:?}"
export PYTHONPATH=/workspace AGENTIC_OUTPUT_DIR=/workspace
mkdir -p "$RESULT_DIR"
printf "%s" "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"
python3 -m infx.serving.client prepare
python3 -m infx.serving.client run'
