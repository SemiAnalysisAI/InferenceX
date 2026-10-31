#!/usr/bin/env bash
# One serving-YAML benchmark point inside its Slurm allocation. infx/launch/drivers/serving.py
# sbatches this file with --chdir=<workspace>; every input is env (see the driver's job_env).
# Steps: DCGM, then the optional storage and router sidecars as --overlap steps, then the engine
# step (bench.sh: engine replicas + client) in the foreground. Ports are picked free on this node
# at run time: pyxis shares the host network with other users' jobs, and nodes are never exclusive.
set -euo pipefail

: "${SERVING_ENGINE_IMAGE:?}" "${SERVING_DCGM_IMAGE:?}" "${SERVING_HF_CACHE_HOST:?}"
WS="$PWD"
RESULTS="$WS/results"
mkdir -p "$RESULTS"
rm -rf "$RESULTS/.dynkv" "$RESULTS/.router_exited" "$RESULTS/.storage_exited"

PICKED=()
pick() {  # pick <var>: a free TCP port in 20000-32767 (below the ephemeral range), not picked before
    local _free  # not "port": a caller's `pick port` must set its own variable
    _free=$(python3 - "${PICKED[@]}" <<'PY'
import random, socket, sys
taken = set(sys.argv[1:])
for port in random.sample(range(20000, 32768), 2000):
    if str(port) in taken:
        continue
    with socket.socket() as probe:
        try:
            probe.bind(("", port))
        except OSError:
            continue
    print(port)
    break
PY
)
    [[ -n "$_free" ]] || { echo "ERROR: no free port in 20000-32767" >&2; exit 1; }
    PICKED+=("$_free")
    printf -v "$1" '%s' "$_free"
}

HF_TARGET="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
CONTAINER=(
    --container-mounts="$WS:/workspace,$SERVING_HF_CACHE_HOST:${HF_TARGET%/}"
    --no-container-mount-home --container-remap-root --container-writable
    --container-workdir=/workspace --no-container-entrypoint
)
SIDECARS=()
cleanup() {
    kill "${SIDECARS[@]}" 2>/dev/null || true
    rm -rf "$RESULTS/.dynkv"
}
trap cleanup EXIT

pick DCGM_PORT
srun --overlap --container-image="$SERVING_DCGM_IMAGE" "${CONTAINER[@]}" \
    dcgm-exporter -a ":$DCGM_PORT" > "$RESULTS/dcgm.log" 2>&1 &
SIDECARS+=("$!")
export AIPERF_GPU_TELEMETRY_URL="http://localhost:$DCGM_PORT/metrics"

if [[ -n "${STORAGE_CMD:-}" ]]; then
    pick STORAGE_MASTER_PORT
    pick STORAGE_META_PORT
    export STORAGE_MASTER_PORT STORAGE_META_PORT
    (
        rc=0
        NVIDIA_VISIBLE_DEVICES=none srun --overlap --container-image="$STORAGE_IMAGE" "${CONTAINER[@]}" \
            bash benchmarks/single_node/serving/sidecar.sh storage > "$RESULTS/storage.log" 2>&1 || rc=$?
        echo "exit $rc" > "$RESULTS/.storage_exited"
    ) &
    SIDECARS+=("$!")
    SECONDS=0
    until (exec 9<>"/dev/tcp/127.0.0.1/$STORAGE_MASTER_PORT") 2>/dev/null; do
        if [[ -f "$RESULTS/.storage_exited" ]]; then
            echo "ERROR: storage sidecar exited before readiness; see results/storage.log" >&2
            exit 1
        fi
        if (( SECONDS > 300 )); then
            echo "ERROR: storage sidecar did not open port $STORAGE_MASTER_PORT within 300 s" >&2
            exit 1
        fi
        sleep 2
    done
fi

pick PORT
export PORT
if [[ -n "${ROUTER_CMD:-}" ]]; then
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/workspace/results/.dynkv DYN_SYSTEM_ENABLED=true
        pick DYN_SYSTEM_PORT  # Dynamo parses it as i16: 20000-32767 fits
    else
        ENGINE_PORTS=""
        for ((replica = 0; replica < ${REPLICAS:-1}; replica++)); do
            pick port
            ENGINE_PORTS+="${ENGINE_PORTS:+ }$port"
        done
        export ENGINE_PORTS ENGINE_PORT="${ENGINE_PORTS%% *}"
    fi
    if [[ "$ROUTER_CMD$ROUTER_ENV" == *'{{router_metrics_port}}'* ]]; then
        pick ROUTER_METRICS_PORT  # bench.sh scrapes it; a fixed metrics_port would collide on a shared node
        export ROUTER_METRICS_PORT
    fi
    (
        rc=0
        srun --overlap --container-image="$ROUTER_IMAGE" "${CONTAINER[@]}" \
            bash benchmarks/single_node/serving/sidecar.sh router > "$RESULTS/router.log" 2>&1 || rc=$?
        echo "exit $rc" > "$RESULTS/.router_exited"
    ) &
    SIDECARS+=("$!")
    # exported after the router step forked, so the frontend does not inherit the engine's port
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_SYSTEM_PORT
    fi
fi

srun --container-image="$SERVING_ENGINE_IMAGE" "${CONTAINER[@]}" bash benchmarks/single_node/serving/bench.sh
