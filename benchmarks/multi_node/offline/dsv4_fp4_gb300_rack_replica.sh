#!/usr/bin/env bash

set -Eeuo pipefail

: "${GLOBAL_BATCH_SIZE:?Replica GLOBAL_BATCH_SIZE is required}"
: "${TRT_BENCH_WORKSPACE:?TRT_BENCH_WORKSPACE is required}"
: "${TRT_BENCH_ALLOCATION_JOB_ID:?Allocation job ID is required}"
: "${TRT_BENCH_REPLICA_NODELIST:?Replica node list is required}"
: "${TRT_BENCH_RACK_REPLICA_COUNT:?Rack replica count is required}"
: "${TRT_BENCH_RACK_REPLICA_INDEX:?Rack replica index is required}"
: "${TRT_BENCH_RACK_GLOBAL_BATCH_SIZE:?Rack global batch size is required}"
: "${TRT_BENCH_RACK_ID:?Rack benchmark ID is required}"
: "${TRT_BENCH_RACK_ROOT_REL:?Rack work root is required}"
: "${TRT_BENCH_FABRIC_CLUSTER_UUID:?Fabric UUID is required}"
: "${TRT_BENCH_FABRIC_CLIQUE_ID:?Fabric clique ID is required}"

TRT_BENCH_CONFIG_PROFILE="rack-tp8-mtp1-engine"
IMAGE="${IMAGE:-nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1}"
MODEL_PATH="${MODEL_PATH:-/scratch/models/DeepSeek-V4-Pro}"
DATASET_REVISION="${DATASET_REVISION:-90f0394333616266d9fe85824ceaf505093cbaa5}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH is required}"
TRT_BENCH_CACHE_ROOT="${TRT_BENCH_CACHE_ROOT:?TRT_BENCH_CACHE_ROOT is required}"
LOAD_BALANCER_ROOT="${LOAD_BALANCER_ROOT:?LOAD_BALANCER_ROOT is required}"
SQUASH_FILE="${SQUASH_FILE:?SQUASH_FILE is required}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-18000}"
COMPLETION_VISIBILITY_TIMEOUT="${TRT_BENCH_COMPLETION_VISIBILITY_TIMEOUT:-180}"
REPLICA_COUNT="$TRT_BENCH_RACK_REPLICA_COUNT"
REPLICA_INDEX="$TRT_BENCH_RACK_REPLICA_INDEX"
RACK_GLOBAL_BATCH_SIZE="$TRT_BENCH_RACK_GLOBAL_BATCH_SIZE"
RACK_BARRIER_TIMEOUT_SECONDS="${TRT_BENCH_RACK_BARRIER_TIMEOUT_SECONDS:-3600}"
REPLICA_WORLD_SIZE=8
REPLICA_PHYSICAL_NODES=2
GPUS_PER_NODE=4
JOB_ID="$TRT_BENCH_ALLOCATION_JOB_ID"
REPLICA_LABEL="$(printf 'r%02d' "$REPLICA_INDEX")"
BENCH_ID="${TRT_BENCH_RACK_ID}-replica$(printf '%02d' "$REPLICA_INDEX")"
HOST_OUTPUT_ROOT="${TRT_BENCH_WORKSPACE}/${TRT_BENCH_RACK_ROOT_REL}/replicas/${REPLICA_LABEL}"
CONTAINER_OUTPUT_ROOT="/workspace/${TRT_BENCH_RACK_ROOT_REL}/replicas/${REPLICA_LABEL}"
RESULT_FILE="${HOST_OUTPUT_ROOT}/offline_result_${BENCH_ID}.json"
COMPLETION_FILE="${HOST_OUTPUT_ROOT}/offline_completion_${BENCH_ID}.json"
WORLD_LOG="${HOST_OUTPUT_ROOT}/offline_world_${BENCH_ID}.log"
TIMING_LOG="${HOST_OUTPUT_ROOT}/offline_timing_${BENCH_ID}.log"
RANK_MAP_FILE="${HOST_OUTPUT_ROOT}/offline_rank_map_${BENCH_ID}.tsv"
WORLD_STEP_PID=""
RANK_ENV_RECORDS=""

log() {
    printf '[offline-trt-rack-replica %s %s] %s\n' \
        "$REPLICA_LABEL" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_failure() {
    local return_code="$1"
    local message="$2"
    python3 - \
        "$RESULT_FILE" \
        "$GLOBAL_BATCH_SIZE" \
        "$return_code" \
        "$BENCH_ID" \
        "$message" <<'PY'
import json
import sys

path, global_batch, return_code, experiment_id, message = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "schema_version": 2,
            "status": "failed",
            "benchmark": {
                "experiment_id": experiment_id,
                "hardware": "GB300 NVL8",
                "hardware_profile": "gb300",
                "benchmark_profile": "rack-tp8-mtp1-engine",
                "active_gpu_count": 8,
                "physical_nodes": 2,
                "global_batch_size": int(global_batch),
                "concurrency": int(global_batch),
            },
            "error": message,
            "return_code": int(return_code),
        },
        stream,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
PY
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    if [[ -n "$WORLD_STEP_PID" ]]; then
        kill "$WORLD_STEP_PID" >/dev/null 2>&1 || true
        wait "$WORLD_STEP_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$RANK_ENV_RECORDS" ]]; then
        rm -f "$RANK_ENV_RECORDS"
    fi
    if [[ ! -f "$RESULT_FILE" ]]; then
        if [[ "$rc" -eq 0 ]]; then
            rc=1
        fi
        write_failure "$rc" \
            "Rack replica launcher exited before the container wrote a result"
    fi
    exit "$rc"
}
trap cleanup EXIT

controller_artifacts_ready() {
    [[ -s "$COMPLETION_FILE" && -s "$RESULT_FILE" ]]
}

wait_for_controller_artifacts() {
    local started="$SECONDS"
    local last_heartbeat=-1
    while (( SECONDS - started < COMPLETION_VISIBILITY_TIMEOUT )); do
        if controller_artifacts_ready; then
            return 0
        fi
        elapsed=$((SECONDS - started))
        if (( elapsed / 15 > last_heartbeat )); then
            last_heartbeat=$((elapsed / 15))
            log "waiting for shared result visibility elapsed=${elapsed}s"
        fi
        sleep 1
    done
    controller_artifacts_ready
}

mkdir -p "$HOST_OUTPUT_ROOT"
rm -f \
    "$RESULT_FILE" \
    "$COMPLETION_FILE" \
    "$WORLD_LOG" \
    "$TIMING_LOG" \
    "$RANK_MAP_FILE"

IFS=',' read -r -a replica_nodes <<<"$TRT_BENCH_REPLICA_NODELIST"
if [[ "${#replica_nodes[@]}" -ne "$REPLICA_PHYSICAL_NODES" ]]; then
    echo "Replica requires two nodes: $TRT_BENCH_REPLICA_NODELIST" >&2
    exit 1
fi

log "validating rank map nodes=$TRT_BENCH_REPLICA_NODELIST"
# shellcheck disable=SC2016
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --mpi=pmix \
    --oversubscribe \
    --cpu-bind=none \
    --nodes="$REPLICA_PHYSICAL_NODES" \
    --ntasks="$REPLICA_WORLD_SIZE" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --nodelist="$TRT_BENCH_REPLICA_NODELIST" \
    --kill-on-bad-exit=1 \
    bash -c \
        'printf "%s\t%s\t%s\t%s\n" "$SLURM_PROCID" "$(hostname)" "$SLURM_LOCALID" "${CUDA_VISIBLE_DEVICES:-unset}"' \
    | sort -n -k1,1 > "$RANK_MAP_FILE"
python3 - \
    "$RANK_MAP_FILE" \
    "$REPLICA_WORLD_SIZE" \
    "$REPLICA_PHYSICAL_NODES" \
    "$GPUS_PER_NODE" <<'PY'
import collections
import csv
import sys

path, world_size, physical_nodes, gpus_per_node = sys.argv[1:]
world_size = int(world_size)
physical_nodes = int(physical_nodes)
gpus_per_node = int(gpus_per_node)
with open(path, encoding="utf-8") as stream:
    rows = list(csv.reader(stream, delimiter="\t"))
if len(rows) != world_size:
    raise SystemExit(f"rank map has {len(rows)} rows, expected {world_size}")
if [int(row[0]) for row in rows] != list(range(world_size)):
    raise SystemExit("rank map does not contain consecutive replica ranks")
hosts = collections.Counter(row[1] for row in rows)
if len(hosts) != physical_nodes or set(hosts.values()) != {gpus_per_node}:
    raise SystemExit(f"invalid replica host placement: {hosts}")
for host in hosts:
    local = sorted(int(row[2]) for row in rows if row[1] == host)
    if local != list(range(gpus_per_node)):
        raise SystemExit(f"{host} has invalid local ranks: {local}")
PY

export BENCH_ID
export GLOBAL_BATCH_SIZE
export MODEL_PATH
export DATASET_PATH
export DATASET_REVISION
export WORKER_TIMEOUT
export IMAGE
export TRT_BENCH_CACHE_ROOT
export TRT_BENCH_GIT_REVISION
export TRT_BENCH_HARDWARE_PROFILE="gb300"
export TRT_BENCH_CONFIG_PROFILE
export TRT_BENCH_EXTERNAL_MPI="1"
export TRT_BENCH_ALLOCATION_JOB_ID="$JOB_ID"
export TRT_BENCH_ALLOCATION_NODE="${replica_nodes[0]}"
export TRT_BENCH_SLURM_NODELIST="$TRT_BENCH_REPLICA_NODELIST"
export TRT_BENCH_OUTPUT_ROOT="$CONTAINER_OUTPUT_ROOT"
TRT_BENCH_COMPLETION_FILE="${CONTAINER_OUTPUT_ROOT}/$(basename "$COMPLETION_FILE")"
TRT_BENCH_EXTERNAL_WORLD_LOG="${CONTAINER_OUTPUT_ROOT}/$(basename "$WORLD_LOG")"
TRT_BENCH_EXTERNAL_TIMING_LOG="${CONTAINER_OUTPUT_ROOT}/$(basename "$TIMING_LOG")"
TRT_BENCH_RANK_MAP_ARTIFACT="${TRT_BENCH_RACK_ROOT_REL}/replicas/${REPLICA_LABEL}/$(basename "$RANK_MAP_FILE")"
export TRT_BENCH_COMPLETION_FILE
export TRT_BENCH_EXTERNAL_WORLD_LOG
export TRT_BENCH_EXTERNAL_TIMING_LOG
export TRT_BENCH_RANK_MAP_ARTIFACT
export TRT_BENCH_TOPOLOGY_ARTIFACT="offline_topology_${TRT_BENCH_RACK_ID}.log"
export TRT_BENCH_RACK_BARRIER_DIR="/workspace/${TRT_BENCH_RACK_ROOT_REL}/barrier"
export TRT_BENCH_RACK_REPLICA_COUNT="$REPLICA_COUNT"
export TRT_BENCH_RACK_REPLICA_INDEX="$REPLICA_INDEX"
export TRT_BENCH_RACK_GLOBAL_BATCH_SIZE="$RACK_GLOBAL_BATCH_SIZE"
export TRT_BENCH_RACK_BARRIER_TIMEOUT_SECONDS="$RACK_BARRIER_TIMEOUT_SECONDS"
export UCX_TLS="cuda_ipc,cuda_copy,sm,self,tcp"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/inferencex-offline-pycache
export PYTHONPATH="/workspace/utils/bench_offline:${PYTHONPATH:-}"
export ENROOT_ALLOW_DEV=yes
export NCCL_GRAPH_MIXING_SUPPORT=0
export MIMALLOC_PURGE_DELAY=-1
export PYTHONWARNINGS="ignore::DeprecationWarning:cutlass.cute.core"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TLLM_LOG_LEVEL=INFO
export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1
export TRTLLM_ENABLE_PDL=1
export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1
export TRTLLM_EPLB_SHM_NAME="offline_${JOB_ID}_${REPLICA_LABEL}_${TRT_BENCH_RACK_ID}"

HOST_EXTERNAL_WORK_DIR="${TRT_BENCH_WORKSPACE}/.offline_work_${BENCH_ID}_${JOB_ID}"
EXTERNAL_WORK_DIR="/workspace/.offline_work_${BENCH_ID}_${JOB_ID}"
FIXED_BATCH_ARM_FILE="${EXTERNAL_WORK_DIR}/fixed_batch_barrier.armed.json"
PERFECT_ROUTER_MARKER="${EXTERNAL_WORK_DIR}/perfect_router.jsonl"
HOST_PERFECT_ROUTER_MARKER="${HOST_EXTERNAL_WORK_DIR}/perfect_router.jsonl"
CUTE_CACHE_DIR="${TRT_BENCH_CACHE_ROOT}/cute-dsl-rack/${REPLICA_LABEL}"
rm -rf "$HOST_EXTERNAL_WORK_DIR"
mkdir -p "$HOST_EXTERNAL_WORK_DIR" "$CUTE_CACHE_DIR"
RANK_ENV_RECORDS="$(mktemp /tmp/offline-trt-rack-rank-env.XXXXXX)"
python3 \
    "$TRT_BENCH_WORKSPACE/utils/bench_offline/emit_rank_environment.py" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --fixed-batch-arm-file "$FIXED_BATCH_ARM_FILE" \
    --marker-file "$PERFECT_ROUTER_MARKER" \
    --cute-cache-dir "$CUTE_CACHE_DIR" \
    --hardware-profile gb300 \
    --config-profile "$TRT_BENCH_CONFIG_PROFILE" \
    --format nul > "$RANK_ENV_RECORDS"
rank_env_exports=0
rank_env_unsets=0
while IFS= read -r -d '' operation; do
    IFS= read -r -d '' name
    if [[ ! "$name" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
        echo "Invalid rank environment variable name: $name" >&2
        exit 1
    fi
    case "$operation" in
        unset)
            unset "$name"
            ((rank_env_unsets += 1))
            ;;
        export)
            IFS= read -r -d '' value
            printf -v "$name" '%s' "$value"
            export "${name?}"
            ((rank_env_exports += 1))
            ;;
        *)
            echo "Invalid rank environment operation: $operation" >&2
            exit 1
            ;;
    esac
done < "$RANK_ENV_RECORDS"
rm -f "$RANK_ENV_RECORDS"
RANK_ENV_RECORDS=""
log "preseeded rank environment exports=$rank_env_exports cleared=$rank_env_unsets"

for rack_integer_name in \
    REPLICA_COUNT \
    REPLICA_INDEX \
    RACK_GLOBAL_BATCH_SIZE \
    RACK_BARRIER_TIMEOUT_SECONDS
do
    rack_integer_value="${!rack_integer_name}"
    if [[ ! "$rack_integer_value" =~ ^[0-9]+$ ]]; then
        echo \
            "Invalid rack synchronization integer ${rack_integer_name}=${rack_integer_value@Q}" \
            >&2
        exit 1
    fi
done
if (( REPLICA_COUNT <= 1 )); then
    echo "Rack replica count must be greater than one: $REPLICA_COUNT" >&2
    exit 1
fi
if (( REPLICA_INDEX < 0 || REPLICA_INDEX >= REPLICA_COUNT )); then
    echo \
        "Rack replica index $REPLICA_INDEX is outside 0..$((REPLICA_COUNT - 1))" \
        >&2
    exit 1
fi
if (( RACK_GLOBAL_BATCH_SIZE <= 0 || RACK_BARRIER_TIMEOUT_SECONDS <= 0 )); then
    echo \
        "Rack global batch and barrier timeout must be positive: global_batch=$RACK_GLOBAL_BATCH_SIZE timeout=$RACK_BARRIER_TIMEOUT_SECONDS" \
        >&2
    exit 1
fi
log "validated rack synchronization environment replica=$REPLICA_INDEX/$REPLICA_COUNT rack_global_batch=$RACK_GLOBAL_BATCH_SIZE engine_global_batch=$GLOBAL_BATCH_SIZE timeout=${RACK_BARRIER_TIMEOUT_SECONDS}s barrier=$TRT_BENCH_RACK_BARRIER_DIR"

CONTAINER_MOUNTS="${TRT_BENCH_WORKSPACE}:/workspace"
CONTAINER_MOUNTS+=",/scratch/models:/scratch/models"
CONTAINER_MOUNTS+=",$(dirname "$DATASET_PATH"):$(dirname "$DATASET_PATH")"
CONTAINER_MOUNTS+=",${TRT_BENCH_CACHE_ROOT}:${TRT_BENCH_CACHE_ROOT}"
CONTAINER_MOUNTS+=",${LOAD_BALANCER_ROOT}:/dsv4-eplb-configs"
CONTAINER_MOUNTS+=",tmpfs:/dev/shm:size=100%"

log "starting TP8/EP8 MTP1 engine global_batch=$GLOBAL_BATCH_SIZE"
world_started="$SECONDS"
last_heartbeat=-1
set +e
# shellcheck disable=SC2016
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --mpi=pmix \
    --oversubscribe \
    --cpu-bind=verbose,none \
    --nodes="$REPLICA_PHYSICAL_NODES" \
    --ntasks="$REPLICA_WORLD_SIZE" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --nodelist="$TRT_BENCH_REPLICA_NODELIST" \
    --kill-on-bad-exit=1 \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --no-container-mount-home \
    --container-workdir=/workspace \
    --no-container-entrypoint \
    --export=ALL,GITHUB_WORKSPACE=/workspace \
    bash -c '
        if [[ "${SLURM_PROCID:-}" == "0" ]]; then
            : > "$TRT_BENCH_EXTERNAL_TIMING_LOG"
            tail --pid="$$" -n +1 -F \
                "$TRT_BENCH_EXTERNAL_TIMING_LOG" &
            if command -v numactl >/dev/null 2>&1; then
                exec numactl -m 0,1 \
                    trtllm-llmapi-launch \
                    bash /workspace/benchmarks/single_node/offline/run_dsv4_trt_container.sh \
                    >> "$TRT_BENCH_EXTERNAL_TIMING_LOG" 2>&1
            fi
            exec trtllm-llmapi-launch \
                bash /workspace/benchmarks/single_node/offline/run_dsv4_trt_container.sh \
                >> "$TRT_BENCH_EXTERNAL_TIMING_LOG" 2>&1
        fi
        if command -v numactl >/dev/null 2>&1; then
            exec numactl -m 0,1 \
                trtllm-llmapi-launch \
                bash /workspace/benchmarks/single_node/offline/run_dsv4_trt_container.sh
        fi
        exec trtllm-llmapi-launch \
            bash /workspace/benchmarks/single_node/offline/run_dsv4_trt_container.sh
    ' > >(tee -a "$WORLD_LOG") 2>&1 &
WORLD_STEP_PID=$!
set -e

while kill -0 "$WORLD_STEP_PID" >/dev/null 2>&1; do
    if controller_artifacts_ready; then
        break
    fi
    world_elapsed=$((SECONDS - world_started))
    if (( world_elapsed / 60 > last_heartbeat )); then
        last_heartbeat=$((world_elapsed / 60))
        rank_events=0
        if [[ -s "$HOST_PERFECT_ROUTER_MARKER" ]]; then
            rank_events="$(wc -l < "$HOST_PERFECT_ROUTER_MARKER")"
        fi
        barrier_ready=no
        if [[ -s "${TRT_BENCH_WORKSPACE}/${TRT_BENCH_RACK_ROOT_REL}/barrier/replica_$(printf '%02d' "$REPLICA_INDEX").ready.json" ]]; then
            barrier_ready=yes
        fi
        log "engine active elapsed=${world_elapsed}s rank_events=$rank_events barrier_ready=$barrier_ready"
    fi
    sleep 1
done

world_rc=0
if ! controller_artifacts_ready; then
    set +e
    wait "$WORLD_STEP_PID"
    world_rc=$?
    set -e
    WORLD_STEP_PID=""
    log "engine step exited return_code=$world_rc; waiting for NFS visibility"
    if ! wait_for_controller_artifacts; then
        write_failure "$world_rc" \
            "Replica engine exited without visible result/completion artifacts"
        exit 1
    fi
fi

read -r completion_rc completion_status result_status < <(
    python3 - "$COMPLETION_FILE" "$RESULT_FILE" <<'PY'
import json
import sys

completion_path, result_path = sys.argv[1:]
with open(completion_path, encoding="utf-8") as stream:
    completion = json.load(stream)
with open(result_path, encoding="utf-8") as stream:
    result = json.load(stream)
print(
    int(completion["return_code"]),
    str(completion["result_status"]),
    str(result.get("status", "missing")),
)
PY
)
log "controller finalized return_code=$completion_rc completion_status=$completion_status result_status=$result_status"

if [[ -n "$WORLD_STEP_PID" ]]; then
    kill "$WORLD_STEP_PID" >/dev/null 2>&1 || true
    set +e
    wait "$WORLD_STEP_PID"
    world_rc=$?
    set -e
    WORLD_STEP_PID=""
fi
log "engine step stopped transport_return_code=$world_rc"

if [[ "$completion_status" != "$result_status" ]]; then
    exit 1
fi
if [[ "$completion_rc" -ne 0 || "$result_status" != "success" ]]; then
    exit 1
fi
log "replica benchmark complete"
