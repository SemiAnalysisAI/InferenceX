#!/usr/bin/env bash

set -Eeuo pipefail

: "${GLOBAL_BATCH_SIZE:?GLOBAL_BATCH_SIZE is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${RUNNER_NAME:?RUNNER_NAME is required}"

TRT_BENCH_WORKSPACE="${TRT_BENCH_WORKSPACE:-$GITHUB_WORKSPACE}"
IMAGE="${IMAGE:-nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1}"
MODEL_PATH="${MODEL_PATH:-/scratch/models/DeepSeek-V4-Pro}"
DATASET_REVISION="${DATASET_REVISION:-90f0394333616266d9fe85824ceaf505093cbaa5}"
DATASET_FILE="longbook_qa_eng.jsonl"
SHARED_ROOT="${TRT_BENCH_SHARED_ROOT:-/data/home/sa-shared/gharunners/offline-trt}"
DATASET_ROOT="${DATASET_ROOT:-${SHARED_ROOT}/datasets/InfiniteBench/${DATASET_REVISION}}"
DATASET_PATH="${DATASET_ROOT}/${DATASET_FILE}"
TRT_BENCH_CACHE_ROOT="${TRT_BENCH_CACHE_ROOT:-${SHARED_ROOT}/cache/dsv4-1.3.0-deepseek-v4-dev.1-sm100a}"
LOAD_BALANCER_ROOT="${LOAD_BALANCER_ROOT:-${SHARED_ROOT}/configs/dsv4-eplb}"
LOAD_BALANCER_FILE="moe_load_balancer_gen_ep8_slots384.yaml"
LOAD_BALANCER_PATH="${LOAD_BALANCER_ROOT}/${LOAD_BALANCER_FILE}"
LOAD_BALANCER_URL="https://raw.githubusercontent.com/NVIDIA/srt-slurm/sa-submission-q2-2026/configs/dsv4-moe-load-balancer-configs/${LOAD_BALANCER_FILE}"
LOAD_BALANCER_SHA256="279558557f3983ebd957d7c5578ce0d61c05f6ab72cda28fb31f0d1c2ef734b5"
SQUASH_ROOT="${SQUASH_ROOT:-/data/home/sa-shared/gharunners/squash}"
SQUASH_FILE="${SQUASH_ROOT}/$(printf '%s' "$IMAGE" | sed 's|[/:@#]|_|g').sqsh"
SALLOC_TIME_LIMIT="${SALLOC_TIME_LIMIT:-360}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-18000}"
BENCH_ID="${BENCH_ID:-rack-tp8x9-mtp1-gbs${GLOBAL_BATCH_SIZE}}"
SLURM_PARTITION="${SLURM_PARTITION:-batch_1}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-benchmark}"
RACK_PHYSICAL_NODES=18
RACK_WORLD_SIZE=72
REPLICA_COUNT=9
REPLICA_PHYSICAL_NODES=2
GPUS_PER_NODE=4
TRT_BENCH_GIT_REVISION="$(git -C "$TRT_BENCH_WORKSPACE" rev-parse HEAD)"
export TRT_BENCH_GIT_REVISION

case "$GLOBAL_BATCH_SIZE" in
    72|288|576|30960|36864)
        ;;
    *)
        echo "Unsupported rack global batch: $GLOBAL_BATCH_SIZE" >&2
        exit 1
        ;;
esac
if (( GLOBAL_BATCH_SIZE % REPLICA_COUNT != 0 )); then
    echo "Rack global batch must divide across $REPLICA_COUNT replicas" >&2
    exit 1
fi
ENGINE_GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / REPLICA_COUNT))

RESULT_FILE="${TRT_BENCH_WORKSPACE}/offline_result_${BENCH_ID}.json"
CONTROLLER_LOG="${TRT_BENCH_WORKSPACE}/offline_controller_${BENCH_ID}.log"
RANK_MAP_FILE="${TRT_BENCH_WORKSPACE}/offline_rank_map_${BENCH_ID}.tsv"
TOPOLOGY_FILE="${TRT_BENCH_WORKSPACE}/offline_topology_${BENCH_ID}.log"
ALLOCATION_LOG="${TRT_BENCH_WORKSPACE}/offline_allocation_${BENCH_ID}.log"
COMPLETION_FILE="${TRT_BENCH_WORKSPACE}/offline_completion_${BENCH_ID}.json"
WORLD_LOG="${TRT_BENCH_WORKSPACE}/offline_world_${BENCH_ID}.log"
DEBUG_ARCHIVE="${TRT_BENCH_WORKSPACE}/offline_debug_${BENCH_ID}.tar.gz"
DEBUG_ARCHIVED=0
JOB_ID=""
RACK_ROOT_REL=""
TELEMETRY_STEP_PID=""
SALLOC_PIPE_PID=""
declare -a REPLICA_PIDS=()

rm -f \
    "$RESULT_FILE" \
    "$CONTROLLER_LOG" \
    "$RANK_MAP_FILE" \
    "$TOPOLOGY_FILE" \
    "$ALLOCATION_LOG" \
    "$COMPLETION_FILE" \
    "$WORLD_LOG" \
    "$DEBUG_ARCHIVE" \
    "${TRT_BENCH_WORKSPACE}/offline_gpu_metrics_${BENCH_ID}_"*.csv
exec > >(tee -a "$WORLD_LOG" "$CONTROLLER_LOG") 2>&1

log() {
    printf '[offline-trt-gb300-rack %s] %s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_completion() {
    local return_code="$1"
    local result_status="$2"
    local temporary="${COMPLETION_FILE}.tmp.$$"
    python3 - "$temporary" "$return_code" "$result_status" <<'PY'
import json
import sys

path, return_code, result_status = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "return_code": int(return_code),
            "result_status": result_status,
        },
        stream,
        sort_keys=True,
    )
    stream.write("\n")
PY
    mv "$temporary" "$COMPLETION_FILE"
}

write_host_failure() {
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
                "mode": "offline_rack_fixed_global_batch_decode",
                "experiment_id": experiment_id,
                "hardware": "GB300 NVL72",
                "hardware_profile": "gb300",
                "benchmark_profile": "rack-tp8x9-mtp1",
                "renderer_hw": "gb300-nv",
                "world_size": 72,
                "active_gpu_count": 72,
                "physical_nodes": 18,
                "gpus_per_node": 4,
                "is_multinode": True,
                "effective_parallelism": "9xDEP8",
                "replica_count": 9,
                "global_batch_size": int(global_batch),
                "concurrency": int(global_batch),
                "local_batch_size": int(global_batch) // 72,
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

archive_debug() {
    if [[ -n "$RACK_ROOT_REL" \
        && -d "${TRT_BENCH_WORKSPACE}/${RACK_ROOT_REL}" ]]; then
        if tar \
            -C "$TRT_BENCH_WORKSPACE" \
            --exclude='corpus.bin' \
            -czf "$DEBUG_ARCHIVE" \
            "$RACK_ROOT_REL" 2>/dev/null; then
            DEBUG_ARCHIVED=1
        fi
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e
    for pid in "${REPLICA_PIDS[@]:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" >/dev/null 2>&1 || true
            wait "$pid" >/dev/null 2>&1 || true
        fi
    done
    if [[ -n "$TELEMETRY_STEP_PID" ]]; then
        kill "$TELEMETRY_STEP_PID" >/dev/null 2>&1 || true
        wait "$TELEMETRY_STEP_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$SALLOC_PIPE_PID" ]]; then
        kill "$SALLOC_PIPE_PID" >/dev/null 2>&1 || true
        wait "$SALLOC_PIPE_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$JOB_ID" ]]; then
        scancel "$JOB_ID" >/dev/null 2>&1 || true
    fi
    if [[ ! -f "$RESULT_FILE" ]]; then
        if [[ "$rc" -eq 0 ]]; then
            rc=1
        fi
        write_host_failure "$rc" \
            "Rack host launcher exited before aggregation completed"
    fi
    if [[ "$DEBUG_ARCHIVED" -ne 1 ]]; then
        archive_debug
    fi
    if [[ ! -f "$COMPLETION_FILE" ]]; then
        result_status="$(
            python3 - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(str(json.load(stream).get("status", "missing")))
PY
        )"
        write_completion "$rc" "$result_status"
    fi
    exit "$rc"
}
trap cleanup EXIT

download_locked() {
    local destination="$1"
    local url="$2"
    local expected_sha256="$3"
    local lock="${destination}.lock"
    mkdir -p "$(dirname "$destination")"
    (
        exec 9>"$lock"
        flock -w 600 9
        if [[ -s "$destination" ]] \
            && printf '%s  %s\n' "$expected_sha256" "$destination" \
                | sha256sum --check --status; then
            log "using cached file: $destination"
            exit 0
        fi
        temporary="${destination}.tmp.$$"
        trap 'rm -f "$temporary"' EXIT
        curl --fail --location --retry 5 --retry-all-errors \
            "$url" \
            --output "$temporary"
        printf '%s  %s\n' "$expected_sha256" "$temporary" \
            | sha256sum --check --status
        mv "$temporary" "$destination"
    )
}

import_squash_on_arm_compute() {
    mkdir -p "$SQUASH_ROOT"
    log "checking ARM64 TRT image cache: $SQUASH_FILE"
    srun \
        --partition="$SLURM_PARTITION" \
        --account="$SLURM_ACCOUNT" \
        --nodes=1 \
        --ntasks=1 \
        --exclusive \
        --time=180 \
        bash -s -- "$SQUASH_FILE" "$IMAGE" <<'BASH'
set -Eeuo pipefail
squash_file="$1"
image="$2"
lock_file="${squash_file}.lock"
exec 9>"$lock_file"
flock -w 900 9
if unsquashfs -l "$squash_file" >/dev/null 2>&1; then
    echo "Using cached ARM64 image: $squash_file"
    exit 0
fi
rm -f "$squash_file"
enroot_local="$(mktemp -d /tmp/enroot-import.XXXXXX)"
trap 'rm -rf "$enroot_local"' EXIT
export ENROOT_TEMP_PATH="$enroot_local/tmp"
export ENROOT_CACHE_PATH="$enroot_local/cache"
export ENROOT_DATA_PATH="$enroot_local/data"
export ENROOT_RUNTIME_PATH="$enroot_local/run"
mkdir -p \
    "$ENROOT_TEMP_PATH" \
    "$ENROOT_CACHE_PATH" \
    "$ENROOT_DATA_PATH" \
    "$ENROOT_RUNTIME_PATH"
enroot import -o "$squash_file" "docker://$image"
BASH
}

mkdir -p "$DATASET_ROOT" "$TRT_BENCH_CACHE_ROOT/cute-dsl-rack"
PYTHONPATH="$TRT_BENCH_WORKSPACE/utils/bench_offline" \
TRT_BENCH_HARDWARE_PROFILE=gb300 \
TRT_BENCH_CONFIG_PROFILE=rack-tp8-mtp1-engine \
python3 - "$ENGINE_GLOBAL_BATCH_SIZE" <<'PY'
import sys

from trt_config import validate_global_batch_size

validate_global_batch_size(int(sys.argv[1]))
PY
download_locked \
    "$DATASET_PATH" \
    "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/${DATASET_REVISION}/${DATASET_FILE}?download=true" \
    "7f53a3aabd5445fa29f168914f13569c348406cfc88a3e495f3f2c13a0966ca4"
download_locked \
    "$LOAD_BALANCER_PATH" \
    "$LOAD_BALANCER_URL" \
    "$LOAD_BALANCER_SHA256"
import_squash_on_arm_compute

base_cute_cache="${TRT_BENCH_CACHE_ROOT}/cute-dsl"
for replica_index in $(seq 0 $((REPLICA_COUNT - 1))); do
    replica_label="$(printf 'r%02d' "$replica_index")"
    replica_cache="${TRT_BENCH_CACHE_ROOT}/cute-dsl-rack/${replica_label}"
    mkdir -p "$replica_cache"
    if [[ -d "$base_cute_cache" ]]; then
        cp -a -n "$base_cute_cache"/. "$replica_cache"/ 2>/dev/null || true
    fi
done

log "requesting full rack nodes=$RACK_PHYSICAL_NODES gpus=$RACK_WORLD_SIZE profile=rack-tp8x9-mtp1 rack_gbs=$GLOBAL_BATCH_SIZE engine_gbs=$ENGINE_GLOBAL_BATCH_SIZE"
set +e
salloc \
    --partition="$SLURM_PARTITION" \
    --account="$SLURM_ACCOUNT" \
    --nodes="$RACK_PHYSICAL_NODES" \
    --ntasks="$RACK_WORLD_SIZE" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --exclusive \
    --time="$SALLOC_TIME_LIMIT" \
    --no-shell \
    --job-name="$RUNNER_NAME" \
    2>&1 | tee "$ALLOCATION_LOG" &
SALLOC_PIPE_PID=$!
set -e
allocation_started="$SECONDS"
while kill -0 "$SALLOC_PIPE_PID" >/dev/null 2>&1; do
    if [[ -z "$JOB_ID" && -s "$ALLOCATION_LOG" ]]; then
        JOB_ID="$(
            sed -n \
                -e 's/.*Pending job allocation \([0-9][0-9]*\).*/\1/p' \
                -e 's/.*Granted job allocation \([0-9][0-9]*\).*/\1/p' \
                "$ALLOCATION_LOG" \
                | tail -n 1
        )"
    fi
    allocation_elapsed=$((SECONDS - allocation_started))
    if (( allocation_elapsed > 0 && allocation_elapsed % 60 == 0 )); then
        if [[ -n "$JOB_ID" ]]; then
            queue_state="$(
                squeue \
                    --jobs="$JOB_ID" \
                    --noheader \
                    --format='state=%T reason=%R nodes=%D start=%S' \
                    2>/dev/null \
                    || true
            )"
            log "waiting for rack allocation job=$JOB_ID elapsed=${allocation_elapsed}s ${queue_state:-state=transitioning}"
        else
            log "waiting for Slurm allocation id elapsed=${allocation_elapsed}s"
        fi
    fi
    sleep 1
done
set +e
wait "$SALLOC_PIPE_PID"
allocation_rc=$?
set -e
SALLOC_PIPE_PID=""
if [[ "$allocation_rc" -ne 0 ]]; then
    exit "$allocation_rc"
fi
granted_job_id="$(
    sed -n \
        's/.*Granted job allocation \([0-9][0-9]*\).*/\1/p' \
        "$ALLOCATION_LOG" \
        | tail -n 1
)"
if [[ -n "$granted_job_id" ]]; then
    JOB_ID="$granted_job_id"
fi
if [[ -z "$JOB_ID" ]]; then
    echo "Could not determine Slurm allocation ID" >&2
    exit 1
fi

node_expression="$(
    squeue --jobs="$JOB_ID" --noheader --format='%N' | head -n 1
)"
mapfile -t allocation_nodes < <(
    scontrol show hostnames "$node_expression"
)
if [[ "${#allocation_nodes[@]}" -ne "$RACK_PHYSICAL_NODES" ]]; then
    echo "Expected $RACK_PHYSICAL_NODES nodes, got ${allocation_nodes[*]}" >&2
    exit 1
fi
node_list="$(
    IFS=' '
    printf '%s' "${allocation_nodes[*]}"
)"
log "allocation job=$JOB_ID nodes=$node_list"

RACK_ROOT_REL=".offline_rack_${BENCH_ID}_${JOB_ID}"
RACK_ROOT="${TRT_BENCH_WORKSPACE}/${RACK_ROOT_REL}"
rm -rf "$RACK_ROOT"
mkdir -p "$RACK_ROOT/barrier" "$RACK_ROOT/replicas"

log "capturing and validating the full 72-rank task map"
# shellcheck disable=SC2016
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --mpi=pmix \
    --oversubscribe \
    --cpu-bind=none \
    --nodes="$RACK_PHYSICAL_NODES" \
    --ntasks="$RACK_WORLD_SIZE" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --kill-on-bad-exit=1 \
    bash -c \
        'printf "%s\t%s\t%s\t%s\n" "$SLURM_PROCID" "$(hostname)" "$SLURM_LOCALID" "${CUDA_VISIBLE_DEVICES:-unset}"' \
    | sort -n -k1,1 > "$RANK_MAP_FILE"
python3 - \
    "$RANK_MAP_FILE" \
    "$RACK_WORLD_SIZE" \
    "$RACK_PHYSICAL_NODES" \
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
    raise SystemExit("rack rank map is not exactly 0..71")
hosts = collections.Counter(row[1] for row in rows)
if len(hosts) != physical_nodes or set(hosts.values()) != {gpus_per_node}:
    raise SystemExit(f"invalid rack host placement: {hosts}")
for host in hosts:
    local = sorted(int(row[2]) for row in rows if row[1] == host)
    if local != list(range(gpus_per_node)):
        raise SystemExit(f"{host} has invalid local ranks: {local}")
print(f"validated rack rank map across {len(hosts)} nodes")
PY

log "capturing and validating the 72-GPU NVLink fabric domain"
FABRIC_PROBE="${TRT_BENCH_WORKSPACE}/utils/bench_offline/gb300_fabric.py"
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --nodes="$RACK_PHYSICAL_NODES" \
    --ntasks="$RACK_PHYSICAL_NODES" \
    --ntasks-per-node=1 \
    --cpu-bind=none \
    --kill-on-bad-exit=1 \
    python3 "$FABRIC_PROBE" probe \
        --expected-gpus="$GPUS_PER_NODE" \
    2>&1 | tee "$TOPOLOGY_FILE"
fabric_validation="$(
    python3 "$FABRIC_PROBE" validate-log \
        "$TOPOLOGY_FILE" \
        --expected-nodes="$RACK_PHYSICAL_NODES" \
        --expected-gpus-per-node="$GPUS_PER_NODE"
)"
printf '__FABRIC_DOMAIN__ %s\n' "$fabric_validation" \
    | tee -a "$TOPOLOGY_FILE"
FABRIC_CLUSTER_UUID="$(
    python3 -c \
        'import json, sys; print(json.load(sys.stdin)["cluster_uuid"])' \
        <<<"$fabric_validation"
)"
FABRIC_CLIQUE_ID="$(
    python3 -c \
        'import json, sys; print(json.load(sys.stdin)["clique_id"])' \
        <<<"$fabric_validation"
)"
log "validated one 72-GPU fabric cluster_uuid=$FABRIC_CLUSTER_UUID clique_id=$FABRIC_CLIQUE_ID"

export BENCH_ID
export TRT_BENCH_TELEMETRY_DIR="$TRT_BENCH_WORKSPACE"
log "starting one-second GPU telemetry on all 18 nodes"
# shellcheck disable=SC2016
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --nodes="$RACK_PHYSICAL_NODES" \
    --ntasks="$RACK_PHYSICAL_NODES" \
    --ntasks-per-node=1 \
    --cpu-bind=none \
    bash -c '
        host="$(hostname)"
        output="${TRT_BENCH_TELEMETRY_DIR}/offline_gpu_metrics_${BENCH_ID}_${host}.csv"
        query="timestamp,index,power.draw,temperature.gpu"
        query+=",clocks.current.sm,clocks.current.memory"
        query+=",utilization.gpu,utilization.memory,memory.used,memory.free"
        exec nvidia-smi --query-gpu="$query" --format=csv -l 1 > "$output"
    ' &
TELEMETRY_STEP_PID=$!

export IMAGE
export MODEL_PATH
export DATASET_PATH
export DATASET_REVISION
export TRT_BENCH_CACHE_ROOT
export LOAD_BALANCER_ROOT
export SQUASH_FILE
export WORKER_TIMEOUT
export TRT_BENCH_WORKSPACE
export TRT_BENCH_RACK_REPLICA_COUNT="$REPLICA_COUNT"
export TRT_BENCH_RACK_GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
export TRT_BENCH_FABRIC_CLUSTER_UUID="$FABRIC_CLUSTER_UUID"
export TRT_BENCH_FABRIC_CLIQUE_ID="$FABRIC_CLIQUE_ID"

log "launching nine TP8/EP8 MTP1 engines"
for replica_index in $(seq 0 $((REPLICA_COUNT - 1))); do
    first_node="${allocation_nodes[$((replica_index * REPLICA_PHYSICAL_NODES))]}"
    second_node="${allocation_nodes[$((replica_index * REPLICA_PHYSICAL_NODES + 1))]}"
    replica_nodes="${first_node},${second_node}"
    replica_label="$(printf 'r%02d' "$replica_index")"
    replica_root="${RACK_ROOT}/replicas/${replica_label}"
    mkdir -p "$replica_root"
    log "launch replica=$replica_label nodes=$replica_nodes engine_gbs=$ENGINE_GLOBAL_BATCH_SIZE"
    GLOBAL_BATCH_SIZE="$ENGINE_GLOBAL_BATCH_SIZE" \
    TRT_BENCH_ALLOCATION_JOB_ID="$JOB_ID" \
    TRT_BENCH_REPLICA_NODELIST="$replica_nodes" \
    TRT_BENCH_RACK_REPLICA_INDEX="$replica_index" \
    TRT_BENCH_RACK_ID="$BENCH_ID" \
    TRT_BENCH_RACK_ROOT_REL="$RACK_ROOT_REL" \
    bash \
        "$TRT_BENCH_WORKSPACE/benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_replica.sh" \
        > "${replica_root}/host_launcher.log" 2>&1 &
    REPLICA_PIDS[replica_index]=$!
done

replica_started="$SECONDS"
last_replica_heartbeat=-1
replica_failure=0
replica_failure_message=""
completed_replicas=0
declare -a REPLICA_DONE=()
while (( completed_replicas < REPLICA_COUNT )); do
    for replica_index in $(seq 0 $((REPLICA_COUNT - 1))); do
        if [[ "${REPLICA_DONE[$replica_index]:-0}" == "1" ]]; then
            continue
        fi
        pid="${REPLICA_PIDS[$replica_index]}"
        if kill -0 "$pid" >/dev/null 2>&1; then
            continue
        fi
        set +e
        wait "$pid"
        replica_rc=$?
        set -e
        REPLICA_DONE[replica_index]=1
        REPLICA_PIDS[replica_index]=""
        ((completed_replicas += 1))
        replica_label="$(printf 'r%02d' "$replica_index")"
        log "replica finished replica=$replica_label return_code=$replica_rc completed=$completed_replicas/$REPLICA_COUNT"
        if [[ "$replica_rc" -ne 0 ]]; then
            replica_failure=1
            replica_failure_message="Replica ${replica_label} failed with return code ${replica_rc}"
            break
        fi
    done
    if [[ "$replica_failure" -eq 1 ]]; then
        break
    fi
    replica_elapsed=$((SECONDS - replica_started))
    if (( replica_elapsed / 60 > last_replica_heartbeat )); then
        last_replica_heartbeat=$((replica_elapsed / 60))
        barrier_ready="$(
            find "$RACK_ROOT/barrier" \
                -maxdepth 1 \
                -name 'replica_*.ready.json' \
                -type f \
                | wc -l
        )"
        visible_results="$(
            find "$RACK_ROOT/replicas" \
                -name 'offline_result_*.json' \
                -type f \
                | wc -l
        )"
        log "rack engines active elapsed=${replica_elapsed}s completed=$completed_replicas/$REPLICA_COUNT barrier_ready=$barrier_ready/$REPLICA_COUNT visible_results=$visible_results/$REPLICA_COUNT"
    fi
    sleep 1
done

if [[ "$replica_failure" -eq 1 ]]; then
    log "$replica_failure_message; terminating remaining replicas"
    for pid in "${REPLICA_PIDS[@]:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
    sleep 2
    write_host_failure 1 "$replica_failure_message"
    exit 1
fi
REPLICA_PIDS=()

log "all replicas succeeded; aggregating one rack fixed-GBS result"
PYTHONPATH="$TRT_BENCH_WORKSPACE/utils/bench_offline" \
python3 \
    "$TRT_BENCH_WORKSPACE/utils/bench_offline/aggregate_rack.py" \
    --replica-root "$RACK_ROOT/replicas" \
    --output "$RESULT_FILE" \
    --experiment-id "$BENCH_ID" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --allocation-job-id "$JOB_ID" \
    --slurm-nodes "$node_list" \
    --fabric-cluster-uuid "$FABRIC_CLUSTER_UUID" \
    --fabric-clique-id "$FABRIC_CLIQUE_ID" \
    --rank-map-artifact "$(basename "$RANK_MAP_FILE")" \
    --topology-artifact "$(basename "$TOPOLOGY_FILE")"

result_status="$(
    python3 - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
aggregate = result["aggregate"]
print(str(result["status"]))
print(
    "rack result "
    f"round_tpot={aggregate['decode_round_tpot_ms']:.3f}ms "
    f"step_tput_per_gpu={aggregate['decode_step_tput_per_gpu']:.2f} "
    f"tokens_per_step={aggregate['observed_tokens_per_step']:.3f} "
    f"output_tput_per_gpu={aggregate['output_tput_per_gpu']:.2f}"
)
PY
)"
printf '%s\n' "$result_status"
status_line="$(head -n 1 <<<"$result_status")"
if [[ "$status_line" != "success" ]]; then
    exit 1
fi
archive_debug
write_completion 0 success

log "canceling rack allocation job=$JOB_ID"
scancel "$JOB_ID" >/dev/null 2>&1 || true
JOB_ID=""
if [[ -n "$TELEMETRY_STEP_PID" ]]; then
    kill "$TELEMETRY_STEP_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_STEP_PID" >/dev/null 2>&1 || true
    TELEMETRY_STEP_PID=""
fi
log "GB300 NVL72 rack benchmark complete"
