#!/usr/bin/env bash

set -Eeuo pipefail

: "${GLOBAL_BATCH_SIZE:?GLOBAL_BATCH_SIZE is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${RUNNER_NAME:?RUNNER_NAME is required}"

IMAGE="${IMAGE:-nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1}"
MODEL_PATH="${MODEL_PATH:-/scratch/models/DeepSeek-V4-Pro}"
DATASET_REVISION="${DATASET_REVISION:-90f0394333616266d9fe85824ceaf505093cbaa5}"
DATASET_FILE="longbook_qa_eng.jsonl"
SHARED_ROOT="${TRT_BENCH_SHARED_ROOT:-/data/home/sa-shared/gharunners/offline-trt}"
DATASET_ROOT="${DATASET_ROOT:-${SHARED_ROOT}/datasets/InfiniteBench/${DATASET_REVISION}}"
DATASET_PATH="${DATASET_ROOT}/${DATASET_FILE}"
TRT_BENCH_CACHE_ROOT="${TRT_BENCH_CACHE_ROOT:-${SHARED_ROOT}/cache/dsv4-1.3.0-deepseek-v4-dev.1-sm100a}"
LOAD_BALANCER_ROOT="${LOAD_BALANCER_ROOT:-${SHARED_ROOT}/configs/dsv4-eplb}"
LOAD_BALANCER_FILE="moe_load_balancer_gen_ep16_slots384.yaml"
LOAD_BALANCER_PATH="${LOAD_BALANCER_ROOT}/${LOAD_BALANCER_FILE}"
LOAD_BALANCER_URL="https://raw.githubusercontent.com/NVIDIA/srt-slurm/sa-submission-q2-2026/configs/dsv4-moe-load-balancer-configs/${LOAD_BALANCER_FILE}"
LOAD_BALANCER_SHA256="278da78f94be418d189015b18625ba2dbdfe03ee4be09e1a685f0e93708f681b"
SQUASH_ROOT="${SQUASH_ROOT:-/data/home/sa-shared/gharunners/squash}"
SQUASH_FILE="${SQUASH_ROOT}/$(printf '%s' "$IMAGE" | sed 's|[/:@#]|_|g').sqsh"
SALLOC_TIME_LIMIT="${SALLOC_TIME_LIMIT:-180}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-7200}"
BENCH_ID="${BENCH_ID:-gbs${GLOBAL_BATCH_SIZE}}"
SLURM_PARTITION="${SLURM_PARTITION:-batch_1}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-benchmark}"
PHYSICAL_NODES=4
GPUS_PER_NODE=4
WORLD_SIZE=16
BENCH_GIT_REVISION="$(
    git -C "$GITHUB_WORKSPACE" rev-parse HEAD
)"

if [[ ! "$BENCH_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid BENCH_ID: $BENCH_ID" >&2
    exit 1
fi

RESULT_FILE="${GITHUB_WORKSPACE}/offline_result_${BENCH_ID}.json"
RANK_MAP_FILE="${GITHUB_WORKSPACE}/offline_rank_map_${BENCH_ID}.tsv"
TOPOLOGY_FILE="${GITHUB_WORKSPACE}/offline_topology_${BENCH_ID}.log"
ALLOCATION_LOG="${GITHUB_WORKSPACE}/offline_allocation_${BENCH_ID}.log"
COMPLETION_FILE="${GITHUB_WORKSPACE}/offline_completion_${BENCH_ID}.json"
JOB_ID=""
TELEMETRY_STEP_PID=""
RANK_ENV_RECORDS=""
SALLOC_PIPE_PID=""
WORLD_STEP_PID=""
rm -f \
    "$RESULT_FILE" \
    "$RANK_MAP_FILE" \
    "$TOPOLOGY_FILE" \
    "$ALLOCATION_LOG" \
    "$COMPLETION_FILE" \
    "${GITHUB_WORKSPACE}/offline_gpu_metrics_${BENCH_ID}_"*.csv

log() {
    printf '[offline-trt-gb300 %s] %s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_host_failure() {
    local return_code="$1"
    python3 - \
        "$RESULT_FILE" \
        "$GLOBAL_BATCH_SIZE" \
        "$return_code" \
        "$BENCH_ID" <<'PY'
import json
import sys

path, global_batch_size, return_code, experiment_id = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "schema_version": 2,
            "status": "failed",
            "benchmark": {
                "global_batch_size": int(global_batch_size),
                "concurrency": int(global_batch_size),
                "experiment_id": experiment_id,
                "hardware": "GB300 NVL16",
                "hardware_profile": "gb300",
                "renderer_hw": "gb300-nv",
                "world_size": 16,
                "active_gpu_count": 16,
                "physical_nodes": 4,
                "gpus_per_node": 4,
                "is_multinode": True,
                "effective_parallelism": "DEP16",
            },
            "error": "GB300 host launcher exited before the container wrote a result",
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
    if [[ -n "$JOB_ID" ]]; then
        scancel "$JOB_ID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$WORLD_STEP_PID" ]]; then
        kill "$WORLD_STEP_PID" >/dev/null 2>&1 || true
        wait "$WORLD_STEP_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$TELEMETRY_STEP_PID" ]]; then
        kill "$TELEMETRY_STEP_PID" >/dev/null 2>&1 || true
        wait "$TELEMETRY_STEP_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$RANK_ENV_RECORDS" ]]; then
        rm -f "$RANK_ENV_RECORDS"
    fi
    if [[ -n "$SALLOC_PIPE_PID" ]]; then
        kill "$SALLOC_PIPE_PID" >/dev/null 2>&1 || true
        wait "$SALLOC_PIPE_PID" >/dev/null 2>&1 || true
    fi
    if [[ ! -f "$RESULT_FILE" ]]; then
        if [[ "$rc" -eq 0 ]]; then
            rc=1
        fi
        write_host_failure "$rc"
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

mkdir -p "$DATASET_ROOT" "$TRT_BENCH_CACHE_ROOT/cute-dsl"
download_locked \
    "$DATASET_PATH" \
    "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/${DATASET_REVISION}/${DATASET_FILE}?download=true" \
    "7f53a3aabd5445fa29f168914f13569c348406cfc88a3e495f3f2c13a0966ca4"
download_locked \
    "$LOAD_BALANCER_PATH" \
    "$LOAD_BALANCER_URL" \
    "$LOAD_BALANCER_SHA256"
import_squash_on_arm_compute

log "requesting ${PHYSICAL_NODES} GB300 nodes and ${WORLD_SIZE} GPUs"
set +e
salloc \
    --partition="$SLURM_PARTITION" \
    --account="$SLURM_ACCOUNT" \
    --nodes="$PHYSICAL_NODES" \
    --ntasks="$WORLD_SIZE" \
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
            log "waiting for GB300 allocation job=$JOB_ID elapsed=${allocation_elapsed}s ${queue_state:-state=transitioning}"
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
allocation_output="$(cat "$ALLOCATION_LOG")"
if [[ "$allocation_rc" -ne 0 ]]; then
    exit "$allocation_rc"
fi
granted_job_id="$(
    printf '%s\n' "$allocation_output" \
        | sed -n 's/.*Granted job allocation \([0-9][0-9]*\).*/\1/p' \
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
if [[ "${#allocation_nodes[@]}" -ne "$PHYSICAL_NODES" ]]; then
    echo "Expected $PHYSICAL_NODES allocation nodes, got ${allocation_nodes[*]}" >&2
    exit 1
fi
node_list="$(
    IFS=' '
    printf '%s' "${allocation_nodes[*]}"
)"
log "allocation job=$JOB_ID nodes=$node_list"

log "capturing and validating the 16-rank task map"
# shellcheck disable=SC2016
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --mpi=pmix \
    --oversubscribe \
    --cpu-bind=none \
    --nodes="$PHYSICAL_NODES" \
    --ntasks="$WORLD_SIZE" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --kill-on-bad-exit=1 \
    bash -c \
        'printf "%s\t%s\t%s\t%s\n" "$SLURM_PROCID" "$(hostname)" "$SLURM_LOCALID" "${CUDA_VISIBLE_DEVICES:-unset}"' \
    | sort -n -k1,1 > "$RANK_MAP_FILE"
python3 - "$RANK_MAP_FILE" <<'PY'
import collections
import csv
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    rows = list(csv.reader(stream, delimiter="\t"))
if len(rows) != 16:
    raise SystemExit(f"rank map has {len(rows)} rows, expected 16")
ranks = [int(row[0]) for row in rows]
if ranks != list(range(16)):
    raise SystemExit(f"rank map is not exactly 0..15: {ranks}")
hosts = collections.Counter(row[1] for row in rows)
if len(hosts) != 4 or set(hosts.values()) != {4}:
    raise SystemExit(f"expected four hosts with four ranks each: {hosts}")
local_ranks = collections.defaultdict(list)
for row in rows:
    local_ranks[row[1]].append(int(row[2]))
for host, values in local_ranks.items():
    if sorted(values) != list(range(4)):
        raise SystemExit(f"{host} local ranks are not 0..3: {values}")
print(f"validated rank map: {dict(hosts)}")
PY
cat "$RANK_MAP_FILE"

log "capturing per-node NVLink fabric state"
FABRIC_PROBE="${GITHUB_WORKSPACE}/utils/bench_offline/gb300_fabric.py"
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --nodes="$PHYSICAL_NODES" \
    --ntasks="$PHYSICAL_NODES" \
    --ntasks-per-node=1 \
    --cpu-bind=none \
    --kill-on-bad-exit=1 \
    python3 "$FABRIC_PROBE" probe \
        --expected-gpus="$GPUS_PER_NODE" \
    2>&1 | tee "$TOPOLOGY_FILE"
fabric_validation="$(
    python3 "$FABRIC_PROBE" validate-log \
        "$TOPOLOGY_FILE" \
        --expected-nodes="$PHYSICAL_NODES" \
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
log "validated one 16-GPU NVLink fabric cluster_uuid=$FABRIC_CLUSTER_UUID clique_id=$FABRIC_CLIQUE_ID"

export BENCH_ID
export TRT_BENCH_TELEMETRY_DIR="$GITHUB_WORKSPACE"
log "starting one-second GPU telemetry on every physical node"
# shellcheck disable=SC2016
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --nodes="$PHYSICAL_NODES" \
    --ntasks="$PHYSICAL_NODES" \
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

export GLOBAL_BATCH_SIZE
export MODEL_PATH
export DATASET_PATH
export DATASET_REVISION
export WORKER_TIMEOUT
export IMAGE
export TRT_BENCH_CACHE_ROOT
export TRT_BENCH_GIT_REVISION="$BENCH_GIT_REVISION"
export TRT_BENCH_HARDWARE_PROFILE="gb300"
export TRT_BENCH_EXTERNAL_MPI="1"
export TRT_BENCH_ALLOCATION_JOB_ID="$JOB_ID"
export TRT_BENCH_ALLOCATION_NODE="${allocation_nodes[0]}"
export TRT_BENCH_SLURM_NODELIST="$node_list"
TRT_BENCH_COMPLETION_FILE="/workspace/$(basename "$COMPLETION_FILE")"
export TRT_BENCH_COMPLETION_FILE
TRT_BENCH_RANK_MAP_ARTIFACT="$(basename "$RANK_MAP_FILE")"
TRT_BENCH_TOPOLOGY_ARTIFACT="$(basename "$TOPOLOGY_FILE")"
export TRT_BENCH_RANK_MAP_ARTIFACT
export TRT_BENCH_TOPOLOGY_ARTIFACT
export TRT_BENCH_FABRIC_CLUSTER_UUID="$FABRIC_CLUSTER_UUID"
export TRT_BENCH_FABRIC_CLIQUE_ID="$FABRIC_CLIQUE_ID"
export UCX_TLS="cuda_ipc,cuda_copy,sm,self,tcp"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/inferencex-offline-pycache
export PYTHONPATH="/workspace/utils/bench_offline:${PYTHONPATH:-}"
export ENROOT_ALLOW_DEV=yes
export NCCL_GRAPH_MIXING_SUPPORT=0
export MIMALLOC_PURGE_DELAY=0
export PYTHONWARNINGS="ignore::DeprecationWarning:cutlass.cute.core"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TLLM_LOG_LEVEL=INFO
export TRTLLM_ENABLE_PDL=1
export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1
export TRTLLM_EPLB_SHM_NAME="offline_${JOB_ID}_${BENCH_ID}"

EXTERNAL_WORK_DIR="/workspace/.offline_work_${BENCH_ID}_${JOB_ID}"
FIXED_BATCH_ARM_FILE="${EXTERNAL_WORK_DIR}/fixed_batch_barrier.armed.json"
PERFECT_ROUTER_MARKER="${EXTERNAL_WORK_DIR}/perfect_router.jsonl"
CUTE_CACHE_DIR="${TRT_BENCH_CACHE_ROOT}/cute-dsl"
RANK_ENV_RECORDS="$(mktemp /tmp/offline-trt-rank-env.XXXXXX)"
python3 \
    "$GITHUB_WORKSPACE/utils/bench_offline/emit_rank_environment.py" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --fixed-batch-arm-file "$FIXED_BATCH_ARM_FILE" \
    --marker-file "$PERFECT_ROUTER_MARKER" \
    --cute-cache-dir "$CUTE_CACHE_DIR" \
    --hardware-profile gb300 \
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
log "preseeded external MPI rank environment exports=$rank_env_exports cleared=$rank_env_unsets"
log "rank marker=$PERFECT_ROUTER_MARKER arm_file=$FIXED_BATCH_ARM_FILE"

CONTAINER_MOUNTS="${GITHUB_WORKSPACE}:/workspace"
CONTAINER_MOUNTS+=",/scratch/models:/scratch/models"
CONTAINER_MOUNTS+=",${DATASET_ROOT}:${DATASET_ROOT}"
CONTAINER_MOUNTS+=",${TRT_BENCH_CACHE_ROOT}:${TRT_BENCH_CACHE_ROOT}"
CONTAINER_MOUNTS+=",${LOAD_BALANCER_ROOT}:/dsv4-eplb-configs"

log "starting external TRT world with trtllm-llmapi-launch"
log "image=$IMAGE model=$MODEL_PATH global_batch=$GLOBAL_BATCH_SIZE"
world_started="$SECONDS"
last_world_heartbeat=-1
set +e
srun \
    --jobid="$JOB_ID" \
    --overlap \
    --mpi=pmix \
    --oversubscribe \
    --cpu-bind=verbose,none \
    --nodes="$PHYSICAL_NODES" \
    --ntasks="$WORLD_SIZE" \
    --ntasks-per-node="$GPUS_PER_NODE" \
    --kill-on-bad-exit=1 \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --no-container-mount-home \
    --container-workdir=/workspace \
    --no-container-entrypoint \
    --export=ALL,GITHUB_WORKSPACE=/workspace \
    bash -c '
        if command -v numactl >/dev/null 2>&1; then
            exec numactl -m 0,1 \
                trtllm-llmapi-launch \
                bash /workspace/benchmarks/single_node/offline/run_dsv4_trt_container.sh
        fi
        exec trtllm-llmapi-launch \
            bash /workspace/benchmarks/single_node/offline/run_dsv4_trt_container.sh
    ' &
WORLD_STEP_PID=$!
set -e

while kill -0 "$WORLD_STEP_PID" >/dev/null 2>&1; do
    if [[ -s "$COMPLETION_FILE" ]]; then
        break
    fi
    world_elapsed=$((SECONDS - world_started))
    if (( world_elapsed / 60 > last_world_heartbeat )); then
        last_world_heartbeat=$((world_elapsed / 60))
        rank_events=0
        if [[ -s "$PERFECT_ROUTER_MARKER" ]]; then
            rank_events="$(wc -l < "$PERFECT_ROUTER_MARKER")"
        fi
        result_ready=no
        if [[ -s "$RESULT_FILE" ]]; then
            result_ready=yes
        fi
        log "external TRT world active elapsed=${world_elapsed}s rank_events=$rank_events result_ready=$result_ready"
    fi
    sleep 1
done

if [[ ! -s "$COMPLETION_FILE" ]]; then
    set +e
    wait "$WORLD_STEP_PID"
    world_rc=$?
    set -e
    WORLD_STEP_PID=""
    for _ in 1 2 3 4 5; do
        if [[ -s "$COMPLETION_FILE" ]]; then
            break
        fi
        sleep 1
    done
    if [[ ! -s "$COMPLETION_FILE" ]]; then
        log "external TRT world exited without controller completion return_code=$world_rc"
        if [[ "$world_rc" -eq 0 ]]; then
            exit 1
        fi
        exit "$world_rc"
    fi
fi

read -r completion_rc completion_status result_status < <(
    python3 - "$COMPLETION_FILE" "$RESULT_FILE" <<'PY'
import json
import sys

completion_path, result_path = sys.argv[1:]
with open(completion_path, encoding="utf-8") as stream:
    completion = json.load(stream)
return_code = int(completion["return_code"])
completion_status = str(completion["result_status"])
try:
    with open(result_path, encoding="utf-8") as stream:
        status = str(json.load(stream).get("status", "missing"))
except FileNotFoundError:
    status = "missing"
print(return_code, completion_status, status)
PY
)
world_elapsed=$((SECONDS - world_started))
log "controller finalized elapsed=${world_elapsed}s return_code=$completion_rc completion_status=$completion_status result_status=$result_status"
log "canceling Slurm allocation job=$JOB_ID to terminate external MPI management ranks"
scancel "$JOB_ID" >/dev/null 2>&1 || true
if [[ -n "$WORLD_STEP_PID" ]]; then
    kill "$WORLD_STEP_PID" >/dev/null 2>&1 || true
    set +e
    wait "$WORLD_STEP_PID"
    world_rc=$?
    set -e
    WORLD_STEP_PID=""
else
    world_rc="${world_rc:-0}"
fi
log "external TRT world stopped transport_return_code=$world_rc"

if [[ "$completion_status" != "$result_status" ]]; then
    log "completion/result status mismatch completion=$completion_status result=$result_status"
    exit 1
fi
if [[ "$completion_rc" -ne 0 || "$result_status" != "success" ]]; then
    exit 1
fi

log "GB300 benchmark command completed"
