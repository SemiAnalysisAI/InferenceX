#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck disable=SC1091
source /workspace/benchmarks/benchmark_lib.sh

: "${GLOBAL_BATCH_SIZE:?GLOBAL_BATCH_SIZE is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${DATASET_REVISION:?DATASET_REVISION is required}"

WORKER_TIMEOUT="${WORKER_TIMEOUT:-7200}"
BENCH_ID="${BENCH_ID:-gbs${GLOBAL_BATCH_SIZE}}"
TRT_BENCH_HARDWARE_PROFILE="${TRT_BENCH_HARDWARE_PROFILE:-b300}"
TRT_BENCH_EXTERNAL_MPI="${TRT_BENCH_EXTERNAL_MPI:-0}"
TRT_BENCH_CACHE_ROOT="${TRT_BENCH_CACHE_ROOT:-/data/trtllm-cache/dsv4-c185066-sm100a}"
if [[ ! "$BENCH_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid BENCH_ID: $BENCH_ID" >&2
    exit 1
fi
ALLOCATION_JOB_ID="${TRT_BENCH_ALLOCATION_JOB_ID:-${SLURM_JOB_ID:-unknown}}"
ALLOCATION_NODE="${TRT_BENCH_ALLOCATION_NODE:-${SLURMD_NODENAME:-unknown}}"
if [[ "$TRT_BENCH_EXTERNAL_MPI" == "1" ]]; then
    WORK_DIR="/workspace/.offline_work_${BENCH_ID}_${ALLOCATION_JOB_ID}"
else
    WORK_DIR="/tmp/inferencex-trt-offline-${BENCH_ID}-${ALLOCATION_JOB_ID}-$$"
fi
RESULT_FILE="/workspace/offline_result_${BENCH_ID}.json"
CONTROLLER_LOG="/workspace/offline_controller_${BENCH_ID}.log"
DEBUG_ARCHIVE="/workspace/offline_debug_${BENCH_ID}.tar.gz"
GPU_METRICS="/workspace/offline_gpu_metrics_${BENCH_ID}.csv"
COMPLETION_FILE="${TRT_BENCH_COMPLETION_FILE:-}"

log() {
    printf '[offline-trt-launcher %s] %s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

start_offline_gpu_monitor() {
    local query
    query="timestamp,index,power.draw,temperature.gpu"
    query+=",clocks.current.sm,clocks.current.memory"
    query+=",utilization.gpu,utilization.memory,memory.used,memory.free"
    GPU_METRICS_CSV="$GPU_METRICS"
    export GPU_METRICS_CSV
    nvidia-smi \
        --query-gpu="$query" \
        --format=csv \
        -l 1 > "$GPU_METRICS" 2>/dev/null &
    GPU_MONITOR_PID=$!
    log "GPU telemetry started pid=$GPU_MONITOR_PID output=$GPU_METRICS"
}

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

finalize() {
    local rc=$?
    trap - EXIT
    set +e
    log "finalizing benchmark global_batch=$GLOBAL_BATCH_SIZE return_code=$rc"
    stop_gpu_monitor
    if [[ -d "${CUTE_DSL_CACHE_DIR:-}" ]]; then
        cache_files="$(
            find "$CUTE_DSL_CACHE_DIR" -type f 2>/dev/null | wc -l
        )"
        log "persistent CuTe cache files=$cache_files path=$CUTE_DSL_CACHE_DIR"
    fi
    if [[ -f "$WORK_DIR/result.json" ]]; then
        cp "$WORK_DIR/result.json" "$RESULT_FILE"
    else
        if [[ "$rc" -eq 0 ]]; then
            rc=1
        fi
        python3 - \
            "$RESULT_FILE" \
            "$GLOBAL_BATCH_SIZE" \
            "$rc" \
            "$BENCH_ID" \
            "$TRT_BENCH_HARDWARE_PROFILE" <<'PY'
import json
import sys

(
    path,
    global_batch_size,
    return_code,
    experiment_id,
    hardware_profile,
) = sys.argv[1:]
world_size = 16 if hardware_profile == "gb300" else 8
hardware = "GB300 NVL16" if hardware_profile == "gb300" else "B300"
with open(path, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "schema_version": 2,
            "status": "failed",
            "benchmark": {
                "global_batch_size": int(global_batch_size),
                "concurrency": int(global_batch_size),
                "experiment_id": experiment_id,
                "hardware": hardware,
                "hardware_profile": hardware_profile,
                "active_gpu_count": world_size,
            },
            "error": "Container benchmark exited before result.json was written",
            "return_code": int(return_code),
        },
        stream,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
PY
    fi
    tar \
        --exclude=corpus.bin \
        -C "$WORK_DIR" \
        -czf "$DEBUG_ARCHIVE" \
        . 2>/dev/null || true
    log "artifacts finalized result=$RESULT_FILE debug=$DEBUG_ARCHIVE"
    if [[ -n "$COMPLETION_FILE" ]]; then
        completion_tmp="${COMPLETION_FILE}.tmp.$$"
        result_status="$(
            python3 - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(str(json.load(stream).get("status", "missing")))
PY
        )"
        log "publishing controller completion return_code=$rc result_status=$result_status path=$COMPLETION_FILE"
        python3 - "$completion_tmp" "$rc" "$result_status" <<'PY'
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
        mv "$completion_tmp" "$COMPLETION_FILE"
    fi
    exit "$rc"
}
trap finalize EXIT

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Missing staged model: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -s "$DATASET_PATH" ]]; then
    echo "Missing pinned InfiniteBench dataset: $DATASET_PATH" >&2
    exit 1
fi

export TRT_BENCH_SLURM_JOB_ID="$ALLOCATION_JOB_ID"
export TRT_BENCH_SLURM_NODE="$ALLOCATION_NODE"
if [[ "$TRT_BENCH_EXTERNAL_MPI" != "1" ]]; then
    while IFS='=' read -r name _; do
        case "$name" in
            SLURM_*|PMIX*|PMI*|OMPI_*|ORTE_*)
                unset "$name"
                ;;
        esac
    done < <(env)
fi

if [[ "$TRT_BENCH_HARDWARE_PROFILE" == "b300" ]]; then
    export NCCL_NVLS_ENABLE=0
fi
export NCCL_GRAPH_MIXING_SUPPORT=0
export MIMALLOC_PURGE_DELAY=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1
export TRTLLM_MHC_ENABLE_FUSED_HC=1
export ENABLE_PERFECT_ROUTER=1
export TRTLLM_ENABLE_PERFECT_ROUTER=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/inferencex-offline-pycache
export PYTHONPATH="/workspace/utils/bench_offline:${PYTHONPATH:-}"
export CUTE_DSL_CACHE_DIR="$TRT_BENCH_CACHE_ROOT/cute-dsl"
# TRT's MPI pool forwards TRTLLM_* variables, so the rank-entry shim restores
# the unprefixed CuTe variable before importing TRT's worker implementation.
export TRTLLM_BENCH_CUTE_DSL_CACHE_DIR="$CUTE_DSL_CACHE_DIR"
mkdir -p "$CUTE_DSL_CACHE_DIR"

log "benchmark start hardware_profile=$TRT_BENCH_HARDWARE_PROFILE global_batch=$GLOBAL_BATCH_SIZE model=$MODEL_PATH"
log "benchmark id=$BENCH_ID execution=fixed-global-batch"
log "dataset revision=$DATASET_REVISION path=$DATASET_PATH"
log "allocation job=$ALLOCATION_JOB_ID node=$ALLOCATION_NODE nodes=${TRT_BENCH_SLURM_NODELIST:-$ALLOCATION_NODE}"
log "external MPI=$TRT_BENCH_EXTERNAL_MPI"
log "worker timeout=${WORKER_TIMEOUT}s work_dir=$WORK_DIR"
log "persistent CuTe cache path=$CUTE_DSL_CACHE_DIR"
nvidia-smi

if [[ "$TRT_BENCH_EXTERNAL_MPI" == "1" ]]; then
    log "per-node GPU telemetry is managed by the host launcher"
else
    log "starting one-second GPU telemetry with used/free memory"
    start_offline_gpu_monitor
fi

log "starting offline benchmark controller"
CONTROLLER_ARGS=(
    --model-path "$MODEL_PATH"
    --dataset "$DATASET_PATH"
    --dataset-revision "$DATASET_REVISION"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --output-dir "$WORK_DIR"
    --worker-timeout "$WORKER_TIMEOUT"
    --experiment-id "$BENCH_ID"
)
python3 -u /workspace/utils/bench_offline/run.py \
    "${CONTROLLER_ARGS[@]}" \
    2>&1 | tee "$CONTROLLER_LOG"
