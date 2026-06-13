#!/usr/bin/env bash

set -Eeuo pipefail

source /workspace/benchmarks/benchmark_lib.sh

: "${CONC:?CONC is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${DATASET_REVISION:?DATASET_REVISION is required}"

WORKER_TIMEOUT="${WORKER_TIMEOUT:-3600}"
BENCH_ID="${BENCH_ID:-conc${CONC}}"
EXPERIMENT_CONFIG_B64="${EXPERIMENT_CONFIG_B64:-}"
TRT_BENCH_CACHE_ROOT="${TRT_BENCH_CACHE_ROOT:-/data/trtllm-cache/dsv4-c185066-sm100a}"
EXECUTION_MODE="serial-tuning"
if [[ -n "$EXPERIMENT_CONFIG_B64" ]]; then
    EXECUTION_MODE="single-candidate"
fi
if [[ ! "$BENCH_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid BENCH_ID: $BENCH_ID" >&2
    exit 1
fi
ALLOCATION_JOB_ID="${SLURM_JOB_ID:-unknown}"
ALLOCATION_NODE="${SLURMD_NODENAME:-unknown}"
WORK_DIR="/tmp/inferencex-trt-offline-${BENCH_ID}-${ALLOCATION_JOB_ID}-$$"
RESULT_FILE="/workspace/offline_result_${BENCH_ID}.json"
CONTROLLER_LOG="/workspace/offline_controller_${BENCH_ID}.log"
DEBUG_ARCHIVE="/workspace/offline_debug_${BENCH_ID}.tar.gz"
PROFILE_ARCHIVE="/workspace/offline_profiles_${BENCH_ID}.tar.gz"
PROFILE_SUMMARY="/workspace/offline_profile_summary_${BENCH_ID}.json"
GPU_METRICS="/workspace/offline_gpu_metrics_${BENCH_ID}.csv"

log() {
    printf '[offline-trt-launcher %s] %s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

mkdir -p "$WORK_DIR"

finalize() {
    local rc=$?
    trap - EXIT
    set +e
    log "finalizing benchmark concurrency=$CONC return_code=$rc"
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
        python3 - "$RESULT_FILE" "$CONC" "$rc" "$BENCH_ID" <<'PY'
import json
import sys

path, concurrency, return_code, experiment_id = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump(
        {
            "schema_version": 1,
            "status": "failed",
            "benchmark": {
                "concurrency": int(concurrency),
                "experiment_id": experiment_id,
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
    if find "$WORK_DIR" \
        -maxdepth 1 \
        -type f \
        -name '*_torch_profile-rank-*.json' \
        -print -quit \
        | grep -q .; then
        if python3 /workspace/utils/bench_offline/summarize_profile.py \
            "$WORK_DIR" \
            --json-out "$WORK_DIR/profile_summary.json" \
            >/dev/null; then
            cp "$WORK_DIR/profile_summary.json" "$PROFILE_SUMMARY"
            log "profile summary finalized path=$PROFILE_SUMMARY"
        else
            log "profile summary failed; raw traces remain available"
        fi
    fi
    profile_file_list="$WORK_DIR/profile_files.txt"
    find "$WORK_DIR" \
        -maxdepth 1 \
        -type f \
        \( \
            -name '*_torch_profile-rank-*.json' \
            -o -name '*_profile_manifest.json' \
            -o -name 'profile_summary.json' \
        \) \
        -printf '%f\n' \
        | sort > "$profile_file_list"
    profile_tar_excludes=()
    if [[ -s "$profile_file_list" ]]; then
        if tar \
            -C "$WORK_DIR" \
            -czf "$PROFILE_ARCHIVE" \
            -T "$profile_file_list" 2>/dev/null; then
            profile_tar_excludes+=(--exclude='*_torch_profile-rank-*.json')
            log "profile artifacts finalized archive=$PROFILE_ARCHIVE"
        else
            rc=1
            log "profile archive failed; preserving traces in debug archive"
            python3 - "$RESULT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    result = json.load(stream)
result.update(
    {
        "status": "failed",
        "failure_kind": "profile_artifact",
        "error": "Profile archive creation failed; traces are in the debug archive",
    }
)
with open(path, "w", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
        fi
    fi
    tar \
        --exclude=corpus.bin \
        "${profile_tar_excludes[@]}" \
        -C "$WORK_DIR" \
        -czf "$DEBUG_ARCHIVE" \
        . 2>/dev/null || true
    log "artifacts finalized result=$RESULT_FILE debug=$DEBUG_ARCHIVE"
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
while IFS='=' read -r name _; do
    case "$name" in
        SLURM_*|PMIX*|PMI*|OMPI_*|ORTE_*)
            unset "$name"
            ;;
    esac
done < <(env)

export NCCL_NVLS_ENABLE=0
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

log "benchmark start concurrency=$CONC model=$MODEL_PATH"
log "benchmark id=$BENCH_ID execution=$EXECUTION_MODE"
log "dataset revision=$DATASET_REVISION path=$DATASET_PATH"
log "allocation job=$ALLOCATION_JOB_ID node=$ALLOCATION_NODE"
log "worker timeout=${WORKER_TIMEOUT}s work_dir=$WORK_DIR"
log "persistent CuTe cache path=$CUTE_DSL_CACHE_DIR"
nvidia-smi

log "starting one-second GPU telemetry"
start_gpu_monitor --output "$GPU_METRICS"

log "starting offline benchmark controller"
CONTROLLER_ARGS=(
    --model-path "$MODEL_PATH"
    --dataset "$DATASET_PATH"
    --dataset-revision "$DATASET_REVISION"
    --concurrency "$CONC"
    --output-dir "$WORK_DIR"
    --worker-timeout "$WORKER_TIMEOUT"
    --tuning-attempts 6
    --experiment-id "$BENCH_ID"
)
if [[ -n "$EXPERIMENT_CONFIG_B64" ]]; then
    EXPERIMENT_CONFIG_PATH="$WORK_DIR/experiment_config.json"
    printf '%s' "$EXPERIMENT_CONFIG_B64" \
        | base64 --decode > "$EXPERIMENT_CONFIG_PATH"
    CONTROLLER_ARGS+=(--experiment-config "$EXPERIMENT_CONFIG_PATH")
    log "decoded single-candidate config path=$EXPERIMENT_CONFIG_PATH"
fi
python3 -u /workspace/utils/bench_offline/run.py \
    "${CONTROLLER_ARGS[@]}" \
    2>&1 | tee "$CONTROLLER_LOG"
