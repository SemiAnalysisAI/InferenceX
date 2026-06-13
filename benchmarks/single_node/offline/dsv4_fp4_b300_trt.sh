#!/usr/bin/env bash

set -Eeuo pipefail

: "${CONC:?CONC is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${RUNNER_NAME:?RUNNER_NAME is required}"

IMAGE="${IMAGE:-ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066}"
MODEL_PATH="${MODEL_PATH:-/scratch/models/DeepSeek-V4-Pro}"
DATASET_REVISION="${DATASET_REVISION:-90f0394333616266d9fe85824ceaf505093cbaa5}"
DATASET_FILE="longbook_qa_eng.jsonl"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/inferencex/InfiniteBench/${DATASET_REVISION}}"
DATASET_PATH="${DATASET_ROOT}/${DATASET_FILE}"
SALLOC_TIME_LIMIT="${SALLOC_TIME_LIMIT:-500}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-3600}"
BENCH_ID="${BENCH_ID:-conc${CONC}}"
EXPERIMENT_CONFIG_B64="${EXPERIMENT_CONFIG_B64:-}"
SLURM_PARTITION="${SLURM_PARTITION:-batch_1}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-benchmark}"
BENCH_GIT_REVISION="${GITHUB_SHA:-}"
if [[ -z "$BENCH_GIT_REVISION" ]]; then
    BENCH_GIT_REVISION="$(git -C "$GITHUB_WORKSPACE" rev-parse HEAD)"
fi
if [[ ! "$BENCH_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Invalid BENCH_ID: $BENCH_ID" >&2
    exit 1
fi
RESULT_FILE="${GITHUB_WORKSPACE}/offline_result_${BENCH_ID}.json"
JOB_ID=""
rm -f "$RESULT_FILE"

cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n "$JOB_ID" ]]; then
        scancel "$JOB_ID" >/dev/null 2>&1 || true
    fi
    if [[ ! -f "$RESULT_FILE" ]]; then
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
            "error": "Host launcher exited before the container wrote a result",
            "return_code": int(return_code),
        },
        stream,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
PY
    fi
    exit "$rc"
}
trap cleanup EXIT

mkdir -p "$DATASET_ROOT"
(
    exec 9>"${DATASET_PATH}.lock"
    flock -w 600 9
    if [[ ! -s "$DATASET_PATH" ]]; then
        temporary="${DATASET_PATH}.tmp.$$"
        trap 'rm -f "$temporary"' EXIT
        curl --fail --location --retry 5 --retry-all-errors \
            "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/${DATASET_REVISION}/${DATASET_FILE}?download=true" \
            --output "$temporary"
        mv "$temporary" "$DATASET_PATH"
    fi
)

SQUASH_FILE="/data/squash/$(printf '%s' "$IMAGE" | sed 's|[/:@#]|_|g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"
(
    exec 9>"$LOCK_FILE"
    flock -w 900 9
    if unsquashfs -l "$SQUASH_FILE" >/dev/null 2>&1; then
        echo "Using cached image: $SQUASH_FILE"
    else
        rm -f "$SQUASH_FILE"
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
        enroot import -o "$SQUASH_FILE" "docker://$IMAGE"
    fi
)

set +e
allocation_output="$(
    salloc \
        --partition="$SLURM_PARTITION" \
        --account="$SLURM_ACCOUNT" \
        --nodes=1 \
        --gres=gpu:8 \
        --exclusive \
        --time="$SALLOC_TIME_LIMIT" \
        --no-shell \
        --job-name="$RUNNER_NAME" 2>&1
)"
allocation_rc=$?
set -e
printf '%s\n' "$allocation_output"
if [[ "$allocation_rc" -ne 0 ]]; then
    exit "$allocation_rc"
fi
JOB_ID="$(
    printf '%s\n' "$allocation_output" \
        | sed -n 's/.*Granted job allocation \([0-9][0-9]*\).*/\1/p' \
        | tail -n 1
)"
if [[ -z "$JOB_ID" ]]; then
    echo "Could not determine Slurm allocation ID" >&2
    exit 1
fi

srun \
    --jobid="$JOB_ID" \
    --mpi=none \
    --kill-on-bad-exit=1 \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace,/scratch/models:/scratch/models,/data/datasets:/data/datasets" \
    --no-container-mount-home \
    --container-workdir=/workspace \
    --no-container-entrypoint \
    --export="ALL,GITHUB_WORKSPACE=/workspace,CONC=$CONC,BENCH_ID=$BENCH_ID,EXPERIMENT_CONFIG_B64=$EXPERIMENT_CONFIG_B64,MODEL_PATH=$MODEL_PATH,DATASET_PATH=$DATASET_PATH,DATASET_REVISION=$DATASET_REVISION,WORKER_TIMEOUT=$WORKER_TIMEOUT,IMAGE=$IMAGE,TRT_BENCH_GIT_REVISION=$BENCH_GIT_REVISION" \
    bash benchmarks/single_node/offline/run_dsv4_trt_container.sh
