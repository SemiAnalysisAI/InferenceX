#!/usr/bin/env bash
# One isolated B200 allocation: build or reuse an image, then test a fresh container.
set -eo pipefail
source benchmarks/benchmark_lib.sh --validation-only
check_env_vars GITHUB_WORKSPACE GITHUB_RUN_ID GITHUB_RUN_ATTEMPT RUNNER_NAME \
    MODEL_PREFIX FRAMEWORK PRECISION GPU_COUNT TP CONC EVAL_ONLY EVAL_FRAMEWORK \
    EVAL_SUITE SLURM_PARTITION SLURM_ACCOUNT B200_SQUASH_DIR TRT_SOURCE_SHA \
    TRT_DEVEL_IMAGE TRT_BUILD_JOBS TRT_CUDA_ARCHS TRT_BUILD_TIME_LIMIT TRT_IMAGE_MODE
[[ "$GITHUB_RUN_ID" =~ ^[0-9]+$ && "$GITHUB_RUN_ATTEMPT" =~ ^[0-9]+$ ]]
[[ "$TRT_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$MODEL_PREFIX/$FRAMEWORK/$PRECISION/$TP/$GPU_COUNT/$CONC" == minimaxm3/trt/fp4/8/8/1 ]]
[[ "$EVAL_ONLY/$EVAL_FRAMEWORK/$EVAL_SUITE" == true/bfcl/bfcl_responses_smoke ]]
[[ "${RUNNER_NAME%%_*}" == b200-nscale-slurm || "${RUNNER_NAME%%_*}" == b200-nscale-compat ]]

PROBE_ID="infx-trt-main-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
JOB_NAME="${RUNNER_NAME}-${PROBE_ID}"
EVIDENCE="$GITHUB_WORKSPACE/trt-main-build-evidence"
BUILD_SCRATCH="/scratch/$PROBE_ID"
CANDIDATE_IMAGE="$B200_SQUASH_DIR/${PROBE_ID}-${TRT_SOURCE_SHA:0:12}.sqsh"
case "$TRT_IMAGE_MODE" in
    build) ;;
    reuse)
        check_env_vars TRT_REUSE_IMAGE TRT_REUSE_IMAGE_SHA256
        [[ "$TRT_REUSE_IMAGE" == "$B200_SQUASH_DIR"/infx-trt-main-*.sqsh ]]
        [[ "$TRT_REUSE_IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]]
        CANDIDATE_IMAGE="$TRT_REUSE_IMAGE"
        ;;
    *) echo "Unsupported experimental TRT image mode: $TRT_IMAGE_MODE" >&2; exit 1 ;;
esac
mkdir -p "$EVIDENCE"
JOB_ID=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$JOB_ID" ]]; then
        scancel "$JOB_ID" || true
    else
        # Covers cancellation while salloc waits; the job name is unique to this attempt.
        scancel --user="$USER" --name="$JOB_NAME" || true
    fi
    printf '%s\n' "$status" > "$EVIDENCE/exit-code.txt"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$EVIDENCE/started-at.txt"
salloc --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
    --nodes=1 --gres="gpu:$GPU_COUNT" --exclusive --mem=0 \
    --time="$TRT_BUILD_TIME_LIMIT" --no-shell --job-name="$JOB_NAME" \
    2>&1 | tee "$EVIDENCE/allocation.log"
JOB_ID=$(sed -n 's/.*Granted job allocation \([0-9][0-9]*\).*/\1/p' "$EVIDENCE/allocation.log" | tail -n1)
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
scontrol show job "$JOB_ID" > "$EVIDENCE/slurm-job.txt"
printf '%s\n' "$CANDIDATE_IMAGE" > "$EVIDENCE/candidate-image-path.txt"
printf '%s\n' "$TRT_IMAGE_MODE" > "$EVIDENCE/image-mode.txt"

if [[ "$TRT_IMAGE_MODE" == build ]]; then
    printf '%s\n' "$BUILD_SCRATCH" > "$EVIDENCE/build-scratch-path.txt"
    srun --jobid="$JOB_ID" --ntasks=1 bash -c '
        set -e
        test ! -e "$1"
        mkdir "$1"
        df -h "$1" "$2"
        available=$(df -Pk "$1" | awk "NR==2 {print \$4}")
        test "$available" -ge 314572800
        nvidia-smi
    ' bash "$BUILD_SCRATCH" "$B200_SQUASH_DIR" | tee "$EVIDENCE/node-preflight.log"

    # Pyxis remaps container root to the runner UID. Only this writable container is changed.
    # Its saved image excludes the mounted source/build/evidence directories.
    timeout --signal=TERM --kill-after=60 4h srun --jobid="$JOB_ID" --ntasks=1 \
        --container-image="$TRT_DEVEL_IMAGE" --container-writable --container-remap-root \
        --container-save="$CANDIDATE_IMAGE" --no-container-mount-home \
        --no-container-entrypoint --container-workdir=/ \
        --container-mounts="$GITHUB_WORKSPACE:/infx:ro,$BUILD_SCRATCH:/trt-build,$EVIDENCE:/build-evidence" \
        --export=ALL bash /infx/experimental/bfcl/build-trt-main.sh \
        2>&1 | tee "$EVIDENCE/build.log"
    test -f "$EVIDENCE/build-completed-at.txt"
else
    srun --jobid="$JOB_ID" --ntasks=1 bash -c '
        set -e
        printf "%s  %s\n" "$2" "$1" | sha256sum --check
        nvidia-smi
    ' bash "$CANDIDATE_IMAGE" "$TRT_REUSE_IMAGE_SHA256" | tee "$EVIDENCE/reused-image-check.log"
fi
srun --jobid="$JOB_ID" --ntasks=1 bash -c '
    set -e
    unsquashfs -s "$1"
    sha256sum "$1"
' bash "$CANDIDATE_IMAGE" | tee "$EVIDENCE/image.sha256"

export MODEL_PATH=/scratch/models/MiniMax-M3-NVFP4 MODEL=/scratch/models/MiniMax-M3-NVFP4
export AIPERF_MMAP_CACHE_HOST_PATH=/data/home/sa-shared/gharunners/aiperf-cache
export TRT_LLM_GIT_COMMIT="$TRT_SOURCE_SHA"
export PORT=8888 AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache
date -u +%FT%TZ > "$EVIDENCE/evaluation-started-at.txt"
srun --jobid="$JOB_ID" --ntasks=1 --container-image="$CANDIDATE_IMAGE" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace,$MODEL_PATH:$MODEL_PATH,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache" \
    --no-container-mount-home --container-workdir=/workspace --no-container-entrypoint \
    --container-env=TRT_LLM_GIT_COMMIT --export=ALL \
    bash -c '
        set -e
        python3 experimental/bfcl/verify-trt-main-image.py
        cp /opt/inferencex-trt-main-build.json trt-main-build-evidence/build-manifest.json
        TRT_LLM_VERSION=$(python3 -c '\''import json; print(json.load(open("/opt/inferencex-trt-main-build.json"))["package_version"])'\'')
        export TRT_LLM_VERSION
        bash benchmarks/single_node/agentic/minimaxm3_fp4_b200_trt_mtp.sh
    ' 2>&1 | tee "$EVIDENCE/evaluation.log"
date -u +%FT%TZ > "$EVIDENCE/evaluation-completed-at.txt"
