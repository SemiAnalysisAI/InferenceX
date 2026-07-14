#!/usr/bin/env bash

# Adapt one bundled DSV4 generation-only case to the InferenceX multi-node contract.

set -euo pipefail

required=(
    CONFIG_FILE RESULT_FILENAME GITHUB_WORKSPACE RUNNER_NAME MODEL
    CONC_LIST ISL OSL PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP
    PREFILL_DP_ATTN DECODE_NUM_WORKERS DECODE_TP DECODE_EP
    DECODE_DP_ATTN TRT_GEN_ONLY_SQUASH_FILE MODEL_PATH
)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable ${name} is unset" >&2
        exit 1
    fi
done

if [[ $(wc -w <<<"${CONC_LIST}") -ne 1 ]]; then
    echo "ERROR: each offline matrix row must select exactly one concurrency: ${CONC_LIST}" >&2
    exit 1
fi
CONCURRENCY=${CONC_LIST}
SOURCE_CONFIG="${GITHUB_WORKSPACE}/${CONFIG_FILE}"
if [[ ! -f "${SOURCE_CONFIG}" ]]; then
    echo "ERROR: source config does not exist: ${SOURCE_CONFIG}" >&2
    exit 1
fi

CASE_NAME=$(basename "${SOURCE_CONFIG}" .yaml)
RUN_KEY=$(printf '%s' "${RESULT_FILENAME}-${CASE_NAME}" | sha1sum | cut -c1-12)
RUN_ROOT="/data/home/sa-shared/gharunners/trt-gen-only/${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUN_KEY}"
LOG_DIR="${RUN_ROOT}/logs"
EFFECTIVE_CONFIG="${RUN_ROOT}/effective_config.yaml"
METADATA_FILE="${RUN_ROOT}/case_metadata.json"
DATASET_ROOT="/data/home/sa-shared/gharunners/datasets/dsv4-trt-offline"
ADAPTER="${GITHUB_WORKSPACE}/utils/bench_offline/trt_disagg_gen_only.py"
SUBMIT="${GITHUB_WORKSPACE}/benchmarks/multi_node/offline/trtllm_gen_only/benchmark/scripts/submit.py"
CONTAINER_MOUNT="/data/:/data/,/scratch/:/scratch/,${GITHUB_WORKSPACE}:/workspace"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${DATASET_ROOT}"

archive_logs() {
    if [[ -d "${RUN_ROOT}" ]]; then
        tar czf "${GITHUB_WORKSPACE}/multinode_server_logs.tar.gz" \
            -C "${RUN_ROOT}" . 2>/dev/null || true
    fi
}
trap archive_logs EXIT

python3 "${ADAPTER}" prepare \
    --source-config "${SOURCE_CONFIG}" \
    --output-config "${EFFECTIVE_CONFIG}" \
    --metadata-output "${METADATA_FILE}" \
    --partition "${SLURM_PARTITION:-batch_1}" \
    --account "${SLURM_ACCOUNT:-benchmark}" \
    --job-time "${SLURM_JOB_TIME:-03:00:00}" \
    --job-name "${RUNNER_NAME}" \
    --container-image "${TRT_GEN_ONLY_SQUASH_FILE}" \
    --container-mount "${CONTAINER_MOUNT}" \
    --model-path "${MODEL_PATH}" \
    --dataset-root "${DATASET_ROOT}" \
    --log-dir "${LOG_DIR}" \
    --decode-steps "${OSL}" \
    --concurrency "${CONCURRENCY}" \
    --prefill-num-workers "${PREFILL_NUM_WORKERS}" \
    --prefill-tp "${PREFILL_TP}" \
    --prefill-ep "${PREFILL_EP}" \
    --prefill-dp-attn "${PREFILL_DP_ATTN}" \
    --decode-num-workers "${DECODE_NUM_WORKERS}" \
    --decode-tp "${DECODE_TP}" \
    --decode-ep "${DECODE_EP}" \
    --decode-dp-attn "${DECODE_DP_ATTN}"

export TRT_GEN_ONLY_DATASET_GENERATOR="/workspace/benchmarks/multi_node/offline/trtllm_gen_only/dataset/gen_dataset.sh"
python3 "${SUBMIT}" \
    --config "${EFFECTIVE_CONFIG}" \
    --log-dir "${LOG_DIR}" \
    --wait

CTX_GPUS=$((PREFILL_NUM_WORKERS * PREFILL_TP))
GEN_GPUS=$((DECODE_NUM_WORKERS * DECODE_TP))
TOTAL_GPUS=$((CTX_GPUS + GEN_GPUS))
RAW_BASENAME="${RESULT_FILENAME}_${CASE_NAME}_conc${CONCURRENCY}_gpus_${TOTAL_GPUS}_ctx_${CTX_GPUS}_gen_${GEN_GPUS}"
RAW_RESULT="${GITHUB_WORKSPACE}/${RAW_BASENAME}.json"

python3 "${ADAPTER}" result \
    --config "${EFFECTIVE_CONFIG}" \
    --log-dir "${LOG_DIR}" \
    --output "${RAW_RESULT}" \
    --source-config "${CASE_NAME}.yaml" \
    --model-id "${MODEL}" \
    --decode-steps "${OSL}"

echo "Offline disaggregated result: ${RAW_RESULT}"
