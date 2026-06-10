#!/usr/bin/env bash

set -u

logs_dir="${1:-}"
workspace="${2:-${GITHUB_WORKSPACE:-$(pwd)}}"

if [[ -z "$logs_dir" ]]; then
    echo "Dynamo SpeedBench AL: missing logs directory argument" >&2
    exit 0
fi

if [[ "${FRAMEWORK:-}" != dynamo* || "${SPEC_DECODING:-none}" != "mtp" ]]; then
    echo "Dynamo SpeedBench AL: skipping FRAMEWORK=${FRAMEWORK:-unknown} SPEC_DECODING=${SPEC_DECODING:-none}"
    exit 0
fi

mtp="${SPEEDBENCH_NUM_SPEC_TOKENS:-${NUM_SPEC_TOKENS:-${SPECULATIVE_DRAFT_TOKENS:-}}}"
if [[ -z "$mtp" && -n "${CONFIG_FILE:-}" ]]; then
    config_path="${CONFIG_FILE%%:*}"
    if [[ -f "$config_path" ]]; then
        mtp="$(sed -n 's/.*num_speculative_tokens[^0-9]*\([0-9][0-9]*\).*/\1/p' "$config_path" | head -1)"
    fi
fi
mtp="${mtp:-2}"

mode="${SPEEDBENCH_THINKING_MODE:-}"
if [[ -z "$mode" ]]; then
    if [[ "${MODEL_PREFIX:-}" == "dsv4" ]]; then
        mode="on"
    else
        mode="off"
    fi
fi

model_name="${MODEL_NAME:-${MODEL:-}}"
if [[ -z "$model_name" ]]; then
    model_name="${SERVED_MODEL_NAME:-unknown}"
fi

output="${workspace}/results_speedbench_al_${mode}_mtp${mtp}.json"
metric_source="dynamo-decode-log-counters"
if [[ -n "${FRAMEWORK:-}" ]]; then
    metric_source="${FRAMEWORK}-decode-log-counters"
fi

echo "Dynamo SpeedBench AL: parsing decode logs from $logs_dir"
python3 "${workspace}/utils/evals/dynamo_speedbench_al_from_logs.py" \
    --logs-dir "$logs_dir" \
    --output "$output" \
    --reference-yaml "${workspace}/benchmarks/speedbench-reference-al.yaml" \
    --model "$model_name" \
    --model-prefix "${MODEL_PREFIX:-}" \
    --thinking-mode "$mode" \
    --num-speculative-tokens "$mtp" \
    --framework "${FRAMEWORK:-dynamo}" \
    --metric-source "$metric_source" || true
