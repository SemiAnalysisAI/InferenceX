#!/usr/bin/env bash

# MiniMax-M3 B300 vLLM SPEED-Bench AL matrix collector for EAGLE3 speculative decoding.
#
# For each thinking mode (on/off) and EAGLE3 level (num_speculative_tokens), measure
# the REAL acceptance length (AL) on one SPEED-Bench category and emit a YAML matrix
# in the golden_al_distribution shape. The synthetic value is injected downstream by
# the throughput recipe, not here.
#
# Draft: Inferact/MiniMax-M3-EAGLE3. The EAGLE3 head is MHA and must use FLASH_ATTN
# (FlashInfer only supports page size 128 through its trtllm-gen kernel, which needs
# GQA/MQA); the target keeps its default FlashInfer backend. --block-size 128 is
# mandatory for the MSA sparse/index cache. --language-model-only frees the vision
# encoder's VRAM (text-only benchmark).
#
# The *_fp4_* filename is only the naming convention speedbench-al.yml requires
# (${model-prefix}_fp4_b300_vllm.sh); the staged MiniMax-M3 weights are unquantized
# BF16, so no quantization-specific flags (--moe-backend marlin, --kv-cache-dtype
# fp8) apply.
#
# Dispatch this collector through speedbench-al.yml.
#
# Required collection settings come from speedbench-al.yml.

set -o pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON DP_ATTENTION EP_SIZE MODEL MODEL_PATH \
    MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

SERVE_MODEL="${MODEL_PATH}"
GPU_MEM_UTIL="0.90"

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"

MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
CONCURRENCY="1"
# Official MiniMax-M3 sampling: temperature 1.0, top_p 0.95, top_k 40.
TEMPERATURE="1.0"
TOP_P="0.95"
TOP_K="40"
# M3 thinking toggles via the thinking_mode chat_template key.
CHAT_TEMPLATE_KWARGS_OFF='{"thinking_mode": "disabled"}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"

export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi
if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi

# dirname(MODEL_PATH) is the read-only staged mount (/scratch/models), so the draft
# must go to a writable workspace dir, not next to the target.
echo "=== Downloading EAGLE3 draft model ($DRAFT_MODEL) ==="
DRAFT_DIR="/workspace/draft_models"
mkdir -p "$DRAFT_DIR"
DRAFT_MODEL_PATH="$DRAFT_DIR/${DRAFT_MODEL##*/}"
if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
    hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
fi

echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP" --enable-expert-parallel)
elif [ "${EP_SIZE}" -gt 1 ]; then
    PARALLEL_ARGS=(--tensor-parallel-size "$TP" --enable-expert-parallel)
else
    # Plain TP per the official MiniMax-M3 recipe. Do NOT force a MoE backend: the
    # staged checkpoint is unquantized BF16, for which marlin is rejected.
    PARALLEL_ARGS=(--tensor-parallel-size "$TP")
fi

fetch_metric() {
    local port="$1" name="$2"
    curl -s "http://localhost:${port}/metrics" \
      | grep -oP "${name}\\{[^}]*\\} \\K[0-9.]+" || echo "0"
}

SERVER_PID=""
_descendants() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        echo "$child"
        _descendants "$child"
    done
}
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        local descendants
        descendants=$(_descendants "$SERVER_PID")
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        local pid
        for pid in $descendants; do
            kill -9 "$pid" 2>/dev/null || true
        done
        local waited=0
        while [[ $waited -lt 120 ]]; do
            local used
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
            if [[ -z "$used" || "$used" -lt 2000 ]]; then break; fi
            sleep 3; waited=$((waited + 3))
        done
        SERVER_PID=""
    fi
}
trap 'cleanup_server' EXIT

start_gpu_monitor

declare -A AL_RESULT

run_cell() {
    local mode="$1" mtp="$2"
    local think_args=()
    if [[ "$mode" == "on" && -n "$CHAT_TEMPLATE_KWARGS_ON" ]]; then
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
    elif [[ "$mode" == "off" && -n "$CHAT_TEMPLATE_KWARGS_OFF" ]]; then
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_OFF")
    fi

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  EAGLE3=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        "${PARALLEL_ARGS[@]}"
        --pipeline-parallel-size 1
        --block-size 128
        --language-model-only
        --max-cudagraph-capture-size 2048
        --trust-remote-code
        --no-enable-prefix-caching
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-model-len 16384
        --max-num-batched-tokens 16384
        --stream-interval 30
        --speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $mtp, \"attention_backend\": \"FLASH_ATTN\"}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
        echo "  -> server failed to start (thinking=$mode eagle3=$mtp), recording N/A"
        AL_RESULT["${mode}_${mtp}"]="N/A"
        cleanup_server
        return
    fi

    local acc_before drf_before acc_after drf_after
    acc_before=$(fetch_metric "$PORT" "vllm:spec_decode_num_accepted_tokens_total")
    drf_before=$(fetch_metric "$PORT" "vllm:spec_decode_num_drafts_total")

    vllm bench serve \
        --model "$SERVE_MODEL" \
        --port "$PORT" \
        --dataset-name speed_bench \
        --dataset-path "$SPEEDBENCH_DIR" \
        --speed-bench-category "$CATEGORY" \
        --speed-bench-output-len "$SPEEDBENCH_OUTPUT_LEN" \
        --num-prompts -1 \
        --max-concurrency "$CONCURRENCY" \
        --save-result \
        --save-detailed \
        --result-dir "$RESULTS_DIR" \
        --result-filename "speedbench_${mode}_mtp${mtp}" \
        --trust-remote-code \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        "${think_args[@]}"

    acc_after=$(fetch_metric "$PORT" "vllm:spec_decode_num_accepted_tokens_total")
    drf_after=$(fetch_metric "$PORT" "vllm:spec_decode_num_drafts_total")

    local delta_acc delta_drf al
    delta_acc=$(awk "BEGIN {printf \"%d\", $acc_after - $acc_before}")
    delta_drf=$(awk "BEGIN {printf \"%d\", $drf_after - $drf_before}")
    if [[ "$delta_drf" -gt 0 ]]; then
        al=$(awk "BEGIN {printf \"%.2f\", 1 + ($delta_acc / $delta_drf)}")
    else
        al="N/A"
    fi
    echo "  -> thinking=$mode EAGLE3=$mtp AL=$al (accepted=$delta_acc drafts=$delta_drf)"
    AL_RESULT["${mode}_${mtp}"]="$al"

    cleanup_server
}

for mode in $THINKING_MODES; do
    for mtp in $MTP_LIST; do
        run_cell "$mode" "$mtp"
    done
done

stop_gpu_monitor

emit_mode_block() {
    local mode="$1"
    for mtp in $MTP_LIST; do
        echo "    $mtp: ${AL_RESULT[${mode}_${mtp}]:-N/A}"
    done
}

{
    echo "# Acceptance Length (AL) reference values measured with SPEED-Bench."
    echo "# dataset: $CATEGORY | temperature: $TEMPERATURE | top_p: $TOP_P | top_k: $TOP_K | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# thinking_off chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_OFF"
    echo "# Measured on $MODEL_KEY (B300, vLLM EAGLE3), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/minimaxm3_fp4_b300_vllm.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (EAGLE3 level); value = golden AL"
    echo "${MODEL_KEY}:"
    if [[ " $THINKING_MODES " == *" on "* ]]; then
        echo "  thinking_on:"
        emit_mode_block on
    fi
    if [[ " $THINKING_MODES " == *" off "* ]]; then
        echo "  thinking_off:"
        emit_mode_block off
    fi
} > "$OUT_YAML"

echo ""
echo "=========================================="
echo "  SPEED-Bench AL matrix written to: $OUT_YAML"
echo "=========================================="
cat "$OUT_YAML"
