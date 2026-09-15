#!/usr/bin/env bash

# Qwen3.5-397B-A17B B300 vLLM SPEED-Bench AL matrix collector.
#
# For each thinking mode (on/off) and MTP level (num_speculative_tokens), measure the
# REAL acceptance length (AL) on one SPEED-Bench category and emit a YAML matrix in
# the golden_al_distribution shape. The synthetic value is injected downstream by the
# throughput recipe, not here. Serves the NVFP4 build (Qwen3.5-397B-A17B-NVFP4).
#
# --max-cudagraph-capture-size 512: Qwen3.5 is a mamba hybrid and a large capture
# size trips the causal_conv1d assert (vLLM PR #34571). Thinking OFF must be passed
# explicitly via enable_thinking (Qwen does not treat "no kwargs" as off).
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

MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
CONCURRENCY="1"
# Model-card sampling DIFFERS by mode and MUST be passed per-mode or the AL is
# measured at the wrong settings:
#   thinking : temperature 0.6, top_p 0.95, top_k 20, presence_penalty 0.0
#   instruct : temperature 0.7, top_p 0.8,  top_k 20, presence_penalty 1.5
TEMPERATURE_ON="0.6";  TOP_P_ON="0.95";  TOP_K_ON="20";  PRESENCE_PENALTY_ON="0.0"
TEMPERATURE_OFF="0.7"; TOP_P_OFF="0.8"; TOP_K_OFF="20"; PRESENCE_PENALTY_OFF="1.5"
# Unset -> vLLM default (deterministic seed=0); vary it to measure temperature>0
# variance.
SEED="${SEED:-}"
# --save-detailed keeps per-request completions to eyeball that thinking_on emits
# <think> and thinking_off does not; off by default (bloats the result JSON).
SAVE_DETAILED="${SAVE_DETAILED:-}"
# Qwen thinking toggles via the enable_thinking chat_template key.
CHAT_TEMPLATE_KWARGS_OFF='{"enable_thinking": false}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
# Flat results dir to match the speedbench-al.yml artifact glob
# (speedbench_results/server_*.log) and its pre-run `rm -rf speedbench_results`.
RESULTS_DIR="/workspace/speedbench_results"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi
if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi

echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

NEED_SHIM=0
if [[ " $THINKING_MODES " == *" on "*  && -n "$CHAT_TEMPLATE_KWARGS_ON"  ]]; then NEED_SHIM=1; fi
if [[ " $THINKING_MODES " == *" off "* && -n "$CHAT_TEMPLATE_KWARGS_OFF" ]]; then NEED_SHIM=1; fi
if [[ "$NEED_SHIM" == "1" ]]; then
    if ! apply_chat_template_kwargs_shim; then
        echo "CRITICAL: --chat-template-kwargs shim failed — aborting"
        exit 1
    fi
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi
EP_ARGS=()
if [ "${EP_SIZE}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
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
    local temp top_p top_k pp
    if [[ "$mode" == "on" ]]; then
        [[ -n "$CHAT_TEMPLATE_KWARGS_ON" ]] && think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
        temp="$TEMPERATURE_ON";  top_p="$TOP_P_ON";  top_k="$TOP_K_ON";  pp="$PRESENCE_PENALTY_ON"
    else
        [[ -n "$CHAT_TEMPLATE_KWARGS_OFF" ]] && think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_OFF")
        temp="$TEMPERATURE_OFF"; top_p="$TOP_P_OFF"; top_k="$TOP_K_OFF"; pp="$PRESENCE_PENALTY_OFF"
    fi
    local seed_args=()
    [[ -n "$SEED" ]] && seed_args=(--seed "$SEED")
    local detail_args=()
    [[ -n "$SAVE_DETAILED" ]] && detail_args=(--save-detailed)

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  MTP=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        "${PARALLEL_ARGS[@]}"
        --pipeline-parallel-size 1
        --kv-cache-dtype fp8
        --trust-remote-code
        --no-enable-prefix-caching
        "${EP_ARGS[@]}"
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
        --enable-auto-tool-choice
        --language-model-only
        --max-cudagraph-capture-size 512
        --max-model-len 16384
        --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $mtp}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
        echo "  -> server failed to start (thinking=$mode mtp=$mtp), recording N/A"
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
        --temperature "$temp" \
        --top-p "$top_p" \
        --top-k "$top_k" \
        --presence-penalty "$pp" \
        "${seed_args[@]}" \
        "${detail_args[@]}" \
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
    echo "  -> thinking=$mode MTP=$mtp AL=$al (accepted=$delta_acc drafts=$delta_drf)"
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
    echo "# dataset: $CATEGORY | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on : temp $TEMPERATURE_ON top_p $TOP_P_ON top_k $TOP_K_ON presence_penalty $PRESENCE_PENALTY_ON | chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# thinking_off: temp $TEMPERATURE_OFF top_p $TOP_P_OFF top_k $TOP_K_OFF presence_penalty $PRESENCE_PENALTY_OFF | chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_OFF"
    echo "# Measured on $MODEL_KEY (B300, vLLM MTP), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/qwen3.5_fp4_b300_vllm.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (MTP level); value = golden AL"
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
