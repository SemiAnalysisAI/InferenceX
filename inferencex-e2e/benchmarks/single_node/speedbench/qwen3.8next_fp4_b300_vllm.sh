#!/usr/bin/env bash

# Qwen3.8-Flash-Next B300 vLLM SPEED-Bench AL matrix collector (native MTP).
#
# For each thinking mode (on/off) and MTP level (num_speculative_tokens), measure the
# REAL acceptance length (AL) on one SPEED-Bench category and emit a YAML matrix in
# the golden_al_distribution shape. The synthetic value is injected downstream by the
# throughput recipe, not here. Qwen3.8-Flash-Next ships a built-in MTP module, so no
# separate draft model: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
#
# Recipe-required serve flags: --no-enable-flashinfer-autotune; --max-num-seqs 256
# (avoids a Mamba-cache capacity error at startup); kv-cache dtype left at default
# for the hybrid GDN + Qwen Sparse Attention architecture (do not force fp8);
# --max-cudagraph-capture-size 512 (mamba-hybrid causal_conv1d capture-size assert);
# --language-model-only (the checkpoint is multimodal, AL is collected on text only).
# The 51B n-gram table fits in HBM at TP4, so VLLM_PLE_CPU_OFFLOAD is not needed.
#
# Dispatch (speedbench-al.yml): the image and thinking-kwargs defaults are DSV4's,
# so override both:
#   gh workflow run speedbench-al.yml \
#     --repo SemiAnalysisAI/InferenceX \
#     --ref BRANCH \
#     -f runner=b300 \
#     -f model=Qwen/Qwen3.8-Flash-Next-FP8 \
#     -f model-prefix=qwen3.8next \
#     -f image=vllm/vllm-openai:qwen38-flash-next \
#     -f 'mtp-list=1 2 3 4 5 6 7 8' \
#     -f 'thinking-modes=off on' \
#     -f 'thinking-kwargs={"enable_thinking": true}' \
#     -f category=coding \
#     -f output-len=4096 \
#     -f open-pr=false
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
MAX_NUM_SEQS="256"

# Plain TP8 is incompatible with the official FP8 checkpoint (128-wide quantization
# blocks, per the vLLM recipe). speedbench-al.yml exports TP=8 unconditionally; fold
# it back to the recipe-validated TP4 unless the caller runs TEP (EP_SIZE>1). AL is
# GPU-count-independent, so collecting on 4 of 8 GPUs does not affect the curve.
if [[ "$TP" == "8" && "${EP_SIZE}" -le 1 ]]; then
    echo "NOTE: TP=8 without expert parallelism is incompatible with the FP8 checkpoint; using recipe-validated TP=4."
    TP=4
fi

MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# AL is concurrency-independent (per-token accept/reject; no spec-disable-by-batch is
# set), so batch the SPEED-Bench pass to stay under the CI wall-time limit; conc=1
# blew the 8h budget on Kimi-K3.
CONCURRENCY="64"
# Model-card sampling DIFFERS by mode and MUST be passed per-mode or the AL is
# measured at the wrong settings:
#   thinking : temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0
#   instruct : temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5
TEMPERATURE_ON="1.0";  TOP_P_ON="0.95";  TOP_K_ON="20";  PRESENCE_PENALTY_ON="0.0"
TEMPERATURE_OFF="0.7"; TOP_P_OFF="0.8"; TOP_K_OFF="20"; PRESENCE_PENALTY_OFF="1.5"
# Unset -> vLLM default (deterministic seed=0); vary it to measure temperature>0
# variance.
SEED="${SEED:-}"
# --save-detailed keeps per-request completions to eyeball that thinking_on emits
# <think> and thinking_off does not; off by default (bloats the result JSON).
SAVE_DETAILED="${SAVE_DETAILED:-}"
# Qwen thinking toggles via enable_thinking (default ON; reasoning_effort
# stays at its xhigh default).
CHAT_TEMPLATE_KWARGS_OFF='{"enable_thinking": false}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
# Flat results dir to match the speedbench-al.yml artifact glob
# (speedbench_results/server_*.log) and its pre-run `rm -rf speedbench_results`.
RESULTS_DIR="/workspace/speedbench_results"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi

# Qwen3.8-Flash-Next-FP8 is not in the launcher's STAGED_MODELS, so MODEL_PATH
# resolves into the writable models dir; download only when it is an empty dir.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi
fi

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
        echo "CRITICAL: --chat-template-kwargs support is missing and the shim failed — aborting"
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
        --trust-remote-code
        --no-enable-prefix-caching
        "${EP_ARGS[@]}"
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
        --enable-auto-tool-choice
        --language-model-only
        --no-enable-flashinfer-autotune
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs "$MAX_NUM_SEQS"
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
    echo "# Measured on $MODEL_KEY (B300, vLLM native MTP), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/qwen3.8next_fp4_b300_vllm.sh (speedbench-al.yml)."
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
