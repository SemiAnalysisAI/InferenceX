#!/usr/bin/env bash

# Qwen3.8-Flash-Next B300 vLLM SPEED-Bench AL matrix collector.
#
# Emits the golden AL matrix (thinking on/off x MTP level) in the same shape as
# benchmarks/speedbench-reference-al.yaml. Adapted from qwen3.5_fp4_b300_vllm.sh.
#
# Model-specific notes (all per the Qwen3.8-Flash-Next card / recipe):
#   - tool-call-parser qwen3_xml (NOT qwen3_coder); reasoning-parser qwen3.
#   - TP is HARD-CODED to 4 (recipe-validated; QSA has only 2 KV heads, TP8 is
#     never validated). AL is parallelism-independent so this only pins the config.
#   - recipe serve extras: --max-num-seqs 256, --gpu-memory-utilization 0.90,
#     --no-enable-flashinfer-autotune; --max-cudagraph-capture-size kept
#     <= max-num-seqs so the hybrid GDN backbone does not trip the capture assert.
#   - thinking on/off via the enable_thinking key; reasoning_effort is
#     xhigh (default) / medium / low — there is no "high".
#   - weights are BF16 Qwen/Qwen3.8-Flash-Next (~423 GB, fits 4x B300); the
#     "_fp4_" in the filename is only the shared naming convention.
#
# Dispatch requirements (NOT set here — pass via speedbench-al.yml inputs):
#   - image: the dedicated vllm/vllm-openai:qwen38-flash-next (PyPI is unsupported).
#   - thinking-kwargs: {"enable_thinking": true, "reasoning_effort": "xhigh"}
#     (the workflow default {"thinking": true, "reasoning_effort": "high"} is
#     invalid here and is rejected below).
#   - model: Qwen/Qwen3.8-Flash-Next.

set -uo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL="${MODEL:?MODEL env var required (e.g. Qwen/Qwen3.8-Flash-Next)}"
SERVE_MODEL="${MODEL_PATH:-$MODEL}"
# TP pinned to 4 (recipe-validated); not driven by the workflow's TP=8.
TP=4
PORT="${PORT:-8888}"

MTP_LIST="${MTP_LIST:-1 2 3 4 5 6 7 8}"
THINKING_MODES="${THINKING_MODES:-off on}"
CATEGORY="${CATEGORY:-coding}"
MODEL_KEY="${MODEL_KEY:-qwen3.8-flash-next}"
SPEEDBENCH_OUTPUT_LEN="${SPEEDBENCH_OUTPUT_LEN:-4096}"
CONCURRENCY="${CONCURRENCY:-16}"
# Sampling per the card, per-mode (min_p 0.0 / repetition_penalty 1.0 are vLLM defaults):
#   thinking : temp 1.0, top_p 0.95, top_k 20, presence_penalty 0.0
#   instruct : temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5
TEMPERATURE_ON="${TEMPERATURE_ON:-1.0}";  TOP_P_ON="${TOP_P_ON:-0.95}";  TOP_K_ON="${TOP_K_ON:-20}";  PRESENCE_PENALTY_ON="${PRESENCE_PENALTY_ON:-0.0}"
TEMPERATURE_OFF="${TEMPERATURE_OFF:-0.7}"; TOP_P_OFF="${TOP_P_OFF:-0.8}"; TOP_K_OFF="${TOP_K_OFF:-20}"; PRESENCE_PENALTY_OFF="${PRESENCE_PENALTY_OFF:-1.5}"
SEED="${SEED:-}"
SAVE_DETAILED="${SAVE_DETAILED:-}"

REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
DEFAULT_CHAT_TEMPLATE_KWARGS_ON="{\"enable_thinking\": true, \"reasoning_effort\": \"$REASONING_EFFORT\"}"
DEFAULT_CHAT_TEMPLATE_KWARGS_OFF='{"enable_thinking": false}'
CHAT_TEMPLATE_KWARGS_ON="${CHAT_TEMPLATE_KWARGS_ON:-$DEFAULT_CHAT_TEMPLATE_KWARGS_ON}"
CHAT_TEMPLATE_KWARGS_OFF="${CHAT_TEMPLATE_KWARGS_OFF:-$DEFAULT_CHAT_TEMPLATE_KWARGS_OFF}"
# Guard the deepseek-shaped workflow default ({"thinking": true, "reasoning_effort":
# "high"}): Qwen needs the enable_thinking key and only accepts xhigh/medium/low.
if [[ "$CHAT_TEMPLATE_KWARGS_ON" != *enable_thinking* || "$CHAT_TEMPLATE_KWARGS_ON" == *'"high"'* ]]; then
    echo "CRITICAL: thinking-on chat_template_kwargs must use enable_thinking and reasoning_effort xhigh/medium/low."
    echo "Got: $CHAT_TEMPLATE_KWARGS_ON  (set the workflow 'thinking-kwargs' input accordingly)"
    exit 1
fi

SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/workspace/speed_bench_data}"
# Flat results dir to match the speedbench-al.yml artifact glob
# (speedbench_results/server_*.log) and its pre-run `rm -rf speedbench_results`.
RESULTS_DIR="${RESULTS_DIR:-/workspace/speedbench_results}"
OUT_YAML="${OUT_YAML:-$RESULTS_DIR/speedbench-reference-al.yaml}"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi

# ---- Resolve target weights ----
# Not in the launcher's STAGED_MODELS, so MODEL_PATH points at the writable models
# dir and the BF16 weights (~423 GB) download once here on the first run. Add the
# basename to STAGED_MODELS once staged to read from the faster read-only mount.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        if [[ ! -w "$(dirname "$MODEL_PATH")" ]]; then
            echo "CRITICAL: $MODEL_PATH is empty and $(dirname "$MODEL_PATH") is not writable."
            echo "This means the basename is listed in the launcher's STAGED_MODELS but the"
            echo "weights were never staged. Either get them staged, or remove it from"
            echo "STAGED_MODELS so MODEL_PATH resolves to the writable models dir instead."
            exit 1
        fi
        echo "=== $MODEL_PATH is empty; downloading $MODEL (~423 GB, first run only) ==="
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi
fi

# ---- Download SPEED-Bench dataset ----
echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)

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
        --max-num-seqs 256
        --gpu-memory-utilization 0.90
        --no-enable-flashinfer-autotune
        --reasoning-parser qwen3
        --tool-call-parser qwen3_xml
        --enable-auto-tool-choice
        --max-cudagraph-capture-size 256
        --max-model-len 16384
        --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $mtp}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    # wait_for_server_ready exits the shell (not return) when the server dies, which
    # would make the N/A branch unreachable and let one bad cell abort the whole
    # matrix. The subshell keeps that exit local: a cell that cannot start its
    # server costs one cell instead of the run.
    if ! (wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"); then
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

# ---- Emit the YAML matrix ----
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
    echo "# Measured on $MODEL_KEY (B300, TP4, vLLM MTP), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/qwen3.8flashnext_fp4_b300_vllm.sh (speedbench-al.yml)."
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
echo "=== Wrote AL matrix to $OUT_YAML ==="
cat "$OUT_YAML"
