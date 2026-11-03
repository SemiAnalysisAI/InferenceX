#!/usr/bin/env bash

set -o pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON DP_ATTENTION EP_SIZE MODEL MODEL_PATH \
    MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

GPU_MEM_UTIL="0.90"
# Startup fails with a Mamba-cache capacity error when this exceeds the available cache blocks.
MAX_NUM_SEQS="256"

# speedbench-al.yml exports TP=8, past the model's 4 KV heads. The model fits on one GPU, and
# AL doesn't depend on the GPU count.
if [[ "$TP" -gt 4 ]]; then
    echo "NOTE: TP=$TP exceeds the model's 4 KV heads; using TP=1."
    TP=1
fi

MODEL_KEY="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
# AL doesn't depend on concurrency, so batch the pass to stay under the CI wall-time limit.
CONCURRENCY="64"
# The model card's sampling differs by thinking mode.
TEMPERATURE_ON="1.0";  TOP_P_ON="0.95";  TOP_K_ON="20";  PRESENCE_PENALTY_ON="0.0"
TEMPERATURE_OFF="0.7"; TOP_P_OFF="0.8"; TOP_K_OFF="20"; PRESENCE_PENALTY_OFF="1.5"
CHAT_TEMPLATE_KWARGS_OFF='{"enable_thinking": false}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi

# Some nodes run another service on the default port; serve on a free one instead.
if ! python3 -c 'import socket, sys; socket.socket().bind(("0.0.0.0", int(sys.argv[1])))' "$PORT" 2>/dev/null; then
    busy_port="$PORT"
    PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("0.0.0.0", 0)); print(s.getsockname()[1])')"
    echo "Port $busy_port is in use on $(hostname); serving on $PORT"
fi

# An unstaged checkpoint lands in the cluster's writable download root; hf download resumes and
# verifies a partial copy.
if [[ ! -e "$MODEL_PATH" || -w "$MODEL_PATH" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
fi

echo "=== Downloading SPEED-Bench dataset ==="
# Pin the image's Hugging Face stack: an unconstrained install can pull huggingface-hub 2.x,
# which transformers 5.x rejects, and then no server starts.
pip install -q -c <(pip freeze | grep -iE '^(huggingface[-_]hub|transformers|tokenizers)==') datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
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

declare -A AL_RESULT

run_cell() {
    local mode="$1" mtp="$2"
    local think_args=()
    local temp top_p top_k pp
    if [[ "$mode" == "on" ]]; then
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
        temp="$TEMPERATURE_ON";  top_p="$TOP_P_ON";  top_k="$TOP_K_ON";  pp="$PRESENCE_PENALTY_ON"
    else
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_OFF")
        temp="$TEMPERATURE_OFF"; top_p="$TOP_P_OFF"; top_k="$TOP_K_OFF"; pp="$PRESENCE_PENALTY_OFF"
    fi

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  MTP=$mtp  category=$CATEGORY"
    echo "=========================================="

    # A larger capture size trips the mamba hybrid's causal_conv1d assert. AL is collected on text
    # only, and the checkpoint is multimodal.
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
        --kv-cache-dtype fp8
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-cudagraph-capture-size 512
        --max-model-len 16384
        --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $mtp}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$MODEL_PATH" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! PYTHONSAFEPATH=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench wait \
            --url "http://0.0.0.0:${PORT}/health" --pid "$SERVER_PID" --log "$server_log"; then
        echo "  -> server failed to start (thinking=$mode mtp=$mtp), recording N/A"
        AL_RESULT["${mode}_${mtp}"]="N/A"
        cleanup_server
        return
    fi

    local acc_before drf_before acc_after drf_after
    acc_before=$(fetch_metric "$PORT" "vllm:spec_decode_num_accepted_tokens_total")
    drf_before=$(fetch_metric "$PORT" "vllm:spec_decode_num_drafts_total")

    vllm bench serve \
        --model "$MODEL_PATH" \
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
    echo "# Auto-generated by benchmarks/single_node/speedbench/qwen3.827b_fp4_b300_vllm.sh (speedbench-al.yml)."
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
