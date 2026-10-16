#!/usr/bin/env bash

# Muse Spark Open B300 SGLang SPEED-Bench AL matrix collector for DFlash2 speculative
# decoding (RadixArk/Muse-Spark-Open-DFlash2).
#
# For each DFlash2 speculative-token count, measure the REAL acceptance length (AL) on one
# SPEED-Bench category and emit a YAML matrix in the golden_al_distribution shape. Server
# flags follow the verified B300 NVFP4 cell of the SGLang Muse Spark Open cookbook (TP4/EP4),
# minus hierarchical cache, which only matters for prefix reuse.
#
# Muse Spark Open has no MTP head, so the speculative draft is the external DFlash2 model.
# DFlash's --speculative-num-draft-tokens is the verification width (current token + drafts),
# so MTP_LIST level N runs with N + 1 draft tokens and keys stay num_speculative_tokens.
# The checkpoint has no thinking-off mode: reasoning_effort only takes low..xhigh, and the
# SGLang client applies the chat template's default (xhigh), so only thinking_on is collected.
#
# AL is SGLang's avg_spec_accept_length (accepted tokens per verify step, bonus token
# included, cumulative since server start), the same quantity as vLLM's
# 1 + accepted/drafts. Each cell restarts the server, so it covers one cell only.
#
# Dispatch this collector through speedbench-al.yml with framework=sglang.

set -o pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY MODEL MODEL_PATH MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES

if [[ "$THINKING_MODES" != "on" ]]; then
    echo "CRITICAL: Muse Spark Open has no thinking-off mode; dispatch with thinking-modes=on (got '$THINKING_MODES')"
    exit 1
fi

SERVE_MODEL="${MODEL_PATH}"
DRAFT_MODEL="${DRAFT_MODEL:-RadixArk/Muse-Spark-Open-DFlash2}"

MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# AL is a per-draft accept/reject property independent of batch size, so batch the
# SPEED-Bench pass to cut wall-clock, as the DSV4 DSpark collector does.
CONCURRENCY="32"
TEMPERATURE="1.0"
TOP_P="0.95"

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"

mkdir -p "$RESULTS_DIR"
nvidia-smi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    if [[ ! -w "$(dirname "$MODEL_PATH")" ]]; then
        echo "CRITICAL: $MODEL_PATH is empty and $(dirname "$MODEL_PATH") is not writable."
        echo "The cluster's models.entries names this checkpoint but the weights were never staged."
        exit 1
    fi
    echo "=== $MODEL_PATH is empty; downloading $MODEL (~484 GB, first run only) ==="
    hf download "$MODEL" --local-dir "$MODEL_PATH"
fi
hf download "$DRAFT_MODEL"

echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

# SGLang's speed-bench loader only accepts the throughput-split categories on the CLI and
# reads prompts from "turns", while prepare.py writes "messages". Filter the qualitative
# split to $CATEGORY and restore "turns" from the user messages.
CATEGORY_JSONL="$SPEEDBENCH_DIR/qualitative_${CATEGORY}.jsonl"
python3 - "$SPEEDBENCH_DIR/qualitative.jsonl" "$CATEGORY_JSONL" "$CATEGORY" <<'EOF'
import json, sys
src, dst, category = sys.argv[1:]
with open(src) as f, open(dst, "w") as out:
    for line in f:
        row = json.loads(line)
        if row.get("category") == category:
            row["turns"] = [m["content"] for m in row["messages"] if m["role"] == "user"]
            out.write(json.dumps(row) + "\n")
EOF
NUM_PROMPTS=$(wc -l < "$CATEGORY_JSONL")
if [[ "$NUM_PROMPTS" -eq 0 ]]; then
    echo "CRITICAL: no SPEED-Bench qualitative rows for category '$CATEGORY'"
    exit 1
fi

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

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  DFLASH=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --model-path "$SERVE_MODEL"
        --host 0.0.0.0 --port "$PORT"
        --tp-size 4 --ep-size 4
        --quantization modelopt_fp4
        --moe-runner-backend flashinfer_trtllm
        --kv-cache-dtype fp8_e4m3
        --mem-fraction-static 0.8
        --trust-remote-code
        --enable-layernorm-sp
        --enable-megamoe-for-layernorm-sp
        --speculative-algorithm DFLASH
        --speculative-draft-model-path "$DRAFT_MODEL"
        --speculative-draft-attention-backend fa4
        --speculative-num-draft-tokens "$((mtp + 1))"
        --mm-feature-transport cpu
        --reasoning-parser muse
        --tool-call-parser muse
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    python3 -m sglang.launch_server "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! PYTHONSAFEPATH=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench wait \
            --url "http://0.0.0.0:${PORT}/health" --pid "$SERVER_PID" --log "$server_log"; then
        echo "  -> server failed to start (thinking=$mode dflash=$mtp), recording N/A"
        AL_RESULT["${mode}_${mtp}"]="N/A"
        cleanup_server
        return
    fi

    local result_file="$RESULTS_DIR/speedbench_${mode}_mtp${mtp}.jsonl"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --model "$SERVE_MODEL" \
        --dataset-name speed-bench \
        --dataset-path "$CATEGORY_JSONL" \
        --speed-bench-output-len "$SPEEDBENCH_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --disable-ignore-eos \
        --output-file "$result_file" \
        --output-details

    local al
    al=$(python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    al = json.loads(f.readlines()[-1]).get("accept_length")
print(f"{al:.2f}" if al else "N/A")
' "$result_file" 2>/dev/null || echo "N/A")
    echo "  -> thinking=$mode DFLASH=$mtp AL=$al"
    AL_RESULT["${mode}_${mtp}"]="$al"

    cleanup_server
}

for mtp in $MTP_LIST; do
    run_cell on "$mtp"
done

{
    echo "# Acceptance Length (AL) reference values measured with SPEED-Bench."
    echo "# dataset: $CATEGORY | temperature: $TEMPERATURE | top_p: $TOP_P | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on: chat template default (reasoning_effort xhigh)"
    echo "# draft: $DRAFT_MODEL"
    echo "# Measured on $MODEL_KEY ($GPU_NAME, SGLang DFlash2), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/musespark_fp4_b300_sglang.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (DFlash2 draft tokens); value = golden AL"
    echo "${MODEL_KEY}:"
    echo "  thinking_on:"
    for mtp in $MTP_LIST; do
        echo "    $mtp: ${AL_RESULT[on_${mtp}]:-N/A}"
    done
} > "$OUT_YAML"

echo ""
echo "=========================================="
echo "  SPEED-Bench AL matrix written to: $OUT_YAML"
echo "=========================================="
cat "$OUT_YAML"
