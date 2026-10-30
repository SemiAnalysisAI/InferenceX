#!/usr/bin/env bash

# GLM-5.3 B300 SGLang SPEED-Bench AL matrix collector for native MTP (NEXTN) speculative
# decoding on the NVFP4 checkpoint the GLM-5.3 SGLang configs serve.
#
# For each MTP level, measure the REAL acceptance length (AL) on one SPEED-Bench category and
# emit a YAML matrix in the golden_al_distribution shape. Server flags follow
# benchmarks/single_node/srt-slurm-recipes/glm5.3/sglang/b300-fp4-mtp/agentic.yaml, with
# speculative-num-steps set to the level and one verification token on top.
#
# SGLang's SPEED-Bench loader renders prompts with the chat template's default, which for
# GLM-5.3 is thinking on, so only thinking_on is collected.
#
# AL is SGLang's accept_length (accepted tokens per verify step, bonus token included), the
# same quantity as vLLM's 1 + accepted/drafts. Each cell restarts the server.
#
# Dispatch this collector through speedbench-al.yml with model-prefix=glm5.3 framework=sglang.

set -o pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY MODEL MODEL_PATH MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES

if [[ "$THINKING_MODES" != "on" ]]; then
    echo "CRITICAL: SGLang's SPEED-Bench loader renders only the default (thinking on); dispatch with thinking-modes=on (got '$THINKING_MODES')"
    exit 1
fi

SERVE_MODEL="${MODEL_PATH}"

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
    echo "=== $MODEL_PATH is empty; downloading $MODEL (first run only) ==="
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
    echo "  Cell: thinking=$mode  MTP=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --model-path "$SERVE_MODEL"
        --host 0.0.0.0 --port "$PORT"
        --tp-size 8 --ep-size 1
        --quantization modelopt_fp4
        --trust-remote-code
        --mem-fraction-static 0.86
        --chunked-prefill-size 8192
        --max-prefill-tokens 8192
        --reasoning-parser glm45
        --tool-call-parser glm47
        --speculative-algorithm NEXTN
        --speculative-num-steps "$mtp"
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens "$((mtp + 1))"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    python3 -m sglang.launch_server "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! PYTHONSAFEPATH=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench wait \
            --url "http://0.0.0.0:${PORT}/health" --pid "$SERVER_PID" --log "$server_log"; then
        echo "  -> server failed to start (thinking=$mode mtp=$mtp), recording N/A"
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
    echo "  -> thinking=$mode MTP=$mtp AL=$al"
    AL_RESULT["${mode}_${mtp}"]="$al"

    cleanup_server
}

for mtp in $MTP_LIST; do
    run_cell on "$mtp"
done

{
    echo "# Acceptance Length (AL) reference values measured with SPEED-Bench."
    echo "# dataset: $CATEGORY | temperature: $TEMPERATURE | top_p: $TOP_P | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on: chat template default"
    echo "# speculative: NEXTN, eagle-topk 1, num-draft-tokens = level + 1; kv_cache_dtype: engine default"
    echo "# Measured on $MODEL_KEY ($GPU_NAME, SGLang MTP), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/glm5.3_fp4_b300_sglang.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (MTP level); value = golden AL"
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
