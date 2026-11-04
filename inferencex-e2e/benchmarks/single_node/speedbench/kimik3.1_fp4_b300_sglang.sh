#!/usr/bin/env bash

set -o pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY MODEL MODEL_PATH MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES

# The chat template thinks by default and the SGLang SPEED-Bench loader passes no
# chat_template_kwargs, so only the thinking-on mode can be collected.
if [[ "$THINKING_MODES" != "on" ]]; then
    echo "CRITICAL: this collector renders only the template default (thinking on); dispatch with thinking-modes=on (got '$THINKING_MODES')"
    exit 1
fi

SERVE_MODEL="${MODEL_PATH}"
DRAFT_MODEL="RadixArk/Kimi-K3.1-Dflash2-Preview"
MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# The draft card fits 32 running requests at verification width 8 on B300; AL does not
# depend on batch size.
CONCURRENCY="32"
SAMPLING='{"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.0}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"
mkdir -p "$RESULTS_DIR"

nvidia-smi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Pyxis containers share the host network, so a stray listener on the node fails every cell's
# bind; stop before the downloads and name the holder.
if [[ -n "$(ss -Hltn "sport = :${PORT}" 2>/dev/null)" ]] \
        || (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
    echo "CRITICAL: port ${PORT} on $(hostname) is already in use by a process outside this job."
    ss -ltnp "sport = :${PORT}" 2>&1 || true
    exit 1
fi

# An unstaged checkpoint lands in the cluster's writable download root; hf download resumes and
# verifies a partial copy, which a non-empty-directory check would accept as complete.
if [[ ! -e "$MODEL_PATH" || -w "$MODEL_PATH" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
fi
hf download "$DRAFT_MODEL"

echo "=== Downloading SPEED-Bench dataset ==="
# Pin the image's Hugging Face stack: an unconstrained install can pull a huggingface-hub
# that the image's transformers rejects, and then no server starts.
pip install -q -c <(pip freeze | grep -iE '^(huggingface[-_]hub|transformers|tokenizers)==') datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"
if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

# SGLang's speed-bench loader reads prompts from "turns", while prepare.py writes "messages".
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

scrape_metrics() {
    curl -sf "http://localhost:${PORT}/metrics" > "$1"
}

# The AL gate's own SGLang counter reader, loaded by path: the infx.results package needs
# dependencies the serving image does not ship.
acceptance_length() {
    python3 - "$repo_root/infx/results/agentic/spec_decode.py" "$1" "$2" <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("spec_decode", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
before, after = (open(path).read() for path in sys.argv[2:])
reading = module.READERS["sglang"](module.increase({"s": before}, {"s": after}))
if not reading or not reading[1]:
    print("N/A")
else:
    accepted, verifications = reading
    print(f"{1 + accepted / verifications:.2f} {int(accepted)} {int(verifications)}")
EOF
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
    local mtp="$1"

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=on  DFLASH=$mtp  category=$CATEGORY"
    echo "=========================================="

    # Server flags follow the draft card's B300 TP8 block-verification command; DFLASH's
    # --speculative-num-draft-tokens is the verification width, the drafts plus one.
    local serve_args=(
        --model-path "$SERVE_MODEL"
        --host 0.0.0.0 --port "$PORT"
        --tp-size 8
        --trust-remote-code
        --weight-loader-prefetch-checkpoints
        --model-loader-extra-config '{"enable_multithread_load": true}'
        --mem-fraction-static 0.85
        --mamba-full-memory-ratio 1.0
        --mamba-ssm-dtype float32
        --kv-cache-dtype fp8_e4m3
        --disable-radix-cache
        --max-running-requests "$CONCURRENCY"
        --context-length 131072
        --reasoning-parser kimi_k3
        --tool-call-parser kimi_k3
        --enable-metrics
        --speculative-algorithm DFLASH
        --speculative-draft-model-path "$DRAFT_MODEL"
        --speculative-draft-attention-backend cutedsl_mla
        --speculative-draft-kv-cache-dtype fp8_e4m3
        --speculative-num-draft-tokens "$((mtp + 1))"
        --speculative-use-block-verification
    )

    local server_log="$RESULTS_DIR/server_on_mtp${mtp}.log"
    python3 -m sglang.launch_server "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! PYTHONSAFEPATH=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench wait \
            --url "http://0.0.0.0:${PORT}/health" --pid "$SERVER_PID" --log "$server_log"; then
        echo "  -> server failed to start (dflash=$mtp), recording N/A"
        AL_RESULT["$mtp"]="N/A"
        cleanup_server
        return
    fi

    local before="$RESULTS_DIR/metrics_on_mtp${mtp}_before.txt"
    local after="$RESULTS_DIR/metrics_on_mtp${mtp}_after.txt"
    if ! scrape_metrics "$before"; then
        echo "  -> no /metrics before the pass (dflash=$mtp), recording N/A"
        AL_RESULT["$mtp"]="N/A"
        cleanup_server
        return
    fi

    # sglang-oai streams /v1/completions like the vLLM collections; its loader renders each
    # prompt with the chat template, and sampling goes in the request body because this
    # backend ignores --temperature and --top-p.
    python3 -m sglang.benchmark.serving \
        --backend sglang-oai \
        --port "$PORT" \
        --model "$SERVE_MODEL" \
        --dataset-name speed-bench \
        --dataset-path "$CATEGORY_JSONL" \
        --speed-bench-output-len "$SPEEDBENCH_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --extra-request-body "$SAMPLING" \
        --disable-ignore-eos \
        --output-file "$RESULTS_DIR/speedbench_on_mtp${mtp}.jsonl" \
        --output-details
    local bench_rc=$?
    if [[ $bench_rc -ne 0 ]]; then
        echo "  -> benchmark client exited rc=$bench_rc (dflash=$mtp); cell will be N/A"
        AL_RESULT["$mtp"]="N/A"
        cleanup_server
        return
    fi

    # Counters lag the last responses by a scheduler step or two.
    sleep 5
    local reading="N/A"
    if scrape_metrics "$after"; then
        reading=$(acceptance_length "$before" "$after" || echo "N/A")
    fi
    local al="${reading%% *}"
    echo "  -> thinking=on DFLASH=$mtp AL=$al (accepted verifications: ${reading#* })"
    AL_RESULT["$mtp"]="$al"

    cleanup_server
}

for mtp in $MTP_LIST; do
    run_cell "$mtp"
done

{
    echo "# Acceptance Length (AL) reference values measured with SPEED-Bench."
    echo "# dataset: $CATEGORY | top_p: 0.95 | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on: temperature=1.0, chat template default (thinking on)"
    echo "# speculative: DFLASH, draft $DRAFT_MODEL, block verification"
    echo "# Measured on $MODEL_KEY ($GPU_NAME, SGLang DFLASH), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/kimik3.1_fp4_b300_sglang.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (DFLASH draft tokens); value = golden AL"
    echo "${MODEL_KEY}:"
    echo "  thinking_on:"
    for mtp in $MTP_LIST; do
        echo "    $mtp: ${AL_RESULT[$mtp]:-N/A}"
    done
} > "$OUT_YAML"

echo ""
echo "=========================================="
echo "  SPEED-Bench AL matrix written to: $OUT_YAML"
echo "=========================================="
cat "$OUT_YAML"

MEASURED=0
for mtp in $MTP_LIST; do
    [[ "${AL_RESULT[$mtp]:-N/A}" != "N/A" ]] && MEASURED=$((MEASURED + 1))
done
if [[ "$MEASURED" -eq 0 ]]; then
    echo "CRITICAL: no cell produced an AL value — see the server logs and the"
    echo "benchmark client output above."
    exit 1
fi
