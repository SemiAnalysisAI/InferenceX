#!/usr/bin/env bash

set -o pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY MODEL MODEL_PATH MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES

if [[ "$THINKING_MODES" != "on" ]]; then
    echo "CRITICAL: the K3.1 DSpark curve this matches has thinking_on only; dispatch with thinking-modes=on (got '$THINKING_MODES')"
    exit 1
fi

SERVE_MODEL="${MODEL_PATH}"
DRAFT_MODEL="RadixArk/Kimi-K3.1-Dflash2-Preview"
MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# Client settings of the K3.1 DSpark curve's vLLM collection, so the two curves differ only
# in the drafter.
CONCURRENCY="64"
CHAT_TEMPLATE_KWARGS='{"thinking": true}'
SAMPLING='{"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.0}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"
mkdir -p "$RESULTS_DIR"

nvidia-smi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Pyxis containers share the host network and some B300 nodes run another service on the
# workflow's port, so serve on a free one instead.
if ! python3 -c 'import socket, sys; socket.socket().bind(("0.0.0.0", int(sys.argv[1])))' "$PORT" 2>/dev/null; then
    busy_port="$PORT"
    PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("0.0.0.0", 0)); print(s.getsockname()[1])')"
    echo "Port $busy_port is in use on $(hostname); serving on $PORT"
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
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/5ac8609a56ac941540b10c92e68d556e6343cd4c/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"
if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

# SGLang's SPEED-Bench loaders reorder the prompts, so this client sends them itself: each
# category row's first user message, in vllm bench serve's order (random.seed(0) shuffle),
# rendered by the AL gate's chat-template renderer and streamed to /v1/completions with
# max_tokens 4096 and the engine-default ignore_eos, exactly as the vLLM collection did.
BENCH_CLIENT="$RESULTS_DIR/speedbench_completions_client.py"
cat > "$BENCH_CLIENT" <<'PYEOF'
import json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from infx.bench.al_gate import render

dataset, category, model, url, concurrency, max_tokens, kwargs, sampling = sys.argv[1:]
prompts = []
with open(dataset) as rows:
    for line in rows:
        row = json.loads(line)
        if row.get("category") == category:
            prompts.append(next(m["content"] for m in row["messages"] if m["role"] == "user"))
if not prompts:
    sys.exit(f"no SPEED-Bench qualitative rows for category {category!r}")
random.Random(0).shuffle(prompts)
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
texts = [render(p, "chat_template", json.loads(kwargs), tokenizer) for p in prompts]
body = {"model": model, "max_tokens": int(max_tokens), "stream": True, **json.loads(sampling)}


def send(text):
    data = json.dumps({**body, "prompt": text}).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        for _ in response:
            pass


with ThreadPoolExecutor(int(concurrency)) as pool:
    list(pool.map(send, texts))
print(f"sent {len(texts)} {category} prompts at concurrency {concurrency}")
PYEOF

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
    # ReplaySSM keeps one KDA state per request instead of one per draft token, which lets
    # 64 running requests fit at the widest widths.
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
        --enable-linear-replayssm-spec
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

    PYTHONSAFEPATH=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 "$BENCH_CLIENT" \
        "$SPEEDBENCH_DIR/qualitative.jsonl" "$CATEGORY" "$SERVE_MODEL" \
        "http://localhost:${PORT}/v1/completions" "$CONCURRENCY" "$SPEEDBENCH_OUTPUT_LEN" \
        "$CHAT_TEMPLATE_KWARGS" "$SAMPLING" 2>&1 | tee "$RESULTS_DIR/client_on_mtp${mtp}.log"
    local bench_rc=${PIPESTATUS[0]}
    if [[ $bench_rc -ne 0 ]]; then
        echo "  -> benchmark client exited rc=$bench_rc (dflash=$mtp); cell will be N/A"
        AL_RESULT["$mtp"]="N/A"
        cleanup_server
        return
    fi

    # The engine counts the last verifications after the responses close, so scrape until
    # the reading settles, as the AL gate does.
    local reading="N/A" previous="" settle
    for settle in $(seq 30); do
        if ! scrape_metrics "$after"; then
            reading="N/A"
            break
        fi
        reading=$(acceptance_length "$before" "$after" || echo "N/A")
        [[ "$reading" == "$previous" ]] && break
        previous="$reading"
        sleep 2
    done
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
    echo "# thinking_on: temperature=1.0, chat_template_kwargs: $CHAT_TEMPLATE_KWARGS"
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
