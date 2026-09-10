#!/usr/bin/env bash
# DSv4.1 Flash golden AL collector; the b300 filename follows speedbench-al.yml.
# Runs on B200 at TP4 using the successful AgentX image and UVA loading path.
# Official recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
set -euo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL="${MODEL:?}"
TP="${TP:-4}"
[[ "$TP" == 4 ]] || { echo "DSv4.1 Flash golden AL requires TP=4" >&2; exit 1; }
MTP_LIST="${MTP_LIST:-1 2 3 4 5 6 7 8}"
THINKING_MODES="${THINKING_MODES:-off on}"
CATEGORY="${CATEGORY:-coding}"
SPEEDBENCH_OUTPUT_LEN="${SPEEDBENCH_OUTPUT_LEN:-4096}"
CONCURRENCY="${CONCURRENCY:-32}"
DEFAULT_THINKING='{"thinking":true,"reasoning_effort":"high"}'
CHAT_TEMPLATE_KWARGS_ON="${CHAT_TEMPLATE_KWARGS_ON:-$DEFAULT_THINKING}"
RESULTS_DIR="${RESULTS_DIR:-/ix/speedbench_results}"
SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/ix/speed_bench_data}"
OUT_YAML="${OUT_YAML:-/ix/speedbench-reference-al.yaml}"
# Collect the requested draft lengths; runtime failures remain visible per length.
for mtp in $MTP_LIST; do
    [[ "$mtp" =~ ^[1-8]$ ]] || { echo 'DSv4.1 Flash draft lengths must be in 1..8' >&2; exit 1; }
done
for mode in $THINKING_MODES; do
    [[ "$mode" == on || "$mode" == off ]] || exit 1
done
mkdir -p "$RESULTS_DIR"
# SPEED-Bench applies its chat template by default. This CLI has no
# --use-chat-template switch (that belongs to other benchmark entry points).
BENCH_ARGS=(
    --model "$MODEL" --tokenizer "${MODEL_PATH:-$MODEL}"
    --dataset-name speed_bench --dataset-path "$SPEEDBENCH_DIR"
    --speed-bench-category "$CATEGORY" --speed-bench-output-len "$SPEEDBENCH_OUTPUT_LEN"
    --num-prompts -1 --max-concurrency "$CONCURRENCY"
    --tokenizer-mode deepseek_v41 --temperature 1.0
    --save-result --save-detailed --result-dir "$RESULTS_DIR"
)
# Parse the exact client arguments with the installed benchmark parser. The
# abbreviated --help output omits valid options, so it cannot validate support.
for kwargs in '{"thinking":false}' "$CHAT_TEMPLATE_KWARGS_ON"; do
    python3 - "${BENCH_ARGS[@]}" --port "${PORT:-8888}" \
        --chat-template-kwargs "$kwargs" --result-filename preflight.json <<'PYEOF'
import sys
from vllm.benchmarks.serve import add_cli_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser(description="SPEED-Bench client preflight")
add_cli_args(parser)
parser.parse_args(sys.argv[1:])
print("SPEED-Bench client arguments accepted by installed vLLM parser")
PYEOF
done
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    MODEL_PATH="$MODEL"
fi
pip install -q datasets tiktoken
curl -fLsS https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
    | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"
[[ -s "$SPEEDBENCH_DIR/qualitative.jsonl" ]]

export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
select_available_server_port
SERVER_PID=""
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        # setsid gives this server and its workers a private process group.
        kill -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        kill -9 -- "-$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        wait_for_server_resources_released
    fi
}
trap cleanup_server EXIT

for mtp in $MTP_LIST; do
    # A fresh server per draft length keeps model configuration and counters isolated.
    # Adaptive verification would make the measured target dependent on profiling.
    SPEC_CONFIG=$(printf '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' "$mtp")
    CAPTURE_SIZE=256
    while (( CAPTURE_SIZE < CONCURRENCY * (mtp + 1) )); do
        CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
    done
    setsid vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
        --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" \
        --language-model-only --tokenizer-mode deepseek_v41 \
        --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
        --reasoning-parser deepseek_v41 --engram-config '{"cpu_offload":true}' \
        --speculative-config "$SPEC_CONFIG" --max-model-len 16384 \
        --no-enable-prefix-caching --max-cudagraph-capture-size "$CAPTURE_SIZE" \
        --disable-uvicorn-access-log > "$RESULTS_DIR/server_dspark${mtp}.log" 2>&1 &
    SERVER_PID=$!
    if ! (wait_for_server_ready --port "$PORT" --server-log "$RESULTS_DIR/server_dspark${mtp}.log" --server-pid "$SERVER_PID"); then
        echo "Draft length $mtp failed startup; retaining logs and attempting the next length" >&2
        cleanup_server
        continue
    fi

    # Keep one server for both modes. Counter deltas exclude startup and other cells.
    for mode in $THINKING_MODES; do
        kwargs='{"thinking":false}'
        [[ "$mode" == off ]] || kwargs="$CHAT_TEMPLATE_KWARGS_ON"
        curl -fSs "http://localhost:$PORT/metrics" > "$RESULTS_DIR/before_${mode}_mtp${mtp}.prom"
        vllm bench serve "${BENCH_ARGS[@]}" --port "$PORT" \
            --chat-template-kwargs "$kwargs" --result-filename "speedbench_${mode}_mtp${mtp}.json"
        curl -fSs "http://localhost:$PORT/metrics" > "$RESULTS_DIR/after_${mode}_mtp${mtp}.prom"
    done
    cleanup_server
done
python3 utils/speedbench_al.py --results-dir "$RESULTS_DIR" --output "$OUT_YAML" \
    --draft-lengths $MTP_LIST --modes $THINKING_MODES --thinking-kwargs "$CHAT_TEMPLATE_KWARGS_ON" \
    --model "$MODEL" --image "${IMAGE:?}" --tp "$TP" --category "$CATEGORY" \
    --output-len "$SPEEDBENCH_OUTPUT_LEN"
cat "$OUT_YAML"
