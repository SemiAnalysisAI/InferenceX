#!/usr/bin/env bash

# MiMo-V2.6-Flash B300 vLLM SPEED-Bench AL matrix collector for the checkpoint's DFlash drafter.
#
# For each thinking mode (on/off) and DFlash draft length (num_speculative_tokens), measure
# the REAL acceptance length (AL) on one SPEED-Bench category and emit a YAML matrix in the
# golden_al_distribution shape.
#
# The drafter is the BF16 dflash/ subfolder of the target checkpoint, served as shipped. Its
# block size of 8 bounds the draft length at 7. Thinking is ON by default in MiMo's chat
# template, so the OFF cell MUST pass enable_thinking:false and the ON cell enable_thinking:true.
#
# AL is 1 + delta(accepted draft tokens) / delta(verification drafts) from vLLM's spec-decode
# counters around each SPEED-Bench pass.
#
# Dispatch this collector through speedbench-al.yml with framework=vllm.

set -o pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON MODEL MODEL_PATH MTP_LIST OUT_YAML PORT \
    SPEEDBENCH_OUTPUT_LEN THINKING_MODES

if ! python3 -c 'import json, sys; sys.exit(json.loads(sys.argv[1]).get("enable_thinking") is not True)' \
        "$CHAT_TEMPLATE_KWARGS_ON"; then
    echo "CRITICAL: dispatch with thinking-kwargs='{\"enable_thinking\": true}' (got '$CHAT_TEMPLATE_KWARGS_ON')"
    exit 1
fi

SERVE_MODEL="${MODEL_PATH}"
DRAFT_MODEL_PATH="$MODEL_PATH/dflash"
MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# AL is a per-draft accept/reject property independent of batch size, so batch the
# SPEED-Bench pass to cut wall-clock.
CONCURRENCY="32"
TEMPERATURE="1.0"
TOP_P="0.95"
CHAT_TEMPLATE_KWARGS_OFF='{"enable_thinking": false}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"

# `vllm bench serve` os.execv's the CLIENT into the Rust vllm-rs binary whenever the
# dataset (speed_bench qualifies) and backend are supported, with no opt-out. The
# Rust flag surface lacks --speed-bench-output-len, --save-detailed and
# --chat-template-kwargs, so every cell would die at argument parsing. Call the
# Python entrypoint directly (vllm.benchmarks.serve.main), as the CLI's own Python
# fallback does. The SERVER keeps the Rust frontend; AL is read from /metrics.
BENCH_DRIVER="$RESULTS_DIR/bench_serve_python.py"
cat > "$BENCH_DRIVER" <<'PYEOF'
# Python-only `vllm bench serve`: bypasses the Rust os.execv delegation in
# vllm/entrypoints/cli/benchmark/serve.py by importing the benchmark directly.
from vllm.benchmarks.serve import add_cli_args, main

try:  # import path moved in newer vLLM
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils import FlexibleArgumentParser

parser = FlexibleArgumentParser(
    description="vllm bench serve (Python entrypoint, no Rust delegation)"
)
add_cli_args(parser)
main(parser.parse_args())
PYEOF
if [[ ! -s "$BENCH_DRIVER" ]]; then
    echo "CRITICAL: could not write the benchmark driver to $BENCH_DRIVER — aborting."
    exit 1
fi

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
if [[ ! -f "$DRAFT_MODEL_PATH/config.json" ]]; then
    echo "CRITICAL: no DFlash drafter at $DRAFT_MODEL_PATH"
    exit 1
fi

DRAFT_BLOCK=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["block_size"])
' "$DRAFT_MODEL_PATH/config.json")
for mtp in $MTP_LIST; do
    if [[ "$mtp" -lt 1 || "$mtp" -ge "$DRAFT_BLOCK" ]]; then
        echo "CRITICAL: DFlash draft length $mtp is outside 1..$((DRAFT_BLOCK - 1)) (block_size $DRAFT_BLOCK)"
        exit 1
    fi
done

echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

# Preflight the client flags every cell uses; otherwise a CLI mismatch only shows up
# as an all-N/A matrix after every server start. --help=all prints every flag group.
BENCH_HELP="$(python3 "$BENCH_DRIVER" --help=all 2>&1)"
for flag in --speed-bench-category --speed-bench-output-len --chat-template-kwargs --save-detailed; do
    if [[ "$BENCH_HELP" != *"$flag"* ]]; then
        echo "CRITICAL: the Python benchmark entrypoint does not support $flag — aborting."
        echo "--- python3 $BENCH_DRIVER --help=all ---"
        echo "$BENCH_HELP"
        exit 1
    fi
done
echo "=== Benchmark client flag preflight OK ==="

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
    local think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_OFF")
    if [[ "$mode" == "on" ]]; then
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
    fi

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  DFLASH=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        --tensor-parallel-size 4
        --trust-remote-code
        --no-enable-prefix-caching
        --gpu-memory-utilization 0.9
        --max-model-len 32768
        --reasoning-parser mimo
        --tool-call-parser mimo
        --enable-auto-tool-choice
        --speculative-config "{\"method\": \"dflash\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $mtp}"
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! PYTHONSAFEPATH=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench wait \
            --url "http://0.0.0.0:${PORT}/health" --pid "$SERVER_PID" --log "$server_log"; then
        echo "  -> server failed to start (thinking=$mode dflash=$mtp), recording N/A"
        AL_RESULT["${mode}_${mtp}"]="N/A"
        cleanup_server
        return
    fi

    local acc_before drf_before acc_after drf_after
    acc_before=$(fetch_metric "$PORT" "vllm:spec_decode_num_accepted_tokens_total")
    drf_before=$(fetch_metric "$PORT" "vllm:spec_decode_num_drafts_total")

    python3 "$BENCH_DRIVER" \
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
        "${think_args[@]}"
    local bench_rc=$?
    if [[ $bench_rc -ne 0 ]]; then
        echo "  -> benchmark client exited rc=$bench_rc (thinking=$mode dflash=$mtp)"
    fi

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
    echo "  -> thinking=$mode DFLASH=$mtp AL=$al (accepted=$delta_acc drafts=$delta_drf)"
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
    echo "# dataset: $CATEGORY | temperature: $TEMPERATURE | top_p: $TOP_P | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# thinking_off chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_OFF"
    echo "# draft: $MODEL/dflash (BF16, greedy drafting)"
    echo "# Measured on $MODEL_KEY ($GPU_NAME, vLLM DFlash), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/mimov2.6flash_fp4_b300_vllm.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (DFlash draft tokens); value = golden AL"
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

# A matrix where every cell is N/A is a failed collection, not a result: fail the
# job so it is not mistaken for a curve worth reviewing.
MEASURED=0
for mode in $THINKING_MODES; do
    for mtp in $MTP_LIST; do
        [[ "${AL_RESULT[${mode}_${mtp}]:-N/A}" != "N/A" ]] && MEASURED=$((MEASURED + 1))
    done
done
if [[ "$MEASURED" -eq 0 ]]; then
    echo "CRITICAL: no cell produced an AL value — see the server logs and the"
    echo "benchmark client output above."
    exit 1
fi
