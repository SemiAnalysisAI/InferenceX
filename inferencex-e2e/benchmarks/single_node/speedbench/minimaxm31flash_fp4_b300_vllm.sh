#!/usr/bin/env bash

# MiniMax-M3.1-Flash vLLM SPEED-Bench AL matrix collector for DSpark speculative decoding.
#
# For each DSpark speculative-token count, measure the REAL acceptance length (AL) on
# one SPEED-Bench category and emit a YAML matrix in the golden_al_distribution shape.
# The DSpark drafter ships inside the checkpoint (dspark/, dspark_block_size 7), so
# --speculative-config takes no draft model path and levels stop at 7.
# The chat template has no thinking on/off switch (thinking is always adaptive), so
# the golden curve is collected for thinking_on only, the mode AgentX resolves.
#
# Serve flags follow the Blackwell TP4 recipe at
# https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash: NVFP4 KV and indexer cache,
# block-size 128 (MSA), and DSpark with probabilistic drafting and block rejection.
#
# The _b300_ in the filename is the speedbench-al.yml naming convention; the golden
# curve is collected on B200 (runner b200-nscale).
#
# Usage (inside the MiniMax-M3.1-Flash vLLM container, on a B200 or B300 node):
#   export MODEL=MiniMaxAI/MiniMax-M3.1-Flash
#   bash benchmarks/single_node/speedbench/minimaxm31flash_fp4_b300_vllm.sh
#
# Required collection settings come from speedbench-al.yml.

set -o pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON DP_ATTENTION EP_SIZE MODEL MODEL_PATH \
    MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

SERVE_MODEL="${MODEL_PATH}"
GPU_MEM_UTIL="0.90"
MAX_MODEL_LEN="16384"
MAX_NUM_SEQS="128"

MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# AL is concurrency-independent (per-token accept/reject; no spec-disable-by-batch is
# set), so batch the SPEED-Bench pass to stay under the CI wall-time limit; conc=1
# blew the 8h budget on an earlier Kimi model.
CONCURRENCY="64"
# Model card sampling: temperature 1.0, top_p 0.95.
TOP_P="0.95"
# speedbench-al.yml exports its thinking-kwargs input as CHAT_TEMPLATE_KWARGS_ON. The
# template only reads reasoning_effort; dispatch with -f 'thinking-kwargs={}' to keep
# the template default the AgentX server sees.
if [[ " $THINKING_MODES " == *" off "* ]]; then
    echo "CRITICAL: MiniMax-M3.1-Flash has no thinking-off mode; dispatch thinking-modes=on."
    exit 1
fi

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"

export VLLM_FLOAT32_MATMUL_PRECISION="high"
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# `vllm bench serve` os.execv's the CLIENT into the Rust vllm-rs binary whenever the
# dataset (speed_bench qualifies) and backend are supported, with no opt-out. The
# Rust flag surface lacks --speed-bench-output-len, --save-detailed and
# --chat-template-kwargs, so every cell would die at argument parsing. Call the
# Python entrypoint directly (vllm.benchmarks.serve.main), as the CLI's own Python
# fallback does. The SERVER keeps the Rust frontend; AL is read from /metrics.
BENCH_DRIVER="$RESULTS_DIR/bench_serve_python.py"
mkdir -p "$RESULTS_DIR"
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
# No `set -e` here, so a failed redirect would otherwise only surface later as a
# confusing "No such file" from the preflight.
if [[ ! -s "$BENCH_DRIVER" ]]; then
    echo "CRITICAL: could not write the benchmark driver to $BENCH_DRIVER — aborting."
    exit 1
fi

nvidia-smi

# CI downloads unstaged weights into the runner's writable model root; this covers
# a standalone run as well.
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

# Preflight the client flags every cell uses; otherwise a CLI mismatch only shows up
# as an all-N/A matrix after eight full server starts (~1h). Probe the driver, not
# `vllm bench serve` (its help exits before the Rust execv), and ask for --help=all
# since plain --help prints only a group summary.
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

# speedbench-al.yml allocates TP=8 GPUs; serve on the first four with the recipe's
# TP4 Blackwell layout. AL does not depend on the parallel layout.
export CUDA_VISIBLE_DEVICES="0,1,2,3"
PARALLEL_ARGS=(--tensor-parallel-size 4)

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
    local temperature=1.0
    if [[ -n "$CHAT_TEMPLATE_KWARGS_ON" && "$CHAT_TEMPLATE_KWARGS_ON" != "{}" ]]; then
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
    fi

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  DSPARK=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        "${PARALLEL_ARGS[@]}"
        --trust-remote-code
        --language-model-only
        --enable-prefix-caching
        --block-size 128
        --kv-cache-dtype nvfp4
        --attention-config '{"indexer_kv_dtype":"nvfp4","use_prefill_query_quantization":true}'
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
        --disable-hybrid-kv-cache-manager
        --reasoning-parser minimax_m3
        --tool-call-parser minimax_m3
        --enable-auto-tool-choice
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-model-len "$MAX_MODEL_LEN"
        --speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $mtp, \"attention_backend\": \"FLASH_ATTN\", \"kv_cache_dtype\": \"fp8\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! PYTHONSAFEPATH=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench wait \
            --url "http://0.0.0.0:${PORT}/health" --pid "$SERVER_PID" --log "$server_log"; then
        echo "  -> server failed to start (thinking=$mode dspark=$mtp), recording N/A"
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
        --temperature "$temperature" \
        --top-p "$TOP_P" \
        "${think_args[@]}"
    local bench_rc=$?
    if [[ $bench_rc -ne 0 ]]; then
        echo "  -> benchmark client exited rc=$bench_rc (thinking=$mode dspark=$mtp); cell will be N/A"
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
    echo "  -> thinking=$mode DSPARK=$mtp AL=$al (accepted=$delta_acc drafts=$delta_drf)"
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
    echo "# dataset: $CATEGORY | top_p: $TOP_P | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on (adaptive): temperature=1.0, chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# Measured on $MODEL_KEY (TP4, vLLM DSpark built-in drafter, probabilistic drafting, block rejection), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/minimaxm31flash_fp4_b300_vllm.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (DSpark level); value = golden AL"
    echo "${MODEL_KEY}:"
    if [[ " $THINKING_MODES " == *" on "* ]]; then
        echo "  thinking_on:"
        emit_mode_block on
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
