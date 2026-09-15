#!/usr/bin/env bash

# Kimi-K3 B300 vLLM SPEED-Bench AL matrix collector for DSpark speculative decoding.
#
# For each DSpark speculative-token count, measure the REAL acceptance length (AL) on
# one SPEED-Bench category and emit a YAML matrix in the golden_al_distribution shape.
# The Inferact/Kimi-K3-DSpark draft head is downloaded to a writable workspace dir;
# the target (moonshotai/Kimi-K3, FP4) is pre-staged at /scratch/models/Kimi-K3.
# K3 is a thinking model (kimi_k3 reasoning parser defaults enable_thinking=True),
# so the golden curve is collected for thinking_on only.
#
# VARIANT of kimik3_fp4_b300_vllm.sh (the baseline): the speculative-config also sets
# draft_sample_method=probabilistic and rejection_sample_method=block so baseline and
# variant AL curves can be measured side by side.
#
# Usage (inside the Kimi-K3 vLLM container, on a B300 node):
#   export MODEL=moonshotai/Kimi-K3
#   bash benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm_probabilistic_sample_method_block_rejection_sample_method.sh
#
# Required collection settings come from speedbench-al.yml.

set -o pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON DP_ATTENTION EP_SIZE MODEL MODEL_PATH \
    MTP_LIST OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

SERVE_MODEL="${MODEL_PATH}"
GPU_MEM_UTIL="0.90"
MAX_MODEL_LEN="16384"
MAX_NUM_SEQS="512"

DRAFT_MODEL="Inferact/Kimi-K3-DSpark"

MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# AL is concurrency-independent (per-token accept/reject; no spec-disable-by-batch is
# set), so batch the SPEED-Bench pass to stay under the CI wall-time limit; conc=1
# blew the 8h budget on Kimi-K2.5.
CONCURRENCY="64"
TOP_P="0.95"
# K3 defaults to thinking ON, so the on-cell kwargs are explicit and the off-cell
# kwargs disable it. speedbench-al.yml's thinking-kwargs input defaults to the DSV4
# value and is exported as CHAT_TEMPLATE_KWARGS_ON; dispatch K3 with
# -f 'thinking-kwargs={"thinking": true}'.
CHAT_TEMPLATE_KWARGS_OFF='{"thinking": false}'

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"

export NCCL_DMABUF_ENABLE="0"
export VLLM_ALLREDUCE_USE_FLASHINFER="1"
export VLLM_USE_RUST_FRONTEND="1"
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

# Kimi-K3 is in the launcher's STAGED_MODELS (read-only /scratch/models/Kimi-K3),
# so this is a no-op in CI; it covers a standalone run with unstaged weights.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi
fi

# dirname(MODEL_PATH) can be the read-only staged mount (/scratch/models), so the
# draft must go to a writable workspace dir, not next to the target.
DRAFT_DIR="/workspace/draft_models"
mkdir -p "$DRAFT_DIR"
DRAFT_MODEL_PATH="$DRAFT_DIR/${DRAFT_MODEL##*/}"
if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
    hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
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
        echo "CRITICAL: --chat-template-kwargs shim failed — aborting"
        exit 1
    fi
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
    local temperature
    if [[ "$mode" == "on" ]]; then
        temperature=1.0
        if [[ -n "$CHAT_TEMPLATE_KWARGS_ON" ]]; then
            think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
        fi
    else
        temperature=0.6
        if [[ -n "$CHAT_TEMPLATE_KWARGS_OFF" ]]; then
            think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_OFF")
        fi
    fi

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  DSPARK=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        "${PARALLEL_ARGS[@]}"
        --pipeline-parallel-size 1
        --trust-remote-code
        --load-format fastsafetensors
        --moe-backend auto
        --enable-prefix-caching
        --kv-cache-dtype fp8
        "${EP_ARGS[@]}"
        --reasoning-parser kimi_k3
        --tool-call-parser kimi_k3
        --enable-auto-tool-choice
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-model-len "$MAX_MODEL_LEN"
        --max-cudagraph-capture-size 256
        --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}'
        --speculative-config "{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $mtp, \"attention_backend\": \"FLASHINFER_MLA\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
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

stop_gpu_monitor

emit_mode_block() {
    local mode="$1"
    for mtp in $MTP_LIST; do
        echo "    $mtp: ${AL_RESULT[${mode}_${mtp}]:-N/A}"
    done
}

{
    echo "# Acceptance Length (AL) reference values measured with SPEED-Bench."
    echo "# dataset: $CATEGORY | top_p: $TOP_P | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on: temperature=1.0, chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    if [[ " $THINKING_MODES " == *" off "* ]]; then
        echo "# thinking_off: temperature=0.6, chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_OFF"
    fi
    echo "# Measured on $MODEL_KEY (B300, vLLM DSpark, draft: $DRAFT_MODEL), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm_probabilistic_sample_method_block_rejection_sample_method.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (DSpark level); value = golden AL"
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
