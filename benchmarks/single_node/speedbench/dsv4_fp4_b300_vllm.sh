#!/usr/bin/env bash

# DSV4-Pro B300 vLLM SPEED-Bench AL matrix collector.
#
# Produces the golden acceptance-length (AL) reference matrix consumed by the
# synthetic-acceptance framework: for each thinking mode (on/off) and each MTP
# level (num_speculative_tokens), measure the AL on a single SPEED-Bench
# category (default: coding) and emit a YAML matrix identical in shape to
# benchmarks/speedbench-reference-al.yaml.
#
# This is the "AL distribution collection" script wired into the
# speedbench-al.yml GitHub Action (workflow_dispatch / push-button).
#
# Usage (inside the vLLM container, on a B300 node):
#   export MODEL=/data/models/dsv4-pro
#   bash benchmarks/single_node/speedbench/dsv4_fp4_b300_vllm.sh
#
# Tunables (env):
#   MTP_LIST          space-separated MTP levels   (default "1 2 3 4 5 6 7 8")
#   THINKING_MODES    space-separated: off|on       (default "off on")
#   CATEGORY          SPEED-Bench category          (default coding)
#   SPEEDBENCH_OUTPUT_LEN  per-request output len   (default 4096)
#   OUT_YAML          output matrix path            (default $RESULTS_DIR/speedbench-reference-al.yaml)

set -uo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL="${MODEL:?MODEL env var required (e.g. /data/models/dsv4-pro)}"
# Serve from the local weights dir resolved by the launcher (MODEL_PATH points
# at the pre-staged copy, e.g. /scratch/models/DeepSeek-V4-Pro). Falls back to
# MODEL for a standalone local run where MODEL is itself a path. A leading "/"
# makes the download guard below a no-op.
SERVE_MODEL="${MODEL_PATH:-$MODEL}"
TP="${TP:-8}"
DP_ATTENTION="${DP_ATTENTION:-false}"
EP_SIZE="${EP_SIZE:-1}"
PORT="${PORT:-8888}"

MTP_LIST="${MTP_LIST:-1 2 3 4 5 6 7 8}"
THINKING_MODES="${THINKING_MODES:-off on}"
CATEGORY="${CATEGORY:-coding}"
# Top-level key in the emitted YAML matrix. Derived from the model by the
# workflow (e.g. deepseek-v4-pro); falls back to the model basename, lowercased.
MODEL_KEY="${MODEL_KEY:-$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')}"
SPEEDBENCH_OUTPUT_LEN="${SPEEDBENCH_OUTPUT_LEN:-4096}"
CONCURRENCY="${CONCURRENCY:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# thinking-on chat_template_kwargs. MUST match the production/golden config:
# the reference matrix (benchmarks/speedbench-reference-al.yaml) was measured
# with reasoning_effort=high.
DEFAULT_CHAT_TEMPLATE_KWARGS_ON='{"thinking": true, "reasoning_effort": "high"}'
CHAT_TEMPLATE_KWARGS_ON="${CHAT_TEMPLATE_KWARGS_ON:-$DEFAULT_CHAT_TEMPLATE_KWARGS_ON}"

SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/workspace/speed_bench_data}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/speedbench_results}"
OUT_YAML="${OUT_YAML:-$RESULTS_DIR/speedbench-reference-al.yaml}"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi
if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi

# ---- Download SPEED-Bench dataset ----
echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

# ---- Temporary shim: add a real --chat-template-kwargs CLI option ----
# Upstream gap (until vllm-project/vllm#44244 lands): speed_bench/CustomDataset
# pre-renders the chat template client-side WITHOUT chat_template_kwargs and
# posts to /v1/completions, so thinking mode cannot be enabled via --extra-body
# or --default-chat-template-kwargs. This wires a proper --chat-template-kwargs
# option through get_samples into CustomDataset.sample's apply_chat_template.
# TODO: delete this whole block once #44244 is released in the benchmark image;
# the patch is idempotent (marker check) so it is safe to leave until then.
apply_chat_template_kwargs_shim() {
    echo "=== Patching vLLM benchmark to add --chat-template-kwargs (temporary shim) ==="
    python3 - <<'PYEOF'
import vllm.benchmarks.serve as S
import vllm.benchmarks.datasets.datasets as D

def patch(mod, edits, marker):
    f = mod.__file__
    src = open(f).read()
    if marker in src:
        print("already patched:", f)
        return
    for old, new in edits:
        n = src.count(old)
        assert n == 1, f"anchor matched {n} times in {f}, aborting:\n{old[:80]}..."
        src = src.replace(old, new, 1)
    open(f, "w").write(src)
    print("patched OK ->", f)

# Edit 1: serve.py -- declare the --chat-template-kwargs argument before --extra-body
serve_old = '''    parser.add_argument(
        "--extra-body",'''
serve_new = '''    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help="JSON dict forwarded to apply_chat_template during "
        "client-side prompt rendering, e.g. to enable reasoning mode.",
    )
    parser.add_argument(
        "--extra-body",'''
patch(S, [(serve_old, serve_new)], marker='"--chat-template-kwargs"')

# Edit 2: datasets.py -- forward args.chat_template_kwargs into the speed_bench .sample() call
disp_old = '''                output_len=args.speed_bench_output_len,
                enable_multimodal_chat=args.enable_multimodal_chat,'''
disp_new = '''                output_len=args.speed_bench_output_len,
                chat_template_kwargs=args.chat_template_kwargs,
                enable_multimodal_chat=args.enable_multimodal_chat,'''

# Edit 3: datasets.py -- forward chat_template_kwargs into CustomDataset.sample's template call
samp_old = '''                # apply template
                if not skip_chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )

                prompt_len = len(tokenizer(prompt).input_ids)'''
samp_new = '''                # apply template
                if not skip_chat_template:
                    _ctk = kwargs.get("chat_template_kwargs") or {}
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        **_ctk,
                    )

                prompt_len = len(tokenizer(prompt).input_ids)'''
patch(D, [(disp_old, disp_new), (samp_old, samp_new)],
      marker="chat_template_kwargs=args.chat_template_kwargs")
PYEOF
}

# Apply the shim once if any thinking-on cell is requested.
if [[ " $THINKING_MODES " == *" on "* ]]; then
    if ! apply_chat_template_kwargs_shim; then
        echo "CRITICAL: --chat-template-kwargs shim failed — aborting"
        exit 1
    fi
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi
EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi
MOE_ARGS=()
if [ "${DP_ATTENTION}" = "true" ]; then
    MOE_ARGS=(--moe-backend deep_gemm_mega_moe)
fi

fetch_metric() {
    local port="$1" name="$2"
    curl -s "http://localhost:${port}/metrics" \
      | grep -oP "${name}\\{[^}]*\\} \\K[0-9.]+" || echo "0"
}

SERVER_PID=""
# List all descendant PIDs of $1 recursively, matched by PARENT pid. This can
# never include this script (the script is an ancestor of the server, not a
# descendant), so it avoids the self-kill a name-based `pkill -f vllm` caused
# (the script filename contains "vllm").
_descendants() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        echo "$child"
        _descendants "$child"
    done
}
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        # Snapshot the server's worker/EngineCore subprocesses BEFORE killing the
        # parent: once the parent dies the children reparent to init and the tree
        # link is lost. Killing the captured PIDs guarantees no orphaned worker
        # survives to hold GPU memory and OOM the next server start.
        local descendants
        descendants=$(_descendants "$SERVER_PID")
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        local pid
        for pid in $descendants; do
            kill -9 "$pid" 2>/dev/null || true
        done
        # Wait for GPU memory to actually free before the next server starts.
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

# Per-cell AL is collected into associative arrays keyed by "mode_mtp".
declare -A AL_RESULT

run_cell() {
    local mode="$1" mtp="$2"
    local think_args=()
    if [[ "$mode" == "on" ]]; then
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
    fi

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  MTP=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        "${PARALLEL_ARGS[@]}"
        --pipeline-parallel-size 1
        --kv-cache-dtype fp8
        --trust-remote-code
        --block-size 256
        --no-enable-prefix-caching
        "${EP_ARGS[@]}"
        "${MOE_ARGS[@]}"
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
        --attention_config.use_fp4_indexer_cache True
        --tokenizer-mode deepseek_v4
        --tool-call-parser deepseek_v4
        --enable-auto-tool-choice
        --reasoning-parser deepseek_v4
        --max-cudagraph-capture-size 2048
        --max-model-len 16384
        --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $mtp}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
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
        --tokenizer-mode deepseek_v4 \
        --temperature "$TEMPERATURE" \
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
    echo "# dataset: $CATEGORY | temperature: $TEMPERATURE | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# Measured on $MODEL_KEY (B300, vLLM MTP), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/dsv4_fp4_b300_vllm.sh (speedbench-al.yml)."
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
