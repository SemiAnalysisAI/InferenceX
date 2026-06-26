#!/usr/bin/env bash

# Kimi-K2.5 B300 vLLM SPEED-Bench AL matrix collector for EAGLE3 speculative
# decoding.
#
# Produces the golden acceptance-length (AL) reference matrix consumed by the
# synthetic-acceptance framework: for each thinking mode (on/off) and each
# EAGLE3 speculative-token count, measure the REAL AL on a single SPEED-Bench
# category (default: coding) and emit a YAML matrix identical in shape to
# benchmarks/speedbench-reference-al.yaml.
#
# Kimi-K2.5 uses the lightseekorg/kimi-k2.5-eagle3-mla draft head (MLA
# variant, recommended by official docs). The draft model is downloaded
# alongside the target checkpoint before the sweep begins.
#
# Differences vs the GLM-5 MTP template (glm5_fp4_b300_vllm.sh):
#   - speculative-config     eagle3 with external draft model (not mtp)
#   - reasoning-parser       kimi_k2        (was glm45)
#   - tool-call-parser       kimi_k2        (was glm47)
#   - thinking toggle        {"thinking": true/false}   (was enable_thinking)
#   - temperature            1.0 (thinking) / 0.6 (instant)  (was fixed 1.0)
#   - NO --chat-template-content-format, --tokenizer-mode, --block-size,
#     or --attention_config.use_fp4_indexer_cache
#   - --language-model-only  (text-only benchmark, no vision)
#
# Usage (inside the Kimi vLLM container, on a B300 node):
#   export MODEL=moonshotai/Kimi-K2.5-NVFP4
#   bash benchmarks/single_node/speedbench/kimik2.5_fp4_b300_vllm.sh
#
# Tunables (env):
#   MTP_LIST          space-separated EAGLE3 spec-token counts (default "1 2 3 4 5 6 7 8")
#   THINKING_MODES    space-separated: off|on       (default "off on")
#   CATEGORY          SPEED-Bench category          (default coding)
#   SPEEDBENCH_OUTPUT_LEN  per-request output len   (default 4096)
#   OUT_YAML          output matrix path            (default $RESULTS_DIR/speedbench-reference-al.yaml)

set -uo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL="${MODEL:?MODEL env var required (e.g. moonshotai/Kimi-K2.5-NVFP4)}"
SERVE_MODEL="${MODEL_PATH:-$MODEL}"
TP="${TP:-8}"
DP_ATTENTION="${DP_ATTENTION:-false}"
EP_SIZE="${EP_SIZE:-1}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

DRAFT_MODEL="lightseekorg/kimi-k2.5-eagle3-mla"

MTP_LIST="${MTP_LIST:-1 2 3 4 5 6 7 8}"
THINKING_MODES="${THINKING_MODES:-off on}"
CATEGORY="${CATEGORY:-coding}"
MODEL_KEY="${MODEL_KEY:-$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')}"
SPEEDBENCH_OUTPUT_LEN="${SPEEDBENCH_OUTPUT_LEN:-4096}"
# AL is concurrency-independent (per-token accept/reject; no spec-disable-by-batch
# is set below), so batch the SPEED-Bench pass to keep wall-time under the CI
# limit. conc=1 made Kimi-K2.5 exceed the 8h budget. 64 captures most of the
# batch-decode speedup before it saturates / KV pressure grows; override via env.
CONCURRENCY="${CONCURRENCY:-64}"
TOP_P="${TOP_P:-0.95}"
# Kimi thinking toggles via the thinking chat_template key (default ON).
DEFAULT_CHAT_TEMPLATE_KWARGS_ON='{"thinking": true}'
DEFAULT_CHAT_TEMPLATE_KWARGS_OFF='{"thinking": false}'
CHAT_TEMPLATE_KWARGS_ON="${CHAT_TEMPLATE_KWARGS_ON:-$DEFAULT_CHAT_TEMPLATE_KWARGS_ON}"
CHAT_TEMPLATE_KWARGS_OFF="${CHAT_TEMPLATE_KWARGS_OFF:-$DEFAULT_CHAT_TEMPLATE_KWARGS_OFF}"

SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/workspace/speed_bench_data}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/speedbench_results}"
OUT_YAML="${OUT_YAML:-$RESULTS_DIR/speedbench-reference-al.yaml}"

# Blackwell NVFP4 checkpoints need FlashInfer FP4 MoE kernels; auto-enable
# when the served model name contains NVFP4 (e.g. nvidia/Kimi-K2.5-NVFP4).
if [[ "$SERVE_MODEL" == *NVFP4* || "$SERVE_MODEL" == *nvfp4* ]]; then
    export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
fi
export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi

# ---- Download target if it is not pre-staged ----
# A pre-staged target lands in the read-only staged mount (/scratch/models);
# only download when MODEL_PATH is an empty writable dir (non-staged run).
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi
fi

# ---- Download EAGLE3 draft model to a WRITABLE dir ----
# The draft must NOT go next to a pre-staged target: dirname(MODEL_PATH) is the
# read-only staged mount (/scratch/models), so writing the draft there fails
# with PermissionError. Use a writable workspace dir regardless of staging.
DRAFT_DIR="${DRAFT_MODEL_DIR:-/workspace/draft_models}"
mkdir -p "$DRAFT_DIR"
DRAFT_MODEL_PATH="$DRAFT_DIR/${DRAFT_MODEL##*/}"
if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
    hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
fi

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
# Model agnostic (forwards whatever dict it is given). TODO: delete once #44244
# is released in the benchmark image; idempotent (marker check), safe to leave.
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

# Apply the shim once if any cell will pass chat_template_kwargs.
NEED_SHIM=0
if [[ " $THINKING_MODES " == *" on "*  && -n "$CHAT_TEMPLATE_KWARGS_ON"  ]]; then NEED_SHIM=1; fi
if [[ " $THINKING_MODES " == *" off "* && -n "$CHAT_TEMPLATE_KWARGS_OFF" ]]; then NEED_SHIM=1; fi
if [[ "$NEED_SHIM" == "1" ]]; then
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
    echo "  Cell: thinking=$mode  EAGLE3=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        "${PARALLEL_ARGS[@]}"
        --pipeline-parallel-size 1
        --kv-cache-dtype fp8
        --trust-remote-code
        --language-model-only
        --no-enable-prefix-caching
        "${EP_ARGS[@]}"
        --reasoning-parser kimi_k2
        --tool-call-parser kimi_k2
        --enable-auto-tool-choice
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-model-len 16384
        --speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $mtp}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"; then
        echo "  -> server failed to start (thinking=$mode eagle3=$mtp), recording N/A"
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
        --temperature "$temperature" \
        --top-p "$TOP_P" \
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
    echo "  -> thinking=$mode EAGLE3=$mtp AL=$al (accepted=$delta_acc drafts=$delta_drf)"
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
    echo "# dataset: $CATEGORY | top_p: $TOP_P | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on: temperature=1.0, chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# thinking_off: temperature=0.6, chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_OFF"
    echo "# Measured on $MODEL_KEY (B300, vLLM EAGLE3), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/kimik2.5_fp4_b300_vllm.sh (speedbench-al.yml)."
    echo "#"
    echo "# key = num_speculative_tokens (EAGLE3 level); value = golden AL"
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
