#!/usr/bin/env bash

# Qwen3.8-Flash-Next B300 vLLM SPEED-Bench AL matrix collector (native MTP).
#
# Produces the golden acceptance-length (AL) reference matrix consumed by the
# synthetic-acceptance framework: for each thinking mode (on/off) and each MTP
# level (num_speculative_tokens), measure the REAL AL on a single SPEED-Bench
# category (default: coding) and emit a YAML matrix identical in shape to the
# other golden_al_distribution curves. This measures real MTP acceptance; the
# synthetic value is injected downstream by the throughput recipe, not here.
#
# Qwen3.8-Flash-Next ships a built-in 1-layer / 4B MTP module (trained with
# multi-steps), so speculative decoding is native — no separate draft model.
# Reference: https://huggingface.co/Qwen/Qwen3.8-Flash-Next and the vLLM recipe
# https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
#
# Adapted from speedbench/qwen3.5_fp4_b300_vllm.sh. Differences vs Qwen3.5:
#   - checkpoint            Qwen/Qwen3.8-Flash-Next-FP8 (official FP8; ~173 GiB).
#                           NOT pre-staged on the B300 cluster -> the launcher
#                           resolves MODEL_PATH into the writable models dir and
#                           this script downloads it there on first run.
#   - TP                    4, not 8. Plain TP8 is incompatible with the FP8
#                           checkpoint's 128-wide quantization blocks (per the
#                           vLLM recipe); TP4 is the recipe-validated Blackwell
#                           full-tray configuration, incl. MTP3. AL is
#                           GPU-count-independent, so collecting on 4 of the
#                           node's 8 GPUs does not affect the curve. The
#                           speedbench-al.yml workflow exports TP=8; that value
#                           is overridden below unless EP_SIZE>1 (TEP) is set.
#   - serve flags           --no-enable-flashinfer-autotune (recipe-required),
#                           --max-num-seqs 256 (avoids a Mamba-cache capacity
#                           error at startup, per the recipe),
#                           --gpu-memory-utilization 0.90 (recipe default)
#   - NO --kv-cache-dtype fp8   the recipe leaves kv-cache dtype at its default
#                           for this new hybrid GDN + Qwen Sparse Attention
#                           architecture; do not force fp8 here.
#   - sampling (model card): thinking  temp 1.0 top_p 0.95 top_k 20 pp 0.0
#                            instruct  temp 0.7 top_p 0.80 top_k 20 pp 1.5
#   - concurrency           64 (kimik3 precedent: AL is per-token accept/reject
#                           and concurrency-independent; batching keeps the
#                           16-cell sweep inside the CI wall-time budget)
#
# Kept from the Qwen3.5 template:
#   - reasoning-parser qwen3, tool-call-parser qwen3_coder (per the model's
#     official serving commands)
#   - --language-model-only  (the checkpoint is multimodal — vision encoder —
#     but golden AL is collected on the text path only)
#   - --max-cudagraph-capture-size 512  (safe carry-over for the mamba-hybrid
#     causal_conv1d capture-size assert; Flash-Next is GDN-based like Qwen3.5)
#   - thinking on/off via the enable_thinking chat_template key (model card:
#     default ON; reasoning_effort left at its xhigh default), OFF passed
#     explicitly
#
# N-gram embedding note: the 51B n-gram table fits in HBM at TP4 on B300, so
# VLLM_PLE_CPU_OFFLOAD is not needed here. It is optional for TP/TEP and only
# required for DEP, which this collector does not use.
#
# Dispatch (speedbench-al.yml) — the defaults for image and thinking-kwargs are
# DSV4's, so override both:
#   gh workflow run speedbench-al.yml \
#     --repo SemiAnalysisAI/InferenceX \
#     --ref BRANCH \
#     -f runner=b300 \
#     -f model=Qwen/Qwen3.8-Flash-Next-FP8 \
#     -f model-prefix=qwen3.8next \
#     -f image=vllm/vllm-openai:qwen38-flash-next \
#     -f 'mtp-list=1 2 3 4 5 6 7 8' \
#     -f 'thinking-modes=off on' \
#     -f 'thinking-kwargs={"enable_thinking": true}' \
#     -f category=coding \
#     -f output-len=4096 \
#     -f open-pr=false
#
# Usage (inside the vLLM container, on a B300 node):
#   export MODEL=Qwen/Qwen3.8-Flash-Next-FP8
#   bash benchmarks/single_node/speedbench/qwen3.8next_fp4_b300_vllm.sh
#
# Tunables (env):
#   MTP_LIST          space-separated MTP levels   (default "1 2 3 4 5 6 7 8")
#   THINKING_MODES    space-separated: off|on       (default "off on")
#   CATEGORY          SPEED-Bench category          (default coding)
#   SPEEDBENCH_OUTPUT_LEN  per-request output len   (default 4096)
#   OUT_YAML          output matrix path            (default $RESULTS_DIR/speedbench-reference-al.yaml)

set -uo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL="${MODEL:?MODEL env var required (e.g. Qwen/Qwen3.8-Flash-Next-FP8)}"
SERVE_MODEL="${MODEL_PATH:-$MODEL}"
TP="${TP:-4}"
DP_ATTENTION="${DP_ATTENTION:-false}"
EP_SIZE="${EP_SIZE:-1}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Plain TP8 is incompatible with the official FP8 checkpoint (128-wide
# quantization blocks, per the vLLM recipe). speedbench-al.yml exports TP=8
# unconditionally; fold that back to the recipe-validated TP4 unless the
# caller explicitly runs TEP (EP_SIZE>1).
if [[ "$TP" == "8" && "${EP_SIZE}" -le 1 ]]; then
    echo "NOTE: TP=8 without expert parallelism is incompatible with the FP8 checkpoint; using recipe-validated TP=4."
    TP=4
fi

MTP_LIST="${MTP_LIST:-1 2 3 4 5 6 7 8}"
THINKING_MODES="${THINKING_MODES:-off on}"
CATEGORY="${CATEGORY:-coding}"
MODEL_KEY="${MODEL_KEY:-$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')}"
SPEEDBENCH_OUTPUT_LEN="${SPEEDBENCH_OUTPUT_LEN:-4096}"
# AL is concurrency-independent (per-token accept/reject; no spec-disable-by-batch
# is set below), so batch the SPEED-Bench pass to keep wall-time under the CI
# limit. Precedent: kimik3_fp4_b300_vllm.sh, where conc=1 blew the 8h budget.
CONCURRENCY="${CONCURRENCY:-64}"
# Provider-recommended sampling — DIFFERS by mode (per the Qwen3.8-Flash-Next
# model card):
#   thinking : temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0
#   instruct : temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5
# (min_p 0.0 / repetition_penalty 1.0 are vLLM defaults.) These MUST be passed
# per-mode or the measured AL is taken at the wrong sampling settings.
TEMPERATURE_ON="${TEMPERATURE_ON:-1.0}";  TOP_P_ON="${TOP_P_ON:-0.95}";  TOP_K_ON="${TOP_K_ON:-20}";  PRESENCE_PENALTY_ON="${PRESENCE_PENALTY_ON:-0.0}"
TEMPERATURE_OFF="${TEMPERATURE_OFF:-0.7}"; TOP_P_OFF="${TOP_P_OFF:-0.8}"; TOP_K_OFF="${TOP_K_OFF:-20}"; PRESENCE_PENALTY_OFF="${PRESENCE_PENALTY_OFF:-1.5}"
# Optional sampling seed for run-to-run variance checks. Unset -> vLLM default
# (deterministic seed=0); set to different values to measure temperature>0 variance.
SEED="${SEED:-}"
# Optional: also save per-request completions (--save-detailed) to eyeball that
# thinking_on actually emits <think> reasoning and thinking_off does not. Off by
# default (bloats the result JSON with all completions). Set SAVE_DETAILED=1.
SAVE_DETAILED="${SAVE_DETAILED:-}"
# Qwen thinking toggles via the enable_thinking chat_template key (default ON
# for Flash-Next). reasoning_effort is left at its model default (xhigh).
# Use separate single-quoted defaults: an inline ${VAR:-{...}} default whose value
# contains "}" is truncated by bash brace parsing (matches upstream fix #1695).
DEFAULT_CHAT_TEMPLATE_KWARGS_ON='{"enable_thinking": true}'
DEFAULT_CHAT_TEMPLATE_KWARGS_OFF='{"enable_thinking": false}'
CHAT_TEMPLATE_KWARGS_ON="${CHAT_TEMPLATE_KWARGS_ON:-$DEFAULT_CHAT_TEMPLATE_KWARGS_ON}"
CHAT_TEMPLATE_KWARGS_OFF="${CHAT_TEMPLATE_KWARGS_OFF:-$DEFAULT_CHAT_TEMPLATE_KWARGS_OFF}"

SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/workspace/speed_bench_data}"
# Flat results dir to match the speedbench-al.yml artifact glob
# (speedbench_results/server_*.log) and its pre-run `rm -rf speedbench_results`.
RESULTS_DIR="${RESULTS_DIR:-/workspace/speedbench_results}"
OUT_YAML="${OUT_YAML:-$RESULTS_DIR/speedbench-reference-al.yaml}"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi

# ---- Download target if it is not pre-staged ----
# Qwen3.8-Flash-Next-FP8 is not in the launcher's STAGED_MODELS list, so the
# launcher resolves MODEL_PATH into the writable models dir (/data/models);
# only download when MODEL_PATH is an empty writable dir (non-staged run).
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi
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
    local temp top_p top_k pp
    if [[ "$mode" == "on" ]]; then
        [[ -n "$CHAT_TEMPLATE_KWARGS_ON" ]] && think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
        temp="$TEMPERATURE_ON";  top_p="$TOP_P_ON";  top_k="$TOP_K_ON";  pp="$PRESENCE_PENALTY_ON"
    else
        [[ -n "$CHAT_TEMPLATE_KWARGS_OFF" ]] && think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_OFF")
        temp="$TEMPERATURE_OFF"; top_p="$TOP_P_OFF"; top_k="$TOP_K_OFF"; pp="$PRESENCE_PENALTY_OFF"
    fi
    local seed_args=()
    [[ -n "$SEED" ]] && seed_args=(--seed "$SEED")
    local detail_args=()
    [[ -n "$SAVE_DETAILED" ]] && detail_args=(--save-detailed)

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  MTP=$mtp  category=$CATEGORY"
    echo "=========================================="

    local serve_args=(
        --host 0.0.0.0 --port "$PORT"
        "${PARALLEL_ARGS[@]}"
        --pipeline-parallel-size 1
        --trust-remote-code
        --no-enable-prefix-caching
        "${EP_ARGS[@]}"
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
        --enable-auto-tool-choice
        --language-model-only
        --no-enable-flashinfer-autotune
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-cudagraph-capture-size 512
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
        --temperature "$temp" \
        --top-p "$top_p" \
        --top-k "$top_k" \
        --presence-penalty "$pp" \
        "${seed_args[@]}" \
        "${detail_args[@]}" \
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
    echo "# dataset: $CATEGORY | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on : temp $TEMPERATURE_ON top_p $TOP_P_ON top_k $TOP_K_ON presence_penalty $PRESENCE_PENALTY_ON | chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# thinking_off: temp $TEMPERATURE_OFF top_p $TOP_P_OFF top_k $TOP_K_OFF presence_penalty $PRESENCE_PENALTY_OFF | chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_OFF"
    echo "# Measured on $MODEL_KEY (B300, vLLM native MTP), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/qwen3.8next_fp4_b300_vllm.sh (speedbench-al.yml)."
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
