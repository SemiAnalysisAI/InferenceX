#!/usr/bin/env bash

# DSV4-Pro B300 vLLM SPEED-Bench AL matrix collector for DSpark speculative decoding.
#
# For each thinking mode (on/off) and DSpark speculative-token count, measure the REAL
# acceptance length (AL) on one SPEED-Bench category and emit a YAML matrix in the
# golden_al_distribution shape. DSpark ships as a separate checkpoint
# (deepseek-ai/DeepSeek-V4-Pro-DSpark, 960 GB) with the draft baked in, so there is no
# external draft head and no "model" key in the speculative-config. Every flag that
# affects drafting is byte-identical to the DSV4 MTP collector so the two AL curves
# stay comparable.
#
# Dispatch this collector through speedbench-al.yml.
#
# Required collection settings come from speedbench-al.yml.

set -o pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON DRAFT_SAMPLE_METHOD MODEL MODEL_PATH MTP_LIST \
    OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

# MODEL_PATH is the launcher-resolved weights dir (writable models dir until the
# checkpoint is staged; see the download block below).
SERVE_MODEL="${MODEL_PATH}"

# Top-level key in the emitted YAML matrix comes from the model basename.
MODEL_KEY="$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')"
# AL is a per-draft accept/reject property independent of batch size, so batch the
# SPEED-Bench pass to cut wall-clock. Nothing sets speculative_disable_by_batch_size,
# so drafting stays on at this batch size.
CONCURRENCY="32"
# Must stay >= CONCURRENCY or requests just queue. Held far below vLLM's default of
# 1024 because that sizes two allocations the memory profiler never sees: the
# rejection sampler's fp32 logits scratch (max_num_seqs * (1 + spec_tokens) * vocab *
# 4B, 4.4 GB at 8 tokens) and the spec-decode CUDA graphs. DSV4-Pro has no room:
# weights + a 100 GiB KV cache already fill 266 of 268 GiB per B300.
MAX_NUM_SEQS="64"
# Reserve device memory for KV cache and speculative verification.
GPU_MEM_UTIL="0.90"
TEMPERATURE="1.0"
# MUST match the golden config: golden_al_distribution/dsv4_mtp.yaml was measured
# with reasoning_effort=high.
# The published recipe uses greedy; probabilistic won at every level on Kimi-K3
# (golden_al_distribution/kimik3_dspark*.yaml). vLLM accepts exactly these two values
# (vllm/config/speculative.py: DraftSampleMethod).
case "$DRAFT_SAMPLE_METHOD" in
    greedy|probabilistic) ;;
    *)
        echo "CRITICAL: DRAFT_SAMPLE_METHOD must be 'greedy' or 'probabilistic' (got '$DRAFT_SAMPLE_METHOD')"
        exit 1
        ;;
esac
# Opt-in rather than tied to draft_sample_method: flipping it to "block" would bundle
# two variables into one measurement, and the forced-AL config has to stay on a
# sampling method TRT-LLM supports too.
REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD:-}"

SPEEDBENCH_DIR="/workspace/speed_bench_data"
RESULTS_DIR="/workspace/speedbench_results"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi

# The DSpark checkpoint is not in the launcher's STAGED_MODELS, so MODEL_PATH resolves
# to the writable models dir and the ~960 GB download runs once. Add the basename to
# STAGED_MODELS once the weights are staged on the read-only mount.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        if [[ ! -w "$(dirname "$MODEL_PATH")" ]]; then
            echo "CRITICAL: $MODEL_PATH is empty and $(dirname "$MODEL_PATH") is not writable."
            echo "This means the basename is listed in the launcher's STAGED_MODELS but the"
            echo "weights were never staged. Either get them staged, or remove it from"
            echo "STAGED_MODELS so MODEL_PATH resolves to the writable models dir instead."
            exit 1
        fi
        echo "=== $MODEL_PATH is empty; downloading $MODEL (~960 GB, first run only) ==="
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

# speed_bench/CustomDataset renders the chat template client-side and posts to
# /v1/completions, so thinking mode must reach apply_chat_template via
# --chat-template-kwargs (native since vllm-project/vllm#44244). Assert rather than
# assume: if the CLI option exists but speed_bench does not forward it, the flag is
# silently ignored and every thinking_on cell reports a non-thinking AL.
assert_chat_template_kwargs_support() {
    echo "=== Checking vLLM benchmark --chat-template-kwargs support ==="
    python3 - <<'PYEOF'
import sys
import vllm.benchmarks.serve as S
import vllm.benchmarks.datasets.datasets as D

def read(mod):
    with open(mod.__file__) as fh:
        return fh.read()

s_src, d_src = read(S), read(D)

missing = []
if '"--chat-template-kwargs"' not in s_src:
    missing.append(f"CLI option in {S.__file__}")
if ('chat_template_kwargs=getattr(args' not in d_src
        and 'chat_template_kwargs=args.chat_template_kwargs' not in d_src):
    missing.append(f"speed_bench forward in {D.__file__}")
if '**(chat_template_kwargs or {})' not in d_src:
    missing.append(f"apply_chat_template unpack in {D.__file__}")

if missing:
    print("CRITICAL: this image lacks native --chat-template-kwargs support:")
    for item in missing:
        print("  missing:", item)
    print("thinking_on cells would silently measure a non-thinking AL. Use an")
    print("image that contains vllm-project/vllm#44244.")
    sys.exit(1)

print("native --chat-template-kwargs support confirmed")
PYEOF
}

if [[ " $THINKING_MODES " == *" on "* ]]; then
    if ! assert_chat_template_kwargs_support; then
        echo "CRITICAL: --chat-template-kwargs preflight failed — aborting"
        exit 1
    fi
fi

# TEP8 as in the published B300 DSpark recipe (vllm-project/recipes). Hard-coded
# rather than driven by EP_SIZE / DP_ATTENTION because speedbench-al.yml exports
# EP_SIZE=1 and DP_ATTENTION=false for every model, which silently turned the recipe
# into plain TP; TP-sharding the FP4 experts costs ~37 GiB per GPU over EP and made
# the num_speculative_tokens=4 cell OOM. AL is unaffected by expert placement.
PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
EP_ARGS=(--enable-expert-parallel)
MOE_ARGS=(--moe-backend deep_gemm_mega_moe)

SPEC_EXTRA=""
if [[ -n "$REJECTION_SAMPLE_METHOD" ]]; then
    SPEC_EXTRA=", \"rejection_sample_method\": \"$REJECTION_SAMPLE_METHOD\""
fi

fetch_metric() {
    local port="$1" name="$2"
    curl -s "http://localhost:${port}/metrics" \
      | grep -oP "${name}\\{[^}]*\\} \\K[0-9.]+" || echo "0"
}

SERVER_PID=""
# Descendant PIDs of $1 by PARENT pid. This can never include this script (an
# ancestor of the server), unlike a name-based `pkill -f vllm`, which self-killed
# because the script filename contains "vllm".
_descendants() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        echo "$child"
        _descendants "$child"
    done
}
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        # Snapshot the worker/EngineCore subprocesses BEFORE killing the parent: once it
        # dies the children reparent to init and the tree link is lost. An orphaned
        # worker holds GPU memory and OOMs the next server start.
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
    if [[ "$mode" == "on" ]]; then
        think_args=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
    fi

    echo ""
    echo "=========================================="
    echo "  Cell: thinking=$mode  DSPARK=$mtp  category=$CATEGORY"
    echo "  draft_sample_method=$DRAFT_SAMPLE_METHOD"
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
        --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $mtp, \"draft_sample_method\": \"$DRAFT_SAMPLE_METHOD\"$SPEC_EXTRA}"
    )

    local server_log="$RESULTS_DIR/server_${mode}_mtp${mtp}.log"
    vllm serve "$SERVE_MODEL" "${serve_args[@]}" > "$server_log" 2>&1 &
    SERVER_PID=$!

    # wait_for_server_ready exits the shell (rather than returning) when the server
    # dies; the subshell keeps that exit local so one bad cell does not abort the matrix.
    if ! (wait_for_server_ready --port "$PORT" --server-log "$server_log" --server-pid "$SERVER_PID"); then
        echo "  -> server failed to start (thinking=$mode dspark=$mtp), recording N/A"
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

SPEC_SUMMARY="method=dspark | draft_sample_method=$DRAFT_SAMPLE_METHOD"
if [[ -n "$REJECTION_SAMPLE_METHOD" ]]; then
    SPEC_SUMMARY="$SPEC_SUMMARY | rejection_sample_method=$REJECTION_SAMPLE_METHOD"
fi

{
    echo "# Acceptance Length (AL) reference values measured with SPEED-Bench."
    echo "# dataset: $CATEGORY | temperature: $TEMPERATURE | output_len: $SPEEDBENCH_OUTPUT_LEN"
    echo "# thinking_on chat_template_kwargs: $CHAT_TEMPLATE_KWARGS_ON"
    echo "# speculative-config: $SPEC_SUMMARY"
    echo "# Measured on $MODEL_KEY (B300, vLLM DSpark), per num_speculative_tokens."
    echo "# Auto-generated by benchmarks/single_node/speedbench/dsv4dspark_fp4_b300_vllm.sh (speedbench-al.yml)."
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
