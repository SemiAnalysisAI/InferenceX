#!/usr/bin/env bash

# DSV4-Pro B300 vLLM SPEED-Bench AL matrix collector for DSpark speculative
# decoding.
#
# Produces the golden acceptance-length (AL) reference matrix consumed by the
# synthetic-acceptance framework: for each thinking mode (on/off) and each
# DSpark speculative-token count, measure the REAL AL on a single SPEED-Bench
# category (default: coding) and emit a YAML matrix identical in shape to the
# other golden_al_distribution curves.
#
# DSpark is DeepSeek's own speculative-decoding scheme and ships as a separate
# checkpoint (deepseek-ai/DeepSeek-V4-Pro-DSpark, 960 GB) with the draft baked
# in — unlike the Kimi-K3 DSpark collector there is no external draft head to
# download and no "model" key in the speculative-config.
#
# Differences vs the DSV4 MTP collector (dsv4_fp4_b300_vllm.sh), which this is
# otherwise a copy of so the two AL curves stay directly comparable:
#   - target model       DeepSeek-V4-Pro-DSpark (was DeepSeek-V4-Pro)
#   - speculative-config method dspark + draft_sample_method (was method mtp)
#   - expert parallel + deep_gemm_mega_moe, per the published B300 DSpark recipe
#   - --max-num-seqs and --gpu-memory-utilization pinned (MTP took the defaults)
# The last two are about fitting in memory, not about drafting: see the TEP block
# and MAX_NUM_SEQS below for what each one was fixing. AL is a per-draft
# accept/reject property, independent of expert placement, batch size and graph
# capture, so they leave "DSpark vs MTP on DSV4-Pro" like-for-like. Every flag
# that does affect drafting is byte-identical to the MTP collector.
#
# Usage (inside the vLLM container, on a B300 node):
#   export MODEL=deepseek-ai/DeepSeek-V4-Pro-DSpark
#   bash benchmarks/single_node/speedbench/dsv4dspark_fp4_b300_vllm.sh
#
# Tunables (env):
#   MTP_LIST          space-separated DSpark spec-token counts (default "1 2 3 4 5 6 7 8")
#   THINKING_MODES    space-separated: off|on       (default "off on")
#   CATEGORY          SPEED-Bench category          (default coding)
#   SPEEDBENCH_OUTPUT_LEN  per-request output len   (default 4096)
#   OUT_YAML          output matrix path            (default $RESULTS_DIR/speedbench-reference-al.yaml)
#   DRAFT_SAMPLE_METHOD    greedy|probabilistic     (default greedy)
#   REJECTION_SAMPLE_METHOD  passed through to speculative-config when non-empty
#                            (default: unset, i.e. the vLLM default)
#   MAX_NUM_SEQS      engine max batch size         (default 64)
#   GPU_MEM_UTIL      --gpu-memory-utilization      (default 0.90)

set -uo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL="${MODEL:?MODEL env var required (e.g. deepseek-ai/DeepSeek-V4-Pro-DSpark)}"
# Serve from the local weights dir resolved by the launcher (MODEL_PATH points
# at the writable models dir, e.g. /data/models/DeepSeek-V4-Pro-DSpark, until the
# checkpoint is staged; see the download block below). Falls back to MODEL for a
# standalone local run where MODEL is itself a path.
SERVE_MODEL="${MODEL_PATH:-$MODEL}"
TP="${TP:-8}"
PORT="${PORT:-8888}"

MTP_LIST="${MTP_LIST:-1 2 3 4 5 6 7 8}"
THINKING_MODES="${THINKING_MODES:-off on}"
CATEGORY="${CATEGORY:-coding}"
# Top-level key in the emitted YAML matrix. Derived from the model by the
# workflow (e.g. deepseek-v4-pro-dspark); falls back to the model basename,
# lowercased.
MODEL_KEY="${MODEL_KEY:-$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')}"
SPEEDBENCH_OUTPUT_LEN="${SPEEDBENCH_OUTPUT_LEN:-4096}"
# AL is a per-draft accept/reject property and is independent of batch size, so
# the SPEED-Bench pass is batched to cut wall-clock. Note this differs from the
# DSV4 MTP collector, which measured the existing deepseek-v4-pro curve at 1;
# nothing here sets speculative_disable_by_batch_size, so drafting stays on at
# this batch size and the curves remain comparable.
CONCURRENCY="${CONCURRENCY:-32}"
# Engine batch size; must stay >= CONCURRENCY or the client's requests just queue.
# Held far below the vLLM default of 1024 because that default sizes two
# allocations the memory profiler never sees — the rejection sampler's fp32 logits
# scratch, max_num_seqs * (1 + num_speculative_tokens) * vocab * 4B, which is
# 2.5 GB at 4 speculative tokens and 4.4 GB at 8, and the spec-decode CUDA graphs,
# which grow with the same product. DSV4-Pro has no room for either: 141.5 GiB of
# weights plus a 100 GiB KV cache already fills 266 of the 268 GiB on each B300,
# and warmup died asking for 2.47 GiB more at num_speculative_tokens=4.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
# vLLM's own default, exposed so the KV cache can be traded for headroom if some
# higher num_speculative_tokens still runs out.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# thinking-on chat_template_kwargs. MUST match the production/golden config:
# the reference matrix (golden_al_distribution/dsv4_mtp.yaml) was measured with
# reasoning_effort=high.
DEFAULT_CHAT_TEMPLATE_KWARGS_ON='{"thinking": true, "reasoning_effort": "high"}'
CHAT_TEMPLATE_KWARGS_ON="${CHAT_TEMPLATE_KWARGS_ON:-$DEFAULT_CHAT_TEMPLATE_KWARGS_ON}"
# The greedy/probabilistic knob Benjamin asked to characterize on DSV4-Pro. The
# published recipe uses greedy; probabilistic won at every level on Kimi-K3
# (golden_al_distribution/kimik3_dspark*.yaml). vLLM accepts exactly these two
# values (vllm/config/speculative.py: DraftSampleMethod).
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-greedy}"
case "$DRAFT_SAMPLE_METHOD" in
    greedy|probabilistic) ;;
    *)
        echo "CRITICAL: DRAFT_SAMPLE_METHOD must be 'greedy' or 'probabilistic' (got '$DRAFT_SAMPLE_METHOD')"
        exit 1
        ;;
esac
# Left unset by default. The K3 probabilistic variant also flipped this to
# "block", but that bundles two variables into one measurement and the forced-AL
# config has to stay on a sampling method TRT-LLM supports too, so it is opt-in
# here rather than tied to draft_sample_method.
REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD:-}"

SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/workspace/speed_bench_data}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/speedbench_results}"
OUT_YAML="${OUT_YAML:-$RESULTS_DIR/speedbench-reference-al.yaml}"

export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$RESULTS_DIR"
nvidia-smi

# ---- Resolve target weights ----
# The DSpark checkpoint is NOT in the launcher's STAGED_MODELS (it is not staged
# on the B300 cluster yet), so MODEL_PATH resolves to the writable models dir and
# the ~960 GB download below runs once, on the first collection. Add the basename
# back to STAGED_MODELS once the weights are staged to read them from the faster
# read-only mount instead.
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

# ---- Download SPEED-Bench dataset ----
echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed — $SPEEDBENCH_DIR/qualitative.jsonl not found"
    exit 1
fi

# ---- Preflight: --chat-template-kwargs must reach the chat template ----
# speed_bench/CustomDataset pre-renders the chat template client-side and posts to
# /v1/completions, so thinking mode cannot be enabled via --extra-body or
# --default-chat-template-kwargs — the kwargs have to reach apply_chat_template.
# vllm-project/vllm#44244 made that native and the images this collector runs on
# carry it, so there is nothing to patch (the older collectors monkey-patched
# site-packages here; that is what this replaces).
#
# It is still worth asserting rather than assuming, because the failure is silent
# in one direction: if the CLI option exists but the speed_bench path does not
# forward it, the flag is accepted and ignored, every thinking_on prompt renders
# without thinking, and the cell reports a non-thinking AL under the thinking_on
# key. A missing CLI option, by contrast, would fail loudly at argument parsing.
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

# Only thinking-on cells pass chat_template_kwargs, so only they need the support.
if [[ " $THINKING_MODES " == *" on "* ]]; then
    if ! assert_chat_template_kwargs_support; then
        echo "CRITICAL: --chat-template-kwargs preflight failed — aborting"
        exit 1
    fi
fi

# TEP8, exactly as the published B300 DSpark recipe (vllm-project/recipes: TP 8 +
# --enable-expert-parallel + --moe-backend deep_gemm_mega_moe).
#
# This is hard-coded rather than driven by the EP_SIZE / DP_ATTENTION knobs the
# MTP collector carries, because speedbench-al.yml exports EP_SIZE=1 and
# DP_ATTENTION=false for every model in the matrix, which silently turned the
# recipe into plain TP. That cost real memory: TP-sharding the FP4 experts loaded
# 141.53 GiB per GPU, 8x that being 1132 GiB against an 831 GiB checkpoint, so
# roughly 37 GiB per GPU went to sharding overhead on weights that expert
# parallel keeps whole. With a ~100 GiB KV cache on top, 266 of the 268 GiB were
# gone before warmup, which is what made the num_speculative_tokens=4 cell OOM.
#
# TP stays 8 (not the DP+EP variant the recipe also lists) to match the MTP
# collector's parallelism. Either way AL is unaffected: expert placement changes
# where a matmul runs, not which draft tokens the target model accepts.
PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
EP_ARGS=(--enable-expert-parallel)
MOE_ARGS=(--moe-backend deep_gemm_mega_moe)

# Optional extra speculative-config keys, rendered once so run_cell only has to
# interpolate num_speculative_tokens.
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

    # wait_for_server_ready exits the shell rather than returning when the server
    # dies, which would make the N/A branch below unreachable and let one bad cell
    # abort the whole matrix. Running it in a subshell keeps that exit local, so a
    # cell that cannot start its server costs one cell instead of the run.
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

# ---- Emit the YAML matrix ----
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
