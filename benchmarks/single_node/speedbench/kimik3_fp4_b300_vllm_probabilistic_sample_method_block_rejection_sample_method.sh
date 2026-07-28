#!/usr/bin/env bash

# Kimi-K3 B300 vLLM SPEED-Bench AL matrix collector for DSpark speculative
# decoding.
#
# VARIANT of kimik3_fp4_b300_vllm.sh (kept intact as the baseline): the
# speculative-config additionally sets draft_sample_method=probabilistic and
# rejection_sample_method=block, mirroring PR #2366's acceptance-rate
# optimization as a separate collector so baseline and variant AL curves can
# be measured side by side.
#
# Produces the golden acceptance-length (AL) reference matrix consumed by the
# synthetic-acceptance framework: for each DSpark speculative-token count,
# measure the REAL AL on a single SPEED-Bench category (default: coding) and
# emit a YAML matrix identical in shape to the other golden_al_distribution
# curves.
#
# Kimi-K3 uses the Inferact/Kimi-K3-DSpark draft head. The draft model is
# downloaded to a writable workspace dir before the sweep begins; the target
# checkpoint (moonshotai/Kimi-K3, FP4) is pre-staged on the B300 cluster at
# /scratch/models/Kimi-K3, so the launcher resolves MODEL_PATH there and no
# target download happens.
#
# Differences vs the Kimi-K2.5 EAGLE3 template (kimik2.5_fp4_b300_vllm.sh):
#   - speculative-config     dspark + attention_backend FLASHINFER_MLA (not eagle3)
#   - reasoning/tool parser  kimi_k3        (was kimi_k2)
#   - thinking modes         "on" only — K3 is a thinking model; the parser
#                            (vllm/reasoning/kimi_k3_reasoning_parser.py)
#                            defaults enable_thinking=True
#   - prefix caching         ENABLED (production recipe) — was disabled
#   - serve flags            --load-format fastsafetensors, --moe-backend auto,
#                            --max-num-seqs, --max-cudagraph-capture-size,
#                            --attention-config, per the K3 production recipe
#   - NO --language-model-only (text-only checkpoint)
#
# Usage (inside the Kimi-K3 vLLM container, on a B300 node):
#   export MODEL=moonshotai/Kimi-K3
#   bash benchmarks/single_node/speedbench/kimik3_fp4_b300_vllm_probabilistic_sample_method_block_rejection_sample_method.sh
#
# Tunables (env):
#   MTP_LIST          space-separated DSpark spec-token counts (default "1 2 3 4 5 6 7 8")
#   THINKING_MODES    space-separated: off|on       (default "on")
#   CATEGORY          SPEED-Bench category          (default coding)
#   SPEEDBENCH_OUTPUT_LEN  per-request output len   (default 4096)
#   OUT_YAML          output matrix path            (default $RESULTS_DIR/speedbench-reference-al.yaml)

set -uo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

MODEL="${MODEL:?MODEL env var required (e.g. moonshotai/Kimi-K3)}"
SERVE_MODEL="${MODEL_PATH:-$MODEL}"
TP="${TP:-8}"
DP_ATTENTION="${DP_ATTENTION:-false}"
EP_SIZE="${EP_SIZE:-1}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"

DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"

MTP_LIST="${MTP_LIST:-1 2 3 4 5 6 7 8}"
# K3 is a thinking model (kimi_k3 reasoning parser defaults enable_thinking=True),
# so the golden curve is collected for thinking_on only.
THINKING_MODES="${THINKING_MODES:-on}"
CATEGORY="${CATEGORY:-coding}"
MODEL_KEY="${MODEL_KEY:-$(basename "$SERVE_MODEL" | tr '[:upper:]' '[:lower:]')}"
SPEEDBENCH_OUTPUT_LEN="${SPEEDBENCH_OUTPUT_LEN:-4096}"
# AL is concurrency-independent (per-token accept/reject; no spec-disable-by-batch
# is set below), so batch the SPEED-Bench pass to keep wall-time under the CI
# limit. Inherited from the Kimi-K2.5 collector, where conc=1 blew the 8h budget.
CONCURRENCY="${CONCURRENCY:-64}"
TOP_P="${TOP_P:-0.95}"
# Kimi thinking toggles via the thinking chat_template key. K3 defaults to
# thinking ON, so the on-cell kwargs are stated explicitly and the off-cell
# kwargs disable it. NOTE: speedbench-al.yml's thinking-kwargs input defaults to
# the DSV4 value ({"thinking": true, "reasoning_effort": "high"}) and is exported
# as CHAT_TEMPLATE_KWARGS_ON — dispatch K3 with -f 'thinking-kwargs={"thinking": true}'.
DEFAULT_CHAT_TEMPLATE_KWARGS_ON='{"thinking": true}'
DEFAULT_CHAT_TEMPLATE_KWARGS_OFF='{"thinking": false}'
CHAT_TEMPLATE_KWARGS_ON="${CHAT_TEMPLATE_KWARGS_ON:-$DEFAULT_CHAT_TEMPLATE_KWARGS_ON}"
CHAT_TEMPLATE_KWARGS_OFF="${CHAT_TEMPLATE_KWARGS_OFF:-$DEFAULT_CHAT_TEMPLATE_KWARGS_OFF}"

SPEEDBENCH_DIR="${SPEEDBENCH_DIR:-/workspace/speed_bench_data}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/speedbench_results}"
OUT_YAML="${OUT_YAML:-$RESULTS_DIR/speedbench-reference-al.yaml}"

# Kimi-K3 production serving environment.
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-1}"
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# `vllm bench serve` delegates the CLIENT to the Rust binary: the kimi-k3 branch's
# vllm/entrypoints/cli/benchmark/serve.py runs _maybe_exec_rust_bench(), which
# os.execv's into vllm-rs whenever the dataset (speed_bench qualifies) and backend
# are supported and the binary exists. There is no env var and no CLI flag to opt
# out, and the Rust flag surface is a subset of the Python one (no
# --speed-bench-output-len — only --speed-bench-max-input-len — no --save-detailed,
# no --chat-template-kwargs), so every cell dies at argument parsing and the whole
# matrix comes back N/A.
#
# Call the Python benchmark entrypoint directly instead, exactly as the CLI's own
# Python fallback does (vllm.benchmarks.serve.main with a parser built by
# add_cli_args), which never reaches the execv. The SERVER keeps the Rust frontend
# per the production recipe; AL is read from the server's /metrics, so the client
# choice does not affect the measurement.
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
# This script runs without `set -e`, so a failed redirect would otherwise only
# surface later as a confusing "No such file" from the preflight.
if [[ ! -s "$BENCH_DRIVER" ]]; then
    echo "CRITICAL: could not write the benchmark driver to $BENCH_DRIVER — aborting."
    exit 1
fi

nvidia-smi

# ---- Download target if it is not pre-staged ----
# Kimi-K3 is in the launcher's STAGED_MODELS, so MODEL_PATH resolves to the
# read-only /scratch/models/Kimi-K3 and this block is a no-op. It still covers a
# standalone run where the weights are not staged.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$SERVE_MODEL" != /* ]]; then hf download "$SERVE_MODEL"; fi
fi

# ---- Download DSpark draft model to a WRITABLE dir ----
# The draft must NOT go next to a pre-staged target: dirname(MODEL_PATH) can be
# the read-only staged mount (/scratch/models), so writing the draft there fails
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

# ---- Conditional shim: ensure --chat-template-kwargs reaches the chat template ----
# speed_bench/CustomDataset pre-renders the chat template client-side and posts to
# /v1/completions, so thinking mode cannot be enabled via --extra-body or
# --default-chat-template-kwargs — the kwargs must reach apply_chat_template.
# vllm-project/vllm#44244 added this natively (serve.py declares the CLI option,
# the speed_bench dispatch forwards it, CustomDataset.sample unpacks it), and the
# kimi-k3 image already carries it. So: probe for native support first and no-op
# when present; only patch the pieces an older image is missing. Idempotent.
apply_chat_template_kwargs_shim() {
    echo "=== Checking vLLM benchmark --chat-template-kwargs support ==="
    python3 - <<'PYEOF'
import sys
import vllm.benchmarks.serve as S
import vllm.benchmarks.datasets.datasets as D

def read(mod):
    with open(mod.__file__) as fh:
        return fh.read()

s_src, d_src = read(S), read(D)

# Native (post-#44244) spellings, plus the ones this shim itself writes.
have_cli = '"--chat-template-kwargs"' in s_src
have_forward = ('chat_template_kwargs=getattr(args' in d_src
                or 'chat_template_kwargs=args.chat_template_kwargs' in d_src)
have_unpack = ('**(chat_template_kwargs or {})' in d_src
               or '**_ctk' in d_src)

if have_cli and have_forward and have_unpack:
    print("native --chat-template-kwargs support present; no patching needed")
    sys.exit(0)

print(f"patching (cli={have_cli} forward={have_forward} unpack={have_unpack})")

def apply(mod, src, edits):
    for old, new in edits:
        n = src.count(old)
        assert n == 1, (
            f"anchor matched {n} times in {mod.__file__}, aborting. This image's "
            f"benchmark source differs from both the pre-#44244 and post-#44244 "
            f"layouts; update the shim.\n{old[:120]}..."
        )
        src = src.replace(old, new, 1)
    with open(mod.__file__, "w") as fh:
        fh.write(src)
    print("patched OK ->", mod.__file__)

if not have_cli:
    # serve.py -- declare the --chat-template-kwargs argument before --extra-body
    apply(S, s_src, [('''    parser.add_argument(
        "--extra-body",''', '''    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help="JSON dict forwarded to apply_chat_template during "
        "client-side prompt rendering, e.g. to enable reasoning mode.",
    )
    parser.add_argument(
        "--extra-body",''')])

d_edits = []
if not have_forward:
    # datasets.py -- forward args.chat_template_kwargs into the speed_bench .sample() call
    d_edits.append(('''                output_len=args.speed_bench_output_len,
                enable_multimodal_chat=args.enable_multimodal_chat,''',
                    '''                output_len=args.speed_bench_output_len,
                chat_template_kwargs=args.chat_template_kwargs,
                enable_multimodal_chat=args.enable_multimodal_chat,'''))
if not have_unpack:
    # datasets.py -- forward chat_template_kwargs into CustomDataset.sample's template call
    d_edits.append(('''                # apply template
                if not skip_chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )

                prompt_len = len(tokenizer(prompt).input_ids)''',
                    '''                # apply template
                if not skip_chat_template:
                    _ctk = kwargs.get("chat_template_kwargs") or {}
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        **_ctk,
                    )

                prompt_len = len(tokenizer(prompt).input_ids)'''))
if d_edits:
    apply(D, d_src, d_edits)
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

# ---- Preflight: the benchmark client must support the flags every cell uses ----
# Cheap up front; without it a CLI mismatch is only visible as an all-N/A matrix
# after eight full server starts (~1h of runner time).
# NOTE: probe the driver, not `vllm bench serve` — the CLI's help exits inside
# argparse before the Rust execv, so its help says nothing about what actually
# runs. Plain --help prints only a group summary, so ask for --help=all.
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
