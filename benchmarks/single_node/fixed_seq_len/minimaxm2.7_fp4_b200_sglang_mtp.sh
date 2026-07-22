#!/usr/bin/env bash
set -Eeuo pipefail

# MiniMax-M2.7 NVFP4 on B200 with the public full-vocabulary EAGLE3 draft.
# This entrypoint serves both fixed-sequence and AgentX wrappers so the target,
# draft, speculative settings, and chat behavior cannot drift between them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    PORT \
    PRECISION \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ "$PRECISION" != "fp4" || "$EP_SIZE" != "1" ]]; then
    echo "MiniMax-M2.7 EAGLE3 expects NVFP4 with pure tensor parallelism, got precision=$PRECISION EP=$EP_SIZE" >&2
    exit 1
fi
if [[ "$TP" != "4" && "$TP" != "8" ]]; then
    echo "MiniMax-M2.7 EAGLE3 supports the configured TP4/TP8 search space, got TP=$TP" >&2
    exit 1
fi

readonly TARGET_MODEL="nvidia/MiniMax-M2.7-NVFP4"
readonly TARGET_REVISION="e79701cb1f9dce8fe5395b9ed2b20170beebecde"
readonly DRAFT_MODEL="asherszhang/MiniMax-M2.7-EAGLE3-draft-vocab200k"
readonly DRAFT_REVISION="252b54f7d05a5d0db7734563ae10e2e6a81d6a7d"

if [[ "$MODEL" != "$TARGET_MODEL" && "$MODEL" != /* ]]; then
    echo "Unexpected target model: $MODEL (expected $TARGET_MODEL)" >&2
    exit 1
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    TARGET_MODEL_PATH="$MODEL_PATH"
elif [[ "$MODEL" == /* ]]; then
    TARGET_MODEL_PATH="$MODEL"
else
    TARGET_MODEL_PATH="$TARGET_MODEL"
fi

if [[ "$TARGET_MODEL_PATH" == /* ]]; then
    MODEL_ROOT=$(dirname "$TARGET_MODEL_PATH")
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$MODEL_ROOT/MiniMax-M2.7-EAGLE3-draft-vocab200k}"
    mkdir -p "$TARGET_MODEL_PATH" "$DRAFT_MODEL_PATH"
    exec 8>"$MODEL_ROOT/.minimax-m2.7-eagle3-stage.lock"
    flock -w 7200 8
    if [[ ! -f "$TARGET_MODEL_PATH/config.json" || \
          ! -f "$TARGET_MODEL_PATH/model.safetensors.index.json" || \
          $(find "$TARGET_MODEL_PATH" -maxdepth 1 -name 'model-*-of-00015.safetensors' | wc -l) -ne 15 ]]; then
        hf download "$TARGET_MODEL" --revision "$TARGET_REVISION" --local-dir "$TARGET_MODEL_PATH"
    fi
    if [[ ! -f "$DRAFT_MODEL_PATH/config.json" || ! -f "$DRAFT_MODEL_PATH/model.safetensors" ]]; then
        hf download "$DRAFT_MODEL" --revision "$DRAFT_REVISION" --local-dir "$DRAFT_MODEL_PATH"
    fi
    flock -u 8
else
    hf download "$TARGET_MODEL" --revision "$TARGET_REVISION"
    hf download "$DRAFT_MODEL" --revision "$DRAFT_REVISION"
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
fi

python3 - "$TARGET_MODEL_PATH" "$DRAFT_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

target_path, draft_path = map(Path, sys.argv[1:])
target = json.loads((target_path / "config.json").read_text())
draft = json.loads((draft_path / "config.json").read_text())
quant = target.get("quantization_config", {})
if quant.get("quant_algo") != "NVFP4" or quant.get("quant_method") != "modelopt":
    raise SystemExit(f"Unexpected NVFP4 target configuration: {quant}")
if target.get("vocab_size") != 200064:
    raise SystemExit(f"Unexpected target vocabulary: {target.get('vocab_size')}")
if draft.get("architectures") != ["LlamaForCausalLMEagle3"]:
    raise SystemExit(f"Unexpected EAGLE3 architecture: {draft.get('architectures')}")
if draft.get("vocab_size") != 200064 or draft.get("num_hidden_layers") != 1:
    raise SystemExit(
        f"Unexpected EAGLE3 draft contract: vocab={draft.get('vocab_size')} "
        f"layers={draft.get('num_hidden_layers')}"
    )
print(f"Validated target={target_path} draft={draft_path}")
PY

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi
nvidia-smi

IS_AGENTIC=false
if [[ "${SCENARIO_TYPE:-fixed-seq-len}" == "agentic-coding" ]]; then
    IS_AGENTIC=true
    check_env_vars DURATION RESULT_DIR KV_OFFLOADING TOTAL_CPU_DRAM_GB
    if [[ "$KV_OFFLOADING" != "none" ]]; then
        echo "This low-concurrency EAGLE3 experiment requires GPU-resident KV cache" >&2
        exit 1
    fi
    export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
    resolve_trace_source
    install_agentic_deps
    OUTPUT_DIR="$RESULT_DIR"
    CONTEXT_LENGTH=196608
    export MAX_MODEL_LEN="$CONTEXT_LENGTH"
    MAX_RUNNING_REQUESTS=$((CONC * 2))
    (( MAX_RUNNING_REQUESTS < 8 )) && MAX_RUNNING_REQUESTS=8
else
    check_env_vars ISL OSL MAX_MODEL_LEN
    OUTPUT_DIR="$PWD"
    CONTEXT_LENGTH="$MAX_MODEL_LEN"
    MAX_RUNNING_REQUESTS=$((CONC * 2))
fi
mkdir -p "$OUTPUT_DIR"

SERVER_LOG="$OUTPUT_DIR/server.log"
SERVER_PID=""
GPU_MONITOR_STARTED=false

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    if [[ "$GPU_MONITOR_STARTED" == "true" ]]; then
        stop_gpu_monitor
    fi
    stop_background_process_tree "$SERVER_PID" "MiniMax-M2.7 EAGLE3 server" 60
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

start_gpu_monitor --output "$OUTPUT_DIR/gpu_metrics.csv"
GPU_MONITOR_STARTED=true

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$TARGET_MODEL_PATH"
    --served-model-name "$TARGET_MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --quantization modelopt_fp4
    --dtype bfloat16
    --attention-backend triton
    --moe-runner-backend flashinfer_cutlass
    --speculative-algorithm EAGLE3
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --speculative-draft-model-quantization unquant
    --speculative-draft-attention-backend triton
    --reasoning-parser minimax
    --tool-call-parser minimax-m2
    --context-length "$CONTEXT_LENGTH"
    --chunked-prefill-size 16384
    --weight-loader-prefetch-checkpoints
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --mem-fraction-static 0.85
    --enable-metrics
    --watchdog-timeout 1800
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$OUTPUT_DIR/sglang_command.txt"
printf '\n' | tee -a "$OUTPUT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

mtp_metrics=$(curl -fsS "http://127.0.0.1:$PORT/metrics")
if ! grep -m1 '^sglang:spec_accept_length' <<< "$mtp_metrics"; then
    echo "EAGLE3 acceptance metrics are absent; refusing to publish a non-speculative result" >&2
    exit 1
fi

if [[ "$IS_AGENTIC" == "true" ]]; then
    # AgentX sends structured chat messages to /v1/chat/completions, so the
    # target checkpoint's native chat template is applied by the server.
    build_replay_cmd "$RESULT_DIR"
    # TP4 can take several minutes to finish a long response that was already
    # admitted near the end of the measurement window. The AIPerf default is
    # only 30 seconds, which cancelled the sole profiling request in the TP4
    # smoke while the healthy server was still decoding it. Bound the drain at
    # 30 minutes without extending the measured request-admission window.
    REPLAY_CMD+=" --benchmark-grace-period 1800"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
else
    # EAGLE3 was trained on chat-formatted prompts; keep the benchmark client
    # on the target checkpoint's native chat template.
    run_benchmark_serving \
        --model "$TARGET_MODEL_PATH" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$((CONC * 10))" \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir "$PWD" \
        --trust-remote-code \
        --use-chat-template

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        run_eval --framework lm-eval --port "$PORT"
        append_lm_eval_summary
    fi
fi
