#!/usr/bin/env bash
set -eo pipefail
set -x

# DeepSeek-V4.1-Flash AgentX on MI355X: TP2 / TP4 with five-token DSpark.
# https://github.com/ROCm/ATOM/blob/53b11c9a665e786798785acbedfdfd4da3fb87c4/recipes/DeepSeek-V4.1-Flash-Agentic.md
# The launcher routes both mtp and draft_model through the _mtp suffix.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EP_SIZE DP_ATTENTION EVAL_ONLY PORT

if [[ "$TP" != 2 && "$TP" != 4 ]] || [[ "$EP_SIZE" != 1 || "$DP_ATTENTION" != false ]]; then
    echo "ERROR: this recipe requires TP=2 or TP=4, EP_SIZE=1, and DP_ATTENTION=false" >&2
    exit 1
fi
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

# Resume incomplete downloads and keep the tokenizer aligned with the server.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
export AGENTIC_TOKENIZER_PATH="$MODEL_PATH"

# Respect the GPU allocation chosen by Slurm, including nonzero GPU pairs.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
export OMP_NUM_THREADS=4
export ATOM_NUMA_BIND=0
export ATOM_DISABLE_MMAP=true
export AITER_LOG_LEVEL=WARNING

resolve_trace_source
install_agentic_deps

# Upstream's one-hour AgentX profile uses five warmup requests per lane.
# build_replay_cmd retains the workflow's explicit duration / fast-mode inputs.
export AIPERF_WARMUP_REQUESTS_PER_LANE=5
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

wait_for_amd_gpu_clean
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
SERVER_PID=""
cleanup_atom_server() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "ATOM server" 60
    exit "$exit_code"
}
trap cleanup_atom_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# At c32, capture every size through 32; all other points use the sparse list.
CAPTURE_SIZES='[1,2,3,4,5,6,7,8,16,32,48,64,128]'
if [[ "$CONC" -eq 32 ]]; then
    CAPTURE_SIZES='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]'
fi

# golden_al_distribution/dsv41flash_dspark.yaml: thinking_on, K5 -> AL 3.51.
# Throughput follows the upstream recipe; accuracy eval verifies real drafts.
SPEC_ARGS=(--method dspark --num-speculative-tokens 5)
if [[ "$EVAL_ONLY" != true ]]; then
    SPEC_ARGS+=(--spec-decode-acceptance-length 3.51)
fi

ATOM_CMD=(
    python3 -u -m atom.entrypoints.openai_server
    --model "$MODEL_PATH" --served-model-name "$MODEL" --trust-remote-code
    --host 0.0.0.0 --server-port "$PORT"
    --tensor-parallel-size "$TP"
    --kv_cache_dtype bf16 --index-cache-dtype fp8
    --gpu-memory-utilization 0.9 --max-num-seqs 128
    --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384
    --enable_prefix_caching --block-size 16
    --state-checkpoint-interval-tokens 8192
    --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAPTURE_SIZES"
    "${SPEC_ARGS[@]}"
    --tool-call-parser dsml_v41
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --port "$PORT"
else
    # AgentX traces contain fully formed chat payloads; preserve native handling.
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
