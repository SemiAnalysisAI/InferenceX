#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash on H100, with the memory caps 80 GB cards need.
#
# This is a copy of the shared dsv41flash_fp4_vllm_mtp.sh rather than a symlink
# to it, because H100 cannot serve 1M context on the shared flags. At 1M context
# the sparse attention indexer allocates a
# [max-num-batched-tokens, max-model-len] logits buffer in
# fp8_fp4_paged_mqa_logits: 8192 x 1048576 x 2 bytes is exactly 16.00 GiB. That
# is a fixed cost paid during startup memory profiling, independent of
# concurrency, so it OOMs at concurrency 1 next to ~35.9 GiB/GPU of resident
# weights (run 34467029236). Capping batched tokens shrinks the buffer
# proportionally; capping --max-model-len would too, but that would force the
# 256k-capped trace corpus onto a 1M-context model.
# https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONUNBUFFERED=1
# The failing allocation left 1.04 GiB reserved but unallocated, and the
# indexer buffer is large enough that fragmentation costs a KV block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The indexer buffer scales linearly with batched tokens: 4096 puts it at
# 8 GiB. Anything larger did not fit alongside the weights on this SKU.
MAX_NUM_BATCHED_TOKENS=4096
# Scheduler headroom for AgentX subagent fan-out. Set explicitly because
# vLLM's default of 1024 sizes sampler and scheduler buffers for a batch this
# recipe never runs.
MAX_NUM_SEQS=$((2 * CONC))
NUM_SPEC_TOKENS=5
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < MAX_NUM_SEQS * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Both throughput and eval use real target verification, never synthetic AL.
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --engram-config '{"cpu_offload":true}'
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
    --max-model-len 1048576
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization 0.92
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
