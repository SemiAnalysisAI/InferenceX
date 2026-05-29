#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5 FP8 on B200 using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

pip install --no-deps "transformers==5.2.0" "huggingface-hub==1.4.1"

export SGL_ENABLE_JIT_DEEPGEMM=1

# ---- Start SGLang server ----------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

echo "Starting SGLang server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1

python3 -m sglang.launch_server \
--model-path=$MODEL \
--host=0.0.0.0 \
--port=$PORT \
--trust-remote-code \
--tensor-parallel-size=$TP \
--data-parallel-size 1 \
--expert-parallel-size 1 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--kv-cache-dtype fp8_e4m3 \
--quantization fp8 \
--attention-backend nsa \
--nsa-decode-backend trtllm \
--nsa-prefill-backend trtllm \
--moe-runner-backend flashinfer_trtllm \
--cuda-graph-max-bs $CONC \
--max-running-requests $CONC \
--mem-fraction-static 0.85 \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--enable-flashinfer-allreduce-fusion \
--disable-radix-cache \
--stream-interval 30 \
--context-length $MAX_MODEL_LEN \
--enable-metrics \
--model-loader-extra-config '{"enable_multithread_load": true}' > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
