#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on B300 using vLLM.
# Mirrors the fixed-seq-len parallelism options (pure TP and DEP) so the
# agentic sweep can probe both interactivity and throughput regimes:
#   pure TP (DP_ATTENTION=false, EP_SIZE=1):  attention TP-sharded across
#       all $TP GPUs in a single engine. Lower TPOT, lower batch.
#   TP+EP   (DP_ATTENTION=false, EP_SIZE>1):  attention TP-sharded, MoE
#       experts EP-sharded within the TP group.
#   DEP     (DP_ATTENTION=true, EP_SIZE>1):   per-DP-rank attention with
#       experts EP-sharded across DP ranks (per the vLLM blog recipe).
#       Highest aggregate throughput at large CONC.
#
# Image is vllm/vllm-openai:v0.20.0-cu130. block_size=256, kv-cache-dtype=fp8,
# FP4 indexer cache enabled, FULL_AND_PIECEWISE cudagraph capture with
# custom_ops=all (per the vLLM blog recipe at https://vllm.ai/blog/deepseek-v4).
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}
DP_ATTENTION=${DP_ATTENTION:-false}
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=1000000
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# DeepSeek-V4-Pro weights are large; engine startup can exceed default 600s.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=""
case "$OFFLOADING" in
    none) ;;
    cpu)
        # B300 compute nodes have ~3.8 TiB host RAM; SLURM cgroup limits
        # individual jobs to a fraction of that. Aim for ~2.2 TB total host
        # CPU pool across the engine(s).
        #
        # SimpleCPUOffloadConnector divides cpu_bytes_to_use by
        # parallel_config.world_size (= TP*PP, NOT including DP — see
        # vllm/config/parallel.py docstring). So:
        #   - DP-attn=true  → each of $TP DP engines has world_size=1 in
        #     its parallel_config; the connector does no internal divide,
        #     and each engine torch.zeros + pin_tensor allocates the full
        #     --kv_offloading_size value. Pre-divide by $TP here so the
        #     aggregate host commit ≈ TOTAL_CPU_DRAM_GB.
        #   - DP-attn=false → single engine with world_size=TP. Pass the
        #     full TOTAL_CPU_DRAM_GB; the connector's internal divide
        #     yields TOTAL/TP per rank, and TP-shared mmap (PR #37206)
        #     keeps the aggregate at TOTAL.
        TOTAL_CPU_DRAM_GB=2200
        if [ "$DP_ATTENTION" = "true" ]; then
            PER_ENGINE_GB=$((TOTAL_CPU_DRAM_GB / TP))
        else
            PER_ENGINE_GB=$TOTAL_CPU_DRAM_GB
        fi
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $PER_ENGINE_GB"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu)" >&2
        exit 1
        ;;
esac

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLOAT32_MATMUL_PRECISION=high

vllm serve "$MODEL" \
--host 0.0.0.0 \
--port "$PORT" \
--trust-remote-code \
--kv-cache-dtype fp8 \
--block-size 256 \
"${PARALLEL_ARGS[@]}" \
"${EP_ARGS[@]}" \
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
--attention_config.use_fp4_indexer_cache=True \
--tokenizer-mode deepseek_v4 \
--tool-call-parser deepseek_v4 \
--enable-auto-tool-choice \
--reasoning-parser deepseek_v4 \
--enable-prefix-caching \
--no-disable-hybrid-kv-cache-manager \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
$OFFLOAD_ARGS > "$SERVER_LOG" 2>&1 &
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
