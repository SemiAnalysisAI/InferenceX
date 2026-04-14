#!/usr/bin/env bash
# A/B test: compare N cold starts (baseline) vs 1 start + N reconfigure cycles.
#
# Run this directly on a GPU node inside a vLLM container that includes the
# /reconfigure endpoint. Mount the inferencex workspace at /workspace.
#
# Required env vars:
#   MODEL   -- HuggingFace model path  (e.g. openai/gpt-oss-120b)
#   TP      -- tensor-parallel size    (e.g. 8)
#   CONC    -- benchmark concurrency   (e.g. 32)
#   ISL     -- input sequence length   (e.g. 1024)
#   OSL     -- output sequence length  (e.g. 1024)
#
# Optional:
#   PORT              -- server port (default 8888)
#   VLLM_EXTRA_ARGS   -- extra args passed to vllm serve (e.g. "--kv-cache-dtype fp8")
#   SKIP_BASELINE     -- set to 1 to skip the baseline phase
#   SKIP_RECONFIG     -- set to 1 to skip the reconfigure phase
#
# Usage:
#   export MODEL=openai/gpt-oss-120b TP=8 CONC=32 ISL=1024 OSL=1024
#   bash benchmarks/test_reconfigure_sweep.sh
set -euo pipefail

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL

PORT=${PORT:-8888}
MAX_MODEL_LEN=$(( ISL + OSL + 256 ))
NUM_PROMPTS=$(( CONC * 10 ))

# Parameter grid to sweep
MNB_VALUES=(4096 8192 16384)
MNS_VALUES=(256 512)
GRID_SIZE=$(( ${#MNB_VALUES[@]} * ${#MNS_VALUES[@]} ))

RESULTS_BASE=/workspace/results_reconfigure_test
RESULTS_A="${RESULTS_BASE}/baseline"
RESULTS_B="${RESULTS_BASE}/reconfig"
mkdir -p "$RESULTS_A" "$RESULTS_B"

SERVER_LOG_DIR="${RESULTS_BASE}/logs"
mkdir -p "$SERVER_LOG_DIR"

start_server() {
    local mnb="$1" mns="$2" log="$3"

    vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.9 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$mnb" \
        --max-num-seqs "$mns" \
        --no-enable-prefix-caching \
        --disable-log-requests \
        ${VLLM_EXTRA_ARGS:-} \
        > "$log" 2>&1 &
    SERVER_PID=$!

    wait_for_server_ready \
        --port "$PORT" \
        --server-log "$log" \
        --server-pid "$SERVER_PID"
}

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
        sleep 2
    fi
}

run_bench() {
    local result_dir="$1" result_name="$2"

    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio 0.8 \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONC" \
        --result-filename "$result_name" \
        --result-dir "$result_dir" \
        --server-pid "$SERVER_PID"
}

trap kill_server EXIT

pip install -q datasets pandas 2>/dev/null || true

# ──────────────────────────────────────────────
# Phase A: Baseline — separate server per config
# ──────────────────────────────────────────────
if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
    echo ""
    echo "###############################################"
    echo "# Phase A: Baseline (${GRID_SIZE} cold starts)"
    echo "###############################################"

    A_START=$(date +%s)
    A_RUN=0

    for mnb in "${MNB_VALUES[@]}"; do
      for mns in "${MNS_VALUES[@]}"; do
        A_RUN=$((A_RUN + 1))
        echo ""
        echo "--- A.$A_RUN: max_num_batched_tokens=$mnb max_num_seqs=$mns ---"

        RUN_START=$(date +%s)
        start_server "$mnb" "$mns" "${SERVER_LOG_DIR}/server_a_${A_RUN}.log"
        READY_TIME=$(date +%s)
        echo "  Startup: $((READY_TIME - RUN_START))s"

        run_bench "$RESULTS_A" "baseline_mnb${mnb}_mns${mns}"

        kill_server
        RUN_END=$(date +%s)
        echo "  Total: $((RUN_END - RUN_START))s"
      done
    done

    A_END=$(date +%s)
    A_TOTAL=$((A_END - A_START))
    echo ""
    echo "Phase A total: ${A_TOTAL}s"
else
    A_TOTAL="(skipped)"
fi

# ──────────────────────────────────────────────
# Phase B: Reconfigure — single server, N cycles
# ──────────────────────────────────────────────
if [[ "${SKIP_RECONFIG:-0}" != "1" ]]; then
    echo ""
    echo "###############################################"
    echo "# Phase B: Reconfigure (1 cold start)"
    echo "###############################################"

    B_START=$(date +%s)

    # Start with the largest values so CUDA graphs and KV cache cover all configs
    INIT_MNB=${MNB_VALUES[-1]}
    INIT_MNS=${MNS_VALUES[-1]}

    echo ""
    echo "--- Starting server (mnb=$INIT_MNB mns=$INIT_MNS) ---"
    STARTUP_START=$(date +%s)
    start_server "$INIT_MNB" "$INIT_MNS" "${SERVER_LOG_DIR}/server_b.log"
    STARTUP_END=$(date +%s)
    echo "  Startup: $((STARTUP_END - STARTUP_START))s"

    B_RUN=0
    for mnb in "${MNB_VALUES[@]}"; do
      for mns in "${MNS_VALUES[@]}"; do
        B_RUN=$((B_RUN + 1))
        echo ""
        echo "--- B.$B_RUN: max_num_batched_tokens=$mnb max_num_seqs=$mns ---"

        RECONF_START=$(date +%s)

        export VLLM_DYNAMIC_RECONFIGURE=1
        export VLLM_MAX_NUM_BATCHED_TOKENS="$mnb"
        export VLLM_MAX_NUM_SEQS="$mns"
        reconfigure_vllm_scheduler "$PORT"

        RECONF_END=$(date +%s)
        echo "  Reconfigure: $((RECONF_END - RECONF_START))s"

        run_bench "$RESULTS_B" "reconfig_mnb${mnb}_mns${mns}"

        RUN_END=$(date +%s)
        echo "  Total: $((RUN_END - RECONF_START))s"
      done
    done

    kill_server

    B_END=$(date +%s)
    B_TOTAL=$((B_END - B_START))
    B_STARTUP=$((STARTUP_END - STARTUP_START))
    echo ""
    echo "Phase B total: ${B_TOTAL}s (startup: ${B_STARTUP}s)"
else
    B_TOTAL="(skipped)"
fi

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Comparison"
echo "=============================================="
echo " Phase A (baseline, ${GRID_SIZE} cold starts):  ${A_TOTAL}s"
echo " Phase B (reconfigure, 1 cold start):   ${B_TOTAL}s"
echo ""
echo " Results saved to: ${RESULTS_BASE}/"
echo "=============================================="
