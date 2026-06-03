#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on B200 using vLLM.
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
#
# OFFLOADING values:
#   none        - vLLM GPU KV only, with DSv4 hybrid KV manager enabled.
#   cpu         - vLLM native OffloadingConnector, with hybrid KV manager enabled.
#   lmcache-mp  - Temporarily disabled for DSv4. LMCache PR #3261 must merge
#                 first so LMCacheMPConnector can support HMA block-id tuples.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

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
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
HYBRID_KV_ARGS=(--no-disable-hybrid-kv-cache-manager)
LMCACHE_PID=""

cleanup_lmcache_server() {
    if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
        kill "$LMCACHE_PID" 2>/dev/null || true
        wait "$LMCACHE_PID" 2>/dev/null || true
    fi
}

trap cleanup_lmcache_server EXIT

wait_for_lmcache_ready() {
    { set +x; } 2>/dev/null
    local attempts="${LMCACHE_READY_ATTEMPTS:-120}"
    local tail_pid=""

    while [ ! -f "$LMCACHE_LOG" ]; do
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before creating log file. Exiting." >&2
            exit 1
        fi
        sleep 1
    done

    tail -f -n +1 "$LMCACHE_LOG" &
    tail_pid=$!

    for ((i = 1; i <= attempts; i++)); do
        if curl --output /dev/null --silent --fail "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck"; then
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            return 0
        fi
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before becoming healthy. Log follows:" >&2
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done

    echo "Timed out waiting for LMCache server healthcheck. Log follows:" >&2
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
    cat "$LMCACHE_LOG" >&2 || true
    exit 1
}

case "$OFFLOADING" in
    none) ;;
    cpu)
        # b200-dgxc compute nodes have ~3.8 TiB host RAM; SLURM cgroup limits
        # individual jobs to a fraction of that. Aim for ~1.2 TB total native
        # CPU offload pool across the engine(s); previously 2.8 TB but every
        # DP-attn worker stalled for 4+ min during pinned-CPU-tensor allocation
        # and the shm_broadcast watchdog killed them (run 26246044726). 150 GB
        # per worker (1.2 TB / 8) completes the alloc within the 60 s window.
        #
        # Native --kv-offloading-size becomes OffloadingConnector's
        # cpu_bytes_to_use. For DP-attn there are $TP independent DP engines,
        # so pre-divide to keep aggregate host commit near TOTAL_CPU_DRAM_GB.
        # For pure TP, vLLM treats the size as the total across TP ranks.
        TOTAL_CPU_DRAM_GB=1200
        if [ "$DP_ATTENTION" = "true" ]; then
            PER_ENGINE_GB=$((TOTAL_CPU_DRAM_GB / TP))
        else
            PER_ENGINE_GB=$TOTAL_CPU_DRAM_GB
        fi
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        OFFLOAD_ARGS=(
            --kv-offloading-backend native
            --kv-offloading-size "$PER_ENGINE_GB"
        )
        ;;
    lmcache-mp)
        { set +x; } 2>/dev/null
        # LMCacheMPConnector needs HMA support before it can run DSv4 with the
        # hybrid KV manager. Re-enable this path after
        # https://github.com/LMCache/LMCache/pull/3261 is merged.
        echo "Error: OFFLOADING=lmcache-mp is disabled for DSv4 until LMCache PR #3261 adds HMA support." >&2
        exit 1

        # LMCache docs recommend MP mode for production: start an external
        # `lmcache server`, then point vLLM's LMCacheMPConnector at it. For
        # vLLM >= 0.20, prefer the LMCache-shipped connector module because it
        # tracks the latest server protocol ahead of vLLM's vendored copy.
        #
        # Important DSv4 caveat: LMCacheMPConnector currently only accepts the
        # non-hybrid KV block layout. The connector raises if vLLM returns the
        # hybrid block-id tuple used by the CSA/HCA hybrid KV manager. This
        # mode therefore disables the hybrid manager; `none` and `cpu` keep it
        # enabled for the normal B200 DSv4 path.
        agentic_pip_install --quiet --no-cache-dir lmcache
        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

        TOTAL_CPU_DRAM_GB=2800
        LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
        LMCACHE_PORT="${LMCACHE_PORT:-5555}"
        LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
        LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-200}"
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$TP}"

        echo "Starting LMCache MP server..."
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --max-workers "$LMCACHE_MAX_WORKERS"
            --eviction-policy LRU
        )
        printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
        printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
        "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
        LMCACHE_PID=$!
        echo "LMCache server PID: $LMCACHE_PID"
        wait_for_lmcache_ready

        HYBRID_KV_ARGS=(--disable-hybrid-kv-cache-manager)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
        )
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu, lmcache-mp)" >&2
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

# --max-num-seqs is per-engine. With DP-attn each DP engine handles only
# CONC/$TP sequences in steady state (the trace replay tool's CONC users
# load-balance across DP ranks), so size the per-engine cap to that.
# Pure TP is a single engine and sees all CONC sequences itself.
if [ "$DP_ATTENTION" = "true" ]; then
    PER_ENGINE_MAX_NUM_SEQS=$(( CONC / TP ))
    [ "$PER_ENGINE_MAX_NUM_SEQS" -lt 1 ] && PER_ENGINE_MAX_NUM_SEQS=1
else
    PER_ENGINE_MAX_NUM_SEQS=$CONC
fi

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLOAT32_MATMUL_PRECISION=high

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --kv-cache-dtype fp8
    --block-size 256
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
    --attention_config.use_fp4_indexer_cache=True
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --enable-prefix-caching
    "${HYBRID_KV_ARGS[@]}"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$PER_ENGINE_MAX_NUM_SEQS"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
