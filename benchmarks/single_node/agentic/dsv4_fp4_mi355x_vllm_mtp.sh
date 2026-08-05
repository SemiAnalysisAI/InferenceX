#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using vLLM.
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
# Serving flags follow the validated MI355X recipe from
# https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro?hardware=mi355x
# https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_vllm.sh
# Image is configured in amd-master.yaml.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires one of these. 
#   KV_OFFLOAD_BACKEND=vllm-native.
#   KV_OFFLOAD_BACKEND=mooncake.
#   KV_OFFLOAD_BACKEND=lmcache.
#   KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

# Force the eval framework to lm-eval for this recipe. run_eval derives its
# default as swebench for agentic scenarios (scenario_default=swebench when
# IS_AGENTIC/SCENARIO_TYPE=agentic-coding), but EVAL_FRAMEWORK takes precedence
# over that default (benchmark_lib.sh: framework=${EVAL_FRAMEWORK:-...}), so
# setting it here makes the effective framework always lm-eval, never swebench.
export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

GPU_COUNT=$TP
if [[ ! "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPU_COUNT must be a positive integer, got '$GPU_COUNT'" >&2
    exit 1
fi
export GPU_COUNT

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
rocm-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Nightly ROCm image may be missing runtime deps; ensure they are present.
agentic_pip_install --quiet Pillow fastapi uvicorn

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# vllm-project/router expands the one HTTP backend into one logical worker per
# DP rank and sends X-data-parallel-rank on forwarded requests. aiperf's
# X-Correlation-ID is stable for every turn of a conversation; alias it to the
# router's preferred X-Session-ID header.
USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_VERSION=0.1.14
    VLLM_ROUTER_POLICY=consistent_hash
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet "vllm-router==$VLLM_ROUTER_VERSION"
fi

# DeepSeek-V4-Pro weights are large; engine startup can exceed default 600s.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# vllm-project/vllm#43447 keeps local SWA prefix-cache tails sparsely, while
# vllm-project/vllm#44774 applies the same reachability policy to Mooncake's
# store mask. 32k matches the trace-replay tuning validated for this workload.
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768

# VLLM_PREFIX_CACHE_RETENTION_INTERVAL only applies to sliding-window/Mamba
# models; this vLLM build raises ValueError if it is set for DSv4.

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
ROUTER_PID=""
MOONCAKE_MASTER_PID=""

OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  vllm-native)
    require_agentic_kv_offload_backend vllm-native
    # ---- vLLM native config ----------------------------------------------------------
    unset VLLM_USE_SIMPLE_KV_OFFLOAD
    # MI355X nodes have ~2.7 TiB of host DRAM available for offload;
    # reserve 2.5 TB for the offload pool (leaves ~200 GB headroom for
    # worker RSS / page cache / slurm cgroup).
    TOTAL_CPU_DRAM_PARTITION_GB="$TOTAL_CPU_DRAM_GB"
    # Use vLLM's regular native KV-offload path (OffloadingConnector),
    # NOT the SimpleCPUOffloadConnector. The "vllm-native" backend resolves to
    # OffloadingConnector by default; setting VLLM_USE_SIMPLE_KV_OFFLOAD=1
    # would switch it to SimpleCPUOffloadConnector. We intentionally leave
    # that env var UNSET here so the regular OffloadingConnector path is
    # used. The shortcut --kv_offloading_backend native + --kv_offloading_size
    # form constructs the KVTransferConfig at engine startup
    # (vllm/config/vllm.py:662).

    # Remove --disable-hybrid-kv-cache-manager and enable hybrid kv cache manager (default)
    # This gives extra cache hit than disabling hybrid kv cache manager
    OFFLOAD_ARGS=(
        --kv_offloading_backend native
        --kv_offloading_size "$TOTAL_CPU_DRAM_PARTITION_GB"
    )

    ;;
  mooncake)
    require_agentic_kv_offload_backend mooncake
    # ---- Mooncake config ----------------------------------------------------------
        # Embedded mode contributes one segment per GPU rank to a shared
        # distributed store, so pre-divide the aggregate host-memory budget.
        PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / GPU_COUNT))

        #MOONCAKE_VERSION=0.3.11.post1
        #apt-get update && apt-get install -y libcurl4 libibverbs1 rdma-core librdmacm1 libnuma1 liburing2
        #agentic_pip_install --quiet --no-cache-dir --no-deps \
        #    --force-reinstall "mooncake-transfer-engine-non-cuda==$MOONCAKE_VERSION"

        git clone https://github.com/kvcache-ai/Mooncake.git
        cd Mooncake
        git checkout v0.3.12
        bash dependencies.sh
        mkdir build
        cd build
        cmake ..
        make -j
        sudo make install # optional, make it ready to be used by vLLM/SGLang
        cd ..
        cd ..

        python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null

        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "2GB",
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": false
}
EOF
# (srok)
  #"protocol": "rdma",
  #"device_name": "mlx5_0",
  #"local_buffer_size": "4GB",
        export MOONCAKE_CONFIG_PATH
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1
        export PYTHONHASHSEED=0
        export MC_SLICE_SIZE=1048576
        # (srok)
        #export MC_WORKERS_PER_CTX=4
        export MC_WORKERS_PER_CTX=8

        MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO=0.80
        MOONCAKE_EVICTION_RATIO=0.10
        MOONCAKE_KV_LEASE_TTL=60s
        #MOONCAKE_KV_LEASE_TTL=3600s

        echo "Starting Mooncake master on port $MOONCAKE_MASTER_PORT..."
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio="$MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO" \
            --eviction_ratio="$MOONCAKE_EVICTION_RATIO" \
            --default_kv_lease_ttl="$MOONCAKE_KV_LEASE_TTL" \
            > "$MOONCAKE_MASTER_LOG" 2>&1 &

        sleep 10
        MOONCAKE_MASTER_PID=$!
        if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died during startup." >&2
            cat "$MOONCAKE_MASTER_LOG" >&2
            exit 1
        fi
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        )

    ;;
  lmcache)
    require_agentic_kv_offload_backend lmcache
    # ---- Lmcache config ----------------------------------------------------------
    LMCACHE_PID=""

    cleanup_lmcache_server() {
        if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
            kill "$LMCACHE_PID" 2>/dev/null || true
            wait "$LMCACHE_PID" 2>/dev/null || true
        fi
    }

    trap cleanup_lmcache_server EXIT

    cleanup_agentic_services() {
        local exit_code=$?
        trap - EXIT INT TERM
        set +e
        stop_background_process_tree "$ROUTER_PID" "vLLM router"
        stop_background_process_tree "$SERVER_PID" "vLLM server" 60
        stop_background_process_tree "$MOONCAKE_MASTER_PID" "Mooncake master"
        exit "$exit_code"
    }
    trap cleanup_agentic_services EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    wait_for_lmcache_ready() {
        { set +x; } 2>/dev/null
        local attempts="${LMCACHE_READY_ATTEMPTS:-120}"
        local tail_pid=""

        while [ ! -f "$LMCACHE_LOG" ]; do
            if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
                echo "LMCache server died before creating log file. Exiting." >&2
                exit 1
            fi
            sleep 10
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
        { set +x; } 2>/dev/null
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        git clone https://github.com/LMCache/LMCache.git
        cd LMCache
        git checkout v0.5.2
        pip install -r requirements/build.txt
        CXX=hipcc BUILD_WITH_HIP=1 pip install -e .   --no-build-isolation
        cd ..

        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

        TOTAL_CPU_DRAM_PARTITION_GB="$TOTAL_CPU_DRAM_GB"
        # Match the B200 Kimi LMCache setup: keep a 2.5 TB semantic CPU KV
        # pool, but let the external MP server own that pool so vLLM does not
        # split --kv-offloading-size across TP ranks through the integrated
        # LMCache backend.
        LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
        LMCACHE_PORT="${LMCACHE_PORT:-5555}"
        LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
        # LMCacheMPConnector concatenates lmcache.mp.host and port into the
        # ZMQ endpoint. Bind the server to a raw host, but pass the connector a
        # ZMQ-style host string.
        LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
        LMCACHE_L1_SIZE_GB="${TOTAL_CPU_DRAM_PARTITION_GB}"
        if [ "$LMCACHE_L1_SIZE_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
            echo "Error: LMCACHE_L1_SIZE_GB=$LMCACHE_L1_SIZE_GB exceeds configured capacity $TOTAL_CPU_DRAM_GB" >&2
            exit 1
        fi
        LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
        # LMCache read locks are leases on chunks that lookup has promised
        # vLLM can retrieve. The default 300s TTL is too short for this
        # long-context agentic queue: TP8/conc32 can spend >300s between
        # lookup and retrieve while GPU KV is saturated, which leaves the
        # object present in L1 but no longer readable. Keep the 2.5 TB pool
        # size unchanged and only extend the lookup-to-retrieve lease.
        LMCACHE_L1_READ_TTL_SECONDS="${LMCACHE_L1_READ_TTL_SECONDS:-7200}"
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
        LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$TP}"
        export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
        export LMCACHE_BLOCKING_TIMEOUT_SECS=1200
        LMCACHE_TX_MODE="lmcache_driven"

        echo "Starting LMCache MP server..."
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --l1-read-ttl-seconds "$LMCACHE_L1_READ_TTL_SECONDS"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --max-workers "$LMCACHE_MAX_WORKERS"
            --eviction-policy LRU
            --supported-transfer-mode "$LMCACHE_TX_MODE"
        )
        printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
        printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
        "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
        LMCACHE_PID=$!
        echo "LMCache server PID: $LMCACHE_PID"
        wait_for_lmcache_ready

        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
        )
    ;;
  *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND '${KV_OFFLOAD_BACKEND:-}' (expected: vllm-native, mooncake, lmcache)" >&2
    exit 1
    ;;
esac
fi

# ---- LLM server config ----------------------------------------------------------
PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi
if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

# DEP8 (TP8 + DP-attention) is a high-concurrency arm tuned separately from the
# smaller DEP arm: larger prefill token budget and long-prefill chunking so
# decode latency stays bounded at high concurrency.
IS_DEP8=false
if [ "$DP_ATTENTION" = "true" ] && [ "$TP" -eq 8 ]; then
    IS_DEP8=true
fi

MODE_ARGS=()
MNBT=8192
if [ "$EP_SIZE" -gt 1 ]; then
    MODE_ARGS+=(--enable-ep-weight-filter)
fi
if [ "$DP_ATTENTION" = "true" ]; then
    MODE_ARGS+=(--prefill-schedule-interval 8)
    if [ "$IS_DEP8" = "true" ]; then
        MODE_ARGS+=(
            --max-num-batched-tokens $((MNBT * 2))
            --long-prefill-token-threshold 16384
        )
    else
        MODE_ARGS+=(--max-num-batched-tokens $MNBT)
    fi
fi

# --max-num-seqs is applied PER scheduler. Under DP-attention each of the $TP
# DP ranks runs its own scheduler and the router spreads sessions across them,
# but size the per-rank cap as CONC directly (not CONC/TP) so each rank can
# hold the full session count. In pure-TP there is a single scheduler across
# all GPUs that sees all CONC sessions, so use 2*CONC directly.
if [ "$DP_ATTENTION" = "true" ]; then
    MAX_NUM_SEQS=$((CONC))
    if [ "$MAX_NUM_SEQS" -lt 1 ]; then
        MAX_NUM_SEQS=1
    fi
else
    MAX_NUM_SEQS=$((2 * CONC))
fi
CONTEXT_LEN=1048576

# MTP: cudagraph capture sizes are in TOKENS. With num_speculative_tokens=N,
# every uniform decode batch of S seqs verifies S*(1+N) tokens, so capture the
# explicit multiples (1+N), 2*(1+N), ..., MAX_NUM_SEQS*(1+N) -- one graph per
# decode batch of 1..MAX_NUM_SEQS seqs. vLLM rounds configured sizes up to
# multiples of (1+N) and dedups (adjust_cudagraph_sizes_for_spec_decode), so a
# plain 1..MAX_NUM_SEQS list would collapse to coverage of only
# MAX_NUM_SEQS/(1+N) seqs and drop the largest decode batches to eager.
NUM_SPEC_TOKENS=3
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))
# Throughput pins synthetic MTP acceptance to the dsv4-pro golden AL (thinking_on,
# num_speculative_tokens=3, golden_al_distribution/dsv4_mtp.yaml). The EVAL_ONLY
# accuracy run uses real target verification instead -- synthetic acceptance
# bypasses verification and corrupts the SWE-bench eval (0.0000 score).
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}"
else
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": 2.49}"
fi

# These MI355X nodes have a stable ~32 GiB/GPU carveout: only ~256/288 GiB is
# free at init, independent of the KV-offload backend (observed identically on
# both the lmcache and GPU-resident kv-none arms). At
# --gpu-memory-utilization 0.95 vLLM requests 273.6 GiB and every DP worker
# hard-fails ("Free memory ... less than desired GPU memory utilization").
# But 0.85 (244.8 GiB) leaves only ~23.9 GiB for KV, which is below the
# 24.06 GiB one request at max_model_len=1M needs once the MTP draft layer's
# extra per-token KV is counted -- engine init then dies with
# "available KV cache memory ... larger than ..." on the tighter (eval-only)
# relaunch. 0.86 (247.7 GiB) adds ~2.6 GiB, nearly all to KV (~26.5 GiB), so
# the KV check clears with margin while still keeping ~8 GiB free-mem headroom
# below the ~256 GiB hard-fail ceiling. Additional 0.9 for mooncake headroom
GPU_MEM_UTIL=0.90

# Long-context forward passes (~370K tokens with fp8 KV + DRAM offload) can exceed
# vLLM's default 300s worker RPC timeout, killing the engine with
# "RPC call to sample_tokens timed out". Widen it.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

echo "Starting vllm server..."
set -x
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# MoE backend. All arms use the AITER fused MoE this recipe was validated
# against.
#
# The DEP arms (DP-attention + EP>1) were tried on AITER's FlyDSL mega-MoE
# (--moe-backend flydsl), which fuses the expert-parallel dispatch, both expert
# GEMMs, and the combine into one pipeline (ROCm/FlyDSL#626). That does not work
# for this model: DSv4-Pro resolves expert_dtype to 'fp4', so its experts are
# built by Mxfp4MoEMethod, and vllm/model_executor/layers/fused_moe/oracle/
# mxfp4.py:map_mxfp4_backend() hard-rejects anything outside its MXFP4 allowlist:
#
#   ValueError: moe_backend='flydsl' is not supported for MXFP4 MoE.
#   Expected one of ['deep_gemm', 'flashinfer_trtllm', 'flashinfer_trtllm_afp8',
#   'flashinfer_cutlass', 'flashinfer_cutlass_afp8', 'triton', 'triton_unfused',
#   'humming', 'marlin', 'aiter', 'aiter_mxfp4_fp8', 'aiter_mxfp4_mxfp4', 'xpu',
#   'cpu', 'emulation']
#
# This is a hard failure at model load, not a fallback -- every worker dies in
# FusedMoE.__init__ before any KV cache is allocated. The rejection is on the
# quantization path, so it is independent of EP size and of whether AITER's
# flydsl_gfx950_mi355x_IntraNode_ep8.json tuning table covers the shape (it does:
# hidden_dim=7168 / topk=6 / local_expert_num=48 matches DSv4-Pro at EP8). The
# tuning table is simply never consulted, because FlyDSL is not reachable for an
# MXFP4-quantized MoE in this vLLM build.
#
# So the DEP8 arm runs on --moe-backend aiter, which is on the allowlist. That
# still answers the question this arm exists for: whether sharding KV per DP rank
# and the 384 routed experts across EP8 frees enough HBM to raise the prefix-cache
# hit rate that dominates AgentX. Revisit flydsl once vLLM wires it into the
# MXFP4 backend oracle. Override with MOE_BACKEND to retest without editing this.
MOE_BACKEND="${MOE_BACKEND:-aiter}"

sleep 180

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --trust-remote-code
    --async-scheduling
    --distributed-executor-backend mp
    --kv-cache-dtype fp8
    --block-size 256
    --max-model-len "$CONTEXT_LEN"
    "${PARALLEL_ARGS[@]}"
    "${MODE_ARGS[@]}"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --moe-backend "$MOE_BACKEND"
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --speculative-config "$SPEC_CONFIG"
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    --enable-auto-tool-choice
    --enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs "$MAX_NUM_SEQS"
    "${OFFLOAD_ARGS[@]}"
)

# (srok), not yet
    #--attention_config.use_fp4_indexer_cache=True
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_VLLM_ROUTER" = "true" ]; then
    echo "Starting native vLLM router on port $PORT for $TP DP ranks..."
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy "$VLLM_ROUTER_POLICY" \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$VLLM_ROUTER_METRICS_PORT" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
