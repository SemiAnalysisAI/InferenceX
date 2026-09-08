#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using ATOM with DSpark speculative decoding.
#
# Companion to kimik3_fp4_mi355x_mtp.sh, which runs the same checkpoint under
# vLLM, so the two arms are directly comparable.
#
# TP=8 ONLY, for the same reason as the vLLM arm: the MXFP4 checkpoint is
# 1.561 TB decimal (~195 GB/GPU across 8 GPUs of the 288 GB part), and TP=4
# would need ~390 GB/GPU and cannot load.
#
# The ATOM image is purpose-built for K3, so apply_k3_container_patches.sh is
# NOT sourced here -- that script reproduces a specific patched vLLM container
# byte-for-byte and does not apply to this stack.
#
# Three serving bands, one fresh server per concurrency point:
#   interactive (1, 2, 4)         DCP1, DSpark 7, no LMCache
#   mid         (8, 12, 14, 16)   DCP8, DSpark 3, LMCache 128 GB/rank
#   throughput  (32, 40, 56, 64)  DCP8, no draft model, LMCache 128 or 192 GB/rank
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, DCP_SIZE, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

DRAFT_MODEL="Inferact/Kimi-K3-DSpark"

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP DCP_SIZE=${DCP_SIZE:-1} CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

if [[ -v SLURM_JOB_ID ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# The 0907 image was built with podman on a host behind a local squid, and the
# build environment leaked into the image config: HTTP_PROXY, HTTPS_PROXY,
# http_proxy, and https_proxy are all baked in as http://127.0.0.1:3128, with
# NO_PROXY covering only localhost. Nothing listens on that port inside the
# container on a benchmark node, so every outbound request -- the uv bootstrap
# and aiperf install in install_agentic_deps, the trace-corpus fetch, and both
# hf downloads below -- would go to a dead proxy before the server is ever
# started. The 0821 and 0903 images carried none of these. Clear them, and keep
# the loopback exemption so the AIPerf client and the /metrics scrape are
# unaffected either way.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="$NO_PROXY"

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 is a 1.56 TB checkpoint and only fits at TP=8 on" >&2
    echo "       288 GB gfx950 parts (~195 GB/GPU). Got TP=$TP." >&2
    exit 1
fi

if [[ -v ROCR_VISIBLE_DEVICES ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

wait_for_amd_gpu_clean

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

# Require the ATOM Prometheus stream in every official result. AIPerf
# deduplicates this endpoint against its automatic localhost discovery.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound. Matches the vLLM K3 arm.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# VRAM space check
wait_for_amd_gpu_clean

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "ATOM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- Per-concurrency knobs --------------------------------------------------
# Three bands, and every knob below moves with the band:
#
#   interactive (1, 2, 4)        the latency floor. Everything GPU-resident, no
#                                decode context parallelism, the deepest draft
#                                the golden curve publishes, 8192-token prefill
#                                step.
#   mid         (8, 12, 14, 16)  decode is KV-bandwidth-bound over 100k+ token
#                                agentic contexts, so DCP8 shards the KV read
#                                across all 8 GPUs, the LMCache DRAM tier backs
#                                the paged KV, and ReplaySSM rebuilds the KDA
#                                recurrent state from the checkpoint ring.
#   throughput  (32, 40, 56, 64) past the throughput knee the draft forward no
#                                longer pays for itself against the resident
#                                batch, so no draft model is loaded and
#                                ReplaySSM goes back off.
#
# DCP itself is not chosen here: it comes from dcp-size in
# configs/amd-master.yaml, which leaves it at 1 for concurrency 1-4 and sets 8
# from concurrency 8 up. Nor is the LMCache tier, which follows kv-offloading in
# the same file.
#
# CUDAGRAPH_MAX_NUM_SEQS is the in-flight window the CUDA graphs are captured
# for and defaults to 2 * CONC, which is what the AgentX client can put in
# flight per lane. Only concurrency 14 pins it, because it runs the concurrency
# 16 server verbatim.
AITER_REUSE_IDENTICAL_COMM_GROUPS=0
CUDAGRAPH_MAX_NUM_SEQS=""
case "$CONC" in
    1|2|4)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=7
        SPEC_DECODE_AL=3.84
        ;;
    8)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=1
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=3.00
        ;;
    12)
        MAX_NUM_SEQS=24
        MAX_NUM_BATCHED_TOKENS=4096
        GPU_MEM_UTIL=0.88
        ATOM_ENABLE_REPLAYSSM=1
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=3.00
        ;;
    14)
        # The concurrency 16 server verbatim, including the pinned CUDA-graph
        # width; only the client concurrency is 14. Deriving the width from
        # 2 * CONC here gives 28 and the server dies during graph warmup.
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=1
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=3.00
        CUDAGRAPH_MAX_NUM_SEQS=32
        ;;
    16)
        MAX_NUM_SEQS=32
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=1
        NUM_SPEC_TOKENS=3
        SPEC_DECODE_AL=3.00
        ;;
    32)
        MAX_NUM_SEQS=64
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        ;;
    40)
        MAX_NUM_SEQS=80
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        ;;
    # The two widest points are the only ones that reuse identical AITER
    # communicator groups, and the only ones given the 192 GB/rank LMCache
    # budget rather than 128.
    56)
        MAX_NUM_SEQS=112
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        AITER_REUSE_IDENTICAL_COMM_GROUPS=1
        ;;
    64)
        MAX_NUM_SEQS=128
        MAX_NUM_BATCHED_TOKENS=8192
        GPU_MEM_UTIL=0.86
        ATOM_ENABLE_REPLAYSSM=0
        NUM_SPEC_TOKENS=0
        SPEC_DECODE_AL=0
        AITER_REUSE_IDENTICAL_COMM_GROUPS=1
        ;;
    *)
        echo "Unsupported CONC=$CONC" >&2
        exit 2
        ;;
esac
export ATOM_ENABLE_REPLAYSSM
export AITER_REUSE_IDENTICAL_COMM_GROUPS

# Full CUDA graphs over the dense range [2 .. window * (1 + draft tokens)]. The
# verify step of a DSpark round submits one row per draft token on top of the
# accepted token, so a spec point needs (1 + draft) times the batch widths a
# non-spec point at the same window does; capturing only up to the window would
# send every speculative decode down the eager path.
CUDAGRAPH_MAX_NUM_SEQS="${CUDAGRAPH_MAX_NUM_SEQS:-$((2 * CONC))}"
GRAPH_MAX=$((CUDAGRAPH_MAX_NUM_SEQS * (1 + NUM_SPEC_TOKENS)))
CUDAGRAPH_CAPTURE_SIZES="[$(seq -s, 2 "$GRAPH_MAX")]"
echo "CUDAGRAPH_MAX_NUM_SEQS=$CUDAGRAPH_MAX_NUM_SEQS GRAPH_MAX=$GRAPH_MAX"

# ---- KV offload -------------------------------------------------------------
# The paged KV rides this LMCache CPU tier from concurrency 8 up. K3 is a
# hybrid, so Kimi Delta Attention also carries a per-request recurrent state
# alongside the paged KV, but that state is now rebuilt in place by ReplaySSM
# from the in-GPU checkpoint ring rather than parked in a second CPU tier, so
# the whole per-rank budget goes to the paged KV.
OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        # TOTAL_CPU_DRAM_GB is the AGGREGATE budget from the matrix generator.
        # LMCACHE_MAX_LOCAL_CPU_SIZE is per rank and every rank allocates its
        # own, so the aggregate is divided by TP as the agentic README requires.
        # Handing a rank the whole aggregate does not just overcommit -- it
        # never finishes pinning and hangs the launch partway through. The two
        # dram-utilization blocks in configs/amd-master.yaml land this on the
        # 128 GB/rank the mid band and concurrency 32/40 were measured with, and
        # the 192 GB/rank concurrency 56 and 64 were measured with.
        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=True
        export LMCACHE_MAX_LOCAL_CPU_SIZE="$((TOTAL_CPU_DRAM_GB / TP))"
        # DCP-locked: the offload hash block is block-size(128) x dcp(8) = 1024,
        # so the KV grid and the state-checkpoint grid coincide and the joint
        # load aims both legs at one boundary. 512 or 2048 misaligns it.
        export LMCACHE_CHUNK_SIZE=1024
        # Kept from the previous submission even though the ATOM recipe no
        # longer lists it. K3 is a hybrid, and if this gate is still off by
        # default the paged KV never reaches the CPU tier at all -- which reads
        # as a tier that is on and idle, not as an error. Setting it is a no-op
        # if the default flipped or the knob went away.
        export OFFLOAD_KV_FOR_HYBRID=1
        # Pin the eight ranks to the two sockets explicitly rather than letting
        # ATOM place them: auto-binding is off precisely so this mapping is what
        # takes effect, and a pinned pool on the wrong socket pays a cross-socket
        # hop on every offload read.
        export LMCACHE_NUMA_MODE=auto
        export ATOM_NUMA_BIND=1
        export ATOM_NUMA_NODE=0,0,0,0,1,1,1,1
        export ATOM_AUTO_NUMA_BIND=0
        # Statistics only -- per-step offload counters in the connector. Kept on
        # because the submitted numbers were measured with it on.
        export OFFLOAD_PROFILE=1
        # The GPU staging buffer defaults to 2 chunks (8 MiB) and one K3 state
        # entry is 54.78 MiB; a buffer too small to hold one entry makes the
        # transfer path decline to build -- one log line, then nothing moves,
        # which reads exactly like a tier that is on and idle.
        export OFFLOAD_GPU_STAGING_CHUNKS=32

        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}"
        )
        ;;
    *)
        echo "Unsupported KV_OFFLOAD_BACKEND: $KV_OFFLOAD_BACKEND (expected empty or lmcache)" >&2
        exit 1
        ;;
esac

# ---- ATOM env ---------------------------------------------------------------
echo "Starting atom server..."
export PYTHONNOUSERSITE=1

# Required by ATOM: without it the aiter kernel logs flood the server log for
# the whole 3600 s replay.
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export AITER_SITUV2_A4W4=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_FLYDSL_STAGE2_FP8=1
# Anchor-only state checkpointing: the demand rung is 47% of checkpoint writes
# but reads back 2.8% of the time, against 85.2% for a prompt-end anchor, so it
# costs more in evictions than its reuse is worth on these traces.
export ATOM_STATE_CHECKPOINT_DEMAND=0

# ---- Speculative ------------------------------------------------------------
# https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml
#  7 draft tokens -> AL 3.84
#  3 draft tokens -> AL 3.00
# Concurrency 32 and up serve without a draft model: past the throughput knee
# the draft forward no longer pays for itself against the resident batch.
SPEC_ARGS=()
if [ "$NUM_SPEC_TOKENS" -gt 0 ]; then
    # Stage the draft into the shared HF cache before the server starts. The
    # ATOM image carries no cache entry for it, and an uncached repo id makes
    # every rank pull the same 7 GB checkpoint at once. The failure is silent:
    # the log stops after "Loading drafter model...", no error is printed, every
    # GPU sits at 0% with the target weights already resident, and the pull runs
    # at whatever the node manages -- measured at ~0.7 MB/s on one cluster,
    # about three hours. A cached drafter loads in under a second. Concurrent
    # downloads against the shared cache can hit transient stale handles.
    for attempt in 1 2 3 4 5; do
        hf download "$DRAFT_MODEL" && break
        if [ "$attempt" = 5 ]; then
            echo "hf download of $DRAFT_MODEL failed after $attempt attempts" >&2
            exit 1
        fi
        echo "hf download attempt $attempt failed; retrying in 60s" >&2
        sleep 60
    done

    SPEC_ARGS=(
        --method dspark
        --draft-model "$DRAFT_MODEL"
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
    )
    if [ "${EVAL_ONLY}" != "true" ]; then
        SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
    fi
fi
echo "SPEC_DECODE_AL=$SPEC_DECODE_AL NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS"

# ---- LLM server -------------------------------------------------------------
ATOM_CMD=(
    python -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    # The AgentX client asks for $MODEL on the wire (benchmark_lib.sh passes
    # --model ${SERVED_MODEL_NAME:-$MODEL}), so register the server under that
    # name rather than whatever MODEL_PATH resolves to. A mismatch 404s at
    # warmup.
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --decode-context-parallel-size "${DCP_SIZE:-1}"
    --kv_cache_dtype fp8
    --block-size 128
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enable_prefix_caching
    # -1 is ladder off, checkpointing on: the prompt-end anchor still places a
    # checkpoint, the fixed-interval grid does not. ATOM defaults to 8192, which
    # would place a rung every 8192 tokens and change STATE pool occupancy and
    # prefix reuse in the linear-attention layers -- most visibly on this trace,
    # where prompts run to several hundred thousand tokens.
    --state-checkpoint-interval-tokens -1
    --level 3
    --cudagraph-mode FULL
    --cudagraph-capture-sizes "$CUDAGRAPH_CAPTURE_SIZES"
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
