#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.5-397B-A17B MXFP4 on MI355X with SGLang
# native EAGLE MTP. Throughput uses the committed golden synthetic
# acceptance length; evaluation retains real target-model verification.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

rocm-smi || true
amd-smi || true

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Resident arms keep the page size the TP2/EP2 and TP4 sweeps were measured
# with. HiCache arms move to 64 because Qwen3.5's hybrid attention/Mamba host
# pools transfer page-first, and page size 1 fails the EAGLE verify-graph
# compile on gfx950.
CACHE_ARGS=()
PAGE_SIZE=16
if require_agentic_kv_offload_backend hicache; then
    PAGE_SIZE=64

    # sgl-project/sglang#30393 (merged 2026-08-06) routes an MTP draft KV cache
    # to either a packed or a sidecar HiCache pool. Qwen3.5 conditional-
    # generation checkpoints keep their language-model attributes in the nested
    # text_config, and SGLang normalizes the draft depth only on the parent HF
    # config, so ModelConfig.num_nextn_predict_layers stays None, the draft is
    # misrouted to the sidecar path, and the scheduler dies during startup with
    #   AttributeError: 'HybridLinearKVPool' object has no attribute 'layer_num'
    # Apply the one-line fix from sgl-project/sglang#34560 until it ships in an
    # MI355X image. Resident runs never reach this routing, so the patch stays
    # scoped to the HiCache arms and leaves the measured resident data alone.
    python3 - /sgl-workspace/sglang/python/sglang/srt/configs/model_config.py <<'PYPATCH'
import sys

path = sys.argv[1]
anchor = 'self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"'
assign = "self.hf_config.num_nextn_predict_layers = 1"
fix = "self.hf_text_config.num_nextn_predict_layers = 1"

with open(path) as fh:
    lines = fh.readlines()

matches = [i for i, line in enumerate(lines) if line.strip() == anchor]
if len(matches) != 1:
    sys.exit(f"sglang#34560: expected 1 Qwen3.5 MTP anchor in {path}, found {len(matches)}")

i = matches[0]
if lines[i + 1].strip() != assign:
    sys.exit(f"sglang#34560: unexpected line after anchor in {path}: {lines[i + 1]!r}")
if lines[i + 2].strip() == fix:
    print("sglang#34560 already applied")
    sys.exit(0)

indent = lines[i + 1][: len(lines[i + 1]) - len(lines[i + 1].lstrip())]
lines.insert(i + 2, f"{indent}{fix}\n")
with open(path, "w") as fh:
    fh.writelines(lines)
print("sglang#34560 applied")
PYPATCH

    # --hicache-size is the per-rank budget SGLang splits across Qwen3.5's two
    # hybrid host pools, not a per-pool figure: on this image at TP4 with
    # --hicache-size 144 the ranks allocated 93.37 GB target KV + 50.65 GB Mamba
    # = 144.02 GB each. Packed NEXTN then adds its single draft layer on top of
    # the 60 transferred target layers. Hold the node to 80% of the workflow
    # DRAM budget so the draft layer, page alignment, and the trace-replay
    # client cannot walk the host into the OOM killer mid-storm.
    HICACHE_ALIGNMENT_RESERVE_GB=$TP
    HICACHE_USABLE_TOTAL_GB=$((TOTAL_CPU_DRAM_GB - HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$HICACHE_USABLE_TOTAL_GB" -lt 1 ]; then
        echo "Error: insufficient DRAM after HiCache alignment reserve." >&2
        exit 1
    fi
    MAX_HICACHE_SIZE_GB=$((HICACHE_USABLE_TOTAL_GB * 80 / 100 * 60 / 61 / TP))
    # 144 GB/rank is the largest pool observed to allocate on this image; 180
    # is a bounded step up from it. Raise once a run confirms the larger pinned
    # allocation stays inside the watchdog.
    HICACHE_MAX_SIZE_GB_PER_RANK=${HICACHE_MAX_SIZE_GB_PER_RANK:-180}
    if [ "$MAX_HICACHE_SIZE_GB" -gt "$HICACHE_MAX_SIZE_GB_PER_RANK" ]; then
        MAX_HICACHE_SIZE_GB="$HICACHE_MAX_SIZE_GB_PER_RANK"
    fi
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -lt 1 ] || [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB outside 1..$MAX_HICACHE_SIZE_GB." >&2
        exit 1
    fi
    PROJECTED_HICACHE_TOTAL_GB=$(((HICACHE_SIZE_GB * TP * 61 + 59) / 60 + HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$PROJECTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: projected HiCache use ${PROJECTED_HICACHE_TOTAL_GB} GB exceeds configured ${TOTAL_CPU_DRAM_GB} GB." >&2
        exit 1
    fi
    echo "HiCache pools: ${HICACHE_SIZE_GB} GB per rank across TP=${TP}; projected node total ${PROJECTED_HICACHE_TOTAL_GB} GB of ${TOTAL_CPU_DRAM_GB} GB."

    # kernel + page_first is the transfer path the hybrid KV/Mamba stack already
    # builds on gfx950 (both host pools and the pool-stack attach complete under
    # it). write_through_selective matches the B300 Qwen3.5 MTP sibling.
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
        --hicache-write-policy write_through_selective
    )
fi

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend aiter
    --mem-fraction-static 0.80
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --page-size "$PAGE_SIZE"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 32768
    --chunked-prefill-size 32768
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
