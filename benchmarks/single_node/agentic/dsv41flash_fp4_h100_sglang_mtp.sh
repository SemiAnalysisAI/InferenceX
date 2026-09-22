#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on H100 with SGLang DSpark. A copy of
# dsv41flash_fp4_sglang_mtp.sh rather than a symlink: H100 is not in the
# cookbook's hardware table, and 80 GB cards cannot hold the resident weights
# plus the row-sharded Engram tables (~23.6 GiB per rank at TP8, measured on
# the vLLM arm) and still leave a KV pool. The Engram tables move to one shared
# host copy, and the prefill chunk is halved so the sparse-attention indexer's
# [chunk, context] scoring buffer fits next to the weights at 1M context.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY SPEC_DECODING
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    # Resolve the downloaded snapshot so upstream Engram page-cache advice
    # can find the checkpoint files instead of treating the HF ID as a path.
    MODEL_PATH=$(python3 -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id=sys.argv[1], local_files_only=True))' "$MODEL")
    export MODEL_PATH
fi

nvidia-smi
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
# Hardware-specific tiling only; checkpoint data, scales and dtypes are unchanged.
# Resolve and verify the installed configs through the nightly's actual loader.
python3 "$(dirname "$0")/install_h100_block32_configs.py" \
    "$(dirname "$0")/kernel_configs/h100_dsv41_block32" "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# Long-context indexer masks change allocation sizes across requests. At C20,
# the stock allocator OOMed on a 2.39 GiB mask with 5.61 GiB reserved but unused.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Agentic warmup dispatches hundreds of large prompts at once and SGLang's
# tokenizer can leave bytes unacknowledged past AIPerf's default 30 s
# TCP_USER_TIMEOUT, so Linux aborts live localhost connections.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# Outlast AIPerf's pooled connections so an inter-turn idle gap cannot race
# Uvicorn's five-second keep-alive closure.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

# AgentX measures the thinking-on regime, which is also the committed golden-AL
# curve. SGLang ships thinking off by default for this model.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV41_REASONING_EFFORT=high

# Row-sharded anonymous host Engram tables allow transparent huge pages;
# the shared memfd layout cannot use them when shmem THP is disabled.
# Checkpoint weights/scales are unchanged, and the KV cache stays on GPU.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts, and
# keep decode graphs covering that fan-out down to the cookbook's 64.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
if (( CUDA_GRAPH_MAX_BS < 64 )); then
    CUDA_GRAPH_MAX_BS=64
elif (( CUDA_GRAPH_MAX_BS > 128 )); then
    CUDA_GRAPH_MAX_BS=128
fi

# The indexer's scoring buffer and the hyper-connection activations scale with
# the prefill chunk times the 1M context. 4096 at mem-fraction 0.8 left 16 GB
# of headroom on the 80 GB card and c8 OOMed in eager extend once five
# requests were live with a 570k-token prompt pending (run 35304509605); 2048
# with 0.7 leaves 24 GB and halves the per-chunk working set.
# Back to 4096 with the static fraction kept at 0.7: at 2048 prefill ran near
# 1,000 tok/s and c4 failed AIPerf's 95% latency-coverage check (TTFT 87.6%,
# ITL 88.4% over the 3600 s window, run 35307250127) while c1/c2 passed.
# 0.7 leaves 24 GB for the doubled per-chunk working set instead of the
# 16 GB that OOMed c8 at 0.8/4096.
CHUNKED_PREFILL_SIZE=4096

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
echo "Using SGLang endpoint ${AIPERF_SERVER_URL}"

# The caller selects native non-speculative serving or the bundled DSpark
# draft. STP and accuracy evals must never inherit synthetic acceptance.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
SPECULATIVE_ARGS=()
# Bound long-prefill decode stalls for DSpark as well as the STP comparison.
SCHEDULING_ARGS=(--prefill-decode-interval 16)
case "$SPEC_DECODING" in
    mtp)
        DSPARK_BLOCK_SIZE=5
        DSV41_GOLDEN_AL=3.51
        SPECULATIVE_ARGS=(--speculative-algorithm DSPARK --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE")
        if [[ "$EVAL_ONLY" != true ]]; then
            export SGLANG_SIMULATE_ACC_LEN="$DSV41_GOLDEN_AL"
            export SGLANG_SIMULATE_ACC_METHOD=match-expected
            export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
        fi
        echo "DSpark block size: $DSPARK_BLOCK_SIZE, golden AL=$DSV41_GOLDEN_AL"
        ;;
    none)
        # Long AgentX prompts otherwise keep prefill ahead of every ready decode.
        # Interleave decode steps without changing requests or context lengths.
        echo "Native non-speculative serving; synthetic acceptance disabled"
        ;;
    *)
        echo "Unsupported SPEC_DECODING=$SPEC_DECODING; expected mtp or none" >&2
        exit 1
        ;;
esac

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    # Native MXFP4 Marlin supports Hopper with BF16 activations; dense FP8
    # operators and shipped DSpark precision remain unchanged.
    --moe-runner-backend marlin
    --mem-fraction-static 0.7
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    # The 14.96 GiB H100 KV budget cannot afford the larger Blackwell tail
    # reserve. At C20, 640 tails retain about 5.3M full tokens while reducing
    # the measured eviction pressure on the default 160-tail SWA pool.
    --swa-prefix-tails "$(( CONC >= 4 ? 32 * CONC : 8 * CONC ))"
    "${SPECULATIVE_ARGS[@]}"
    "${SCHEDULING_ARGS[@]}"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --reasoning-parser auto
    --tool-call-parser auto
    # Draft-token forward passes under long-context agentic load block the
    # scheduler long enough to trip the 1800 s default watchdog mid-warmup.
    --watchdog-timeout 3600
    --enable-metrics
)
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
