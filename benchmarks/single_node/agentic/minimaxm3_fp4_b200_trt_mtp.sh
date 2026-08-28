#!/usr/bin/env bash
set -euo pipefail
set -x

# MiniMax-M3 NVFP4 B200 AgentX with EAGLE3-GQA. Throughput uses forced
# synthetic acceptance while EVAL_ONLY respects the verifier's actual result.
# DRAM KV offload uses TRT-LLM's native secondary-memory pool, with the fixed
# kv_cache_config.host_cache_size documented below.
# KV_OFFLOADING / KV_OFFLOAD_BACKEND come from the master config through
# benchmark-tmpl.yml; do not override them here.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC PORT KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EVAL_ONLY

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
NUM_SPEC_TOKENS=3
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/prometheus/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="trtllm_kv_cache_utilization"
# Golden AL for the GQA draft head: golden_al_distribution/minimaxm3_eagle3_gqa.yaml
# minimax-m3.thinking_on[3].

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="/lustre/fsw/gharunners/models/${DRAFT_MODEL##*/}"
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        mkdir -p "$DRAFT_MODEL_PATH"
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
    hf download "$DRAFT_MODEL"
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
fi

nvidia-smi
resolve_trace_source
install_agentic_deps

# kv_cache_config.host_cache_size is pinned per topology in ser.yaml below
# (200 GiB at TP8, 250 GiB at TP4 -- see $mem_off), NOT derived from
# TOTAL_CPU_DRAM_GB.

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "TRTLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CAPTURE_TOKENS_LIST=(1 512 1024 2048)
CAPTURE_TOKENS_LIST=$(printf "%s, " "${CAPTURE_TOKENS_LIST[@]}")

MAX_BATCH=$CONC
if (( MAX_BATCH <= 20 )); then
    CAPTURE_BATCH_LIST=( $(seq 1 $MAX_BATCH) )
elif (( MAX_BATCH == 25 )); then
    CAPTURE_BATCH_LIST=( $(seq 1 15) 17 19 21 23 25 )
elif (( MAX_BATCH == 30 )); then
    CAPTURE_BATCH_LIST=( $(seq 1 12) $(seq 14 2 24) 27 30 )
elif (( MAX_BATCH == 35 )); then
    CAPTURE_BATCH_LIST=( $(seq 1 11) $(seq 14 3 35) )
elif (( MAX_BATCH == 40 )); then
    CAPTURE_BATCH_LIST=( $(seq 2 2 40) )
elif (( MAX_BATCH == 45 )); then
    CAPTURE_BATCH_LIST=( $(seq 2 2 18) $(seq 21 3 45) )
elif (( MAX_BATCH == 50 )); then
    CAPTURE_BATCH_LIST=( $(seq 2 2 8) 11 $(seq 14 2 18) $(seq 21 3 48) 50 )
elif (( MAX_BATCH == 55 )); then
    CAPTURE_BATCH_LIST=( $(seq 2 3 14) $(seq 16 2 18) $(seq 21 3 48) 50 53 55 )
fi
CAPTURE_BATCH_LIST=$(printf "%s, " "${CAPTURE_BATCH_LIST[@]}")

if [[ $TP == 8 ]]; then
    mem_off=214748364800
else
    mem_off=268435456000
fi

cat << EOF > ser.yaml
max_seq_len: 1048576
max_num_tokens: 16384
max_batch_size: $MAX_BATCH
cuda_graph_config:
    enable_padding: true
    batch_sizes: [${CAPTURE_BATCH_LIST%, }]
torch_compile_config:
    enable_fullgraph: true
    enable_inductor: false
    enable_piecewise_cuda_graph: true
    capture_num_tokens: [${CAPTURE_TOKENS_LIST%, }]
    enable_userbuffers: true
    max_num_streams: 3
moe_config:
    backend: TRTLLM
    use_low_precision_moe_combine: true
sparse_attention_config:
    algorithm: minimax_m3
    implementation: msa
    indexer_kv_dtype: fp8
    sparse_disable_index_value: true
    fuse_qkv_index_projection: true
kv_cache_config:
    free_gpu_memory_fraction: 0.94
    enable_block_reuse: true
    tokens_per_block: 128
    use_kv_cache_manager_v2: true
    dtype: fp8
    event_buffer_max_size: 0
    host_cache_size: $mem_off
speculative_config:
    decoding_type: Eagle3
    max_draft_len: 3
    speculative_model: $DRAFT_MODEL_PATH
scheduler_config:
    capacity_scheduler_policy: MAX_UTILIZATION
enable_chunked_prefill: true
enable_autotuner: true
trust_remote_code: true
reasoning_parser: minimax_m3
stream_interval: 20
print_iter_log: true
enable_iter_perf_stats: true
return_perf_metrics: true
num_postprocess_workers: 8
enable_attention_dp: false
EOF

export TLLM_LOG_LEVEL=INFO
export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1
export TLLM_PROFILE_LOG_RANKS=all
# aiperf resolves its tokenizer by HF repo id ($MODEL), not by path, so do NOT
# set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE here.
export PYTHONNOUSERSITE=1
export TRTLLM_ENABLE_PDL=1
export ENROOT_ALLOW_DEV=yes
export NCCL_GRAPH_MIXING_SUPPORT=0
export MIMALLOC_PURGE_DELAY=0
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRTLLM_SERVE_ENABLE_MSGSPEC=1
export TRTLLM_TORCH_COMPILE_CONTEXT_ONLY=1
# Throughput pins the committed MiniMax-M3 EAGLE3-GQA golden AL to 2.78:
# one target token plus 1.78 accepted draft tokens. The force knob overwrites
# the verifier's accepted-token count, so accuracy evals must leave it disabled.
if [ "$EVAL_ONLY" = "true" ]; then
    unset TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS
else
    export TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=1.78
fi

{ set +x; } 2>/dev/null
# Launch through mpirun, as every other TRT-LLM benchmark in this repo does.
TRTLLM_CMD=(
    mpirun -n 1 --oversubscribe --allow-run-as-root
    trtllm-serve "$MODEL_PATH"
    --tp_size "$TP"
    --host 0.0.0.0
    --port "$PORT"
    --chat_template "$MODEL_PATH/chat_template.jinja"
    --config ser.yaml
)
printf '%q ' "${TRTLLM_CMD[@]}" | tee "$RESULT_DIR/trtllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/trtllm_command.txt"
"${TRTLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
set -x

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
