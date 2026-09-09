#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.8-Flash-Next FP8 on MI355X using
# SGLang with MTP speculative decoding. Day-zero recipe for the hybrid
# GDN + QSA architecture on CDNA4 (gfx950).
#
# MI355X is ROCm, so this arm is FP8 (Qwen/Qwen3.8-Flash-Next-FP8).
# NVFP4 is not available on AMD GPUs. Attention backend is aiter (ROCm).
#
# The SGLang cookbook (docs.sglang.io) verified flags for MI355X are:
#   --tp-size 8 --attention-backend aiter --page-size 32 --kv-cache-dtype auto
#   --chunked-prefill-size 16384 --watchdog-timeout 1200 --mem-fraction-static 0.9
#
# Speculative decoding uses NEXTN (the Qwen3.8 Flash Next native MTP head),
# 3 steps, eagle-topk 1, 4 draft tokens — mirroring the H200 agentic recipe.
#
# Required env vars:
#   MODEL, TP, CONC, EP_SIZE, KV_OFFLOADING,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION

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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k
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

# ---- HiCache (host-DRAM KV offload) ----------------------------------------
CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    TOTAL_CPU_DRAM_GB="$REQUESTED_HICACHE_TOTAL_GB"
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
    MAX_HICACHE_SIZE_GB=$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB exceeds configured per-pool limit $MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
    if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
        echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB, TP=$TP, HICACHE_HOST_POOL_COUNT=$HICACHE_HOST_POOL_COUNT" >&2
        exit 1
    fi
    echo "HiCache CPU pool: ${HICACHE_SIZE_GB} GB per rank per host pool across TP=${TP}, host_pool_count=${HICACHE_HOST_POOL_COUNT}"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
    )
fi

# ---- Parallelism ------------------------------------------------------------
PARALLEL_ARGS=(
    --tp-size "$TP"
    --dp-size 1
    --ep-size "$EP_SIZE"
)

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128

# ---- SGLang runtime patches (FP8 PLE weight_scale — SGLang PR #36497) -------
# The qwen38flashnext Docker image predates the FP8 PLE weight_scale fix.
# Rather than rebuilding the image we apply 5 targeted in-place patches to the
# container's sglang source before launching the server.
#
# Patched file: /workspace/sglang-qwen-next/python/sglang/srt/models/qwen4_exp.py
# Commit ref:   sgl-project/sglang@3003ddf1574ef5004e21a10e36aaabc364766921
#
# The patches are idempotent — safe to run multiple times, already-applied
# patches are skipped.
_SGLANG_ROOT="${SGLANG_ROOT:-/workspace/sglang-qwen-next/python/sglang}"
_QWEN4_EXP="${_SGLANG_ROOT}/srt/models/qwen4_exp.py"

if [[ -f "$_QWEN4_EXP" ]]; then
    echo "Applying SGLang runtime patches to $_QWEN4_EXP ..."

    # Patch 1: Register weight_scale buffer in Qwen4ExpNGramEmbedding.__init__
    if ! grep -q 'weight_scale.*torch.ones' "$_QWEN4_EXP"; then
        echo "  PATCH 1: Registering weight_scale buffer in Qwen4ExpNGramEmbedding"
        python3 -c "
import sys
with open('$_QWEN4_EXP', 'r') as f:
    content = f.read()
old = 'self.short_conv_dilation = self.ple_embedding.ngram_size'
if old not in content:
    print('  PATCH 1: Target line not found, skipping', file=sys.stderr)
    sys.exit(0)
new = '''self.ple_embedding.ngram_embedding.register_buffer(
            \"weight_scale\", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )
        self.short_conv_dilation = self.ple_embedding.ngram_size'''
content = content.replace(old, new, 1)
with open('$_QWEN4_EXP', 'w') as f:
    f.write(content)
print('  PATCH 1: Applied successfully')
"
    else
        echo "  PATCH 1: Already applied"
    fi

    # Patch 2: Multiply embeddings by weight_scale in forward()
    if ! grep -q 'embeddings \* self.ngram_embedding.weight_scale' "$_QWEN4_EXP"; then
        echo "  PATCH 2: Adding weight_scale multiply in forward()"
        python3 -c "
import sys
with open('$_QWEN4_EXP', 'r') as f:
    content = f.read()
old = 'embeddings = self.ngram_embedding(lookup_ids)\n        return self._finish_embedding_lookup('
if old not in content:
    print('  PATCH 2: Target pattern not found, skipping', file=sys.stderr)
    sys.exit(0)
new = '''embeddings = self.ngram_embedding(lookup_ids)
        embeddings = embeddings * self.ngram_embedding.weight_scale
        return self._finish_embedding_lookup('''
content = content.replace(old, new, 1)
with open('$_QWEN4_EXP', 'w') as f:
    f.write(content)
print('  PATCH 2: Applied successfully')
"
    else
        echo "  PATCH 2: Already applied"
    fi

    # Patch 3: Add weight_scale to PLE buffer allowlist in _load_qwen4_exp_ple_buffer
    if ! grep -q '"weight_scale"' "$_QWEN4_EXP" 2>/dev/null || \
       ! python3 -c "
with open('$_QWEN4_EXP') as f:
    content = f.read()
assert '\"weight_scale\",' in content and 'ngram_heads_vocab_sizes' in content
idx_ws = content.index('\"weight_scale\",')
idx_nv = content.index('\"ngram_heads_vocab_sizes\"')
assert abs(idx_ws - idx_nv) < 200
" 2>/dev/null; then
        echo "  PATCH 3: Adding weight_scale to PLE buffer allowlist"
        python3 -c "
import sys
with open('$_QWEN4_EXP', 'r') as f:
    content = f.read()
old = '''\"ngram_heads_vocab_sizes\",
        }:'''
if old not in content:
    print('  PATCH 3: Target pattern not found, skipping', file=sys.stderr)
    sys.exit(0)
new = '''\"ngram_heads_vocab_sizes\",
            \"weight_scale\",
        }:'''
content = content.replace(old, new, 1)
with open('$_QWEN4_EXP', 'w') as f:
    f.write(content)
print('  PATCH 3: Applied successfully')
"
    else
        echo "  PATCH 3: Already applied"
    fi

    # Patch 4: weight_scale multiply in offloaded PLE reduce path
    if ! grep -q 'ngram_embedding.reduce(embeddings)\n.*\* self.ple_embedding.ngram_embedding.weight_scale' "$_QWEN4_EXP" 2>/dev/null; then
        echo "  PATCH 4: Adding weight_scale multiply in offloaded PLE reduce path"
        python3 -c "
import sys
with open('$_QWEN4_EXP', 'r') as f:
    content = f.read()
old = 'embeddings = self.ple_embedding.ngram_embedding.reduce(embeddings)\n        embeddings = self.ple_embedding._finish_embedding_lookup('
if old not in content:
    print('  PATCH 4: Target pattern not found, skipping', file=sys.stderr)
    sys.exit(0)
new = '''embeddings = self.ple_embedding.ngram_embedding.reduce(embeddings)
        embeddings = embeddings * self.ple_embedding.ngram_embedding.weight_scale
        embeddings = self.ple_embedding._finish_embedding_lookup('''
content = content.replace(old, new, 1)
with open('$_QWEN4_EXP', 'w') as f:
    f.write(content)
print('  PATCH 4: Applied successfully')
"
    else
        echo "  PATCH 4: Already applied"
    fi

    # Patch 5: Register weight_scale in Qwen4ExpPinnedHostEmbedding
    if ! grep -q 'register_buffer.*"weight_scale".*embedding.weight_scale' "$_QWEN4_EXP" 2>/dev/null; then
        echo "  PATCH 5: Registering weight_scale in Qwen4ExpPinnedHostEmbedding"
        python3 -c "
import sys
with open('$_QWEN4_EXP', 'r') as f:
    content = f.read()
old = 'cpu_weight.weight_loader = self.weight_loader\n        self.register_parameter(\"weight\", cpu_weight)'
if old not in content:
    print('  PATCH 5: Target pattern not found, skipping', file=sys.stderr)
    sys.exit(0)
new = '''cpu_weight.weight_loader = self.weight_loader
        self.register_parameter(\"weight\", cpu_weight)
        self.register_buffer(\"weight_scale\", embedding.weight_scale, persistent=True)'''
content = content.replace(old, new, 1)
with open('$_QWEN4_EXP', 'w') as f:
    f.write(content)
print('  PATCH 5: Applied successfully')
"
    else
        echo "  PATCH 5: Already applied"
    fi

    echo "All SGLang runtime patches applied."
else
    echo "WARNING: qwen4_exp.py not found at $_QWEN4_EXP; running with container defaults" >&2
fi

# ---- ROCm / aiter environment -----------------------------------------------
export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT8
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
export SGLANG_ENABLE_SPEC_V2=1

# ---- Speculative decoding (NEXTN / MTP) -------------------------------------
SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

# golden_al_distribution/qwen3.8next_mtp.yaml:
# qwen3.8-flash-next-fp8.thinking_on[3] = 2.32.
# AgentX replays run with thinking on, so the thinking_on row is the right one.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.32
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

# ---- multi_tokenizer cached_tokens_details patch ----------------------------
SGLANG_MULTI_TOKENIZER=/sgl-workspace/sglang/python/sglang/srt/managers/multi_tokenizer_mixin.py
if [ -f "$SGLANG_MULTI_TOKENIZER" ]; then
    if ! sed -n '/elif isinstance(output, BatchStrOutput):/,/input_token_logprobs_val=_extract_field_by_index/p' "$SGLANG_MULTI_TOKENIZER" \
        | grep -q 'cached_tokens_details=_extract_field_by_index'; then
        sed -i '/elif isinstance(output, BatchStrOutput):/,/cached_tokens=_extract_field_by_index(output, "cached_tokens", i),/ {
            /cached_tokens=_extract_field_by_index(output, "cached_tokens", i),/a\
                cached_tokens_details=_extract_field_by_index(\
                    output, "cached_tokens_details", i\
                ),
        }' "$SGLANG_MULTI_TOKENIZER"
    fi
fi

# ---- Launch server -----------------------------------------------------------
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend aiter
    --page-size 32
    --kv-cache-dtype auto
    --mem-fraction-static 0.90
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --chunked-prefill-size 16384
    --mamba-ssm-dtype bfloat16
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    "${SPEC_ARGS[@]}"
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
