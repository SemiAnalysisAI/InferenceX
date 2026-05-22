#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using SGLang.
# Adapted from benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh (fixed-seq-len
# sibling) with the agentic harness (build_replay_cmd / write_agentic_result_json
# / analyze_benchmark_distributions) swapped in for run_benchmark_serving.
#
# This launcher does NOT support CPU offload. SGLang's KV offload paths are
# different from vLLM's SimpleCPUOffloadConnector, and the matching agentic
# config (dsv4-fp4-mi355x-sglang-agentic) only sweeps offloading=none.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
EP_SIZE=${EP_SIZE:-1}
DP_ATTENTION=${DP_ATTENTION:-false}
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=1000000
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Reject anything other than none: this launcher has no SGLang CPU-offload
# wiring (different surface than vLLM's SimpleCPUOffloadConnector).
case "$OFFLOADING" in
    none) ;;
    *)
        echo "Error: dsv4_fp4_mi355x_sglang.sh only supports OFFLOADING=none (got '$OFFLOADING')" >&2
        exit 1
        ;;
esac

# Transformers in the container doesn't recognize the `deepseek_v4` model_type.
# PR #23608's fallback in hf_transformers_utils.get_config tries to handle this
# by writing a patched config to /tmp, but in practice isn't catching the error
# in this image. Patch the cached config.json directly instead: set model_type
# to `deepseek_v3` so AutoConfig.from_pretrained succeeds, and keep
# architectures=['DeepseekV4ForCausalLM'] so SGLang dispatches to its native
# DSv4 model class (python/sglang/srt/models/deepseek_v4.py).
python3 << PYEOF
import json
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="$MODEL", filename="config.json")
with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
else:
    print(f"No patch needed: model_type is {config.get('model_type')!r}")
PYEOF

# DSv4 FP4-experts path. Mirrors the env block in the fixed-seq-len sibling
# (benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh), which tracks the active
# block in python/run_dsv4.sh on the amd/deepseek_v4 branch:
#   SGLANG_DSV4_FP4_EXPERTS=True   -> route experts through FP4 kernels
#   SGLANG_FORCE_TRITON_MOE_FP8=0  -> dispatch MoE through aiter and apply
#                                    the swiglu_limit clamp in the triton
#                                    MoE fallback path.
export SGLANG_REASONING_EFFORT=max
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_OPT_USE_OLD_COMPRESSOR=true
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_OPT_USE_AITER_MHC_PRE=true
export SGLANG_OPT_USE_AITER_MHC_POST=true
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_FORCE_TRITON_MOE_FP8=0
export SGLANG_HACK_FLASHMLA_BACKEND=tilelang
export SGLANG_OPT_USE_TILELANG_INDEXER=true
export SGLANG_OPT_USE_TRITON_SWA_PREPARE=true

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# Parallelism: pure TP, TP+EP, or DEP (DP-attn + EP). Matches the dsv4 b200
# vllm agentic launcher so the agentic sweep can probe both interactivity and
# throughput regimes.
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
    )
fi
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

# --max-running-requests is per-engine. With DP-attn each DP engine handles
# only CONC/$TP sequences in steady state (the agentic harness load-balances
# users across DP ranks), so size the per-engine cap to that.
# Pure TP is a single engine and sees all CONC sequences itself.
if [ "$DP_ATTENTION" = "true" ]; then
    PER_ENGINE_MAX_RUNNING=$(( CONC / TP ))
    [ "$PER_ENGINE_MAX_RUNNING" -lt 1 ] && PER_ENGINE_MAX_RUNNING=1
else
    PER_ENGINE_MAX_RUNNING=$CONC
fi

echo "Starting sglang server..."
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host=0.0.0.0 \
    --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend compressed \
    --max-running-requests "$PER_ENGINE_MAX_RUNNING" \
    --cuda-graph-max-bs "$PER_ENGINE_MAX_RUNNING" \
    --page-size 256 \
    --context-length "$MAX_MODEL_LEN" \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja" \
    --watchdog-timeout 1800 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
