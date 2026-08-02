#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 (MXFP4) on B300 using vLLM with
# DSpark speculative decoding at level 2, probabilistic drafting, synthetic
# acceptance pinned to the committed golden AL (real verification for evals).
#
# MEASURED, for anyone reading the throughput numbers: this draft head's REAL
# acceptance collapses on the agentic corpus. Its native window is 32k
# (rope_parameters yarn, factor 32, original_max_position_embeddings 32768,
# stretched to the declared 1M), and at the corpus's 100k-330k ISL the measured
# acceptance length is 1.16-2.01 with 13-41% position-1 acceptance -- versus 4.18
# AL and 0.826 position-1 on short-context work under the identical image, draft
# and spec config. Synthetic acceptance is the AgentX-prescribed way to measure
# system performance at a fixed acceptance target (see the SPEC_CONFIG block), so
# these throughput numbers are NOT evidence that the draft predicts well at
# agentic context lengths. It does not.
#
# MTP sibling of kimik3_fp4_b300_vllm.sh. Everything about the target server is
# identical; the deltas are:
#   - --speculative-config  method "dspark", draft Inferact/Kimi-K3-DSpark,
#                           attention_backend FLASHINFER_MLA,
#                           draft_sample_method probabilistic,
#                           rejection_sample_method block
#   - cudagraph capture sizes are expressed in TOKENS, not sequences
#
# This script also serves the MoonEP expert-parallel arm
# (kimik3-fp4-b300-vllm-agentic-dspark-moonep, EP_SIZE=8): the launcher derives
# the script name from model-prefix/precision/framework/spec-decoding, so both
# master-config keys resolve here and the EP arm is selected by EP_SIZE. Its
# deltas relative to the pure-TP8 profile:
#   - --enable-expert-parallel --all2all-backend moonep  (EP=8: EP = TP x DP,
#     TP=8, DP=1; the vLLM patch forces sequence-parallel MoE for moonep so the
#     8 ranks dispatch distinct sequence slices instead of duplicating expert
#     compute)
#   - --load-format auto rather than fastsafetensors: fastsafetensors' large
#     staging buffers OOM against MoonEP's out-of-band VMM expert-weight
#     allocations. auto loads the 1.5 TB checkpoint in ~231 s off the staged
#     mount (bring-up run 38792), well inside VLLM_ENGINE_READY_TIMEOUT_S.
#   - MoonEP itself is built from source at a pinned commit and the vLLM moonep
#     backend is overlaid onto the image's installed vllm package at runtime
#     (see setup_moonep) -- the stock vllm/vllm-openai:kimi-k3 image contains
#     neither.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
# Optional:
#   EP_SIZE  (unset/1 = pure TP8; 8 = MoonEP expert-parallel arm)
#
# TP8 is the only single-node layout. The MXFP4 checkpoint is ~1.5 TB on disk;
# at TP4 that is ~375 GB of weights per GPU against B300's 288 GB of HBM, so
# only the full 8-GPU shard (~188 GB/GPU) fits. Do not add TP4/TP2 arms.
#
# Chat formatting: AGENTS.md requires MTP scripts to pass --use-chat-template to
# run_benchmark_serving. Agentic recipes never call it -- the replay drives
# AIPerf against /v1/chat/completions, so prompts are already chat-formatted and
# spec-decode acceptance is measured on chat-shaped inputs. Nothing to add here.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

# The 2.8T MXFP4 checkpoint only fits across all 8 B300s (see header).
if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on B300 requires TP=8 (a ~1.5 TB MXFP4 checkpoint does not fit at TP<8), got TP='$TP'" >&2
    exit 1
fi

# EP_SIZE > 1 selects the MoonEP arm. EP = TP x DP and DP=1 on this single-node
# layout, so the only valid EP is exactly TP (=8); anything else is a config bug.
if [[ -n "${EP_SIZE:-}" && "${EP_SIZE}" -gt 1 ]]; then
    if [ "$EP_SIZE" -ne "$TP" ]; then
        echo "Error: the MoonEP arm requires EP_SIZE == TP (EP = TP x DP with DP=1), got EP_SIZE='$EP_SIZE' TP='$TP'" >&2
        exit 1
    fi
    MOONEP_ENABLED=1
else
    MOONEP_ENABLED=0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE.
# Either way, MODEL_PATH is what the server is launched with.
#
# The draft must NOT land next to a pre-staged target: Kimi-K3 is in the
# b300-nv launcher's STAGED_MODELS, so dirname(MODEL_PATH) is the read-only
# /scratch/models mount and writing there fails with PermissionError. Use the
# launcher's writable models dir instead, which also caches the draft across
# the jobs in a sweep (same placement as DeepSeek-V4-Pro-DSpark).
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="${WRITABLE_MODELS_DIR:-/data/models}/${DRAFT_MODEL##*/}"
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
    hf download "$DRAFT_MODEL"
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- MoonEP build + vLLM backend overlay (EP arm only) -----------------------
# The stock vllm/vllm-openai:kimi-k3 image ships neither the MoonEP package nor
# a vLLM that knows the moonep all2all backend. Both are added inside the
# container at runtime (the container filesystem is a discarded overlay), at a
# pinned MoonEP commit plus a one-line C++ fix, then the checked-in vLLM diff is
# overlaid onto the installed package. Mirrors the validated bring-up flow in
# /data/home/sa-shared/moonep-b300/bringup.sbatch on the b300-nv login host.
MOONEP_COMMIT="0f385f038fc33bec22e3bcf5a07a8a22693e754c"
MOONEP_SRC="/opt/MoonEP"
# Resolve to an absolute path before any cd: $0 is relative to the container
# workdir (/workspace).
MOONEP_VLLM_PATCH="$(cd "$(dirname "$0")" && pwd)/patches/kimik3_moonep_vllm.patch"

setup_moonep() {
    if [[ ! -f "$MOONEP_VLLM_PATCH" ]]; then
        echo "Error: missing vLLM overlay patch at $MOONEP_VLLM_PATCH" >&2
        exit 1
    fi

    # install_agentic_deps has already ensured git exists in the image.
    rm -rf "$MOONEP_SRC"
    git clone https://github.com/MoonshotAI/MoonEP.git "$MOONEP_SRC"
    git -C "$MOONEP_SRC" checkout "$MOONEP_COMMIT"

    # torch 2.13 marks freshly-created TensorImpls metadata-immutable;
    # MoonEP's shared-buffer view constructor must opt back in before
    # resizing. Idempotent single-line insert.
    python3 - "$MOONEP_SRC/csrc/nvl_shared_buffer.cuh" <<'PYPATCH'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "    impl->set_sizes_contiguous(shape);"
fix = "    impl->set_allow_tensor_metadata_change(true);\n"
if "set_allow_tensor_metadata_change" not in s:
    assert anchor in s, f"anchor line not found in {p}"
    s = s.replace(anchor, fix + anchor, 1)
    open(p, "w").write(s)
    print("PATCHED moonep for torch 2.13")
else:
    print("moonep already patched for torch 2.13")
PYPATCH

    # --no-deps keeps the image's nvidia-cutlass-dsl (4.6.0): MoonEP pins
    # ==4.4.2, but downgrading breaks vllm_flash_attn/cute (quack needs
    # cutlass._mlir_helpers) and, below 4.5, flashinfer. MoonEP's own suite
    # passes on 4.6.0.
    pip install --no-build-isolation --no-deps -e "$MOONEP_SRC"
    python3 -c "import flashinfer; print('flashinfer import OK')"
    python3 -c "import moonep; from moonep._C import nvl_multicast_supported as m; print('moonep ok, multicast', m())"

    # Overlay the vLLM moonep backend onto the image's installed package.
    # --forward makes a re-run on an already-patched tree a no-op instead of
    # prompting; any FAILED hunk or .rej is fatal.
    local patch_log="$RESULT_DIR/moonep_vllm_patch.log"
    pushd /usr/local/lib/python3.12/dist-packages
    if ! patch -p1 --forward --batch < "$MOONEP_VLLM_PATCH" > "$patch_log" 2>&1 \
        || grep -q "FAILED" "$patch_log"; then
        echo "Error: MoonEP vLLM overlay did not apply cleanly" >&2
        tail -30 "$patch_log" >&2
        find . -name "*.rej" -print >&2
        popd
        exit 1
    fi
    popd

    python3 - <<'PYCHECK'
from vllm.utils.import_utils import has_moonep
assert has_moonep(), "has_moonep() is False after install"
from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import MoonEPPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.experts.moonep_deep_gemm_moe import MoonEPDeepGemmFP4Experts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_moonep_mxfp4 import MoonEPCompressedTensorsMxfp4MoEMethod
print("moonep backend imports OK")
PYCHECK
}

# ---- Kimi-K3 production serving environment ---------------------------------
export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
# Required by the upstream recipe on BOTH its blackwell and nvidia paths
# (recipes.vllm.ai/moonshotai/Kimi-K3), and never set here before. It defaults to
# 0, and it is threaded into LatentMoERunner as
# runner_args={"enable_k3_latent_moe_tail_fusion": ...} at
# vllm/models/kimi_k3/nvidia/model.py:549 -- the same runner whose shared-experts
# output buffer asserted (fused_moe/runner/shared_experts.py:165, all 8 TP ranks)
# when the wider flag alignment was tried. With this unset we have been running a
# MoE tail path upstream never exercises, so enable it on its own before
# re-attempting any of the other upstream flags.
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_USE_RUST_FRONTEND=1
# Loading ~1.5 TB of MXFP4 shards off the staged mount takes well past the
# default readiness window even with fastsafetensors.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONNOUSERSITE=1
# AIPerf pins one pooled keep-alive connection per agentic session and reuses it
# across turns, while the Rust frontend's default VLLM_HTTP_TIMEOUT_KEEP_ALIVE is
# 5s. An inter-turn idle gap longer than that lets the client reuse a socket at
# the moment the server closes it -> aiohttp ServerDisconnectedError -> AIPerf
# treats it as a terminal warmup failure and aborts the whole job. Outlast the
# client pool so the race cannot occur.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
# Agentic warmup dispatches large prompts at once; allow up to 15 minutes of TCP
# progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- MoonEP arm: install backend, pick EP flags ------------------------------
# MoonEP's expert weights are CUDA VMM allocations outside the torch caching
# allocator, so the utilization budget has to cover them alongside the KV
# cache. Measured on b300 with Kimi-K3 at TP8/EP8: 0.90 leaves -2.36 GiB for
# KV and the engine refuses to start; 0.96 yields 508,586 KV tokens.
if [ "$MOONEP_ENABLED" = "1" ]; then
    # 0.96 fits a short context but leaves only 11.91 GiB of KV, and the
    # 1M max-model-len this recipe uses needs 16.61 GiB. MoonEP's expert
    # weights are VMM allocations outside the torch allocator, and the
    # 128-row expert padding that DeepGEMM's tight scale stride forces costs
    # ~24 GiB/rank on top -- so this arm needs a higher budget than -dspark.
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.98}"
else
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
fi

if [ "$MOONEP_ENABLED" = "1" ]; then
    setup_moonep
    # fastsafetensors' large staging buffers OOM against MoonEP's out-of-band
    # VMM expert-weight allocations; plain auto loading fits and takes ~231 s.
    LOAD_FORMAT=auto
    EP_ARGS=(--enable-expert-parallel --all2all-backend moonep)
else
    LOAD_FORMAT=fastsafetensors
    EP_ARGS=()
fi

# ---- KV offloading ----------------------------------------------------------
# The generated TOTAL_CPU_DRAM_GB budget is the aggregate host-DRAM pool for the
# node; SimpleCPUOffloadConnector is sized per rank. At dram-utilization 0.63 on
# cluster:b300-nv this resolves to ~220 GiB per rank across the 8 TP ranks.
OFFLOAD_ARGS=()
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
        # Identical prefixes must hash to identical block keys run-to-run.
        export PYTHONHASHSEED=42
        # lazy_offload must be a JSON boolean, not a quoted string: the
        # connector does bool(extra_config.get("lazy_offload", False)), and
        # bool("false") is True in Python.
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":${CPU_BYTES_PER_RANK},\"lazy_offload\":false}}"
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected empty or vllm-simple)" >&2
        exit 1
        ;;
esac

# ---- DSpark speculative decoding -------------------------------------------
# Probabilistic drafting at DSpark level 2, with synthetic acceptance pinned to
# the committed golden AL, per the AgentX policy in
# golden_al_distribution/README.md: "a submission may choose any supported draft
# length, but it may not substitute a different acceptance target", because
# "InferenceX is evaluating inference-system performance, not the ability to
# fine-tune a benchmark-specific speculative head". AL is workload-dependent by
# the README's own admission, so the prescribed acceptance -- not the acceptance
# this corpus happens to produce -- is what makes submissions comparable.
#
# Level 2 rather than 7. Draft length IS ours to choose, and the verify width is
# a real system cost that synthetic acceptance does not remove: at K=7 the target
# verifies 8 positions per step, which loses to no-speculation once the GPU
# saturates even at the forced golden AL of 3.84. Using cost ~ 1 + m(K-1) + f,
# K=2 clears break-even at both ends of the m range while K=3 is a wash at
# saturation and K=7 is negative:
#     K=2  golden AL 2.51  ->  ~1.7x low conc, ~1.2x saturated
#     K=3  golden AL 3.00  ->  ~1.7x low conc, ~0.95x saturated
#     K=7  golden AL 3.84  ->  ~1.3x low conc, ~0.54x saturated
# The drafter's KV footprint (~25% of GPU KV) is unaffected by K and remains a
# genuine cost of running spec decoding at all on 100k+ prompts.
NUM_SPEC_TOKENS=2
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))
# Committed golden AL at K=2 on the probabilistic curve we run
# (golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml:
# thinking_on 2 -> 2.51). The greedy/standard curve's K=2 value is 2.45 -- do not
# mix curves.
SYNTHETIC_ACCEPT_LEN=2.51

# Throughput runs pin synthetic acceptance; the EVAL_ONLY accuracy run must use
# real target verification instead. Synthetic acceptance commits drafted tokens
# regardless of the target's logits, so the generated text is wrong and the
# SWE-bench eval scores 0.0000 -- the same split dsv4_fp4_b300_vllm_mtp.sh makes
# (and which kimik2.5_fp4_b300_mtp.sh omits; follow dsv4, not kimik2.5).
# rejection_sample_method=block does real verification, so it is what EVAL_ONLY
# uses. vLLM rejects synthetic_acceptance_length unless the method is 'synthetic'.
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASHINFER_MLA\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}"
else
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASHINFER_MLA\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

# Agentic fan-out: the scheduler headroom convention shared by the other agentic
# recipes. Upstream pins 32 instead, but the aligned flag set crashed the engine
# core on every concurrency (see the serve-flag note below), so this stays on the
# value that ran green. Capture decode graphs only to this bound — a 93-layer
# 2.8T model makes capturing vLLM's full 2048-wide ladder prohibitively slow.
MAX_NUM_SEQS=$((2 * CONC))

# With spec decoding, cudagraph capture sizes are in TOKENS. vLLM rounds
# configured sizes up to multiples of (1 + num_speculative_tokens) and dedups
# them (adjust_cudagraph_sizes_for_spec_decode), so a plain 1..MAX_NUM_SEQS list
# would collapse to covering only MAX_NUM_SEQS/(1+N) sequences and drop the
# largest decode batches to eager. Enumerate the multiples explicitly so there
# is one graph per decode batch of 1..MAX_NUM_SEQS sequences. The graph mode is
# left at the recipe default that the non-MTP K3 arm already validated.
CUDA_GRAPH_CAPTURE_SIZES=""
for ((num_seqs = 1; num_seqs <= MAX_NUM_SEQS; num_seqs++)); do
    if [ -n "$CUDA_GRAPH_CAPTURE_SIZES" ]; then
        CUDA_GRAPH_CAPTURE_SIZES+=","
    fi
    CUDA_GRAPH_CAPTURE_SIZES+="$((num_seqs * TOKENS_PER_SEQ))"
done
COMPILATION_CONFIG="{\"cudagraph_capture_sizes\":[${CUDA_GRAPH_CAPTURE_SIZES}]}"

echo "Starting vllm server..."

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    # Adopting the full upstream flag set (gpu-memory-utilization 0.95,
    # max-num-seqs 32, max-model-len 1000000, max-num-batched-tokens 32768,
    # --no-enable-flashinfer-autotune, VLLM_USE_V2_MODEL_RUNNER=1) killed the
    # engine core on every concurrency with an assertion in the FlashInfer MoE
    # runner's shared-experts output buffer
    # (fused_moe/runner/shared_experts.py:165, all 8 TP ranks at once). Only
    # mla_prefill_backend=TRTLLM_RAGGED is retained from that set; the rest stay
    # on the values that ran 12/12 green in run 30326393603.
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len 1048576
    --trust-remote-code
    --load-format "$LOAD_FORMAT"
    --moe-backend auto
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    # FP8 KV cache requires the prefill query quantization flag. MLA prefill
    # runs on TRTLLM_RAGGED per the upstream Blackwell override.
    --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'
    --speculative-config "$SPEC_CONFIG"
    --compilation-config "$COMPILATION_CONFIG"
    --disable-uvicorn-access-log
    "${EP_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
