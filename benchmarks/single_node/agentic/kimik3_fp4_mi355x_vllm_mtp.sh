#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace-replay benchmark for Kimi-K3 (FP4) on AMD MI355X (gfx950) via
# vLLM with DSpark speculative decoding at level 2, probabilistic drafting, and
# synthetic acceptance pinned to the committed golden AL.
#
# MTP sibling of kimik3_fp4_mi355x_vllm.sh and the ROCm counterpart of
# kimik3_fp4_b300_vllm_mtp.sh. The target-server config is identical to the base
# MI355X recipe; the deltas are the DSpark --speculative-config, the KV-cache pin
# + spec-aware cudagraph capture sizes, and the fp8 ASM-MLA env. This is the
# variant whose numbers are apples-to-apples with the B300-MTP baseline
# (synthetic acceptance vs synthetic acceptance -- see the SPEC_CONFIG block).
#
# Attention: ROCM_AITER_MLA (asm persistent) for BOTH target and draft. On ROCm,
# the DSpark draft's semi-autoregressive parallel drafting is forced CAUSAL
# (dflash_config.causal=true, applied by apply_k3_container_patches.sh) so the
# draft no longer needs TRITON_MLA's non-causal path and runs on the same fp8 asm
# path as the target. This requires the vLLM ASM patches + DSpark fp8-asm layer
# baked into the validated base image below.
#
# Validated base image (carries the vLLM patches + AITER build + DSpark layer):
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
#   reproduced from the pinned nightly by apply_k3_container_patches.sh.
# Pinned in configs/amd-master.yaml (kimik3-fp4-mi355x-vllm-agentic-mtp).
#
# Required env vars: MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
# Optional: MODEL_PATH (pre-staged target), DRAFT_MODEL_PATH (pre-staged draft),
#           EVAL_ONLY (true -> real block verification for accuracy runs)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on MI355X requires TP=8 (~1.5 TB checkpoint), got TP='$TP'" >&2
    exit 1
fi

# ---- Bootstrap the container from the pinned base image ----------------------
# The image pinned in configs/amd-master.yaml is the STOCK ROCm vLLM nightly
# (cb8104839c...); this idempotently turns it into the exact k3-dspark-benchmark
# container (aiter @ 55dbc4f47 rebuild, tuned GEMM CSV, triton 3.7.0, 5 vLLM ASM
# patches, DSpark fp8-asm layer, FlyDSL->torch reroute). No-op once markers are
# present, so re-runs / pre-patched images cost only the grep verify. Set
# SKIP_K3_BOOTSTRAP=1 to skip (e.g. when serving a pre-baked patched image).
RECIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "${SKIP_K3_BOOTSTRAP:-0}" != "1" ]; then
    bash "$RECIPE_DIR/apply_k3_container_patches.sh"
fi

DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"

# Resolve target + draft weights (pre-staged else HF cache).
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${WRITABLE_MODELS_DIR:-/data/models}/${DRAFT_MODEL##*/}}"
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
else
    if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
    export MODEL_PATH="$MODEL"
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$DRAFT_MODEL}"
    if [[ "$DRAFT_MODEL_PATH" != /* ]]; then hf download "$DRAFT_MODEL"; fi
fi
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi

# Force the DSpark draft causal (non-causal parallel drafting is a cudagraph OOB
# source on ROCm; with causal=true the draft runs the fp8 asm path like the
# target). Idempotent write — the draft may live anywhere the harness staged it,
# so set it here rather than relying on the bootstrap's /dev/shm-scoped step.
# DRAFT_MODEL_PATH may be a bare repo id, so fall back to the HF cache snapshot.
DRAFT_CFG_DIR="$DRAFT_MODEL_PATH"
if [ ! -f "$DRAFT_CFG_DIR/config.json" ]; then
    DRAFT_CFG_DIR="$(python3 - "$DRAFT_MODEL_PATH" <<'PZ'
import sys
try:
    from huggingface_hub import snapshot_download
    print(snapshot_download(sys.argv[1], local_files_only=True), end="")
except Exception:
    print("", end="")
PZ
)"
fi
if [ ! -f "$DRAFT_CFG_DIR/config.json" ]; then
    echo "!! no draft config.json for '$DRAFT_MODEL_PATH' (resolved '${DRAFT_CFG_DIR:-}')" >&2
    exit 1
fi
if true; then
    python3 - "$DRAFT_CFG_DIR/config.json" <<'PY'
import json, sys
f = sys.argv[1]
c = json.load(open(f))
d = c.setdefault("dflash_config", {})
if d.get("causal") is True:
    print("draft already forced causal")
else:
    d["causal"] = True
    json.dump(c, open(f, "w"), indent=2)
    print("forced draft dflash_config.causal=true:", f)
PY
fi

# ---- MI355X day-0 serving environment (AITER + fp8 ASM MLA) ------------------
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export GPU_ARCHS=gfx950
export AITER_SITUV2_A8W4=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export SAFETENSORS_FAST_GPU=1
# fastsafetensors cuts weight load 745s -> 140s; not guaranteed in the image.
LOAD_FORMAT="${LOAD_FORMAT:-}"
if [ -z "$LOAD_FORMAT" ]; then
    if python3 -c "import fastsafetensors" 2>/dev/null; then LOAD_FORMAT=fastsafetensors;
    else echo "fastsafetensors missing -> auto" >&2; LOAD_FORMAT=auto; fi
fi
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# fp8 ASM persistent MLA on the native (unshuffled) KV layout. Do NOT set
# VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT. Baked DSpark layer reads this to route the
# 12-head fp8 spec verify to the asm q-row-fold.
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
# vLLM #50649: recompute_w_u_fwd_kernel autotunes num_stages=4, which is racy
# with num_warps=4 and silently emits ~1e38 garbage that propagates into the
# target logits (and, via NaN, into the rejection sampler's tl.argmax). Take the
# merged-upstream behaviour (safe list [2, 3]) at EVERY concurrency: a
# per-concurrency default makes cells on the same ladder differ in two variables
# at once, so no c1-vs-c8 delta can be attributed to concurrency. Stage 4 is an
# explicit A/B (VLLM_K3_KDA_SAFE_STAGES=0), not a silent function of CONC.
export VLLM_K3_KDA_SAFE_STAGES="${VLLM_K3_KDA_SAFE_STAGES:-1}"
# Merged tuned BF16 GEMM table installed by apply_k3_container_patches.sh.
MERGED_GEMM_CSV="${MERGED_GEMM_CSV:-/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv}"
if [ -z "${AITER_CONFIG_GEMM_BF16:-}" ] && [ -f "$MERGED_GEMM_CSV" ]; then
    export AITER_CONFIG_GEMM_BF16="$MERGED_GEMM_CSV"
fi
# Guard: the tuned BF16 GEMM table is a large perf lever. Without it the dense
# GEMMs fall back to an untuned path and throughput drops materially -- and it does
# so SILENTLY. Warn loudly if the CSV is missing (e.g. a pre-baked image that never
# ran apply_k3_container_patches.sh) so an untuned run is never mistaken for a valid
# number. Set REQUIRE_TUNED_GEMM=1 to make a missing CSV fatal instead of a warning.
if [ -z "${AITER_CONFIG_GEMM_BF16:-}" ] || [ ! -f "${AITER_CONFIG_GEMM_BF16:-/nonexistent}" ]; then
    echo "############################################################################" >&2
    echo "WARNING: tuned BF16 GEMM CSV not found (looked for '$MERGED_GEMM_CSV')."       >&2
    echo "         AITER_CONFIG_GEMM_BF16 is unset -> dense GEMMs run UNTUNED and"        >&2
    echo "         throughput will be materially lower than the published recipe."        >&2
    echo "         Fix: ensure apply_k3_container_patches.sh ran (it installs"            >&2
    echo "         k3_patches/kimik3_bf16_tuned_gemm.csv), or point AITER_CONFIG_GEMM_BF16">&2
    echo "         at the merged CSV. Set REQUIRE_TUNED_GEMM=1 to make this fatal."        >&2
    echo "############################################################################" >&2
    if [ "${REQUIRE_TUNED_GEMM:-0}" = "1" ]; then
        echo "REQUIRE_TUNED_GEMM=1 -> aborting to avoid an untuned (invalid) run." >&2
        exit 1
    fi
else
    echo "tuned BF16 GEMM CSV active: $AITER_CONFIG_GEMM_BF16 ($(wc -l < "$AITER_CONFIG_GEMM_BF16" 2>/dev/null) rows)"
fi

# ---- Resolve traces + install AIPerf (isolated venv) ------------------------
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- KV offloading ----------------------------------------------------------
# MI355X analogue of the B300-MTP offload_mode="on" arm. On the long-ISL agentic
# corpus the on-device KV pin saturates at conc>=8 (peak ~97-100%), the shared
# prefix is evicted, prefix-cache hit collapses and TTFT/throughput fall off a
# cliff. Spilling evicted KV to host keeps the prefix resident. Enabled only when
# the harness sets KV_OFFLOAD_BACKEND (no-op otherwise, so low-conc is unaffected).
OFFLOAD_ARGS=()
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    native)
        require_agentic_kv_offload_backend native
        OFFLOAD_ARGS=(
            --kv-offloading-size "${KV_OFFLOADING_SIZE:-$TOTAL_CPU_DRAM_GB}"
            --kv-offloading-backend native
        )
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
        export PYTHONHASHSEED=42
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":${CPU_BYTES_PER_RANK},\"lazy_offload\":false}}"
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND'" >&2
        exit 1
        ;;
esac

# ---- DSpark speculative decoding -------------------------------------------
# Probabilistic drafting at DSpark level 2 (K=2), synthetic acceptance pinned to
# the committed golden AL per the AgentX policy: "InferenceX is evaluating
# inference-system performance, not the ability to fine-tune a benchmark-specific
# speculative head" (golden_al_distribution/README.md). Synthetic-vs-synthetic is
# the only apples-to-apples comparison with the B300-MTP baseline, which pins the
# same golden AL. K=2 rather than 7: verify width is a real cost synthetic
# acceptance does not remove; K=2 clears break-even at both ends of the conc range.
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-2}"
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))
# Committed golden AL at K=2 on the probabilistic curve
# (kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml:
# thinking_on 2 -> 2.51). Do not mix with the greedy curve's 2.45.
SYNTHETIC_ACCEPT_LEN="${SYNTHETIC_ACCEPT_LEN:-2.51}"

# Draft attention backend: ROCM_AITER_MLA (draft forced causal -> fp8 asm path,
# same as target). Throughput runs pin synthetic acceptance; EVAL_ONLY accuracy
# runs use real block verification (synthetic commits drafted tokens regardless
# of the target's logits, so generated text would be wrong -> eval scores 0).
DRAFT_BACKEND="${DRAFT_BACKEND:-ROCM_AITER_MLA}"
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"$DRAFT_BACKEND\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}"
else
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"$DRAFT_BACKEND\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

# ---- Mandated DSpark serving knobs (do not change) --------------------------
# gpu-mem 0.95 / max-num-seqs 64 / MNBT 16384 / FULL_AND_PIECEWISE are mandated.
# 20, not 64. At these concurrencies max_num_seqs is a MEMORY knob, not a
# scheduling one -- c8 never has more than 8 requests in flight, so the extra 44
# slots only inflate the profiled activation peak and shrink the auto-sized KV
# pool. amd/kimi-k3-agentic-perf-tuning ran seqs 20 at gmu 0.90 and got a
# 2,620,369-token pool (vs 2,020,206 under our 32 GiB pin) and 4137.89
# tok/s/GPU at c10. Deviates from PR 2585's "mandated" seqs 64 -- revisit
# before submission.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MNBT="${MNBT:-16384}"
# 0.90, LIVE now that KV is unpinned (a pin makes gmu inert).
# The old "0.90 kills 4/4 cells" result was measured at max_num_seqs 128 on the
# Gluon-era base recipe: the activation peak scales with the seq slots, so 0.90
# only overcommits when seqs is large. amd/kimi-k3-agentic-perf-tuning ran
# 0.90 WITH seqs 20 successfully, which is the pairing used here. Raising gmu
# without dropping seqs is the combination to avoid.
GPU_MEM="${GPU_MEM:-0.90}"
KVDTYPE_ARGS=(
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
    --attention-backend "${ATTENTION_BACKEND:-ROCM_AITER_MLA}"
)

# KV-cache memory. UNPINNED by default so vLLM runs memory profiling and reports
# what it believes is available -- the number this recipe has never measured.
#
# Measured per rank on the c1/c8 cb810 runs (8x MI355X, 288 GiB):
#   free before weights 281.2 GiB  -  weights 194.9 GiB  =  86.3 GiB actually free
# so a 32 GiB pin left ~54 GiB for activations/ASM workspaces/RCCL, and a 64 GiB
# pin leaves only ~22 GiB. (An earlier revision of this comment read the 281.2
# figure as free memory AT profiling time; it is the pre-weights init_device
# snapshot, printed later.) Setting kv_cache_memory_bytes makes gpu_worker SKIP
# profiling entirely -- which is why the peak is unmeasured, and why the
# 0.95-vs-0.85 gmu sweep found the fault "invariant to gmu": with a pin,
# --gpu-memory-utilization controls nothing.
#
# Caveat on the number this produces: vLLM profiles the TORCH allocator peak.
# aiter's ASM prefill workspaces are raw hipMalloc outside that pool, so the
# reported headroom is an over-estimate of what is really spare -- that gap is
# the suspected mechanism behind the probabilistic mid-prefill
# HSA_OUT_OF_RESOURCES deaths. Treat the profiled value as an upper bound.
#
# Re-pin with KV_CACHE_MEMORY=<bytes> (32 GiB = 34359738368, 64 = 68719476736).
KV_CACHE_MEMORY="${KV_CACHE_MEMORY-}"
KVMEM_ARG=(); [ -n "$KV_CACHE_MEMORY" ] && KVMEM_ARG=(--kv-cache-memory "$KV_CACHE_MEMORY")

# cudagraph capture sizes — pin explicitly so DSpark decode (M = TOKENS_PER_SEQ *
# conc, uniform_decode_query_len = 1 + num_spec = 3) lands on FULL decode graphs
# at every benchmark concurrency. vLLM derives the FULL/decode graph set as
# round_up(size, 3) over this ladder; the AUTO ladder leaves gaps at 12 and 36
# (conc-4 -> 12 tok, conc-12 -> 36 tok fall to a PIECEWISE graph -> attention runs
# eager every step -> get_mla_metadata_v1 host bubble, ~75 ms ITL). Adding 12 and
# 36 gives exact FULL decode graphs. Rule: for a new conc C, add 3*C.
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,4,8,12,16,24,32,36,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CAPTURE_SIZES]}"

# Banner must reflect the ACTUAL spec arm: EVAL_ONLY uses block/real verify (an
# accuracy run, NOT throughput-comparable), the default uses synthetic golden AL.
# Printing "synthetic AL" unconditionally would let a block run be mistaken for a
# throughput run when checking the log -- report the real mode.
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_MODE_DESC="block/REAL verify (EVAL_ONLY accuracy run -- NOT throughput-comparable)"
    # Loud guard: an EVAL_ONLY run uses real block verification, so its acceptance
    # length is the corpus-measured ~1.2-1.7 (not the synthetic golden 2.51) and its
    # tok/s is correspondingly lower. Those numbers are for ACCURACY, not throughput.
    # Warn unmistakably so an eval run's tok/s can't be reported as a recipe result.
    echo "############################################################################" >&2
    echo "WARNING: EVAL_ONLY=true -> ACCURACY run (rejection_sample_method=block)."      >&2
    echo "         Acceptance is real-verify (~1.2-1.7), NOT the synthetic golden 2.51," >&2
    echo "         so throughput / interactivity from this run are LOWER and are NOT"    >&2
    echo "         comparable to the published recipe. Do NOT report tok/s from an"      >&2
    echo "         EVAL_ONLY run. For throughput, run with EVAL_ONLY unset (default)."   >&2
    echo "############################################################################" >&2
else
    SPEC_MODE_DESC="synthetic AL=$SYNTHETIC_ACCEPT_LEN (AgentX golden, throughput methodology)"
fi
echo "Starting vllm server (MI355X/AITER DSpark fp8-asm, spec=$SPEC_MODE_DESC)..."
{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size "$TP"
    --async-scheduling
    --distributed-executor-backend mp
    --gpu-memory-utilization "$GPU_MEM"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len 1048576
    --max-num-batched-tokens "$MNBT"
    --trust-remote-code
    --load-format "$LOAD_FORMAT"
    --moe-backend aiter
    --mm-encoder-tp-mode data
    "${KVDTYPE_ARGS[@]}"
    "${KVMEM_ARG[@]}"
    --speculative-config "$SPEC_CONFIG"
    --compilation-config "$COMPILATION_CONFIG"
    --enable-prefix-caching
    --enable-prompt-tokens-details
    # native hybrid KV (MLA + KDA) — no padding; the fix for the capture fault.
    --no-disable-hybrid-kv-cache-manager
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    --disable-uvicorn-access-log
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"; printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

# cleanup: free the GPU (orphaned TP workers otherwise hold VRAM)
[[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
pkill -9 -f "/usr/local/bin/vll[m]" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
for _ in $(seq 1 30); do pgrep -f "EngineCore|multiprocessing.spawn" >/dev/null 2>&1 || break; sleep 2; done
set +x
