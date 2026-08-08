#!/usr/bin/env bash
set -eo pipefail

# Fast A/B probe for DeepSeek-V4-Pro FP4 on MI355X via vLLM + MTP.
#
# WHY THIS EXISTS
# ---------------
# dsv4_fp4_mi355x_vllm_mtp.sh is the reportable recipe, but one arm of it is a
# ~30 min agentic trace replay (aiperf, 393 Weka trajectories, 1800s profiling
# phase) on top of a ~15 min engine start. That is the right harness for a
# submission and the wrong one for "did this flag help?".
#
# This script keeps the SERVER side byte-for-byte comparable to the agentic
# recipe -- same parallelism construction, same compilation config, same
# speculative config, same AITER env -- and swaps only the CLIENT side for a
# short fixed-seq-len sweep (utils/bench_serving, random dataset). That is the
# same driver the fixed_seq_len/dsv4_fp4_mi355x_vllm_mtp.sh sibling uses.
#
# WHAT 10/10 DOES AND DOES NOT MEASURE
# ------------------------------------
# ISL=10 / OSL=10 is a dispatch-overhead probe, not a workload. With 10 output
# tokens and num_speculative_tokens=3 a request is ~3 decode steps, so the
# measurement is dominated by TTFT, scheduler ramp and per-step launch cost --
# exactly the regime where the MI355X ~4 us kernel dispatch floor shows up, and
# exactly what the RoPE+KV fusion and cudagraph knobs move. It is deliberately
# NOT sensitive to: prefix cache behaviour, KV capacity, long-prefill chunking,
# or anything the agentic 24:1 prefill:decode mix stresses. Do not read an e2e
# throughput conclusion off this script -- read a "did the per-step cost move"
# conclusion off it, then confirm on the real recipe.
#
# VARIANTS
# --------
#   VARIANT=srok    PR #2381 head as authored (seungrokj): gpu-mem-util 0.90,
#                   no RoPE+KV fusion, pure TP8. Runs on the image that PR
#                   pins in amd-master.yaml.
#   VARIANT=mine    this branch's changes: gpu-mem-util 0.86 (derived for the
#                   kv-offloading:none arms), RoPE+KV fusion on.
#   VARIANT=dep8    mine + DP-attention + EP8 (the arm amd-master.yaml adds),
#                   fronted by vllm-router on round_robin. The policy is not
#                   incidental -- it is worth 2.9x here. See the Router block.
#                   Needs the vllm-router, same as the agentic recipe.
#   VARIANT=custom  touch nothing; every knob comes from the environment.
# Any individual knob set in the environment wins over the variant preset.
#
# IMAGES ARE NOT THE SAME ACROSS VARIANTS
# ---------------------------------------
# srok runs on the vLLM nightly its PR pins; mine/dep8 run on a build carrying
# the AITER 4269 shared-expert work. Each variant reports the image it ran on
# (VARIANT_IMAGE below) and the launcher script selects it. This means an
# srok-vs-mine delta is engine-build + flags together, NOT flags alone -- so
# for a clean flag-only attribution also run both on one image:
#     VARIANT=srok IMAGE=<mine's image> ...
# The three-way with per-variant images answers "which configuration is
# fastest today"; the same-image pair answers "which change caused it".
#
# USAGE
#   MODEL_PATH=/mnt/models/DeepSeek-V4-Pro VARIANT=mine CONC=16 \
#     ./dsv4_fp4_mi355x_vllm_mtp_quick.sh
#
# This is a developer probe, not a CI recipe: it is intentionally NOT wired
# into configs/amd-master.yaml.

# benchmark_lib.sh inspects BASH_SOURCE[1] and treats any caller under an
# */agentic/* path as a trace replay. This file lives next to the agentic
# recipe it probes, so it trips that check even though it drives a fixed
# seq-len sweep. Two consequences, both handled here rather than by moving the
# file (keeping it beside its sibling is the point):
#   1. The lib hard-requires KV_OFFLOADING. This probe is GPU-resident by
#      construction, so declare it explicitly.
#   2. The lib unsets MAX_MODEL_LEN ("agentic replays must use the model's
#      native context limit"). Correct for a replay, wrong for a 10/10 probe
#      where a 1M context would cost minutes of startup for no measurement.
#      Stash any caller-supplied value across the source and restore it below.
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
_QUICK_MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

source "$(dirname "$0")/../../benchmark_lib.sh"

if [[ -n "$_QUICK_MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN="$_QUICK_MAX_MODEL_LEN"
fi
unset _QUICK_MAX_MODEL_LEN

# ---- Model resolution --------------------------------------------------------
# Prefer an already-materialised local checkpoint; only fall back to a download.
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d /mnt/models/DeepSeek-V4-Pro ]]; then
        MODEL_PATH=/mnt/models/DeepSeek-V4-Pro
    else
        hf download "$MODEL"
        MODEL_PATH="$MODEL"
    fi
fi
export MODEL_PATH

# ---- Variant presets ---------------------------------------------------------
# The per-variant container image. This script runs INSIDE the container, so it
# only records which image a variant expects; the launcher
# (dsv4_quick_ab_g05.sh) is what actually selects it. Recording it here keeps
# the mapping in one place and stamps it into the result dir.
#   srok    -> the nightly PR #2381 pins in amd-master.yaml
#   others  -> the AITER 4269 shared-expert build
SROK_IMAGE="${SROK_IMAGE:-vllm/vllm-openai-rocm:nightly-b88916617d3d2249bff0dae5cecb6b727c980a20}"
OPTIMAL_IMAGE="${OPTIMAL_IMAGE:-jiahcao/vllm-dsv4:optimal-fse-aiter4269}"

VARIANT="${VARIANT:-mine}"
case "$VARIANT" in
    srok)
        : "${GPU_MEM_UTIL:=0.90}"
        : "${ROCM_ROPE_KVCACHE_FUSION:=0}"
        : "${DP_ATTENTION:=false}"
        : "${EP_SIZE:=1}"
        : "${VARIANT_IMAGE:=$SROK_IMAGE}"
        ;;
    mine)
        : "${GPU_MEM_UTIL:=0.86}"
        : "${ROCM_ROPE_KVCACHE_FUSION:=1}"
        : "${DP_ATTENTION:=false}"
        : "${EP_SIZE:=1}"
        : "${VARIANT_IMAGE:=$OPTIMAL_IMAGE}"
        ;;
    dep8)
        : "${GPU_MEM_UTIL:=0.86}"
        : "${ROCM_ROPE_KVCACHE_FUSION:=1}"
        : "${DP_ATTENTION:=true}"
        : "${EP_SIZE:=8}"
        : "${VARIANT_IMAGE:=$OPTIMAL_IMAGE}"
        ;;
    custom)
        : "${GPU_MEM_UTIL:=0.86}"
        : "${ROCM_ROPE_KVCACHE_FUSION:=0}"
        : "${DP_ATTENTION:=false}"
        : "${EP_SIZE:=1}"
        : "${VARIANT_IMAGE:=$OPTIMAL_IMAGE}"
        ;;
    *)
        echo "Error: unknown VARIANT '$VARIANT' (expected: srok, mine, dep8, custom)" >&2
        exit 1
        ;;
esac

# aiter is the default everywhere; flydsl_mega_moe is the fused alternative
# (EP dispatch + both expert GEMMs + EP combine in one launch chain). It hard-
# requires TP=1 + expert parallel + fp4 experts, so it is only selectable on the
# dep8 variant -- the check below rejects it on the TP8 arms rather than letting
# vLLM raise NotImplementedError nine minutes into weight load.
MOE_BACKEND="${MOE_BACKEND:-aiter}"
if [ "$MOE_BACKEND" = "flydsl_mega_moe" ] && { [ "$DP_ATTENTION" != "true" ] || [ "$EP_SIZE" -le 1 ]; }; then
    echo "Error: MOE_BACKEND=flydsl_mega_moe requires DP_ATTENTION=true and EP_SIZE>1 (use VARIANT=dep8)." >&2
    exit 1
fi

TP="${TP:-8}"
CONC="${CONC:-16}"
ISL="${ISL:-10}"
OSL="${OSL:-10}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"
# 10/10 needs almost nothing, but the DSv4 chat template that --dsv4 applies
# adds a system preamble on top of the 10 sampled tokens. 8192 is far above
# that and keeps engine init fast: max_model_len drives both the KV block
# count and the cudagraph capture work, and the agentic recipe's 1M context
# costs minutes of startup we do not need for a per-step probe.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
NUM_PROMPTS="${NUM_PROMPTS:-$((CONC * 10))}"
PORT="${PORT:-8888}"
# run_benchmark_serving resolves utils/bench_serving/benchmark_serving.py under
# this dir. benchmark_lib.sh defaults INFMAX_CONTAINER_WORKSPACE to /workspace,
# which only exists inside the CI container; derive the repo root instead so a
# bare-metal run finds the driver. An explicit BENCH_SERVING_DIR still wins.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "${BENCH_SERVING_DIR:-}" ]]; then
    if [[ -f "${INFMAX_CONTAINER_WORKSPACE:-}/utils/bench_serving/benchmark_serving.py" ]]; then
        BENCH_SERVING_DIR="$INFMAX_CONTAINER_WORKSPACE"
    else
        BENCH_SERVING_DIR="$_REPO_ROOT"
    fi
fi
if [[ ! -f "$BENCH_SERVING_DIR/utils/bench_serving/benchmark_serving.py" ]]; then
    echo "Error: benchmark_serving.py not found under BENCH_SERVING_DIR=$BENCH_SERVING_DIR" >&2
    exit 1
fi

RESULT_DIR="${RESULT_DIR:-$PWD/dsv4_quick_${VARIANT}_c${CONC}}"
RESULT_FILENAME="${RESULT_FILENAME:-dsv4_quick_${VARIANT}_tp${TP}_c${CONC}_isl${ISL}_osl${OSL}}"
mkdir -p "$RESULT_DIR"

# run_benchmark_serving and run_eval both branch on these; give them values so
# the script works standalone, outside the CI harness that normally exports them.
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# ---- AITER env ---------------------------------------------------------------
# Kept identical to the agentic recipe so a delta measured here transfers.
#
# VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is NOT optional for this
# checkpoint: DSv4-Pro ships MXFP4 routed experts alongside FP8 shared experts,
# and without the fusion the shared-expert weights fail their shape assertion
# at load. The validated manual runs all exported it; the checked-in agentic
# script did not, which is a bug fixed alongside this file.
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-1}"
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# ---- Parallelism -------------------------------------------------------------
# Same construction as the agentic recipe.
PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi
if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

MODE_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    MODE_ARGS+=(--enable-ep-weight-filter)
fi
if [ "$DP_ATTENTION" = "true" ]; then
    MODE_ARGS+=(--prefill-schedule-interval 8)
fi

if [ "$DP_ATTENTION" = "true" ]; then
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-$CONC}"
else
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-$((2 * CONC))}"
fi
[ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1

# ---- Router (DP-attention only) ----------------------------------------------
USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
ROUTER_PID=""
if [ "$DP_ATTENTION" = "true" ]; then
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    agentic_pip_install --quiet "vllm-router==${VLLM_ROUTER_VERSION:-0.1.14}"

    # round_robin, not the consistent_hash the other agentic recipes hardcode.
    #
    # The DP ranks run the MoE layers in lockstep: each step every rank pads its
    # batch out to the busiest rank's token count, so a step costs whatever the
    # fullest rank costs and the idle ranks burn their slice on dummy tokens.
    # Balance is therefore a throughput input, not just a latency one.
    #
    # consistent_hash spreads by request content, which for short requests is
    # uneven enough to leave most ranks padding. Measured on this probe at c48
    # (10/10, MI355X, DSv4-Pro FP4 + MTP) -- only the policy differs:
    #
    #   TP8 baseline              543 tok/s   TPOT 43.5ms
    #   DEP + consistent_hash     261 tok/s   TPOT 94.6ms
    #   DEP + round_robin         751 tok/s   TPOT 23.5ms
    #
    # The tell that this was maldistribution rather than an underfed engine:
    # the DEP deficit held at a near-constant 2.2-2.4x across c16 and c48
    # instead of shrinking as concurrency grew.
    #
    # Scope: this default is right for THIS probe -- fixed short random prompts
    # with prefix caching off, where hashing buys nothing. consistent_hash earns
    # its keep on session affinity and prefix-cache hits, so on a real agentic
    # trace (long, multi-turn, prefix-reusing) the ranking may well invert, and
    # cache_aware (vllm-router's own default) is the third option to measure.
    # Do not propagate this to the full agentic recipes on this evidence alone.
    # Override with VLLM_ROUTER_POLICY=<random|round_robin|cache_aware|
    # power_of_two|consistent_hash>.
    : "${VLLM_ROUTER_POLICY:=round_robin}"
fi

# Preflight the ports. These nodes are shared and --network host means a
# neighbour's server on the same port kills ours -- but only after the ~9min
# weight load and graph capture, since the bind happens last. Checking up front
# turns a 9-minute waste into an immediate, self-explaining failure. The
# DP-attention path needs the pair (router on PORT, backend on PORT+1).
for _p in "$PORT" $([ "$USE_VLLM_ROUTER" = true ] && echo "$VLLM_BACKEND_PORT"); do
    if ss -ltn 2>/dev/null | grep -q ":$_p "; then
        echo "Error: port $_p is already in use on this node." >&2
        echo "       Pick a free pair with PORT=<n> (DP-attention also uses n+1)." >&2
        exit 1
    fi
done
unset _p

# ---- Compilation -------------------------------------------------------------
# See the long note in dsv4_fp4_mi355x_vllm_mtp.sh: the gate on ROCm's
# fuse_rope_kvcache / fuse_rope_kvcache_cat_mla is splitting_ops, not the
# custom-op list, so an explicit empty splitting_ops is what turns them on.
# FULL_DECODE_ONLY survives the empty list (the downgrade branch only rewrites
# PIECEWISE and FULL_AND_PIECEWISE).
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}"
if [ "${ROCM_ROPE_KVCACHE_FUSION:-0}" = "1" ]; then
    COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"splitting_ops\":[],\"pass_config\":{\"fuse_rope_kvcache\":true,\"fuse_rope_kvcache_cat_mla\":true}}"
fi

# ---- Speculative config ------------------------------------------------------
# Synthetic acceptance is the default so an A/B measures engine cost with
# acceptance-rate variance removed; 2.49 is the dsv4-pro golden AL used by the
# reportable recipe. Set MTP_SYNTHETIC=0 for real target verification when the
# question is about acceptance itself rather than per-step cost.
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
if [ "${MTP_SYNTHETIC:-1}" = "1" ]; then
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": 2.49}"
else
    SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}"
fi

# ---- EPLB --------------------------------------------------------------------
# Off by default: it only does anything under expert parallelism, and it needs
# a patched DSv4 amd/model.py (the stock one does not implement the
# MixtureOfExperts protocol, so the runner never registers the model and EPLB
# is a silent no-op -- not an error).
#
# Both MoE backends are supported: aiter via FusedMoE's own EPLB hooks,
# flydsl_mega_moe via the ones the patch adds to DeepseekV4MegaMoEExperts.
#
# NUM_REDUNDANT_EXPERTS must keep (n_routed_experts + N) % ep_size == 0, i.e. a
# multiple of 8 for this 384-expert checkpoint on 8 ranks. The patched model
# raises on a bad value before the ~9min weight load rather than letting it
# assert inside rebalance_execute much later.
EPLB_ARGS=()
if [ "${ENABLE_EPLB:-0}" = "1" ]; then
    if [ "$EP_SIZE" -le 1 ]; then
        echo "Error: ENABLE_EPLB=1 needs expert parallelism (use VARIANT=dep8)." >&2
        exit 1
    fi
    NUM_REDUNDANT_EXPERTS="${NUM_REDUNDANT_EXPERTS:-8}"
    EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-3000}"
    EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-1000}"
    EPLB_ARGS=(
        --enable-eplb
        --eplb-config "{\"num_redundant_experts\": $NUM_REDUNDANT_EXPERTS, \"step_interval\": $EPLB_STEP_INTERVAL, \"window_size\": $EPLB_WINDOW_SIZE, \"log_balancedness\": true, \"log_balancedness_interval\": ${EPLB_LOG_INTERVAL:-100}}"
    )
fi

# ---- Prefix caching ----------------------------------------------------------
# Off by default, matching the fixed_seq_len siblings. With --dsv4 every prompt
# shares a chat preamble, so leaving it on would let run 2 of an A/B start from
# a warmer cache than run 1 and quietly bias the comparison.
PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
if [ "${ENABLE_PREFIX_CACHING:-0}" = "1" ]; then
    PREFIX_CACHE_ARGS=(--enable-prefix-caching)
fi

# ---- Launch ------------------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
SERVER_PID=""

cleanup_quick() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$ROUTER_PID" "vLLM router"
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_gpu_monitor
    exit "$exit_code"
}
trap cleanup_quick EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Replaces the agentic recipe's unconditional `sleep 180`: poll until a prior
# job's VRAM is actually released instead of guessing at 3 minutes.
wait_for_amd_gpu_clean || exit 1
rocm-smi
start_gpu_monitor --interval 1 --output "$RESULT_DIR/gpu_metrics.csv"

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --trust-remote-code
    --async-scheduling
    --distributed-executor-backend mp
    --kv-cache-dtype fp8
    --block-size 256
    --max-model-len "$MAX_MODEL_LEN"
    "${PARALLEL_ARGS[@]}"
    "${MODE_ARGS[@]}"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --moe-backend "$MOE_BACKEND"
    --compilation-config "$COMPILATION_CONFIG"
    --speculative-config "$SPEC_CONFIG"
    --tokenizer-mode deepseek_v4
    --reasoning-parser deepseek_v4
    "${PREFIX_CACHE_ARGS[@]}"
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs "$MAX_NUM_SEQS"
    "${EPLB_ARGS[@]}"
)

{
    echo "VARIANT=$VARIANT TP=$TP CONC=$CONC ISL=$ISL OSL=$OSL"
    echo "GPU_MEM_UTIL=$GPU_MEM_UTIL ROCM_ROPE_KVCACHE_FUSION=${ROCM_ROPE_KVCACHE_FUSION} DP_ATTENTION=$DP_ATTENTION EP_SIZE=$EP_SIZE"
    echo "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=$VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"
    echo "MTP_SYNTHETIC=${MTP_SYNTHETIC:-1} NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS"
    echo "MOE_BACKEND=$MOE_BACKEND"
    echo "MAX_NUM_SEQS=$MAX_NUM_SEQS"
    echo "ENABLE_EPLB=${ENABLE_EPLB:-0}${ENABLE_EPLB:+ NUM_REDUNDANT_EXPERTS=${NUM_REDUNDANT_EXPERTS:-} EPLB_STEP_INTERVAL=${EPLB_STEP_INTERVAL:-}}"
    # Router policy decides how requests spread over the DP ranks, and DP runs
    # the MoE layers in lockstep -- every rank pads to the busiest one's token
    # count each step. An uneven spread is therefore a throughput cost, not just
    # a latency one, so the policy belongs in the run record.
    if [ "$USE_VLLM_ROUTER" = "true" ]; then
        echo "VLLM_ROUTER_POLICY=$VLLM_ROUTER_POLICY"
    fi
    echo "VARIANT_IMAGE=$VARIANT_IMAGE"
    # What is actually running, independent of what the variant expected: the
    # launcher can be pointed at any image, and a mismatch between these two
    # lines is the first thing to check when an A/B looks wrong.
    echo "RUNNING_IMAGE=${QUICK_RUNNING_IMAGE:-unknown}"
    echo "VLLM_VERSION=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
    printf '%q ' "${VLLM_CMD[@]}"
    printf '\n'
} | tee "$RESULT_DIR/vllm_command.txt"

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

# --dsv4 routes prompts through DeepSeek-V4 chat encoding (auto-enables
# --use-chat-template); MTP acceptance is trained on chat-formatted input and
# silently regresses on raw random tokens.
#
# --tokenizer is not redundant with --model. benchmark_serving.py falls back to
# args.model for the tokenizer id (tokenizer_id = args.tokenizer or args.model),
# and --model here is the *served* name -- an HF repo id. With HF_HUB_OFFLINE=1
# and no Hub cache the deepseek_v4 loader cannot resolve it and dies with a
# misleading "you need sentencepiece or tiktoken installed". Both are in fact
# installed; the id simply was not on disk. Point it at the mounted checkpoint,
# which carries tokenizer.json and encoding/.
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$RESULT_DIR" \
    --bench-serving-dir "$BENCH_SERVING_DIR" \
    --tokenizer "$MODEL_PATH" \
    --tokenizer-mode deepseek_v4 \
    --trust-remote-code \
    --dsv4

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

echo "Results: $RESULT_DIR/$RESULT_FILENAME.json"
