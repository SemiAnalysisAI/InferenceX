#!/usr/bin/env bash

# [DO NOT MERGE — experimental] MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM
# recipe that validates vllm-project/vllm#45639 ("[ROCm][M3] Enable AITER AR +
# Gemma-RMS fusion for MiniMax-M3") on real MI355X hardware before an image
# rebuild. It applies #45639 in-place to the shipped vllm/vllm-openai-rocm:minimax-m3
# image, then serves with the AITER fused all-reduce + RMSNorm path enabled.
#
# Mirrors minimaxm3_fp8_mi355x.sh otherwise (--block-size 128, --language-model-only,
# TRITON_ATTN, fp8 KV on gfx950). The #45639-specific knobs:
#   VLLM_ROCM_USE_AITER=1                                  (AITER kernels)
#   --compilation-config custom_ops=["-minimax_gemma_rms_norm"]  (allow IR lowering)
#   --compilation-config pass_config.fuse_allreduce_rms=true     (the fusion pass)
# The fusion needs TP>1; this recipe is swept at TP8.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# ---- Apply vllm-project/vllm#45639 in-place -------------------------------
# The shipped minimax-m3 image predates #45639 (base m3_release). Apply the
# vendored diff to the installed vllm. Idempotent: if it is already applied
# (reverse-applies cleanly) we proceed; if it neither applies cleanly nor is
# already applied, the image has drifted from the PR base — hard-fail so we never
# silently benchmark an unpatched server.
PATCH_FILE="$(cd "$(dirname "$0")/patches" && pwd)/vllm-45639-aiter-ar-gemma-rms.diff"
command -v patch >/dev/null 2>&1 || { apt-get update -q -y && apt-get install -q -y patch; }
VLLM_SP="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
if ( cd "$VLLM_SP" && patch -p1 -R --dry-run < "$PATCH_FILE" >/dev/null 2>&1 ); then
    echo "[vllm#45639] already applied to $VLLM_SP/vllm"
elif ( cd "$VLLM_SP" && patch -p1 --dry-run < "$PATCH_FILE" >/dev/null 2>&1 ); then
    ( cd "$VLLM_SP" && patch -p1 < "$PATCH_FILE" )
    echo "[vllm#45639] applied to $VLLM_SP/vllm"
else
    echo "FATAL: vllm#45639 patch neither applies cleanly nor is already applied" >&2
    echo "       ($VLLM_SP/vllm has drifted from the PR's m3_release base)" >&2
    exit 1
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
# #45639: AITER fused all-reduce + Gemma-RMSNorm.
export VLLM_ROCM_USE_AITER=1
# DEBUG so the server log carries the fusion-pass match/replace counts
# ("RocmAiterAllReduceFusionPass Replaced N patterns", "fusion pass matches: {}")
# in addition to the (default-level) registration bail warnings.
export VLLM_LOGGING_LEVEL=DEBUG

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --compilation-config '{"custom_ops": ["-minimax_gemma_rms_norm"], "pass_config": {"fuse_allreduce_rms": true}}' \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- #45639 AITER AR + Gemma-RMS fusion diagnostics (definitive) ----------
# Engine init (incl. torch.compile fusion passes) has finished by now, so the
# fusion-pass logging is in the server log. Two questions, answered from the log:
#   1) Did the pass REGISTER? Any of these warning_once strings => it registered
#      ZERO patterns (match count is 0 by construction):
#        "AllReduce fusion pass is disabled", "AITER allreduce fusion must be
#        initialized", "AITER allreduce-rmsnorm fusion disabled: aiter<0.1.12"
#        (the M3/6144 one), "Custom Allreduce is required".
#   2) Did it MATCH+REPLACE? "RocmAiterAllReduceFusionPass Replaced N patterns"
#      (N>0 => matched & replaced; N==0 => matched nothing) and the per-pass
#      "fusion pass matches: {...}" table.
set +x
echo "================ #45639 fusion-pass verdict ================"
echo "--- [1] registration bail warnings (presence => registered 0 patterns) ---"
grep -nE "AllReduce fusion pass is disabled|AITER allreduce fusion must be initialized|AITER allreduce-rmsnorm fusion disabled|Custom Allreduce is required" "$SERVER_LOG" \
    || echo "  (none — no registration bail)"
echo "--- [2] match / replace counts ---"
grep -nE "RocmAiterAllReduceFusionPass Replaced [0-9]+ patterns|fusion pass matches:" "$SERVER_LOG" \
    || echo "  (no 'Replaced N patterns' / 'fusion pass matches' line found)"
echo "==========================================================="
set -x

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
