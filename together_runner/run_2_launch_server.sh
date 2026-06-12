#!/usr/bin/env bash
# Step 2/3 — Launch the inference server ($ENGINE) inside the container, with
# staged progress monitoring, zero-download when weights are pre-staged, and a
# tuning switch (sglang only).
#   bash run_2_launch_server.sh
#   SMOKE=1 bash run_2_launch_server.sh   # forces fast path (sglang tuning off)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/bench_lib.sh"

check_env_vars ENGINE CONTAINER_NAME PORT PROFILE MODEL TP || exit 1
SMOKE="${SMOKE:-0}"
SERVER_LOG="/root/server.log"

# SMOKE (gate-2) always takes the fast path regardless of config.
[[ "$SMOKE" == "1" ]] && ENABLE_TUNING=0

# --- Resolve model source: prefer pre-staged local weights (zero download) ---
STAGED="$MODELS_ROOT/$PROFILE"
if [[ -n "${MODEL_PATH:-}" && -d "$MODEL_PATH" ]]; then
    SRC="$MODEL_PATH"; trlog "using MODEL_PATH=$SRC (skip download)"
elif [[ -f "$STAGED/.ready" ]]; then
    SRC="$STAGED"; trlog "using pre-staged weights $SRC (.ready present, skip download)"
elif [[ "$MODEL" == /* ]]; then
    SRC="$MODEL"; trlog "MODEL is a local path $SRC (skip download)"
else
    SRC="$MODEL"
    trwarn "weights NOT pre-staged — server will download $MODEL on first start (slow)."
    trwarn "run prestage_weights.sh first to avoid this."
fi

# max-model-len from ISL/OSL (mirrors gptoss_fp4_b200.sh).
calc_max_model_len() {
  if [[ "$ISL" == 1024 && "$OSL" == 1024 ]]; then echo $((ISL + OSL + 20))
  elif [[ "$ISL" == 8192 || "$OSL" == 8192 ]]; then echo $((ISL + OSL + 256))
  else echo "${MAX_MODEL_LEN:-10240}"; fi
}

# --- build launch command (INNER) + startup plan per (ENGINE, PROFILE) ---
PRELAUNCH=""   # optional in-container command run before the server (e.g. write config)
case "$ENGINE" in
  sglang)
    TUNE_ARGS=()
    if [[ "$ENABLE_TUNING" == "1" ]]; then
        trlog "ENABLE_TUNING=1 — autotune KEPT (slower start, peak perf; baselines)"
        STAGE_PLAN="engine-init weight-load autotune graph-capture"
    else
        TUNE_ARGS+=(--disable-flashinfer-autotune)
        trlog "ENABLE_TUNING=0 — autotune SKIPPED (--disable-flashinfer-autotune)"
        STAGE_PLAN="engine-init weight-load graph-capture"
    fi
    case "$PROFILE" in
      smoke)
        ARGS=(--model-path "$SRC" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT"
              --tensor-parallel-size "$TP" --trust-remote-code "${TUNE_ARGS[@]}") ;;
      dsr1-fp4)
        check_env_vars EP_SIZE || exit 1
        SCHED_RECV=10; (( CONC >= 16 )) && SCHED_RECV=30
        ARGS=(--model-path "$SRC" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT" --trust-remote-code
              --tensor-parallel-size="$TP" --data-parallel-size=1
              --cuda-graph-max-bs 256 --max-running-requests 256 --mem-fraction-static 0.85
              --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 16384
              --ep-size "$EP_SIZE" --quantization modelopt_fp4
              --enable-flashinfer-allreduce-fusion --scheduler-recv-interval "$SCHED_RECV"
              --enable-symm-mem --disable-radix-cache --attention-backend trtllm_mla
              --moe-runner-backend flashinfer_trtllm --stream-interval 10 "${TUNE_ARGS[@]}") ;;
      *) trerr "unknown sglang PROFILE='$PROFILE' (smoke|dsr1-fp4)"; exit 1 ;;
    esac
    INNER="PYTHONNOUSERSITE=1 python3 -m sglang.launch_server $(printf '%q ' "${ARGS[@]}")"
    ;;

  vllm)
    [[ "$ENABLE_TUNING" == "1" ]] && trlog "note: ENABLE_TUNING is a label only for vLLM (no --disable flag); FlashInfer MoE still autotunes"
    # gptoss-fp4 autotunes the FlashInfer MoE at startup (VLLM_USE_FLASHINFER_MOE_*);
    # smoke (small dense model) has no MoE autotune. Include it for gptoss only.
    if [[ "$PROFILE" == "gptoss-fp4" ]]; then
        STAGE_PLAN="engine-init weight-load autotune graph-capture"
    else
        STAGE_PLAN="engine-init weight-load graph-capture"
    fi
    VENV="VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 TORCH_CUDA_ARCH_LIST=10.0 PYTHONNOUSERSITE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600"
    # OOM guard (KLAUD_DEBUG §2): VLLM_OOM_GUARD=1 disables the cudagraph mem profiler.
    [[ "${VLLM_OOM_GUARD:-0}" == "1" ]] && VENV="$VENV VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
    case "$PROFILE" in
      gptoss-fp4)
        MML=$(calc_max_model_len)
        PRELAUNCH=$(cat <<PRE
cat > /root/vllm_config.yaml <<'YAML'
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true}}'
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: $MML
YAML
PRE
)
        ARGS=(serve "$SRC" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT"
              --config /root/vllm_config.yaml
              --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}"
              --tensor-parallel-size "$TP" --max-num-seqs 512 --trust-remote-code) ;;
      smoke)
        ARGS=(serve "$SRC" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT"
              --tensor-parallel-size "$TP" --trust-remote-code) ;;
      *) trerr "unknown vllm PROFILE='$PROFILE' (gptoss-fp4|smoke)"; exit 1 ;;
    esac
    INNER="$VENV vllm $(printf '%q ' "${ARGS[@]}")"
    ;;

  *) trerr "unknown ENGINE='$ENGINE' (sglang|vllm)"; exit 1 ;;
esac

trlog "engine=$ENGINE profile=$PROFILE model_src=$SRC tp=$TP port=$PORT smoke=$SMOKE"

run_timer_start
# Optional pre-launch (e.g. write vLLM config.yaml), then launch detached.
[[ -n "$PRELAUNCH" ]] && docker exec "$CONTAINER_NAME" bash -c "$PRELAUNCH"
docker exec -d "$CONTAINER_NAME" bash -c "$INNER > $SERVER_LOG 2>&1"

# Staged startup monitor (engine-aware markers; readiness via /health).
if monitor_server_until_ready "$CONTAINER_NAME" "$PORT" "$SERVER_LOG" 3600; then
    trlog "server READY in $(_fmt_dur "$(run_elapsed)") — next: bash run_3_test_client.sh"
else
    trerr "server failed to start (see log dump above)."
    exit 1
fi
