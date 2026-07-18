#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
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

# Install amd-quark for MXFP4 quantization support
# need to manually install due to ROCm vLLM bug
# https://github.com/vllm-project/vllm/issues/35633
pip install amd-quark

ensure_aiter_mxfp4_moe_backend() {
  # Temporary bridge until vllm-project/vllm#48683 lands in the official ROCm
  # nightly. AITER v0.1.16.post3 has Kimi FP4 tuning rows but lacks the
  # RadeonFlow MXFP4 MoE backend files used by the best MI355X path.
  if python3 - <<'PY'
import site
from pathlib import Path
root = next(Path(p) for p in site.getsitepackages()
            if p.endswith("dist-packages"))
required = [
    root / "aiter/ops/moe_mxfp4_aux.py",
    root / "aiter/aot/flydsl/mxfp4_moe.py",
]
raise SystemExit(0 if all(p.exists() for p in required) else 1)
PY
  then
    echo "AITER MXFP4 MoE backend already present; skipping runtime AITER bump."
    return
  fi

  local aiter_branch="${AITER_RUNTIME_BUMP_VERSION:-v0.1.16.post4}"
  local aiter_version="${aiter_branch#v}"
  local aiter_repo="${AITER_RUNTIME_BUMP_REPO:-https://github.com/ROCm/aiter.git}"
  local aiter_src="/tmp/aiter-${aiter_branch}"
  local wheel_cache="${AITER_RUNTIME_WHEEL_CACHE:-/tmp/aiter-wheels}"

  mkdir -p "$wheel_cache"
  if compgen -G "$wheel_cache/amd_aiter-${aiter_version}-*.whl" >/dev/null; then
    echo "Installing cached AITER ${aiter_branch} wheel from $wheel_cache."
    python3 -m pip install flydsl==0.2.2
    python3 -m pip install --force-reinstall --no-deps "$wheel_cache"/amd_aiter-"${aiter_version}"-*.whl
    return
  fi

  echo "Building AITER ${aiter_branch} wheel for RadeonFlow MXFP4 MoE backend."
  rm -rf "$aiter_src"
  rm -rf /tmp/aiter-dist
  git clone --recursive --depth 1 --branch "$aiter_branch" "$aiter_repo" "$aiter_src"
  python3 -m pip install pyyaml flydsl==0.2.2
  (
    cd "$aiter_src"
    AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=gfx950 \
      python3 setup.py bdist_wheel --dist-dir=/tmp/aiter-dist
  )
  cp /tmp/aiter-dist/*.whl "$wheel_cache"/
  python3 -m pip install --force-reinstall --no-deps /tmp/aiter-dist/*.whl
}

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

ensure_aiter_mxfp4_moe_backend

SERVER_LOG=/workspace/server.log

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
# Disable that features to avoid crashes.
# This is related to the changes in the driver at:
# https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=true
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=true
export AITER_MXFP4_INTERMEDIATE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

if [ "${EP_SIZE:-0}" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# following AMD andy luo's recipe
# https://x.com/linluo77/status/2017024513595301985

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
--kv-cache-dtype fp8 \
--block-size=16 \
--moe-backend aiter \
--trust-remote-code \
--no-enable-prefix-caching \
--mm-encoder-tp-mode data > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
