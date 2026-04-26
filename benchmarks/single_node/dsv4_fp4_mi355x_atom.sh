#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE"

# PR1 invariants. The YAML constrains these to 1, but a manual invocation with
# different env vars would silently produce wrong output (kv_cache[:1,...]
# hardcode in deepseek_v4.py corrupts state at batch>1; expert-parallel serving
# is not validated by the PR's repro). Fail fast instead.
if [ "$CONC" -ne 1 ]; then
    echo "FATAL: ROCm/ATOM#650 PR1 is single-sequence only; CONC must be 1, got $CONC" >&2
    exit 1
fi
if [ "$EP_SIZE" -ne 1 ]; then
    echo "FATAL: ROCm/ATOM#650 PR1 has not validated expert parallel serving; EP_SIZE must be 1, got $EP_SIZE" >&2
    exit 1
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export OMP_NUM_THREADS=1

# DSv4-specific ATOM env vars (from ROCm/ATOM#650 repro command).
# The aiter fused_moe path is broken on gfx950 with a16w4+Swiglu, so PR1
# requires the triton matmul_ogs path. AITER_LOG_LEVEL quiets the noisy
# warmup logs that otherwise drown out the server-ready signal.
export ATOM_USE_TRITON_MOE=1
export AITER_LOG_LEVEL=WARNING

# Apply ROCm/ATOM#650 (DSv4 PR1 skeleton) over the image's wheel-installed
# atom. The chosen base image ships atom as a built wheel, not editable, so
# we overlay an editable install from the PR branch at a pinned SHA. Bump
# this SHA when the PR moves; do not track the branch tip (the run becomes
# a moving target if the branch is force-pushed).
ATOM_PR_SHA="cdbff359d3db7afd3801e28b38fc71253121ee84"
export ATOM_PR_DIR="/tmp/atom-pr650"

if [ ! -d "$ATOM_PR_DIR/.git" ]; then
    git clone --filter=blob:none https://github.com/ROCm/ATOM.git "$ATOM_PR_DIR"
fi
(
    cd "$ATOM_PR_DIR"
    # Try a targeted fetch first (fast); fall back to fetching the PR ref if
    # the server doesn't allow fetching the SHA directly.
    git fetch --depth=1 origin "$ATOM_PR_SHA" 2>/dev/null \
        || git fetch --depth=1 origin pull/650/head
    git checkout --force "$ATOM_PR_SHA"
    test "$(git rev-parse HEAD)" = "$ATOM_PR_SHA"
    # --no-deps: don't churn the image's pinned ROCm/torch/triton/aiter.
    # --force-reinstall: replace the wheel-installed atom with the editable copy.
    pip install --no-deps --force-reinstall -e .
)

# PR #650's repro explicitly reinstalls triton_kernels editable. Conditional
# in case the path differs in the chosen image; safe no-op if already present.
if [ -d /triton-test/python/triton_kernels/ ]; then
    pip install --no-deps -e /triton-test/python/triton_kernels/
fi

# Preflight version checks. The chosen base image
# (atom0.1.2.post, rebuilt 2026-04-23) was tagged after ATOM pinned
# transformers==5.2.0 (commit 67d6cb61, 2026-03-13), so transformers compat
# is expected; we still assert it explicitly to fail fast with a clear
# message rather than timing out wait_for_server_ready on a confusing
# import error inside the server log. The two non-trivial deps the PR
# introduces are transformers' deepseek_v3 config class (mapped from
# deepseek_v4 in atom/config.py) and triton_kernels.CDNA4MXScaleLayout
# (renamed from GFX950MXScaleLayout in fused_moe_triton.py).
python3 - <<'PYEOF'
import importlib, os, sys
import atom

# Verify the editable install actually took effect — Python could still be
# importing the wheel-installed atom if pip's --force-reinstall silently no-op'd
# (e.g., the wheel and the editable copy share a setup.py path mismatch).
atom_path = os.path.abspath(atom.__file__)
expected = os.path.abspath(os.environ["ATOM_PR_DIR"])
print(f"atom imported from: {atom_path}")
if expected not in atom_path:
    sys.exit(f"FATAL: atom is importing from {atom_path}, not from PR checkout {expected}. "
             f"The pip --force-reinstall -e . did not take effect.")

import transformers
print(f"transformers version: {transformers.__version__}")

# Use CONFIG_MAPPING directly: AutoConfig.for_model() returns an instance
# (transformers 5.2.0 source: `return config_class(*args, **kwargs)`), not a
# class, so `.__name__` would AttributeError. CONFIG_MAPPING maps model_type
# to the config class directly and is unambiguous.
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "deepseek_v3" not in CONFIG_MAPPING:
    sys.exit(f"FATAL: transformers in this image cannot resolve deepseek_v3 model_type. "
             f"ATOM PR #650 maps deepseek_v4 -> deepseek_v3 in _CONFIG_REGISTRY and needs "
             f"transformers to know the v3 schema. Available types: "
             f"{sorted(k for k in CONFIG_MAPPING if 'deepseek' in k)}")
print(f"deepseek_v3 config class: {CONFIG_MAPPING['deepseek_v3'].__name__}")

try:
    layout_mod = importlib.import_module("triton_kernels.tensor_details.layout")
    if not hasattr(layout_mod, "CDNA4MXScaleLayout"):
        avail = [n for n in dir(layout_mod) if "Layout" in n]
        sys.exit(f"FATAL: triton_kernels.tensor_details.layout has no CDNA4MXScaleLayout. "
                 f"PR #650's fused_moe_triton.py change renamed GFX950MXScaleLayout -> "
                 f"CDNA4MXScaleLayout, but this image's triton_kernels still uses the old "
                 f"name. Available Layout classes: {avail}")
    print("triton_kernels.CDNA4MXScaleLayout: present")
except ModuleNotFoundError as e:
    sys.exit(f"FATAL: triton_kernels not importable. PR #650's MoE path needs it. Error: {e}")
PYEOF

# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=""
else
    CALCULATED_MAX_MODEL_LEN=" --max-model-len 10240 "
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CALCULATED_MAX_MODEL_LEN=" --max-model-len $EVAL_MAX_MODEL_LEN "
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x

BLOCK_SIZE=${BLOCK_SIZE:-16}
# --enforce-eager is required: ROCm/ATOM#650 (PR1 skeleton) has no CUDAGraph
# support yet (deferred to a follow-up PR). --max-num-seqs 1 caps the path
# at the single-sequence ceiling that PR1 supports — the model_runner has a
# hardcoded kv_cache[:1,...] that silently corrupts state for batch>1.
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    -tp $TP \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --max-num-seqs 1 > $SERVER_LOG 2>&1 &

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
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
